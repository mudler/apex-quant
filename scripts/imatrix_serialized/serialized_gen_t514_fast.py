"""Band-serialized torch-imatrix generator (transformers-5.14) — throughput-optimized.

Numerically equivalent to serialized_gen_t514.py (Sum x^2 is order/batch-invariant),
but restructured so the GPU stays busy instead of stalling on host syncs:

  1. GPU-resident accumulators. sums/counts live on-GPU (float64) and are copied
     to host exactly once, after the last band — vs the reference tool's
     ~1M per-expert `.cpu().numpy()` calls (64 experts x 40 layers x 126 chunks),
     each a GPU->CPU sync that idled the card. This is the dominant win.
  2. GPU-resident hidden cache. The [nch,ctx,d] activation cache (~198 MB bf16
     for 126x512x1536) stays on-GPU between bands, so there is no per-chunk
     host<->device copy.
  3. Vectorized gate/up expert stats. Each token adds x^2 to each of its top-k
     experts via a single index_add_ over all (token,expert) assignments — no
     per-expert Python loop. (down_exps still loops experts, because each expert
     applies different gate_up weights to compute its intermediate, but it
     accumulates on-GPU with no sync.)
  4. Chunk batching (--batch): several ctx-windows per forward for bigger GEMMs.
     Scan memory scales with batch, so keep it modest in banded mode.

Validate against the reference tool with compare.py (expect corr ~1.0, tiny L1).
"""
import argparse, json, os, re, resource
import numpy as np, torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridDecoderLayer, GraniteMoeHybridRotaryEmbedding,
    GraniteMoeHybridExperts, GraniteMoeHybridTopKRouter,
)
from transformers.masking_utils import create_causal_mask
from safetensors import safe_open
from imatrix_io import write_imatrix, read_imatrix
from gen_imatrix_t514 import map_linear, expert_names

_RENAME = {
    "block_sparse_moe.experts.gate_up_proj": "block_sparse_moe.input_linear.weight",
    "block_sparse_moe.experts.down_proj":    "block_sparse_moe.output_linear.weight",
    "block_sparse_moe.router.weight":        "block_sparse_moe.router.layer.weight",
}


class ShardReader:
    def __init__(self, d):
        self.dir = d
        self.wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.handles = {}

    def get(self, name):
        fn = self.wm[name]
        if fn not in self.handles:
            self.handles[fn] = safe_open(os.path.join(self.dir, fn), framework="pt")
        return self.handles[fn].get_tensor(name)


def materialize(module, prefix, reader, dtype, device):
    sd, missing = {}, []
    for k in module.state_dict().keys():
        raw = f"{prefix}.{_RENAME.get(k, k)}"
        if raw in reader.wm:
            sd[k] = reader.get(raw).to(dtype)
        else:
            missing.append(k)
    if missing:
        raise KeyError(f"{prefix}: no checkpoint tensor for {missing}")
    module.load_state_dict(sd, strict=False, assign=True)
    return module.to(device).eval()


class Acc:
    """On-GPU float64 accumulators; host copy only at the end."""
    def __init__(self, device):
        self.device = device
        self.sums = {}   # name -> [nmat, in] f64 on GPU
        self.counts = {}  # name -> [nmat]    f64 on GPU

    def ensure(self, name, nmat, nin):
        if name not in self.sums:
            self.sums[name] = torch.zeros((nmat, nin), dtype=torch.float64, device=self.device)
            self.counts[name] = torch.zeros(nmat, dtype=torch.float64, device=self.device)
        return self.sums[name], self.counts[name]

    def to_entries(self):
        entries = {}
        for name, s in self.sums.items():
            sums = s.cpu().numpy().astype(np.float32)
            counts = self.counts[name].cpu().numpy().astype(np.float32)
            entries[name] = {"in_sum2": sums if sums.shape[0] > 1 else sums[0], "counts": counts}
        return entries


def make_hooks(named_mods, acc, known):
    def note(names):
        if known is not None:
            for g in names:
                if g not in known:
                    print(f"  WARN mapped name not in ground truth: {g}")

    handles = []
    for hf_name, mod in named_mods:
        if isinstance(mod, torch.nn.Linear):
            names = map_linear(hf_name)
            if names is None:
                continue
            note(names)
            def pre_linear(m, a, names=names):
                x = a[0].detach().reshape(-1, a[0].shape[-1]).float()
                s = (x * x).sum(0).double()
                n = x.shape[0]
                for nm in names:
                    S, C = acc.ensure(nm, 1, x.shape[-1])
                    S[0] += s; C[0] += n
            handles.append(mod.register_forward_pre_hook(pre_linear))
        elif isinstance(mod, GraniteMoeHybridExperts):
            en = expert_names(hf_name); note(list(en.values()))
            ne = mod.num_experts
            def pre_experts(m, a, en=en, ne=ne):
                hidden_states, top_k_index = a[0], a[1]
                x = hidden_states.detach().float()            # [T, hidden]
                T, K = top_k_index.shape
                x2 = x * x                                    # [T, hidden]
                # gate/up: each token adds x^2 to each of its K experts (one index_add)
                assign = top_k_index.reshape(-1)              # [T*K]
                tok = torch.arange(T, device=x.device).repeat_interleave(K)
                Sg, Cg = acc.ensure(en["gate"], ne, x.shape[-1])
                Su, Cu = acc.ensure(en["up"], ne, x.shape[-1])
                contrib = x2.index_select(0, tok).double()    # [T*K, hidden]
                Sg.index_add_(0, assign, contrib); Su.index_add_(0, assign, contrib)
                ones = torch.ones(T * K, dtype=torch.float64, device=x.device)
                Cg.index_add_(0, assign, ones); Cu.index_add_(0, assign, ones)
                # down: per-expert intermediate (distinct weights) — loop, but no sync
                Sd, Cd = acc.ensure(en["down"], ne, m.down_proj.shape[-1])
                for e in torch.unique(top_k_index):
                    tok_idx = (top_k_index == e).any(dim=-1).nonzero(as_tuple=True)[0]
                    cur = x.index_select(0, tok_idx).to(m.gate_up_proj.dtype)
                    g, u = torch.nn.functional.linear(cur, m.gate_up_proj[e]).chunk(2, dim=-1)
                    inter = (m.act_fn(g) * u).float()
                    Sd[e] += (inter * inter).sum(0).double(); Cd[e] += tok_idx.numel()
            handles.append(mod.register_forward_pre_hook(pre_experts))
        elif isinstance(mod, GraniteMoeHybridTopKRouter):
            gname = f"blk.{re.match(r'.*layers\.(\d+)\.', hf_name).group(1)}.ffn_gate_inp.weight"
            note([gname])
            def pre_router(m, a, gname=gname):
                x = a[0].detach().reshape(-1, a[0].shape[-1]).float()
                S, C = acc.ensure(gname, 1, x.shape[-1])
                S[0] += (x * x).sum(0).double(); C[0] += x.shape[0]
            handles.append(mod.register_forward_pre_hook(pre_router))
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=126)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--band", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2, help="ctx-windows per forward (scan mem scales with this)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset-label", default="calibration_datav3")
    ap.add_argument("--ground-truth", default=None)
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.model)
    dev = args.device
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    reader = ShardReader(args.model)
    known = set(read_imatrix(args.ground_truth)[1]) if args.ground_truth else None
    nlayers = cfg.num_hidden_layers
    ltype = cfg.layer_types

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(open(args.calib, encoding="utf-8", errors="ignore").read(), return_tensors="pt").input_ids[0]
    nch = min(args.chunks, ids.shape[0] // args.ctx)
    chunks = torch.stack([ids[c*args.ctx:(c+1)*args.ctx] for c in range(nch)])
    print(f"layers={nlayers} band={args.band} batch={args.batch} chunks={nch} ctx={args.ctx} dtype={dtype}")

    embed_w = reader.get("model.embed_tokens.weight").to(dtype)
    # hidden cache lives on-GPU
    hid = (torch.nn.functional.embedding(chunks, embed_w) * cfg.embedding_multiplier).to(dtype).to(dev)
    del embed_w
    print(f"hidden cache on {dev}: {tuple(hid.shape)} ({hid.element_size()*hid.nelement()/1e6:.0f} MB)")

    rotary = GraniteMoeHybridRotaryEmbedding(cfg).to(dev)
    pos_ids = torch.arange(args.ctx, device=dev).unsqueeze(0)
    pos_emb = rotary(hid[:1], pos_ids)  # cos/sin [1,ctx,dim] broadcast over batch
    mask_cache = {}
    def causal_for(bs):
        if bs not in mask_cache:
            mask_cache[bs] = create_causal_mask(config=cfg, inputs_embeds=hid[:bs],
                                                attention_mask=None, past_key_values=None)
        return mask_cache[bs]

    acc = Acc(dev)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for b0 in range(0, nlayers, args.band):
            band = list(range(b0, min(b0 + args.band, nlayers)))
            layers = []
            for i in band:
                L = GraniteMoeHybridDecoderLayer(cfg, i)
                materialize(L, f"model.layers.{i}", reader, dtype, dev)
                layers.append((i, L))
            named = [(f"model.layers.{i}.{n}", m) for i, L in layers for n, m in L.named_modules()]
            handles = make_hooks(named, acc, known)
            for s in range(0, nch, args.batch):
                sl = slice(s, min(s + args.batch, nch))
                h = hid[sl]
                mask = causal_for(h.shape[0])
                for i, L in layers:
                    m = mask if ltype[i] == "full_attention" else None
                    out = L(h, attention_mask=m, past_key_values=None, position_embeddings=pos_emb)
                    h = out[0] if isinstance(out, tuple) else out
                hid[sl] = h
            for hh in handles:
                hh.remove()
            for _, L in layers:
                del L
            layers.clear()
            if dev == "cuda":
                torch.cuda.empty_cache()
            print(f"  band {band[0]}-{band[-1]} done", flush=True)

    write_imatrix(args.out, acc.to_entries(), [args.dataset_label], nch, args.ctx)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    line = f"wrote {len(acc.sums)} entries -> {args.out} | peak RSS {peak_rss:.1f} GB"
    if dev == "cuda":
        line += f" | peak GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB"
    print(line)


if __name__ == "__main__":
    main()
