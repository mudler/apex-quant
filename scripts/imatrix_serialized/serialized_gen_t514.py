"""Band-serialized (VRAM-bounded) torch-imatrix generator, transformers-5.14.

Same idea as serialized_gen.py: process the model in BANDS of `--band` decoder
layers so peak *GPU* residency is band-size layers, not the whole model. Adapted
to the transformers-5.14 GraniteMoeHybrid rewrite:

  * experts are now a single `GraniteMoeHybridExperts` (params gate_up_proj/
    down_proj, functional per-expert linears) + a bare-Parameter router; hooks
    replay the routing and recompute the down input (see gen_imatrix_t514.py).
  * the on-disk checkpoint still uses the OLD fused key names
    (block_sparse_moe.input_linear/output_linear/router.layer); from_pretrained
    remaps them at load, but our manual per-layer materialize must do it itself.
    The remap is a pure same-shape rename (verified): see _RENAME.
  * config uses layer_types with values 'linear_attention'/'full_attention'
    (not 'mamba'/'attention'); only full_attention layers take the causal mask.

VRAM-bounded: peak GPU = band layers + one naive-Mamba scan intermediate. The
full model is held in CPU RAM via the shard reader's lazy tensors (each weight
read from disk once); the hidden-state cache lives on CPU between bands.
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

# module state_dict key -> raw checkpoint key (same shape, pure rename)
_RENAME = {
    "block_sparse_moe.experts.gate_up_proj": "block_sparse_moe.input_linear.weight",
    "block_sparse_moe.experts.down_proj":    "block_sparse_moe.output_linear.weight",
    "block_sparse_moe.router.weight":        "block_sparse_moe.router.layer.weight",
}


class ShardReader:
    def __init__(self, d):
        self.dir = d
        idx = os.path.join(d, "model.safetensors.index.json")
        self.wm = json.load(open(idx))["weight_map"]
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


def make_hooks(named_mods, acc, known):
    def ensure(name, nmat, nin):
        if name not in acc:
            acc[name] = {"sums": np.zeros((nmat, nin), np.float64), "counts": np.zeros(nmat, np.int64)}
        return acc[name]

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
                continue  # lm_head etc. never appears inside a decoder layer
            note(names)
            def pre_linear(m, a, names=names):
                x = a[0].detach().float().reshape(-1, a[0].shape[-1])
                s = (x * x).sum(0).double().cpu().numpy()
                for nm in names:
                    e = ensure(nm, 1, x.shape[-1]); e["sums"][0] += s; e["counts"][0] += x.shape[0]
            handles.append(mod.register_forward_pre_hook(pre_linear))
        elif isinstance(mod, GraniteMoeHybridExperts):
            en = expert_names(hf_name); note(list(en.values()))
            ne = mod.num_experts
            def pre_experts(m, a, en=en, ne=ne):
                hidden_states, top_k_index = a[0], a[1]
                x = hidden_states.detach().float()
                gate = ensure(en["gate"], ne, x.shape[-1]); up = ensure(en["up"], ne, x.shape[-1])
                with torch.no_grad():
                    mask = torch.nn.functional.one_hot(top_k_index, num_classes=ne).permute(2, 1, 0)
                    for eidx in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
                        e = eidx[0]
                        if e == ne:
                            continue
                        _, tok_idx = torch.where(mask[e])
                        cur = x[tok_idx]
                        s = (cur * cur).sum(0).double().cpu().numpy()
                        gate["sums"][e] += s; gate["counts"][e] += cur.shape[0]
                        up["sums"][e] += s;   up["counts"][e] += cur.shape[0]
                        gu = torch.nn.functional.linear(cur.to(m.gate_up_proj.dtype), m.gate_up_proj[e])
                        g, u = gu.chunk(2, dim=-1)
                        inter = (m.act_fn(g) * u).float()
                        dn = ensure(en["down"], ne, inter.shape[-1])
                        dn["sums"][e] += (inter * inter).sum(0).double().cpu().numpy()
                        dn["counts"][e] += inter.shape[0]
            handles.append(mod.register_forward_pre_hook(pre_experts))
        elif isinstance(mod, GraniteMoeHybridTopKRouter):
            gname = f"blk.{re.match(r'.*layers\.(\d+)\.', hf_name).group(1)}.ffn_gate_inp.weight"
            note([gname])
            def pre_router(m, a, gname=gname):
                x = a[0].detach().float().reshape(-1, a[0].shape[-1])
                s = (x * x).sum(0).double().cpu().numpy()
                e = ensure(gname, 1, x.shape[-1]); e["sums"][0] += s; e["counts"][0] += x.shape[0]
            handles.append(mod.register_forward_pre_hook(pre_router))
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=126)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--band", type=int, default=4, help="layers resident at once (VRAM knob)")
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
    print(f"layers={nlayers} band={args.band} chunks={nch} ctx={args.ctx} dtype={dtype}")

    embed_w = reader.get("model.embed_tokens.weight").to(dtype)
    hid = (torch.nn.functional.embedding(chunks, embed_w) * cfg.embedding_multiplier).to(dtype)  # [nch,ctx,d] CPU
    del embed_w
    print(f"hidden cache: {tuple(hid.shape)} ({hid.element_size()*hid.nelement()/1e6:.0f} MB CPU)")

    rotary = GraniteMoeHybridRotaryEmbedding(cfg).to(dev)
    pos_ids = torch.arange(args.ctx, device=dev).unsqueeze(0)
    dummy = hid[:1].to(dev)
    pos_emb = rotary(dummy, pos_ids)
    causal = create_causal_mask(config=cfg, inputs_embeds=dummy, attention_mask=None, past_key_values=None)

    acc = {}
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
            for c in range(nch):
                h = hid[c:c+1].to(dev)
                for i, L in layers:
                    mask = causal if ltype[i] == "full_attention" else None
                    out = L(h, attention_mask=mask, past_key_values=None, position_embeddings=pos_emb)
                    h = out[0] if isinstance(out, tuple) else out
                hid[c:c+1] = h.to(hid.dtype).cpu()
            for hh in handles:
                hh.remove()
            for _, L in layers:
                del L
            layers.clear()
            if dev == "cuda":
                torch.cuda.empty_cache()
            print(f"  band {band[0]}-{band[-1]} done", flush=True)

    entries = {}
    for name, dd in acc.items():
        sums = dd["sums"].astype(np.float32)
        entries[name] = {"in_sum2": sums if sums.shape[0] > 1 else sums[0],
                         "counts": dd["counts"].astype(np.float32)}
    write_imatrix(args.out, entries, [args.dataset_label], nch, args.ctx)

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    line = f"wrote {len(entries)} entries -> {args.out} | peak RSS {peak_rss:.1f} GB"
    if dev == "cuda":
        line += f" | peak GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB"
    print(line)


if __name__ == "__main__":
    main()
