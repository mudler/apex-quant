"""Band-serialized (memory-bounded) Path-A imatrix generator for Llama4 (Scout).

Same idea as serialized_gen.py (granite) but for the llama4_text arch, which is
different enough to need its own module:
  - Experts are a single Llama4TextExperts module doing bmm on 3D Parameters
    (gate_up_proj, down_proj) -- NOT per-expert Linears. So the down-proj input
    is internal; we recompute the gated intermediate inside the hook from the
    captured input + the band's loaded weights.
  - MoE folds the router score into the expert input (hidden.repeat * router_scores),
    so gate/up expert stats are router-weighted. This MATCHES llama.cpp, which special
    cases this arch -- src/llama-graph.cpp:1837
        const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4;
    and at L1976 multiplies the sigmoid-ed weights into `cur` BEFORE the gate/up
    mul_mat_id (the generic MoE path instead weights the expert OUTPUT at L2121, which
    is what leaves gate/up unweighted for every other arch). So a weighted stat here is
    the correct reproduction, not a divergence.
    Row counts come from the router's exact top-k indices: transformers hands every
    expert all T rows with non-selected ones scaled to zero, so they must be masked
    back out to match llama.cpp's one-count-per-(token, selected-expert) slot.
    There is still no ground-truth Scout imatrix to correlation-check against (bf16
    doesn't fit 128 GB -- the whole point), so this validates by QUANT COHERENCE, not
    correlation.
  - Attention interleaves RoPE / NoPE and uses chunked attention (chunk 8192);
    at ctx<=8192 chunked == plain causal, and Llama4TextAttention applies RoPE
    only on rope layers internally, so we pass one causal mask + rotary to all.

Peak resident weights = band_size layers. Each weight read from disk once.
"""
import argparse, json, os, glob, resource
import numpy as np, torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama4.modeling_llama4 import (
    Llama4TextDecoderLayer, Llama4TextRotaryEmbedding, Llama4TextExperts,
    Llama4Router,
)
from transformers.masking_utils import create_causal_mask
from safetensors import safe_open
from calib import load_calibration_chunks, add_calib_args
from imatrix_io import write_imatrix


def map_name(rel: str, blk: str):
    """rel = module path relative to the layer, e.g. 'self_attn.q_proj'."""
    table = {
        "self_attn.q_proj": [f"{blk}.attn_q.weight"],
        "self_attn.k_proj": [f"{blk}.attn_k.weight"],
        "self_attn.v_proj": [f"{blk}.attn_v.weight"],
        "self_attn.o_proj": [f"{blk}.attn_output.weight"],
        "feed_forward.router": [f"{blk}.ffn_gate_inp.weight"],
        "feed_forward.shared_expert.gate_proj": [f"{blk}.ffn_gate_shexp.weight"],
        "feed_forward.shared_expert.up_proj": [f"{blk}.ffn_up_shexp.weight"],
        "feed_forward.shared_expert.down_proj": [f"{blk}.ffn_down_shexp.weight"],
    }
    return table.get(rel)


class ShardReader:
    def __init__(self, d):
        self.dir = d
        idx = os.path.join(d, "model.safetensors.index.json")
        with open(idx) as f:
            self.wm = json.load(f)["weight_map"]
        self.handles = {}
        # detect weight-key prefix for the text decoder layers
        self.prefix = "language_model.model" if any(
            k.startswith("language_model.model.layers.") for k in self.wm) else "model"

    def get(self, name):
        fn = self.wm[name]
        if fn not in self.handles:
            self.handles[fn] = safe_open(os.path.join(self.dir, fn), framework="pt")
        return self.handles[fn].get_tensor(name)


def make_ensure(acc):
    """Accumulator slot factory: acc[name] = {sums [nmat, nin], counts [nmat]}."""
    def ensure(name, nmat, nin):
        if name not in acc:
            acc[name] = {"sums": np.zeros((nmat, nin), np.float64), "counts": np.zeros(nmat, np.int64)}
        return acc[name]
    return ensure


def hook_linear(mod, names, ensure):
    def pre(m, a, names=names):
        x = a[0].detach().float().reshape(-1, a[0].shape[-1])
        s = (x * x).sum(0).double().cpu().numpy()
        for nm in names:
            e = ensure(nm, 1, x.shape[-1]); e["sums"][0] += s; e["counts"][0] += x.shape[0]
    return mod.register_forward_pre_hook(pre)


def hook_router_select(mod, blk, sel):
    """Capture the router's exact top-k selection for `blk` into sel[blk]."""
    def post(m, a, out):
        router_logits = out[1]
        sel[blk] = torch.topk(router_logits, m.top_k, dim=1).indices.detach()
    return mod.register_forward_hook(post)


def hook_experts(mod, blk, ensure, sel):
    gate_n, up_n, down_n = f"{blk}.ffn_gate_exps.weight", f"{blk}.ffn_up_exps.weight", f"{blk}.ffn_down_exps.weight"
    def pre(m, a):
        E = m.num_experts
        X = a[0].detach().view(E, -1, m.hidden_size)            # [E,T,H] (router-scaled)
        T = X.shape[1]
        # Exact selection mask from the router's top-k, NOT a nonzero-row heuristic.
        # transformers hands every expert all T rows with the non-selected ones
        # scaled to 0, so counting `abs().sum(-1) > 0` rows happened to work but
        # breaks on a genuinely-zero activation row or a router score that
        # underflows to 0 in bf16. llama.cpp counts one per (token, selected-expert)
        # slot, which is exactly what the top-k indices give.
        idx = sel[blk]                                          # [T, top_k]
        keep = torch.zeros(T, E, dtype=torch.bool, device=X.device)
        keep.scatter_(1, idx, True)
        mask = keep.t().unsqueeze(-1)                           # [E,T,1]
        cnt = keep.sum(0).cpu().numpy()                         # [E] exact row counts

        # NOTE: the router-weighted input IS correct for Llama-4. llama.cpp special
        # cases this arch -- llama-graph.cpp:1837
        #   const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4;
        # and at L1976 multiplies the sigmoid-ed weights into `cur` BEFORE the
        # gate/up mul_mat_id, so llama-imatrix also records a weighted stat here.
        xin = X.float() * mask
        s_in = (xin * xin).sum(1).double().cpu().numpy()        # [E,H]  gate/up input stat
        for nm in (gate_n, up_n):
            e = ensure(nm, E, m.hidden_size); e["sums"] += s_in; e["counts"] += cnt
        # recompute gated intermediate to get down_proj input (also from the
        # weighted input, matching the graph order above)
        gate_up = torch.bmm(X.to(m.gate_up_proj.dtype), m.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        inter = (up * m.act_fn(gate)).float() * mask            # [E,T,inter]
        s_dn = (inter * inter).sum(1).double().cpu().numpy()    # [E,inter]
        e = ensure(down_n, E, inter.shape[-1]); e["sums"] += s_dn; e["counts"] += cnt
    return mod.register_forward_pre_hook(pre)


def materialize(module, prefix, reader, dtype, device):
    # Every module parameter MUST come from the checkpoint -- see the note in
    # serialized_gen.py: a skipped key leaves random weights in the band and the
    # resulting imatrix is meaningless with no error anywhere.
    sd, missing = {}, []
    for k in module.state_dict().keys():
        full = f"{prefix}.{k}"
        if full in reader.wm:
            sd[k] = reader.get(full).to(dtype)
        else:
            missing.append(k)
    if missing:
        raise KeyError(f"{prefix}: no checkpoint tensor for {missing}")
    module.load_state_dict(sd, strict=False, assign=True)
    return module.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    add_calib_args(ap)
    ap.add_argument("--band", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-impl", default="eager",
                    help="eager/sdpa/flash_attention_2; must be set because "
                         "create_causal_mask() returns None when "
                         "config._attn_implementation is unset, which makes the "
                         "standalone decoder layers attend bidirectionally")
    args = ap.parse_args()

    full_cfg = AutoConfig.from_pretrained(args.model)
    # MUST be set before create_causal_mask(): it returns None when
    # config._attn_implementation is unset, and a standalone decoder layer given
    # attention_mask=None then attends BIDIRECTIONALLY, silently producing a
    # non-causal imatrix for attn_output (and everything downstream of it).
    full_cfg._attn_implementation = args.attn_impl
    cfg = full_cfg.get_text_config()
    cfg._attn_implementation = args.attn_impl
    dev = args.device
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    reader = ShardReader(args.model)
    P = reader.prefix
    nlayers = cfg.num_hidden_layers
    moe_layers = set(getattr(cfg, "moe_layers", range(nlayers)))
    print(f"llama4: layers={nlayers} experts={cfg.num_local_experts} band={args.band} "
          f"prefix={P} dtype={dtype}")

    acc = {}
    ensure = make_ensure(acc)
    # Llama4TextMoe.forward runs the router BEFORE the experts, so a forward hook on
    # the router lands the exact top-k selection here in time for the experts hook.
    sel = {}

    # tokenize + window
    tok = AutoTokenizer.from_pretrained(args.model)
    chunks, calib_info = load_calibration_chunks(tok, args.calib, args.chunks, args.ctx,
                                                add_bos=args.bos_per_chunk)
    nch = chunks.shape[0]
    print(f"calib: {calib_info}")
    print(f"chunks={nch} ctx={args.ctx}")

    # embed -> hidden cache (CPU); Llama4 has no embedding_multiplier
    embed_w = reader.get(f"{P}.embed_tokens.weight").to(dtype)
    hid = torch.nn.functional.embedding(chunks, embed_w).to(dtype)
    del embed_w
    print(f"hidden cache {tuple(hid.shape)} ({hid.element_size()*hid.nelement()/1e6:.0f} MB)")

    rotary = Llama4TextRotaryEmbedding(cfg).to(dev)
    pos_ids = torch.arange(args.ctx, device=dev).unsqueeze(0)
    dummy = hid[:1].to(dev)
    pos_emb = rotary(dummy, pos_ids)
    causal = create_causal_mask(config=cfg, inputs_embeds=dummy, attention_mask=None, past_key_values=None)

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for b0 in range(0, nlayers, args.band):
            band = list(range(b0, min(b0 + args.band, nlayers)))
            layers, handles = [], []
            for i in band:
                L = materialize(Llama4TextDecoderLayer(cfg, i), f"{P}.layers.{i}", reader, dtype, dev)
                layers.append((i, L))
                blk = f"blk.{i}"
                for rel, mod in L.named_modules():
                    if isinstance(mod, Llama4TextExperts):
                        handles.append(hook_experts(mod, blk, ensure, sel))
                        continue
                    if isinstance(mod, Llama4Router):
                        # router doubles as ffn_gate_inp (input stat) and as the source
                        # of the exact expert selection the experts hook needs
                        handles.append(hook_router_select(mod, blk, sel))
                    names = map_name(rel, blk)
                    if names:
                        handles.append(hook_linear(mod, names, ensure))
            for c in range(nch):
                h = hid[c:c+1].to(dev)
                for i, L in layers:
                    h = L(h, attention_mask=causal, position_ids=pos_ids,
                          past_key_values=None, use_cache=False, position_embeddings=pos_emb)
                hid[c:c+1] = h.to(hid.dtype).cpu()
            for hh in handles: hh.remove()
            # Drop EVERY reference to the band's modules before empty_cache(), or the
            # cache release is a no-op: the hook closures and `layers` keep them alive
            # and `del L` only unbinds the loop variable.
            handles.clear(); layers.clear()
            if dev == "cuda": torch.cuda.empty_cache()
            print(f"  band {band[0]}-{band[-1]} done")

    entries = {}
    for name, d in acc.items():
        sums = d["sums"].astype(np.float32)
        entries[name] = {"in_sum2": sums if sums.shape[0] > 1 else sums[0],
                         "counts": d["counts"].astype(np.float32)}
    write_imatrix(args.out, entries, ["calibration_datav3"], nch, args.ctx)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    line = f"wrote {len(entries)} entries -> {args.out} | peak RSS {peak_rss:.1f} GB"
    if dev == "cuda":
        line += f" | peak GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB"
    print(line)


if __name__ == "__main__":
    main()
