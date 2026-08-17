"""Band-serialized (memory-bounded) Path-A imatrix generator for GLM-4.5-Air
(zai-org/GLM-4.5-Air, model_type glm4_moe).

GLM-4.5-Air facts:
  46 layers, hidden_size 4096, head_dim 128, partial_rotary_factor 0.5
  128 routed experts per layer, 1 shared expert, 8 active experts per token
  moe_intermediate_size = 1408 (NOT divisible by 256 -> K-quants fall back)
  first_k_dense_replace = 1 -> layer 0 uses a dense MLP (Glm4MoeMLP), rest use Glm4MoeMoE

Peak memory is band-bounded: only `--band` layers resident on GPU at once.
Each weight tensor is read from safetensors once; activations are cached CPU-side
between bands.

EXPERIMENTAL / UNTESTED against a real GLM-4.5 checkpoint. Requires a transformers
version whose glm4_moe module exposes the fused `Glm4MoeExperts` class (it does not
exist in 5.5.x, which packages Glm4MoeMoE/Glm4MoeNaiveMoe instead); the import is
guarded so the failure names the reason instead of raising a bare ImportError.
"""
import argparse, json, os, glob, re
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
try:
    from transformers.models.glm4_moe.modeling_glm4_moe import (
        Glm4MoeDecoderLayer, Glm4MoeMoE, Glm4MoeExperts,
        Glm4MoeMLP, Glm4MoeRotaryEmbedding,
    )
    _ARCH_IMPORT_ERR = None
except ImportError as _e:                # keep the pure math importable/testable
    Glm4MoeDecoderLayer = Glm4MoeMoE = Glm4MoeExperts = None
    Glm4MoeMLP = Glm4MoeRotaryEmbedding = None
    _ARCH_IMPORT_ERR = _e
from transformers.masking_utils import create_causal_mask
from transformers.activations import ACT2FN
from safetensors import safe_open

import imatrix_io
from calib import load_calibration_chunks, add_calib_args
from imatrix_io import write_imatrix, read_imatrix


# ---- HF dotted module name -> list of ggml tensor names --------------------
def map_name(hf: str):
    m = re.match(r".*layers\.(\d+)\.(.+)$", hf)
    if not m:
        return None
    i, tail = m.group(1), m.group(2)
    b = f"blk.{i}"
    table = {
        # attention
        "self_attn.q_proj": [f"{b}.attn_q.weight"],
        "self_attn.k_proj": [f"{b}.attn_k.weight"],
        "self_attn.v_proj": [f"{b}.attn_v.weight"],
        "self_attn.o_proj": [f"{b}.attn_output.weight"],
        # dense MLP (layer 0 with first_k_dense_replace=1)
        "mlp.gate_proj": [f"{b}.ffn_gate.weight"],
        "mlp.up_proj": [f"{b}.ffn_up.weight"],
        "mlp.down_proj": [f"{b}.ffn_down.weight"],
        # MoE gate
        "mlp.gate": [f"{b}.ffn_gate_inp.weight"],
        # NOTE: "mlp.experts" is deliberately absent -- the three routed-expert
        # tensors do NOT share one input stat (gate/up take hidden_size inputs,
        # down takes moe_intermediate_size gated-intermediate inputs), so they are
        # resolved by expert_names() and handled by the dedicated hook below.
        # shared experts
        "mlp.shared_experts.gate_proj": [f"{b}.ffn_gate_shexp.weight"],
        "mlp.shared_experts.up_proj": [f"{b}.ffn_up_shexp.weight"],
        "mlp.shared_experts.down_proj": [f"{b}.ffn_down_shexp.weight"],
    }
    return table.get(tail)


def expert_names(hf: str):
    """ggml names for the three routed-expert tensors of an mlp.experts module."""
    m = re.match(r".*layers\.(\d+)\.", hf)
    if not m:
        raise ValueError(f"no layer index in {hf!r}")
    b = f"blk.{m.group(1)}"
    return {"gate": f"{b}.ffn_gate_exps.weight", "up": f"{b}.ffn_up_exps.weight",
            "down": f"{b}.ffn_down_exps.weight"}


def apply_gate_up(x, w_e, hidden):
    """Apply one expert's fused gate_up weight to rows x [T, hidden] -> [T, 2*interm].

    Fused MoE weights appear in two layouts across transformers arch modules:
      [2*interm, hidden]  -- nn.Linear convention (F.linear)
      [hidden, 2*interm]  -- bmm convention (x @ W)
    GLM-4.5's layout is transformers-version dependent, so derive it from the shape
    and hard-fail rather than guess: guessing wrong silently yields a garbage
    ffn_down_exps statistic, which is exactly the failure mode this path must avoid.
    """
    if w_e.ndim != 2:
        raise ValueError(f"expected 2-D per-expert gate_up weight, got {tuple(w_e.shape)}")
    a, b = w_e.shape
    if a == b:
        raise ValueError(f"ambiguous square gate_up layout {tuple(w_e.shape)} (hidden={hidden})")
    if b == hidden:
        return torch.nn.functional.linear(x, w_e)
    if a == hidden:
        return x @ w_e
    raise ValueError(f"gate_up weight {tuple(w_e.shape)} has no hidden={hidden} axis")


def expert_stats(x, topk_idx, gate_up, act_fn, num_experts, hidden):
    """Per-expert Σx² for the three routed-expert tensors of one MoE layer.

    Returns (sum_in2 [E, hidden], sum_inter2 [E, interm], counts [E]) where
      * sum_in2    feeds ffn_gate_exps and ffn_up_exps -- the hidden-dim input each
                   selected expert's gate/up MUL_MAT_ID actually sees;
      * sum_inter2 feeds ffn_down_exps -- act_fn(gate)*up, the gated intermediate
                   that is the down projection's input. It is NOT the hidden-dim
                   stat and has moe_intermediate_size entries, not hidden_size.
      * counts     is the number of tokens routed to each expert, i.e. the row count
                   of every one of the three matmuls (llama.cpp counts one per
                   (token, selected-expert) slot).

    Pure tensor math, no module dependency, so it is testable without a checkpoint.
    """
    x = x.reshape(-1, x.shape[-1])
    T = x.shape[0]
    if x.shape[-1] != hidden:
        raise ValueError(f"expert input dim {x.shape[-1]} != hidden {hidden}")
    dev = x.device
    xf = x.float()
    k = topk_idx.shape[-1]
    assign = topk_idx.reshape(-1).long()
    tok = torch.arange(T, device=dev).repeat_interleave(k)

    sum_in2 = torch.zeros(num_experts, hidden, dtype=torch.float64, device=dev)
    counts = torch.zeros(num_experts, dtype=torch.float64, device=dev)
    sq = (xf * xf).index_select(0, tok).double()
    sum_in2.index_add_(0, assign, sq)
    counts.index_add_(0, assign, torch.ones(assign.numel(), dtype=torch.float64, device=dev))

    # gate_up is [E, 2*interm, hidden] or [E, hidden, 2*interm]; the non-hidden axis
    # is the fused output, half of which is the gated intermediate.
    fused = gate_up.shape[-2] if gate_up.shape[-1] == hidden else gate_up.shape[-1]
    sum_inter2 = torch.zeros(num_experts, fused // 2, dtype=torch.float64, device=dev)
    for e in torch.unique(assign).tolist():
        rows = (topk_idx == e).any(dim=-1).nonzero(as_tuple=True)[0]
        cur = x.index_select(0, rows).to(gate_up.dtype)
        g, u = apply_gate_up(cur, gate_up[e], hidden).chunk(2, dim=-1)
        inter = (act_fn(g) * u).float()
        sum_inter2[e] += (inter * inter).sum(0).double()
    return sum_in2, sum_inter2, counts


class ShardReader:
    def __init__(self, d):
        self.dir = d
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                self.wm = json.load(f)["weight_map"]
        else:
            f = os.path.basename(glob.glob(os.path.join(d, "*.safetensors"))[0])
            with safe_open(os.path.join(d, f), framework="pt") as sf:
                self.wm = {k: f for k in sf.keys()}
        self.handles = {}

    def get(self, name):
        fn = self.wm[name]
        if fn not in self.handles:
            self.handles[fn] = safe_open(os.path.join(self.dir, fn), framework="pt")
        return self.handles[fn].get_tensor(name)


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


def make_hooks(named_mods, acc, known, cfg):
    def ensure(name, nmat, nin):
        if name not in acc:
            acc[name] = {"sums": np.zeros((nmat, nin), np.float64), "counts": np.zeros(nmat, np.int64)}
        return acc[name]
    handles = []
    for hf_name, mod in named_mods:
        if Glm4MoeExperts is not None and isinstance(mod, Glm4MoeExperts):
            en = expert_names(hf_name)
            if known is not None:
                for g in en.values():
                    if g not in known:
                        print(f"  WARN unmapped: {g} ({hf_name})")
            n_exp = mod.num_experts
            act = getattr(mod, "act_fn", None) or ACT2FN[cfg.hidden_act]
            # forward signature: (hidden_states, top_k_index, top_k_weights)
            def pre_experts(m, a, en=en, ne=n_exp, act=act):
                inp, topk_idx = a[0], a[1]
                x = inp.detach()  # [tokens, hidden]
                s_in, s_inter, cnt = expert_stats(
                    x, topk_idx, m.gate_up_proj, act, ne, cfg.hidden_size)
                s_in = s_in.cpu().numpy(); s_inter = s_inter.cpu().numpy()
                cnt = cnt.cpu().numpy().astype(np.int64)
                for nm in (en["gate"], en["up"]):
                    e = ensure(nm, ne, s_in.shape[-1])
                    e["sums"] += s_in
                    e["counts"] += cnt
                e = ensure(en["down"], ne, s_inter.shape[-1])
                e["sums"] += s_inter
                e["counts"] += cnt
            handles.append(mod.register_forward_pre_hook(pre_experts))
            continue
        ggml = map_name(hf_name)
        if ggml is None:
            continue
        if known is not None:
            for g in ggml:
                if g not in known:
                    print(f"  WARN unmapped: {g} ({hf_name})")
        def pre_linear(m, a, names=ggml):
            x = a[0].detach().float().reshape(-1, a[0].shape[-1])
            s = (x * x).sum(0).double().cpu().numpy()
            for nm in names:
                e = ensure(nm, 1, x.shape[-1])
                e["sums"][0] += s
                e["counts"][0] += x.shape[0]
        handles.append(mod.register_forward_pre_hook(pre_linear))
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    add_calib_args(ap)
    ap.add_argument("--band", type=int, default=1, help="layers resident at once (memory knob)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ground-truth", default=None)
    ap.add_argument("--attn-impl", default="eager", help="eager/sdpa/flash_attention_2")
    # OFF by default: trust_remote_code executes arbitrary Python from the model repo,
    # and the pipeline downloads whatever MODEL_ID a config names. GLM-4.5 is supported
    # natively by transformers, so this is only needed for a repo shipping custom code.
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="allow executing model-repo code in AutoConfig/AutoTokenizer")
    args = ap.parse_args()

    if _ARCH_IMPORT_ERR is not None:
        raise RuntimeError(
            "glm4_moe generator needs a transformers whose modeling_glm4_moe exposes "
            f"the fused Glm4MoeExperts class; installed transformers does not ({_ARCH_IMPORT_ERR})")

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    cfg._attn_implementation = args.attn_impl
    dev = args.device
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    reader = ShardReader(args.model)
    known = set(read_imatrix(args.ground_truth)[1]) if args.ground_truth else None
    nlayers = cfg.num_hidden_layers
    d = cfg.hidden_size

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    chunks, calib_info = load_calibration_chunks(tok, args.calib, args.chunks, args.ctx,
                                                add_bos=args.bos_per_chunk)
    nch = chunks.shape[0]
    print(f"calib: {calib_info}")
    print(f"layers={nlayers} band={args.band} chunks={nch} ctx={args.ctx} dtype={dtype} attn={args.attn_impl}")

    # --- embed all chunks -> hidden cache on CPU ---
    embed_w = reader.get("model.embed_tokens.weight").to(dtype)
    hid = torch.nn.functional.embedding(chunks, embed_w).to(dtype)  # [nch, ctx, d] CPU
    del embed_w
    print(f"hidden cache: {tuple(hid.shape)} ({hid.element_size()*hid.nelement()/1e6:.0f} MB CPU)")

    # --- rotary + causal mask ---
    rotary = Glm4MoeRotaryEmbedding(cfg).to(dev)
    pos_ids = torch.arange(args.ctx, device=dev).unsqueeze(0)
    dummy = hid[:1].to(dev)
    pos_emb = rotary(dummy, pos_ids)
    causal = create_causal_mask(config=cfg, inputs_embeds=dummy, attention_mask=None, past_key_values=None)

    acc = {}
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # --- band loop ---
    with torch.no_grad():
        for b0 in range(0, nlayers, args.band):
            band = list(range(b0, min(b0 + args.band, nlayers)))
            layers = []
            for i in band:
                L = Glm4MoeDecoderLayer(cfg, layer_idx=i)
                materialize(L, f"model.layers.{i}", reader, dtype, dev)
                layers.append((i, L))
            named = [(f"model.layers.{i}.{n}", m) for i, L in layers for n, m in L.named_modules()]
            handles = make_hooks(named, acc, known, cfg)
            for c in range(nch):
                h = hid[c:c+1].to(dev)
                for i, L in layers:
                    h = L(h, attention_mask=causal, position_embeddings=pos_emb, use_cache=False)
                hid[c:c+1] = h.to(hid.dtype).cpu()
            for hh in handles:
                hh.remove()
            # Drop EVERY reference to the band's modules before empty_cache(), or the
            # cache release is a no-op: `named` holds the submodules and `del L` only
            # unbinds the loop variable.
            handles.clear(); named.clear(); layers.clear()
            if dev == "cuda":
                torch.cuda.empty_cache()
            print(f"  band {band[0]}-{band[-1]} done")

    # write
    entries = {}
    for name, dd in acc.items():
        sums = dd["sums"].astype(np.float32)
        entries[name] = {"in_sum2": sums if sums.shape[0] > 1 else sums[0],
                         "counts": dd["counts"].astype(np.float32)}
    write_imatrix(args.out, entries, ["calibration_sample"], nch, args.ctx)

    line = f"wrote {len(entries)} entries -> {args.out}"
    if dev == "cuda":
        line += f" | peak GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB"
    print(line)


if __name__ == "__main__":
    main()
