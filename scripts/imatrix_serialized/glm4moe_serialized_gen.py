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
"""
import argparse, json, os, glob, re
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.glm4_moe.modeling_glm4_moe import (
    Glm4MoeDecoderLayer, Glm4MoeMoE, Glm4MoeExperts,
    Glm4MoeMLP, Glm4MoeRotaryEmbedding,
)
from transformers.masking_utils import create_causal_mask
from safetensors import safe_open

import imatrix_io
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
        # routed experts: fused gate_up + separate down
        # Glm4MoeExperts holds gate_up_proj [num_experts, 2*interm, hidden]
        # and down_proj [num_experts, hidden, interm]
        "mlp.experts": [f"{b}.ffn_gate_exps.weight", f"{b}.ffn_up_exps.weight", f"{b}.ffn_down_exps.weight"],
        # shared experts
        "mlp.shared_experts.gate_proj": [f"{b}.ffn_gate_shexp.weight"],
        "mlp.shared_experts.up_proj": [f"{b}.ffn_up_shexp.weight"],
        "mlp.shared_experts.down_proj": [f"{b}.ffn_down_shexp.weight"],
    }
    return table.get(tail)


class ShardReader:
    def __init__(self, d):
        self.dir = d
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.wm = json.load(open(idx))["weight_map"]
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
    sd = {}
    for k in module.state_dict().keys():
        full = f"{prefix}.{k}"
        if full in reader.wm:
            sd[k] = reader.get(full).to(dtype)
    module.load_state_dict(sd, strict=False, assign=True)
    return module.to(device).eval()


def make_hooks(named_mods, acc, known):
    def ensure(name, nmat, nin):
        if name not in acc:
            acc[name] = {"sums": np.zeros((nmat, nin), np.float64), "counts": np.zeros(nmat, np.int64)}
        return acc[name]
    handles = []
    for hf_name, mod in named_mods:
        ggml = map_name(hf_name)
        if ggml is None:
            continue
        if known is not None:
            for g in ggml:
                if g not in known:
                    print(f"  WARN unmapped: {g} ({hf_name})")
        if isinstance(mod, Glm4MoeExperts):
            n_exp = mod.num_experts
            # forward signature: (hidden_states, top_k_index, top_k_weights)
            def pre_experts(m, a, names=ggml, ne=n_exp):
                inp, topk_idx, topk_w = a[0], a[1], a[2]
                x = inp.detach().float()  # [tokens, hidden]
                # A token contributes its input to each of the k selected experts.
                # We accumulate unweighted second moments (same as llama.cpp imatrix).
                k = topk_idx.shape[-1]
                # Reshape x so each selected expert slot gets its own row.
                x_rep = x.unsqueeze(1).expand(-1, k, -1).reshape(-1, x.shape[-1])
                flat_idx = topk_idx.reshape(-1).long()
                # Sum per expert using scatter_add
                sq = x_rep * x_rep
                per_expert_sum = torch.zeros(ne, x.shape[-1], dtype=torch.float64, device=x.device)
                per_expert_count = torch.zeros(ne, dtype=torch.float64, device=x.device)
                flat_idx_exp = flat_idx.unsqueeze(-1).expand(-1, x.shape[-1])
                per_expert_sum.scatter_add_(0, flat_idx_exp, sq.double())
                # Count each (token, expert) slot once
                per_expert_count.scatter_add_(0, flat_idx, torch.ones(flat_idx.shape[0], dtype=torch.float64, device=x.device))
                for nm in names:
                    e = ensure(nm, ne, x.shape[-1])
                    e["sums"] += per_expert_sum.cpu().numpy()
                    e["counts"] += per_expert_count.cpu().numpy().astype(np.int64)
            handles.append(mod.register_forward_pre_hook(pre_experts))
        else:
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
    ap.add_argument("--chunks", type=int, default=126)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--band", type=int, default=1, help="layers resident at once (memory knob)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ground-truth", default=None)
    ap.add_argument("--attn-impl", default="eager", help="eager/sdpa/flash_attention_2")
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    cfg._attn_implementation = args.attn_impl
    dev = args.device
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    reader = ShardReader(args.model)
    known = set(read_imatrix(args.ground_truth)[1]) if args.ground_truth else None
    nlayers = cfg.num_hidden_layers
    d = cfg.hidden_size

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text = open(args.calib, encoding="utf-8", errors="ignore").read()
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    nch = min(args.chunks, ids.shape[0] // args.ctx)
    chunks = torch.stack([ids[c*args.ctx:(c+1)*args.ctx] for c in range(nch)])
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
            handles = make_hooks(named, acc, known)
            for c in range(nch):
                h = hid[c:c+1].to(dev)
                for i, L in layers:
                    h = L(h, attention_mask=causal, position_embeddings=pos_emb, use_cache=False)
                hid[c:c+1] = h.to(hid.dtype).cpu()
            for hh in handles:
                hh.remove()
            for _, L in layers:
                del L
            layers.clear()
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
