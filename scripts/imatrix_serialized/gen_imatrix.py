"""Path-A imatrix generator: compute llama.cpp-compatible imatrix from an HF
model via forward hooks (per-input-channel sum of squared activations).

Memory-bounded: hooks capture only each matmul's input; you can load the model
with accelerate offload (device_map='auto', max_memory=...) and it still works,
because the statistic is per-tensor and additive over tokens.

Validated against llama.cpp's own imatrix for granite-4.0-h-tiny.
"""
import argparse, re, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridParallelExperts,
)
from imatrix_io import write_imatrix, read_imatrix


# ---- HF dotted module name -> list of ggml tensor names --------------------
def map_name(hf: str):
    m = re.match(r".*layers\.(\d+)\.(.+)$", hf)
    if not m:
        return None
    i, tail = m.group(1), m.group(2)
    b = f"blk.{i}"
    table = {
        "self_attn.q_proj": [f"{b}.attn_q.weight"],
        "self_attn.k_proj": [f"{b}.attn_k.weight"],
        "self_attn.v_proj": [f"{b}.attn_v.weight"],
        "self_attn.o_proj": [f"{b}.attn_output.weight"],
        "mamba.in_proj": [f"{b}.ssm_in.weight"],
        "mamba.out_proj": [f"{b}.ssm_out.weight"],
        "block_sparse_moe.router.layer": [f"{b}.ffn_gate_inp.weight"],
        # ParallelExperts: input_linear feeds BOTH gate and up (same input)
        "block_sparse_moe.input_linear": [f"{b}.ffn_gate_exps.weight", f"{b}.ffn_up_exps.weight"],
        "block_sparse_moe.output_linear": [f"{b}.ffn_down_exps.weight"],
        "shared_mlp.input_linear": [f"{b}.ffn_gate_shexp.weight", f"{b}.ffn_up_shexp.weight"],
        "shared_mlp.output_linear": [f"{b}.ffn_down_shexp.weight"],
    }
    return table.get(tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=126)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-memory", default=None, help="e.g. '10GiB' to force offload (demo memory-bound)")
    ap.add_argument("--ground-truth", default=None, help="real imatrix to cross-check name set")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    cpu = args.device == "cpu"
    dtype = torch.float32 if cpu else torch.bfloat16
    load_kw = dict(torch_dtype=dtype, low_cpu_mem_usage=True)
    if args.max_memory:
        load_kw.update(device_map="auto", max_memory={0: args.max_memory, "cpu": "200GiB"})
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    if not args.max_memory:
        model = model.to(args.device)
    model.eval()

    # accumulators: name -> {"sums": np.float64[nmat,in], "counts": np.int64[nmat]}
    acc = {}
    known = None
    if args.ground_truth:
        _, gt = read_imatrix(args.ground_truth)
        known = set(gt)

    def ensure(name, nmat, nin):
        if name not in acc:
            acc[name] = {"sums": np.zeros((nmat, nin), np.float64), "counts": np.zeros(nmat, np.int64)}
        return acc[name]

    handles = []
    mapped, skipped = [], []
    for hf_name, mod in model.named_modules():
        ggml = map_name(hf_name)
        if ggml is None:
            if isinstance(mod, (torch.nn.Linear, GraniteMoeHybridParallelExperts)):
                skipped.append(hf_name)
            continue
        if known is not None:
            for g in ggml:
                if g not in known:
                    print(f"  WARN mapped name not in ground truth: {g}  (from {hf_name})")
        mapped.append((hf_name, ggml))

        if isinstance(mod, GraniteMoeHybridParallelExperts):
            n_exp = mod.num_experts
            def pre_experts(m, a, names=ggml, ne=n_exp):
                inp, expert_size = a[0], a[1]
                x = inp.detach().float()  # [sum(expert_size), in]
                parts = torch.split(x, list(expert_size), dim=0) if sum(expert_size) else []
                for nm in names:
                    e = ensure(nm, ne, x.shape[-1])
                    for ei, p in enumerate(parts):
                        if p.numel():
                            e["sums"][ei] += (p * p).sum(0).double().cpu().numpy()
                            e["counts"][ei] += p.shape[0]
            handles.append(mod.register_forward_pre_hook(pre_experts))
        else:  # nn.Linear
            def pre_linear(m, a, names=ggml):
                x = a[0].detach().float().reshape(-1, a[0].shape[-1])
                s = (x * x).sum(0).double().cpu().numpy()
                for nm in names:
                    e = ensure(nm, 1, x.shape[-1])
                    e["sums"][0] += s
                    e["counts"][0] += x.shape[0]
            handles.append(mod.register_forward_pre_hook(pre_linear))

    print(f"hooked {len(mapped)} modules; skipped (not mapped) Linears: {skipped}")

    # tokenize whole calibration file, window into ctx-sized chunks
    text = open(args.calib, encoding="utf-8", errors="ignore").read()
    ids = tok(text, return_tensors="pt").input_ids[0]
    nchunks = min(args.chunks, ids.shape[0] // args.ctx)
    print(f"calib tokens={ids.shape[0]}, running {nchunks} chunks of {args.ctx}")
    dev = 0 if args.max_memory else args.device
    print(f"device={args.device} dtype={dtype}")
    with torch.no_grad():
        for c in range(nchunks):
            chunk = ids[c * args.ctx : (c + 1) * args.ctx].unsqueeze(0).to(dev)
            model(chunk)
            if (c + 1) % 20 == 0:
                print(f"  chunk {c+1}/{nchunks}")

    for h in handles:
        h.remove()

    # write imatrix: sums stored raw, counts per matrix
    entries = {}
    for name, d in acc.items():
        sums = d["sums"].astype(np.float32)          # [nmat, in]
        counts = d["counts"].astype(np.float32)      # [nmat]
        entries[name] = {"in_sum2": sums if sums.shape[0] > 1 else sums[0], "counts": counts}
    write_imatrix(args.out, entries, ["calibration_datav3"], nchunks, args.ctx)
    print(f"wrote {len(entries)} entries -> {args.out}")


if __name__ == "__main__":
    main()
