"""torch-imatrix generator, transformers-5.14 variant.

Same statistic and output format as gen_imatrix.py (the validated
transformers-5.5.1 tool), adapted to the 5.14 GraniteMoeHybrid rewrite:

  * `GraniteMoeHybridParallelExperts` (fused input_linear/output_linear, called
    with (inputs, expert_size)) was replaced by `GraniteMoeHybridExperts`, a
    module holding `gate_up_proj`/`down_proj` *parameters* and doing per-expert
    `F.linear` inside its forward. There are no expert Linear submodules to
    pre-hook, so we hook the Experts module itself, replay the router's
    per-expert token grouping, and accumulate Sum x^2 for:
        - gate/up experts input  = hidden_states[tokens routed to e]   (dim hidden)
        - down experts input     = act_fn(gate) * up   (recomputed)     (dim inter)
    gate and up share one input (identical stats), matching the original.
  * shared MLP still uses real nn.Linear (input_linear/output_linear) and the
    attention/mamba projections are real nn.Linear -> handled generically.
  * the router in 5.14 is a bare nn.Parameter (F.linear), no submodule; its
    ggml tensor `ffn_gate_inp` is stored F32 and does not consume an imatrix
    (confirmed: it is byte-identical across imatrices), so we skip it.

Mapping is derived from the *loaded* model's named_modules() and cross-checked
against a ground-truth imatrix tensor set; any unmapped Linear or name not in
ground truth aborts before the (expensive) calibration run.
"""
import argparse, re, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridExperts,
    GraniteMoeHybridTopKRouter,
)
from imatrix_io import write_imatrix, read_imatrix


def map_linear(hf: str):
    """HF dotted module name (an nn.Linear) -> list of ggml tensor names, or None."""
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
        # shared MLP: input_linear feeds BOTH gate and up (same input)
        "shared_mlp.input_linear": [f"{b}.ffn_gate_shexp.weight", f"{b}.ffn_up_shexp.weight"],
        "shared_mlp.output_linear": [f"{b}.ffn_down_shexp.weight"],
    }
    return table.get(tail)


def expert_names(hf: str):
    """HF dotted name of a GraniteMoeHybridExperts module -> per-role ggml names."""
    m = re.match(r".*layers\.(\d+)\.", hf)
    b = f"blk.{m.group(1)}"
    return {
        "gate": f"{b}.ffn_gate_exps.weight",
        "up": f"{b}.ffn_up_exps.weight",
        "down": f"{b}.ffn_down_exps.weight",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=126)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-memory", default=None, help="e.g. '12GiB' to cap GPU and offload rest")
    ap.add_argument("--dataset-label", default="calibration_datav3")
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

    known = None
    if args.ground_truth:
        _, gt = read_imatrix(args.ground_truth)
        known = set(gt)

    acc = {}  # name -> {"sums": f64[nmat,in], "counts": i64[nmat]}
    def ensure(name, nmat, nin):
        if name not in acc:
            acc[name] = {"sums": np.zeros((nmat, nin), np.float64), "counts": np.zeros(nmat, np.int64)}
        return acc[name]

    def check(ggml_names):
        if known is None:
            return
        for g in ggml_names:
            if g not in known:
                print(f"  WARN mapped name not in ground truth: {g}")

    handles = []
    mapped_lin, mapped_exp, skipped = [], [], []
    for hf_name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            names = map_linear(hf_name)
            if names is None:
                skipped.append(hf_name)
                continue
            check(names)
            mapped_lin.append(hf_name)
            def pre_linear(m, a, names=names):
                x = a[0].detach().float().reshape(-1, a[0].shape[-1])
                s = (x * x).sum(0).double().cpu().numpy()
                for nm in names:
                    e = ensure(nm, 1, x.shape[-1])
                    e["sums"][0] += s
                    e["counts"][0] += x.shape[0]
            handles.append(mod.register_forward_pre_hook(pre_linear))
        elif isinstance(mod, GraniteMoeHybridExperts):
            en = expert_names(hf_name)
            check(list(en.values()))
            mapped_exp.append(hf_name)
            ne = mod.num_experts
            def pre_experts(m, a, en=en, ne=ne):
                # forward(hidden_states, top_k_index, top_k_weights)
                hidden_states, top_k_index = a[0], a[1]
                x = hidden_states.detach().float()                 # [tokens, hidden]
                gate = ensure(en["gate"], ne, x.shape[-1])
                up = ensure(en["up"], ne, x.shape[-1])
                # down input dim = intermediate; lazily sized on first expert
                down = None
                with torch.no_grad():
                    mask = torch.nn.functional.one_hot(top_k_index, num_classes=ne).permute(2, 1, 0)
                    hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
                    for eidx in hit:
                        e = eidx[0]
                        if e == ne:
                            continue
                        _, tok_idx = torch.where(mask[e])
                        cur = x[tok_idx]                            # [n_e, hidden] gate/up input
                        s = (cur * cur).sum(0).double().cpu().numpy()
                        gate["sums"][e] += s; gate["counts"][e] += cur.shape[0]
                        up["sums"][e] += s;   up["counts"][e] += cur.shape[0]
                        # recompute down input exactly as the module does
                        gu = torch.nn.functional.linear(cur.to(m.gate_up_proj.dtype), m.gate_up_proj[e])
                        g, u = gu.chunk(2, dim=-1)
                        inter = (m.act_fn(g) * u).float()          # [n_e, inter] down input
                        nonlocal_down = ensure(en["down"], ne, inter.shape[-1])
                        sd = (inter * inter).sum(0).double().cpu().numpy()
                        nonlocal_down["sums"][e] += sd; nonlocal_down["counts"][e] += inter.shape[0]
            handles.append(mod.register_forward_pre_hook(pre_experts))
        elif isinstance(mod, GraniteMoeHybridTopKRouter):
            # router is a bare nn.Parameter (F.linear) in 5.14, not a Linear
            # submodule; its input maps to ffn_gate_inp (stored F32 in the quant,
            # imatrix-unused, but present in the shipped imatrix -> capture it).
            mr = re.match(r".*layers\.(\d+)\.", hf_name)
            gname = f"blk.{mr.group(1)}.ffn_gate_inp.weight"
            check([gname])
            mapped_lin.append(hf_name)
            def pre_router(m, a, gname=gname):
                x = a[0].detach().float().reshape(-1, a[0].shape[-1])
                s = (x * x).sum(0).double().cpu().numpy()
                e = ensure(gname, 1, x.shape[-1])
                e["sums"][0] += s
                e["counts"][0] += x.shape[0]
            handles.append(mod.register_forward_pre_hook(pre_router))

    print(f"hooked {len(mapped_lin)} Linear/router + {len(mapped_exp)} Experts modules")
    # lm_head is tied to token_embd (tie_word_embeddings) and quantized without an
    # imatrix — the original 5.5.1 tool skipped it too. Any *other* unmapped Linear
    # is a real gap and aborts before the expensive run.
    INTENTIONAL_SKIP = ("lm_head",)
    fatal = [s for s in skipped if not s.endswith(INTENTIONAL_SKIP)]
    if skipped:
        print(f"skipped (intentional) Linears: {skipped}")
    if fatal:
        print(f"UNMAPPED Linears ({len(fatal)}): {fatal}")
        sys.exit("ABORT: unmapped Linear tensors — fix map_linear() before running")

    text = open(args.calib, encoding="utf-8", errors="ignore").read()
    ids = tok(text, return_tensors="pt").input_ids[0]
    nchunks = min(args.chunks, ids.shape[0] // args.ctx)
    print(f"calib tokens={ids.shape[0]}, running {nchunks} chunks of {args.ctx}; device={args.device} dtype={dtype}")
    dev = 0 if args.max_memory else args.device
    with torch.no_grad():
        for c in range(nchunks):
            chunk = ids[c * args.ctx:(c + 1) * args.ctx].unsqueeze(0).to(dev)
            model(chunk)
            if (c + 1) % 10 == 0:
                print(f"  chunk {c+1}/{nchunks}", flush=True)

    for h in handles:
        h.remove()

    entries = {}
    for name, d in acc.items():
        sums = d["sums"].astype(np.float32)
        counts = d["counts"].astype(np.float32)
        entries[name] = {"in_sum2": sums if sums.shape[0] > 1 else sums[0], "counts": counts}
    write_imatrix(args.out, entries, [args.dataset_label], nchunks, args.ctx)
    print(f"wrote {len(entries)} entries -> {args.out}")


if __name__ == "__main__":
    main()
