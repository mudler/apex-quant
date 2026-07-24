"""Band-serialized (memory-bounded) Path-A imatrix generator.

Instead of a full forward (whole model resident), process the model in BANDS of
`--band` layers: load a band's weights from the safetensors shards ONCE, batch
ALL calibration chunks through it, cache inter-band activations, free the band.

Peak resident weights = band_size layers (not the whole model). Each weight is
read from disk exactly once. Mathematically identical to the full forward
(each decoder block is self-contained given its input activations).

Validated to match the full-forward Path-A imatrix on granite-4.0-h-tiny.
"""
import argparse, json, os, glob, resource
import numpy as np, torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridDecoderLayer, GraniteMoeHybridRotaryEmbedding,
    GraniteMoeHybridParallelExperts,
)
from transformers.masking_utils import create_causal_mask
from safetensors import safe_open
from imatrix_io import write_imatrix, read_imatrix
from gen_imatrix import map_name


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
        if isinstance(mod, GraniteMoeHybridParallelExperts):
            def pre(m, a, names=ggml, ne=mod.num_experts):
                inp, esz = a[0], a[1]
                x = inp.detach().float()
                parts = torch.split(x, list(esz), dim=0) if sum(esz) else []
                for nm in names:
                    e = ensure(nm, ne, x.shape[-1])
                    for ei, p in enumerate(parts):
                        if p.numel():
                            e["sums"][ei] += (p * p).sum(0).double().cpu().numpy()
                            e["counts"][ei] += p.shape[0]
            handles.append(mod.register_forward_pre_hook(pre))
        else:
            def pre(m, a, names=ggml):
                x = a[0].detach().float().reshape(-1, a[0].shape[-1])
                s = (x * x).sum(0).double().cpu().numpy()
                for nm in names:
                    e = ensure(nm, 1, x.shape[-1])
                    e["sums"][0] += s
                    e["counts"][0] += x.shape[0]
            handles.append(mod.register_forward_pre_hook(pre))
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=126)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--band", type=int, default=4, help="layers resident at once (memory knob)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ground-truth", default=None)
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.model)
    dev = args.device
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    reader = ShardReader(args.model)
    known = set(read_imatrix(args.ground_truth)[1]) if args.ground_truth else None
    nlayers = cfg.num_hidden_layers
    d = cfg.hidden_size
    block_type = cfg.layers_block_type

    # tokenize + window
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(open(args.calib, encoding="utf-8", errors="ignore").read(), return_tensors="pt").input_ids[0]
    nch = min(args.chunks, ids.shape[0] // args.ctx)
    chunks = torch.stack([ids[c*args.ctx:(c+1)*args.ctx] for c in range(nch)])  # [nch, ctx]
    print(f"layers={nlayers} band={args.band} chunks={nch} ctx={args.ctx} dtype={dtype}")

    # --- embed all chunks -> hidden cache on CPU ---
    embed_w = reader.get("model.embed_tokens.weight").to(dtype)
    hid = (torch.nn.functional.embedding(chunks, embed_w) * cfg.embedding_multiplier).to(dtype)  # [nch,ctx,d] CPU
    del embed_w
    print(f"hidden cache: {tuple(hid.shape)} ({hid.element_size()*hid.nelement()/1e6:.0f} MB CPU)")

    # --- rotary + causal mask (built once; position_ids identical per chunk) ---
    rotary = GraniteMoeHybridRotaryEmbedding(cfg).to(dev)
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
                L = GraniteMoeHybridDecoderLayer(cfg, i)
                materialize(L, f"model.layers.{i}", reader, dtype, dev)
                layers.append((i, L))
            named = [(f"model.layers.{i}.{n}", m) for i, L in layers for n, m in L.named_modules()]
            handles = make_hooks(named, acc, known)
            for c in range(nch):
                h = hid[c:c+1].to(dev)
                for i, L in layers:
                    mask = None if block_type[i] == "mamba" else causal
                    h = L(h, attention_mask=mask, past_key_values=None, position_embeddings=pos_emb)
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
    write_imatrix(args.out, entries, ["calibration_datav3"], nch, args.ctx)

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # GB (linux: KB)
    line = f"wrote {len(entries)} entries -> {args.out} | peak RSS {peak_rss:.1f} GB"
    if dev == "cuda":
        line += f" | peak GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB"
    print(line)


if __name__ == "__main__":
    main()
