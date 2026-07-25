"""Band-serialized (VRAM-bounded) torch-imatrix generator for Qwen3.5-MoE (Qwen3_5Moe).

Qwen3.5-35B-A3B is a hybrid: GatedDeltaNet linear-attention layers + periodic full
attention (every `full_attention_interval`), a 256-routed + shared-expert MoE on every
layer, wrapped in a multimodal shell (text weights under `model.language_model.`) with a
separate `mtp` head we ignore.

Naming validated 510/510 against bartowski's public GGUF imatrix. The GatedDeltaNet is
all nn.Linear projections, so it needs no custom hook — only a name map:
  in_proj_qkv->attn_qkv, in_proj_z->attn_gate, in_proj_a->ssm_alpha,
  in_proj_b->ssm_beta, out_proj->ssm_out.
Experts use the same fused gate_up_proj/down_proj layout as granite-5.14, so the expert
hook (routing replay + down-input recompute) ports directly. The main model stores
experts fused, so materialize needs NO key remap — only the language_model prefix.
"""
import argparse, json, os, re, resource
import numpy as np, torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer, Qwen3_5MoeExperts, Qwen3_5MoeTopKRouter,
    Qwen3_5MoeTextRotaryEmbedding, Qwen3_5MoeRMSNorm,
)
from transformers.masking_utils import create_causal_mask
from safetensors import safe_open
from imatrix_io import write_imatrix, read_imatrix


def map_linear(hf):
    m = re.match(r".*layers\.(\d+)\.(.+)$", hf)
    if not m:
        return None
    b, tail = f"blk.{m.group(1)}", m.group(2)
    T = {
        "self_attn.q_proj": [f"{b}.attn_q.weight"], "self_attn.k_proj": [f"{b}.attn_k.weight"],
        "self_attn.v_proj": [f"{b}.attn_v.weight"], "self_attn.o_proj": [f"{b}.attn_output.weight"],
        "linear_attn.in_proj_qkv": [f"{b}.attn_qkv.weight"], "linear_attn.in_proj_z": [f"{b}.attn_gate.weight"],
        "linear_attn.in_proj_a": [f"{b}.ssm_alpha.weight"], "linear_attn.in_proj_b": [f"{b}.ssm_beta.weight"],
        "linear_attn.out_proj": [f"{b}.ssm_out.weight"],
        "mlp.shared_expert.gate_proj": [f"{b}.ffn_gate_shexp.weight"],
        "mlp.shared_expert.up_proj": [f"{b}.ffn_up_shexp.weight"],
        "mlp.shared_expert.down_proj": [f"{b}.ffn_down_shexp.weight"],
        "mlp.shared_expert_gate": [f"{b}.ffn_gate_inp_shexp.weight"],
    }
    return T.get(tail)


def expert_names(hf):
    m = re.match(r".*layers\.(\d+)\.", hf)
    if not m:
        raise ValueError(f"no layer index in {hf!r}")
    b = f"blk.{m.group(1)}"
    return {"gate": f"{b}.ffn_gate_exps.weight", "up": f"{b}.ffn_up_exps.weight", "down": f"{b}.ffn_down_exps.weight"}


class ShardReader:
    def __init__(self, d):
        self.dir = d
        self.wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.handles = {}
        cand = ["model.language_model", "language_model.model", "model"]
        self.prefix = next(p for p in cand if any(k.startswith(p + ".layers.") for k in self.wm))

    def get(self, name):
        fn = self.wm[name]
        if fn not in self.handles:
            self.handles[fn] = safe_open(os.path.join(self.dir, fn), framework="pt")
        return self.handles[fn].get_tensor(name)


def materialize(module, prefix, reader, dtype, device):
    sd, missing = {}, []
    for k in module.state_dict().keys():
        raw = f"{prefix}.{k}"
        if raw in reader.wm:
            sd[k] = reader.get(raw).to(dtype)
        else:
            missing.append(k)
    if missing:
        raise KeyError(f"{prefix}: missing {missing[:4]} (+{len(missing)-4})" if len(missing) > 4 else f"{prefix}: missing {missing}")
    module.load_state_dict(sd, strict=False, assign=True)
    return module.to(device).eval()


class Acc:
    def __init__(self, device):
        self.device = device; self.sums = {}; self.counts = {}
    def ensure(self, name, nmat, nin):
        if name not in self.sums:
            self.sums[name] = torch.zeros((nmat, nin), dtype=torch.float64, device=self.device)
            self.counts[name] = torch.zeros(nmat, dtype=torch.float64, device=self.device)
        return self.sums[name], self.counts[name]
    def to_entries(self):
        e = {}
        for name, s in self.sums.items():
            sums = s.cpu().numpy().astype(np.float32); counts = self.counts[name].cpu().numpy().astype(np.float32)
            e[name] = {"in_sum2": sums if sums.shape[0] > 1 else sums[0], "counts": counts}
        return e


def make_hooks(named_mods, acc, known):
    def note(names):
        if known is not None:
            for g in names:
                if g not in known:
                    print(f"  WARN not in ground truth: {g}")
    handles = []
    for hf, mod in named_mods:
        if isinstance(mod, torch.nn.Linear):
            names = map_linear(hf)
            if names is None:
                continue
            note(names)
            def pre_linear(m, a, names=names):
                x = a[0].detach().reshape(-1, a[0].shape[-1]).float()
                s = (x * x).sum(0).double(); n = x.shape[0]
                for nm in names:
                    S, C = acc.ensure(nm, 1, x.shape[-1]); S[0] += s; C[0] += n
            handles.append(mod.register_forward_pre_hook(pre_linear))
        elif isinstance(mod, Qwen3_5MoeExperts):
            en = expert_names(hf); note(list(en.values())); ne = mod.num_experts
            def pre_experts(m, a, en=en, ne=ne):
                hidden_states, top_k_index = a[0], a[1]
                x = hidden_states.detach().float()
                T, K = top_k_index.shape
                x2 = x * x
                assign = top_k_index.reshape(-1)
                tok = torch.arange(T, device=x.device).repeat_interleave(K)
                Sg, Cg = acc.ensure(en["gate"], ne, x.shape[-1]); Su, Cu = acc.ensure(en["up"], ne, x.shape[-1])
                contrib = x2.index_select(0, tok).double()
                Sg.index_add_(0, assign, contrib); Su.index_add_(0, assign, contrib)
                ones = torch.ones(T * K, dtype=torch.float64, device=x.device)
                Cg.index_add_(0, assign, ones); Cu.index_add_(0, assign, ones)
                Sd, Cd = acc.ensure(en["down"], ne, m.down_proj.shape[-1])
                for e in torch.unique(top_k_index):
                    idx = (top_k_index == e).any(dim=-1).nonzero(as_tuple=True)[0]
                    cur = x.index_select(0, idx).to(m.gate_up_proj.dtype)
                    g, u = torch.nn.functional.linear(cur, m.gate_up_proj[e]).chunk(2, dim=-1)
                    inter = (m.act_fn(g) * u).float()
                    Sd[e] += (inter * inter).sum(0).double(); Cd[e] += idx.numel()
            handles.append(mod.register_forward_pre_hook(pre_experts))
        elif isinstance(mod, Qwen3_5MoeTopKRouter):
            _m = re.match(r".*layers\.(\d+)\.", hf)
            gname = f"blk.{_m.group(1)}.ffn_gate_inp.weight"
            note([gname])
            def pre_router(m, a, gname=gname):
                x = a[0].detach().reshape(-1, a[0].shape[-1]).float()
                S, C = acc.ensure(gname, 1, x.shape[-1]); S[0] += (x * x).sum(0).double(); C[0] += x.shape[0]
            handles.append(mod.register_forward_pre_hook(pre_router))
    return handles


def materialize_mtp_layer(layer, reader, dtype, device):
    """Materialize the MTP decoder layer from mtp.layers.0.*. Handles both
    checkpoint layouts: 3.5 stores the MTP experts PER-EXPERT, while 3.6 stores
    them FUSED (gate_up_proj/down_proj) exactly like the main model."""
    pre = "mtp.layers.0"
    sd = {}
    ne = layer.mlp.experts.num_experts
    for k in layer.state_dict().keys():
        if k == "mlp.experts.gate_up_proj":
            if f"{pre}.mlp.experts.gate_up_proj" in reader.wm:  # 3.6: fused
                sd[k] = reader.get(f"{pre}.mlp.experts.gate_up_proj").to(dtype)
            else:                                                # 3.5: per-expert
                g = [reader.get(f"{pre}.mlp.experts.{e}.gate_proj.weight") for e in range(ne)]
                u = [reader.get(f"{pre}.mlp.experts.{e}.up_proj.weight") for e in range(ne)]
                sd[k] = torch.stack([torch.cat([g[e], u[e]], 0) for e in range(ne)]).to(dtype)
        elif k == "mlp.experts.down_proj":
            if f"{pre}.mlp.experts.down_proj" in reader.wm:      # 3.6: fused
                sd[k] = reader.get(f"{pre}.mlp.experts.down_proj").to(dtype)
            else:                                                # 3.5: per-expert
                d = [reader.get(f"{pre}.mlp.experts.{e}.down_proj.weight") for e in range(ne)]
                sd[k] = torch.stack(d).to(dtype)
        elif f"{pre}.{k}" in reader.wm:
            sd[k] = reader.get(f"{pre}.{k}").to(dtype)
    layer.load_state_dict(sd, strict=False, assign=True)
    return layer.to(device).eval()


def process_mtp(cfg, reader, hid, chunks, pos_emb, acc, known, dtype, dev, batch):
    """Cover the nextn/MTP head. transformers-5.14 has no MTP module, so we build it
    from raw checkpoint tensors and run the standard Qwen/DeepSeek MTP forward:

        h'_i = fc( concat[ enorm(embed(t_{i+1})) , hnorm(h_i) ] )   # 'eh_proj' order
        out  = mtp_decoder_layer(h'_i)                              # full-attn MoE layer

    where h_i is the main model's final hidden state (== `hid` after the band loop).
    The MTP block maps to ggml blk.{n_layers}: its decoder-layer tensors reuse the
    normal map (attn_*, ffn_*_exps/shexp, ffn_gate_inp*), plus fc -> nextn.eh_proj.

    NOTE: unlike the 510 main tensors, these values cannot be correlation-checked
    against a public imatrix (none covers MTP) and transformers cannot run the head,
    so this forward is reconstructed from the checkpoint. Names/shapes are validated
    against the gguf; the concat order and token/hidden shift follow the standard
    formulation and are documented assumptions.
    """
    P, H, L = reader.prefix, cfg.hidden_size, cfg.num_hidden_layers
    blk = f"blk.{L}"
    print(f"[mtp] building nextn head -> {blk}.*")
    # full-attention decoder layer for mtp.layers.0
    full_idx = next(i for i, t in enumerate(cfg.layer_types) if t == "full_attention")
    layer = materialize_mtp_layer(Qwen3_5MoeDecoderLayer(cfg, full_idx), reader, dtype, dev)
    fc = torch.nn.Linear(2 * H, H, bias=False)
    fc.load_state_dict({"weight": reader.get("mtp.fc.weight").to(dtype)}, assign=True); fc = fc.to(dev).eval()
    def rms(key):
        n = Qwen3_5MoeRMSNorm(H, eps=cfg.rms_norm_eps)
        n.load_state_dict({"weight": reader.get(key).to(dtype)}, assign=True); return n.to(dev).eval()
    enorm, hnorm = rms("mtp.pre_fc_norm_embedding.weight"), rms("mtp.pre_fc_norm_hidden.weight")
    embed_w = reader.get(f"{P}.embed_tokens.weight").to(dtype).to(dev)

    named = [(f"x.layers.{L}.{n}", m) for n, m in layer.named_modules()]
    handles = make_hooks(named, acc, known)
    ehname = f"{blk}.nextn.eh_proj.weight"
    if known is not None and ehname not in known:
        print(f"  WARN not in ground truth: {ehname}")
    def pre_fc(m, a):
        x = a[0].detach().reshape(-1, a[0].shape[-1]).float()
        S, C = acc.ensure(ehname, 1, x.shape[-1]); S[0] += (x * x).sum(0).double(); C[0] += x.shape[0]
    handles.append(fc.register_forward_pre_hook(pre_fc))

    nch = chunks.shape[0]
    with torch.no_grad():
        for s in range(0, nch, batch):
            sl = slice(s, min(s + batch, nch))
            h = hid[sl].to(dev)                                  # main final hidden [b,T,H]
            ids = chunks[sl].to(dev)
            emb = torch.nn.functional.embedding(ids, embed_w)    # [b,T,H]
            # position i predicts t_{i+2} from h_i and emb(t_{i+1}); align by one
            cat = torch.cat([enorm(emb[:, 1:]), hnorm(h[:, :-1])], dim=-1)  # [b,T-1,2H] (eh order)
            hp = fc(cat)
            bs, tm1 = hp.shape[0], hp.shape[1]
            mask = create_causal_mask(config=cfg, inputs_embeds=hp, attention_mask=None, past_key_values=None)
            out = layer(hp, position_embeddings=(pos_emb[0][:, :tm1], pos_emb[1][:, :tm1]),
                        attention_mask=mask, position_ids=torch.arange(tm1, device=dev).unsqueeze(0))
            _ = out[0] if isinstance(out, tuple) else out
    for hh in handles:
        hh.remove()
    del layer, fc
    if dev == "cuda":
        torch.cuda.empty_cache()
    print(f"[mtp] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=126)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--band", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset-label", default="calibration_datav3")
    ap.add_argument("--ground-truth", default=None)
    ap.add_argument("--mtp", action="store_true", help="also cover the nextn/MTP head -> blk.{n_layers}.*")
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.model).get_text_config()
    dev, dtype = args.device, (torch.bfloat16 if args.device == "cuda" else torch.float32)
    reader = ShardReader(args.model); P = reader.prefix
    known = set(read_imatrix(args.ground_truth)[1]) if args.ground_truth else None
    nlayers = cfg.num_hidden_layers
    ltype = cfg.layer_types
    print(f"qwen3.5: layers={nlayers} experts={getattr(cfg,'num_experts',None)} band={args.band} "
          f"batch={args.batch} prefix={P} dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(open(args.calib, encoding="utf-8", errors="ignore").read(), return_tensors="pt").input_ids[0]
    nch = min(args.chunks, ids.shape[0] // args.ctx)
    chunks = torch.stack([ids[c*args.ctx:(c+1)*args.ctx] for c in range(nch)])
    print(f"chunks={nch} ctx={args.ctx}")

    embed_w = reader.get(f"{P}.embed_tokens.weight").to(dtype)
    hid = torch.nn.functional.embedding(chunks, embed_w).to(dtype).to(dev)   # Qwen: no embedding multiplier
    del embed_w
    print(f"hidden cache on {dev}: {tuple(hid.shape)} ({hid.element_size()*hid.nelement()/1e6:.0f} MB)")

    rotary = Qwen3_5MoeTextRotaryEmbedding(cfg).to(dev)
    pos_ids = torch.arange(args.ctx, device=dev).unsqueeze(0)
    pos_emb = rotary(hid[:1], pos_ids)
    mask_cache = {}
    def causal_for(bs):
        if bs not in mask_cache:
            mask_cache[bs] = create_causal_mask(config=cfg, inputs_embeds=hid[:bs], attention_mask=None, past_key_values=None)
        return mask_cache[bs]

    acc = Acc(dev)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for b0 in range(0, nlayers, args.band):
            band = list(range(b0, min(b0 + args.band, nlayers)))
            layers = []
            for i in band:
                L = materialize(Qwen3_5MoeDecoderLayer(cfg, i), f"{P}.layers.{i}", reader, dtype, dev)
                layers.append((i, L))
            named = [(f"{P}.layers.{i}.{n}", m) for i, L in layers for n, m in L.named_modules()]
            handles = make_hooks(named, acc, known)
            for s in range(0, nch, args.batch):
                sl = slice(s, min(s + args.batch, nch))
                h = hid[sl]
                mask = causal_for(h.shape[0])
                for i, L in layers:
                    m = mask if ltype[i] == "full_attention" else None
                    out = L(h, position_embeddings=pos_emb, attention_mask=m,
                            position_ids=pos_ids, past_key_values=None)
                    h = out[0] if isinstance(out, tuple) else out
                hid[sl] = h
            for hh in handles:
                hh.remove()
            for _, L in layers:
                del L
            if dev == "cuda":
                torch.cuda.empty_cache()
            print(f"  band {band[0]}-{band[-1]} done", flush=True)

    if args.mtp:
        process_mtp(cfg, reader, hid, chunks, pos_emb, acc, known, dtype, dev, args.batch)

    write_imatrix(args.out, acc.to_entries(), [args.dataset_label], nch, args.ctx)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    line = f"wrote {len(acc.sums)} entries -> {args.out} | peak RSS {peak_rss:.1f} GB"
    if dev == "cuda":
        line += f" | peak GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB"
    print(line)


if __name__ == "__main__":
    main()
