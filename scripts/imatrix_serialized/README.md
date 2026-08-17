# Serialized imatrix backend (`IMATRIX_BACKEND=serialized`)

A memory-bounded PyTorch reimplementation of llama.cpp's importance-matrix
computation, done via forward hooks on the HF safetensors. Motivation:
`llama-imatrix` needs the whole model resident (RAM+VRAM), which is impractical for
large MoE / hybrid models (and, for hybrid SSM architectures like `qwen3_5_moe`, the
llama.cpp imatrix path falls back to a serial CPU scan). This decouples the imatrix
from llama.cpp's memory model: peak resident weights = `IMATRIX_BAND` layers,
independent of model size.

Wired into `apex_pipeline.sh` Phase 6 as an **opt-in** backend
(`IMATRIX_BACKEND=serialized`, default `llama` unchanged). Reads the HF safetensors
fetched in Phase 2 and writes the same `imatrix.dat` Phase 7 consumes. torch /
transformers / accelerate are only needed for this backend — see `requirements.txt`.

## What the imatrix is
Per weight tensor `[out, in]`, a per-input-channel second moment `Σ_t x_j²`, stored
in a GGUF as two F32 tensors per weight:
- `<name>.in_sum2` `[nmat, in]` — raw Σx² (nmat = n_experts, or 1 for dense)
- `<name>.counts`  `[nmat, 1]` — token count per expert/matrix

Size is set by `Σ nmat×in` (independent of `out` and of calibration size).

## Files
| File | Role |
|---|---|
| `imatrix_io.py` | read/write GGUF imatrix (bit-exact round-trip) |
| `calib.py` | shared calibration tokenization/windowing incl. llama.cpp's per-chunk BOS |
| `dispatch.py` | picks the generator for a checkpoint's `model_type` by probing the installed transformers; what Phase 6 calls |
| `gen_imatrix.py` | full-forward generator: hook every Linear / fused-expert module, accumulate Σx² |
| `serialized_gen.py` | **band-serialized** generator (GraniteMoeHybrid on transformers 5.5.x) |
| `compare.py` | per-tensor correlation + L1 error vs a reference imatrix |
| `check_gate_up.py` | diagnostic: confirms tensors sharing a graph input share an imatrix stat (takes a file path) |

Tests — all synthetic, none need a checkpoint/GPU/network, all run in CI:

| Test | Covers |
|---|---|
| `test_roundtrip.py` | GGUF format layer: write → read → bit-exact; llama.cpp's `in_sum2`/`counts` layout; metadata keys. numpy+gguf only |
| `test_calib.py` | windowing, chunk cap, per-chunk BOS parity with llama.cpp, short-corpus hard-fail |
| `test_glm4_expert_stats.py` | routed-expert Σx² shapes/values vs an independent per-expert reference; both fused weight layouts; hard-fail cases |
| `test_llama4_expert_counts.py` | drives the Llama-4 hooks through a real `Llama4TextMoe`: exact top-k row counts, router-weighted convention |
| `test_dispatch.py` | every arch entry resolves to an existing script; unknown/unavailable arch messages |
| `test_causal_mask.py` | that every serialized generator sets `_attn_implementation`, without which `create_causal_mask()` returns `None` and the standalone layers attend bidirectionally |

### Per-architecture generator variants
Per-arch name mapping is the extension surface. **You do not pick these by hand** —
`dispatch.py` reads `model_type` from the checkpoint's `config.json` and probes the
installed transformers for the symbols each variant imports, because the version ranges
are genuinely misleading (5.5.1 already has `Qwen3_5MoeExperts` and `Llama4Router`, but
not `GraniteMoeHybridExperts`). To see what your install resolves to:

```bash
python3 scripts/imatrix_serialized/dispatch.py --model <hf_dir> --print-script
```

| Script | Architecture | needs | Status |
|---|---|---|---|
| `serialized_gen.py`, `gen_imatrix.py` | GraniteMoeHybrid | `GraniteMoeHybridParallelExperts` (5.5.x) | ✅ validated |
| `serialized_gen_t514.py`, `_t514_fast.py`, `gen_imatrix_t514.py` | GraniteMoeHybrid | `GraniteMoeHybridExperts` (5.14+; `_fast` VRAM-bounded) | ✅ validated |
| `serialized_gen_qwen35.py` | `qwen3_5_moe` (Qwen3.5/3.6-35B-A3B; GatedDeltaNet + MoE + NextN/MTP head) | `Qwen3_5MoeExperts` (present in 5.5.1) | ✅ validated |
| `glm4moe_serialized_gen.py` | GLM-4.5-MoE | fused `Glm4MoeExperts` (absent in 5.5.x) | ⚠️ experimental (untested) |
| `serialized_gen_llama4.py` | Llama-4 | `Llama4TextExperts` (present in 5.5.1) | ⚠️ experimental (untested) |

Every generator hard-fails (`KeyError`) if any band/head parameter has no checkpoint
tensor. A skipped key would leave randomly-initialized weights in the band and emit a
plausible-looking but meaningless imatrix, so a name mismatch is never survivable.

**Llama-4 note:** its routed gate/up statistic is *router-weighted*, and that is
correct, not a divergence — llama.cpp special-cases this arch
(`weight_before_ffn = arch == LLM_ARCH_LLAMA4`, `src/llama-graph.cpp:1837`) and folds
the sigmoid-ed weights into the expert input **before** the gate/up `mul_mat_id`
(L1976), where the generic MoE path instead weights the expert *output* (L2121).

## Setup

> **Install a CUDA torch build, not the default CPU wheel.** `pip install torch`
> pulls the **CPU-only** wheel, and the generator will then run entirely on CPU
> (`--device cuda` will error or, on CPU, crawl — orders of magnitude slower, and
> the hybrid-SSM path is especially slow on CPU). Install torch from the CUDA index
> matching your driver, e.g.:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick cuXXX for your CUDA
> pip install transformers accelerate numpy gguf                          # rest of requirements.txt
> python3 -c "import torch; assert torch.cuda.is_available(), 'CPU-only torch — reinstall from the CUDA index'"
> ```

Then set `IMATRIX_BACKEND=serialized` (optionally `IMATRIX_BAND=<n>`,
`IMATRIX_DEVICE=cuda`, `IMATRIX_BATCH=<n>`, `IMATRIX_MTP=true`) when invoking
`apex_pipeline.sh`. Phase 6 calls `dispatch.py`, which selects the generator from the
checkpoint's `model_type`. Standalone, either let dispatch pick:
```bash
python3 scripts/imatrix_serialized/dispatch.py --model <hf_dir> \
  --calib <calibration.txt> --out imatrix.dat --band 1 --device cuda
```
or run a variant directly (`serialized_gen.py`, `serialized_gen_qwen35.py`, ...) with
the same flags.

### Key flags
- **`--attn-impl`** (default `eager`) — the attention implementation stamped onto the
  config *before* masks are built. This is not cosmetic: `create_causal_mask()` returns
  `None` when `config._attn_implementation` is unset, and because these generators drive
  standalone decoder layers rather than a full model, a `None` mask makes them attend
  **bidirectionally** — a silently non-causal imatrix. On a synthetic 4-layer
  GraniteMoeHybrid, `blk.3.attn_output` disagreed with the full-forward generator by
  7.5e-01 (corr 0.54) unset, and by 3.2e-06 set.
- **`--band N` / `--batch N`** — throughput knobs. `band=1 batch=1` is the memory-minimal
  but *slowest* corner: the GPU starves between per-layer load + host↔device transfer cycles.
  Raise `--batch` for bigger GEMMs, and `--band` to amortize the per-band weight load over
  more compute. Both are **result-invariant** (Σx² is order/batch/band-invariant) — tune
  them purely for speed vs. VRAM, the imatrix comes out identical.
- **`--no-bos-per-chunk`** — opt out of forcing BOS at position 0 of every chunk.
  llama.cpp *does* force it (`tools/imatrix/imatrix.cpp:865-866` overwrites the first
  token of each chunk when the vocab has `add_bos`), so forcing it is the default here
  and only this flag reproduces a plain stream-window. Note the published validation
  numbers predate the fix, i.e. they were measured stream-windowed; the effect is ~1
  token in `ctx` (0.2% at `ctx=512`).
- **`--mtp`** *(generators for models with a NextN/MTP head, e.g. `serialized_gen_qwen35.py`)* —
  also compute importance for the multi-token-prediction head (`blk.{n_layers}.*`).
  `llama-imatrix` cannot cover it; omit the flag to match a head-less reference file.

  The `nextn.eh_proj` input ordering is **verified against llama.cpp**, not assumed:
  `src/models/qwen35moe.cpp` builds the head as
  `ggml_concat(ctx0, e_norm, h_norm, /*dim=*/0)` and `ggml_compute_forward_concat`
  writes `src0` at the low indices, so input channels `[0, n_embd)` are the
  embedding-norm half and `[n_embd, 2*n_embd)` the hidden-norm half. The generator
  builds `torch.cat([enorm(emb), hnorm(h)], dim=-1)` — same order. The GGUF name comes
  from `conversion/qwen.py`, which renames `mtp.fc` → `layers.{n_layer}.eh_proj` with no
  transpose, so the channel order carries through to `blk.{n_layers}.nextn.eh_proj.weight`.

  The one-position shift is verified too, caller-side: `common/speculative.cpp` submits
  the token sampled *at* a position together with that position's hidden state in one
  batch row, i.e. `embed(t_{i+1})` with `h_i` — matching the generator's
  `emb[:, 1:]` / `h[:, :-1]`. So no MTP imatrix needs rebuilding on this account; what
  is still unvalidated is the numerical result (no public imatrix covers MTP), not the
  wiring.

### Calibration corpus
Use a **diverse** calibration file — mixed prose, code, and math — not wikitext alone.
The imatrix records per-channel activation statistics, so channels that only fire on
(e.g.) code or non-English text get no signal from an all-Wikipedia corpus, which
matters most for coding/agentic and multilingual models. Bartowski's
`calibration_datav3` is a good general default; weight toward your target domain.
Calibration diversity changes the imatrix *values*, not its size or tensor coverage.

The calibration corpus also governs *reproducibility* and *specialization*: re-running with
the **same** calibration reproduces an imatrix to ~0.99 median per-tensor correlation, while a
**different** corpus legitimately shifts the values. That shift concentrates in the routed
**expert** projections (`ffn_*_exps`) — the experts are what specialize by domain — whereas
attention and shared-expert paths stay ~1.0. So a code-weighted calibration measurably favors
coding channels (at a small cost elsewhere); reach for it deliberately when specializing for a
single domain, and use the *same* calibration when your goal is to reproduce a reference.

## Validation

### granite-4.0-h-tiny — end-to-end PPL parity
Same i-quality config + Q6_K base + eval, swapping only the imatrix (200×512 windows):

| imatrix | i-quality PPL |
|---|---|
| **serialized (this tool)** | **8.8966 ± 0.107** |
| llama.cpp (ground truth) | 8.9030 ± 0.107 |

Difference (0.006) is an order of magnitude inside the error bars — statistically
indistinguishable. Format bit-exact; names 368/368 exact; median per-tensor
correlation 0.998 (0.995 band-serialized). band=4 peak GPU 6.4 GB (vs ~13 GB full).

### Qwen3.5-35B-A3B (`qwen3_5_moe`) — second architecture
The headline apex-quant arch. Validated against
[bartowski's canonical llama.cpp imatrix](https://huggingface.co/bartowski/Qwen_Qwen3.5-35B-A3B-GGUF):

- **Name mapping:** 523 tensors covered, `only-real = []` (strict superset — also
  covers the NextN/MTP head, which `llama-imatrix` does not); median per-tensor
  correlation **0.966** (only `ssm_out` low, see below).
- **PPL parity** (i-compact, 200×512 windows): bf16 6.620 · llama.cpp-imatrix 6.756 ·
  **serialized-imatrix 6.775** — the two quants differ by 0.02 PPL, inside the ±0.073
  error bars. Statistically indistinguishable.

Published for verification:
[Qwen3.5](https://huggingface.co/Myric/Qwen3.5-35B-A3B-APEX-GGUF) ·
[Qwen3.6](https://huggingface.co/Myric/Qwen3.6-35B-A3B-APEX-GGUF) (both include the
serialized imatrix).

**Note on `down_*` / `ssm_out`:** these correlate poorly per-tensor (inputs are
post-nonlinearity: SiLU-gated intermediates, SSM scan output). It does **not** affect
PPL — the quantizer needs only relative per-channel importance, which is preserved.

## Extending to a new architecture
`map_name()` maps HF dotted module names → ggml tensor names and dispatches by module
type (`nn.Linear` → all-token; fused-expert module → per-expert split). For a new arch,
add its name mapping (and its fused-expert hook if different). Cross-check against a
real imatrix's tensor-name set via `compare.py` / `--ground-truth` — any unmapped name
warns. Then add the arch to the `ARCHES` table in `dispatch.py` (script + the
transformers symbols it imports) so Phase 6 can reach it; `test_dispatch.py` asserts
every table entry points at a script that exists and resolves on the installed
transformers.

## Known limitations
**Not yet measured where it matters most.** The published PPL comparisons were run at
Q6_K (`i-quality`) and Q4_K (`i-compact`), the rungs where the imatrix has the least
influence. The ladder goes down to `i-nano`=`iq2_xxs` and `i-micro`=`iq1_m`, and the
discriminating test is one of those by KL-divergence vs bf16
(`llama-perplexity --kl-divergence`), which is far more sensitive to imatrix quality
than PPL. Until that number exists, "does not affect PPL" for the poorly-correlating
`down_*`/`ssm_out` tensors is an argument from a measurement that cannot show it.
Relatedly, the ±0.107 error bar quoted for the Granite comparison is the *absolute*
PPL standard error; since both quants were scored on identical windows off an identical
base, the correct statistic is the standard error of the paired per-window NLL
difference, which is much tighter.

Prototype favors correctness: eager attention, per-band CPU↔GPU activation transfers.
Production TODOs: sdpa, batched chunks, keep activations on-GPU between bands,
`causal-conv1d`. `safe_open` mmaps the shards so peak *RSS* shows ~model size, but that
is reclaimable page cache — the committed working set (and GPU) is band-bounded.
