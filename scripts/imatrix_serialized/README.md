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
| `gen_imatrix.py` | full-forward generator: hook every Linear / fused-expert module, accumulate Σx² |
| `serialized_gen.py` | **band-serialized** generator (the memory-bounded backend the pipeline calls) |
| `compare.py` | per-tensor correlation + L1 error vs a reference imatrix |
| `test_roundtrip.py` | read real → write → read → assert identical |
| `check_gate_up.py` | confirms gate/up experts share input stats (corr 1.0) |

### Per-architecture generator variants
Per-arch name mapping is the extension surface. Variants shipped:

| Script | Architecture | transformers | Status |
|---|---|---|---|
| `serialized_gen.py`, `gen_imatrix.py` | GraniteMoeHybrid | 5.5.x (`…ParallelExperts`) | ✅ validated |
| `serialized_gen_t514.py`, `_t514_fast.py`, `gen_imatrix_t514.py` | GraniteMoeHybrid | 5.14+ (`…Experts`; `_fast` VRAM-bounded) | ✅ validated |
| `serialized_gen_qwen35.py` | `qwen3_5_moe` (Qwen3.5/3.6-35B-A3B; GatedDeltaNet + MoE + NextN/MTP head) | 5.14+ | ✅ validated |
| `glm4moe_serialized_gen.py` | GLM-4.5-MoE | 5.5.x | ⚠️ experimental (untested) |
| `serialized_gen_llama4.py` | Llama-4 | 5.5.x | ⚠️ experimental (untested) |

## Setup
Install the optional deps into an environment with a **CUDA** torch build matching
your GPU (`requirements.txt`), then set `IMATRIX_BACKEND=serialized` (optionally
`IMATRIX_BAND=<n>`, `IMATRIX_DEVICE=cuda`) when invoking `apex_pipeline.sh`. Standalone:
```bash
python3 scripts/imatrix_serialized/serialized_gen.py --model <hf_dir> \
  --calib <calibration.txt> --out imatrix.dat --band 1 --device cuda
```

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
warns. Per-arch dispatch in the pipeline is a natural follow-up (currently the Phase-6
call targets `serialized_gen.py`; other arches run their variant directly).

## Known limitations
Prototype favors correctness: eager attention, per-band CPU↔GPU activation transfers.
Production TODOs: sdpa, batched chunks, keep activations on-GPU between bands,
`causal-conv1d`. `safe_open` mmaps the shards so peak *RSS* shows ~model size, but that
is reclaimable page cache — the committed working set (and GPU) is band-bounded.
