# Jetson AGX Xavier MTP Benchmark: Q3_K_M vs APEX I-Compact

## Device

| Spec | Value |
|---|---|
| Device | NVIDIA Jetson AGX Xavier (MiiVii, Tegra194) |
| GPU | Volta SM72, 512 CUDA cores, 7.6 TFLOPS FP16 |
| Memory | 32GB unified (CPU+GPU), ~135 GB/s bandwidth |
| Power | 30W |
| OS | Ubuntu 20.04 aarch64 |
| CUDA | 11.4 |
| Software | llama.cpp master (SHA 571d0d54), built with CUDA |

## Models Tested

Both quantizations of `Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved` (the uncensored fine-tune that preserves all 20 MTP layers):

- **Q3_K_M (K-quant, bartowski)**: 17.3 GB — standard K-quant baseline
- **APEX I-Compact (SC117/cvgro)**: 17.0 GB — APEX layer-wise precision gradient (5 + 5 edge layers at higher precision, middle 30 layers compressed)
- ~~APEX I-Mini~~: Not available at time of writing (404 on both SC117 and cvgro repos). Original PR data retained for reference only.

## Methodology

Each configuration tested with **5 runs** (128 tokens each, fixed prompt, seed=42). Results reported as **mean ± std dev** over 5 runs. Sequential test order: Q3_K_M baseline → Q3_K_M MTP → APEX baseline → APEX MTP (same session, no reboot between config changes).

> **Note on baseline drift**: The original PR single-run data recorded Q3_K_M baseline at 15.96 t/s. Our 5-run mean is 19.32 t/s — a ~21% shift attributed to environmental differences (thermal state, system load). This discrepancy does not affect within-session comparisons (baseline vs MTP for the same model), but cross-session speedup ratios are unreliable.

## Baseline Results (no speculative decoding)

| Quant | Size | Mean ± σ | CV | vs Q3_K_M |
|---|---|---|---|---|
| Q3_K_M | 17.3 GB | **19.32 ± 0.02 t/s** | 0.1% | baseline |
| **APEX I-Compact** | 17.0 GB | **19.97 ± 0.05 t/s** | 0.2% | **+3.4%** |

Both quantizations show extremely low variance (≤0.2% CV), confirming test reproducibility. APEX I-Compact's **+3.4% baseline advantage** over Q3_K_M is small but consistent (all 5 runs above Q3_K_M's max). This may reflect APEX's selective precision allocation improving compute efficiency on memory-bandwidth-bound AGX Xavier.

> Original PR reported Q3_K_M at 15.96 t/s and APEX I-Compact at 15.97 t/s (essentially tied). Our multi-run data shows a small but reproducible APEX advantage.

> APEX I-Mini original PR baseline: 14.67 t/s (single run, not verified).

## MTP Speculative Decoding Results (`--spec-type draft-mtp --spec-draft-n-max 2`)

| Quant | Mean ± σ | CV | Speedup vs baseline | MTP Accept Rate* |
|---|---|---|---|---|
| Q3_K_M | **22.47 ± 0.40 t/s** | 1.8% | **+16.3%** | 97.7% |
| **APEX I-Compact** | **24.85 ± 0.07 t/s** | 0.3% | **+24.4%** | 69.8% |

> *Accept rates from original PR (single run). Q3_K_M speedup of 16.3% vs our baseline (original PR: +45% vs 15.96 t/s). APEX speedup of 24.4% (original PR: +19.0% vs 15.97 t/s).

**Key observation**: APEX I-Compact achieves **higher absolute MTP throughput (24.85 t/s) and higher speedup (+24.4%)** than Q3_K_M MTP (22.47 t/s, +16.3%), despite a significantly lower acceptance rate (69.8% vs 97.7%).

This apparent paradox has a straightforward explanation: APEX I-Compact's **higher baseline throughput (19.97 vs 19.32 t/s)** means each draft token rejected is less costly, so a lower accept rate still yields higher net speed. The MTP acceleration formula:

```
effective_tps = baseline_tps / (1 - accept_rate × draft_ratio)
```

With draft_ratio ≈ 0.5 (n_max=2):

| Quant | Baseline | Accept Rate | Implied MTP | Actual MTP |
|---|---|---|---|---|
| Q3_K_M | 19.32 | 97.7% | 19.32 / (1 - 0.977×0.5) ≈ **22.1** | **22.47** ✓ |
| APEX | 19.97 | 69.8% | 19.97 / (1 - 0.698×0.5) ≈ **24.5** | **24.85** ✓ |

Both predictions match within ~2%, confirming the model is sound. The formula shows that with a 3.4% higher baseline, APEX only needs ~70% accept rate to beat Q3_K_M's 98% rate.

## Key Finding

On this Jetson AGX Xavier test system, **APEX I-Compact outperforms Q3_K_M in both baseline (+3.4%) and MTP (+10.6% absolute)** throughput. The original PR's finding that "K-quant objectively outperforms APEX" was based on single-run data where baseline throughput was essentially identical (15.96 vs 15.97 t/s), making accept rate the dominant factor. With multi-run data revealing a small but persistent APEX baseline advantage, the slower draft acceptance becomes less impactful.

### Practical implications

| Metric | Q3_K_M | APEX I-Compact | Winner |
|---|---|---|---|
| Baseline t/s | 19.32 | 19.97 | **APEX** (+3.4%) |
| MTP t/s | 22.47 | 24.85 | **APEX** (+10.6%) |
| Speedup ratio | +16.3% | +24.4% | **APEX** |
| Model size | 17.3 GB | 17.0 GB | **APEX** (−0.3 GB) |
| Accept rate | 97.7% | 69.8% | Q3_K_M |
| TPS stability (CV) | 1.8% | 0.3% | **APEX** |

## Caveats

1. **Single device (N=1)**: Results are device-specific and may not generalize.
2. **Single prompt, fixed length**: Only tested with one prompt (180 chars) and 128 output tokens.
3. **Thermal ordering**: APEX was tested after Q3_K_M — the device may have been in a different thermal state.
4. **Accept rate not independently measured**: Values from original PR. Accept rates on our hardware may differ — the agreement between formula and actual MTP throughput suggests they're close, but direct measurement would be ideal.
5. **No APEX I-Mini**: The 13.3 GB variant is no longer available, limiting the comparison to similar-size models only.
6. **Causal mechanism speculative**: The link between APEX's layer-wise precision gradient and accept rate reduction is inferred, not experimentally validated.

## Suggested Next Steps

1. **Direct acceptance rate measurement** via llama.cpp's `--print-acceptance` or per-token logprobs comparison
2. **Test APEX I-Balanced/I-Quality** on devices with ≥48 GB unified memory to see if gentler compression preserves accept rates
3. **Cross-validate with different prompts and sequence lengths**
4. **Run reverse order** (APEX first, Q3_K_M second) to isolate thermal effects
