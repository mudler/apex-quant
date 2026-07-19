# Jetson AGX Xavier MTP Benchmark: Q3_K_M vs APEX Compact vs APEX Mini

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

All three are quantizations of `Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved` (the uncensored fine-tune that preserves all 20 MTP layers).

## Baseline Results (no speculative decoding)

| Quant | Size | Tokens/sec | vs Q3_K_M |
|---|---|---|---|
| bartowski Q3_K_M (K-quant) | 17.3 GB | **15.96 t/s** | baseline |
| APEX I-Compact (cvgro) | 16.9 GB | 15.97 t/s | +0.1% |
| APEX I-Mini (cvgro) | 13.3 GB | 14.67 t/s | -8.1% |

## MTP Speculative Decoding Results

| Quant | MTP t/s | Speedup | MTP Accept Rate | Eff. BW Util |
|---|---|---|---|---|
| **Q3_K_M** | **23.14 t/s** | **+45.0%** | **97.7%** | **386 GB/s** |
| APEX I-Compact | 19.01 t/s | +19.0% | 69.8% | 254 GB/s |
| APEX I-Mini | 18.31 t/s | +24.8% | 70.5% | 245 GB/s |

## Key Finding

**Q3_K_M achieves 97.7% MTP acceptance rate** because K-quant preserves the model's probability distribution across all 256 MoE experts. APEX quantizations use a layer-wise precision gradient (5+5 edge layers preserved, middle 30 layers aggressively compressed), which distorts the expert routing distribution. The MTP draft token acceptance rate drops from 97.7% to ~70%, severely reducing speculative decoding efficiency.

On a bandwidth-constrained edge device like Jetson Xavier (135 GB/s), the effective bandwidth utilization is:

- Q3_K_M: 135 × (1 / (1 - 0.977 × 0.667)) ≈ **386 GB/s** (2.86× effective)
- APEX Compact: 135 × (1 / (1 - 0.698 × 0.667)) ≈ **254 GB/s** (1.88× effective)

APEX wastes ~34% of available bandwidth potential due to MTP distribution mismatch.

## Conclusion

For edge devices with unified memory and tight bandwidth budgets, **K-quant (Q3_K_M) objectively outperforms APEX Compact/I-Mini** for Qwen3.6-35B-A3B when using MTP speculative decoding. The 0.4 GB size savings from APEX I-Compact (16.9 vs 17.3 GB) come at the cost of a 45% → 19% MTP speedup collapse.

APEX I-Balanced (24 GB) or I-Quality (22 GB) would likely preserve MTP acceptance rates better, but both exceed the 32 GB unified memory limit of Jetson-class devices.
