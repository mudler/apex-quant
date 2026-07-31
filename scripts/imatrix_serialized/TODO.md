# Outstanding work — serialized imatrix backend (PR #18)

State as of commit `84ec9a7` on `feat/serialized-imatrix`. The review that drove that
commit is `pullrequestreview-4823632517` on
https://github.com/localai-org/apex-quant/pull/18

Everything below is *not done*. What IS done is in the `84ec9a7` commit message.

## Blocked on GPU time + real checkpoints

These are the two remaining items from the reviewer's own "what would get this to merge"
list. Both need hardware I did not have in the session that wrote this.

- [ ] **iq1_m or iq2_xxs KLD-vs-bf16 on Qwen3.5.** The single number that answers whether
      the 0.966 median correlation matters. Published PPL comparisons were run at Q6_K
      (`i-quality`) and Q4_K (`i-compact`) — the rungs where the imatrix has the least
      influence. Use `llama-perplexity --kl-divergence` against bf16, at `i-nano`
      (`iq2_xxs`) or `i-micro` (`iq1_m`). The concern is specifically `ffn_down_exps`:
      dominant tensor group by parameter count in a 256-expert MoE, among the most
      quantization-sensitive, and one of the poor per-tensor correlators.
- [ ] **Redo the Granite error bar as a paired statistic.** The "0.006 is inside ±0.107"
      claim uses the *absolute* PPL standard error. Both quants were scored on identical
      windows off an identical base, so the correct test is the standard error of the
      **paired per-window NLL difference**, which is much tighter. Needs the per-window
      NLLs from the original runs (or a re-run). The absolute effects (0.07% / 0.28%) are
      small enough that it probably survives, but as presented the claim isn't established.

## Consequences of the causal-attention fix — decide before trusting old numbers

`84ec9a7` fixed all five serialized generators attending **bidirectionally**
(`create_causal_mask()` returns `None` when `config._attn_implementation` is unset; these
generators drive standalone decoder layers, so a `None` mask is non-causal). Effect
measured on a synthetic 4-layer GraniteMoeHybrid: `blk.3.attn_output` went from 7.5e-01
relative disagreement with the full-forward generator (corr 0.54) to 3.2e-06.

- [ ] **Re-run the Granite validation.** The published 8.897-vs-8.903 PPL and
      368/368 / median-corr-0.998 numbers were produced with non-causal attention on the
      attention layers. On a mostly-mamba hybrid the affected tensors are a minority,
      which is plausibly why a 0.998 median didn't surface it — but the numbers should be
      regenerated before they're cited again.
- [ ] **Re-check the Qwen3.5 validation for the same reason** (corr 0.966 vs bartowski's
      canonical imatrix, i-compact PPL 6.775). Qwen3.5 has proportionally more full
      attention layers than Granite-hybrid, so the shift may be larger there.
- [ ] **Published artifacts** built on the old imatrices:
      https://huggingface.co/Myric/Qwen3.5-35B-A3B-APEX-GGUF and
      https://huggingface.co/Myric/Qwen3.6-35B-A3B-APEX-GGUF — decide whether to rebuild.
- [ ] Note also that per-chunk BOS is now forced (llama.cpp parity,
      `imatrix.cpp:865-866`); all pre-`84ec9a7` numbers were stream-windowed.
      `--no-bos-per-chunk` reproduces the old behaviour if you want an apples-to-apples
      comparison rather than a fresh baseline.

## Reply to the reviewer

- [ ] **Correct the Llama-4 finding (review item 4).** The reviewer says the generator
      "can't reproduce llama.cpp's statistic as written" because gate/up comes out
      router-weighted where llama.cpp accumulates unweighted. That is true of the generic
      MoE path but **not** of Llama-4, which llama.cpp special-cases:
      `const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4;` at
      `src/llama-graph.cpp:1837`, then L1976 multiplies the sigmoid-ed weights into `cur`
      **before** the gate/up `mul_mat_id`. The generic path weights the expert *output* at
      L2121. So a weighted stat is correct here; `test_llama4_expert_counts.py` pins it.
      The row-count fragility they flagged alongside it was real and is fixed.
- [ ] Report the non-causal-attention bug — it's more consequential than anything in the
      review and was found by building an end-to-end reproduction they didn't have.
- [ ] Mention that `Qwen3_5MoeExperts` exists in transformers 5.5.1, so the qwen3_5_moe
      path never needed 5.14+ (the README row was wrong); `dispatch.py` probes for symbols
      rather than versions for exactly this reason.

## Deferred refactor (reviewer's quality list, not a merge blocker)

- [ ] **Collapse the five near-duplicate ~200-line generators** into one core module plus
      per-arch name-map and hook adapters. Deliberately not attempted in `84ec9a7`: no
      Granite-hybrid / Qwen3.5 / GLM / Llama-4 checkpoint was available locally, so a
      refactor that size could not be verified against the validated paths. The
      testable pieces were extracted instead (`calib.py`, `dispatch.py`, the Llama-4 hooks
      are now module-level). `ShardReader` and the `ensure`/accumulator are still copied
      across all of them and are the obvious next extraction.

## Still untested against real checkpoints

- [ ] **GLM-4.5** (`glm4moe_serialized_gen.py`). The `down_exps` mapping bug is fixed and
      `test_glm4_expert_stats.py` covers the math synthetically, but nothing has run
      against a real GLM-4.5 checkpoint. Also needs a transformers with fused
      `Glm4MoeExperts` — absent in 5.5.x, so the import is guarded and `main()` explains.
- [ ] **Llama-4** (`serialized_gen_llama4.py`). Row counts and the weighting convention
      are now covered by a test through a real `Llama4TextMoe`, but no Scout run exists
      (bf16 doesn't fit 128 GB, which is the point of the backend). Validates by quant
      coherence, not correlation.
- [ ] Both are marked ⚠️ experimental in the README; keep them there until run.

## Context worth not re-deriving

- The user is **not** shipping their transformers 5.14 transition yet. `dispatch.py`
  therefore resolves Granite to `serialized_gen.py` (5.5.x) on the current install, and
  the `_t514*` variants simply don't resolve until a newer transformers is present. Don't
  "fix" this by pinning 5.14.
- Local llama.cpp checkout for cross-referencing: `/home/bryan/llama.cpp` (was at
  `6f3c0a79`). The convert script is now a package — model classes live in
  `conversion/*.py`, not `convert_hf_to_gguf.py`.
- No usable local checkpoint for these arches. `/home/bryan/models/granite-moe` is
  `granitemoe`, **not** `granitemoehybrid`. The end-to-end verification in `84ec9a7` used
  a synthetic 4-layer GraniteMoeHybrid built with random weights — that recipe is worth
  rebuilding rather than hunting for a real tiny checkpoint.
- Installed transformers is 5.5.1; torch 2.11.0+cu130.
