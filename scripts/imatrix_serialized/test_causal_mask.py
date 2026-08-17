"""Guards against a silently NON-CAUSAL imatrix. No checkpoint needed.

The bug: transformers' create_causal_mask() returns None when
config._attn_implementation is unset (it assumes the attention backend applies
causality itself, as sdpa does via is_causal). The band-serialized generators do not
run a model -- they invoke standalone decoder layers -- so a None mask means those
layers attend BIDIRECTIONALLY. Nothing errors; you just get an imatrix whose
attn_output statistic (and everything downstream of it) was computed with future
tokens visible.

Measured on a synthetic 4-layer GraniteMoeHybrid: serialized vs full-forward
blk.3.attn_output disagreed by 7.5e-01 (corr 0.54) with the implementation unset, and
by 3.2e-06 once it was set. So this is worth a test rather than a comment.

Run: python3 test_causal_mask.py   (or under pytest)
"""
import os
import re

import torch
from transformers.masking_utils import create_causal_mask
from transformers.models.granitemoehybrid.configuration_granitemoehybrid import (
    GraniteMoeHybridConfig,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# Every generator that builds masks itself and drives standalone decoder layers.
SERIALIZED_GENERATORS = [
    "serialized_gen.py",
    "serialized_gen_t514.py",
    "serialized_gen_t514_fast.py",
    "serialized_gen_qwen35.py",
    "serialized_gen_llama4.py",
    "glm4moe_serialized_gen.py",
]


def _cfg():
    return GraniteMoeHybridConfig(hidden_size=64, num_hidden_layers=2,
                                  num_attention_heads=4, num_key_value_heads=2,
                                  intermediate_size=128, mamba_n_heads=4,
                                  mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                                  layer_types=["attention", "attention"])


def test_unset_implementation_really_yields_no_mask():
    """Pins the transformers behaviour the bug depends on, so a change is visible."""
    cfg = _cfg()
    assert getattr(cfg, "_attn_implementation", None) in (None, "eager"), \
        "fixture assumption changed"
    cfg._attn_implementation = None
    m = create_causal_mask(config=cfg, inputs_embeds=torch.zeros(1, 6, cfg.hidden_size),
                           attention_mask=None, past_key_values=None)
    assert m is None, f"expected None (the trap), got {type(m)}"


def test_setting_eager_yields_a_real_causal_mask():
    cfg = _cfg()
    cfg._attn_implementation = "eager"
    m = create_causal_mask(config=cfg, inputs_embeds=torch.zeros(1, 6, cfg.hidden_size),
                           attention_mask=None, past_key_values=None)
    assert m is not None, "no mask -> standalone layers would attend bidirectionally"
    assert m.shape[-2:] == (6, 6), m.shape
    # strictly lower-triangular-inclusive: position i must not see j > i
    allowed = (m[0, 0] == 0) if m.dtype != torch.bool else m[0, 0]
    expect = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    assert torch.equal(allowed.bool(), expect), f"mask is not causal:\n{allowed.int()}"


def test_every_serialized_generator_sets_the_attn_implementation():
    """A generator that forgets this line emits a non-causal imatrix with no error."""
    missing = []
    for fn in SERIALIZED_GENERATORS:
        path = os.path.join(HERE, fn)
        with open(path) as f:
            src = f.read()
        if not re.search(r"_attn_implementation\s*=", src):
            missing.append(fn)
    assert not missing, f"generators never set config._attn_implementation: {missing}"


def test_generators_expose_the_attn_impl_flag():
    """The knob should be visible, not buried, since it changes numerics."""
    missing = [fn for fn in SERIALIZED_GENERATORS
               if "--attn-impl" not in open(os.path.join(HERE, fn)).read()]
    assert not missing, f"generators missing the --attn-impl flag: {missing}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
