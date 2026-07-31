"""Drives the Llama-4 hooks through a real (tiny, random) Llama4TextMoe on CPU.

Locks down the two things review round 3 raised about serialized_gen_llama4.py:

 1. Row counts must be llama.cpp's one-per-(token, selected-expert) slot, taken from
    the router's exact top-k -- not the old `abs().sum(-1) > 0` nonzero-row heuristic,
    which miscounts a genuinely-zero activation row or an underflowed router score.
 2. The router-weighted gate/up statistic is CORRECT for this arch. llama.cpp sets
    `weight_before_ffn = arch == LLM_ARCH_LLAMA4` (src/llama-graph.cpp:1837) and
    multiplies the sigmoid-ed weights into the expert input BEFORE the gate/up
    mul_mat_id (L1976), so a weighted stat is the faithful reproduction. This test
    pins the weighted behaviour so a future "fix" to unweighted fails loudly.

Run: python3 test_llama4_expert_counts.py   (or under pytest)
"""
import numpy as np
import torch

from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

from serialized_gen_llama4 import make_ensure, hook_experts, hook_router_select

H, INTERM, NE, TOPK, T = 32, 12, 6, 2, 20
BLK = "blk.0"


def _moe(seed=0):
    torch.manual_seed(seed)
    cfg = Llama4TextConfig(hidden_size=H, intermediate_size=INTERM, num_local_experts=NE,
                           num_experts_per_tok=TOPK, num_hidden_layers=1,
                           num_attention_heads=4, num_key_value_heads=2)
    moe = Llama4TextMoe(cfg).eval()
    for p in moe.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return moe


def _run(moe, x):
    acc = {}
    sel = {}
    ensure = make_ensure(acc)
    handles = [hook_router_select(moe.router, BLK, sel),
               hook_experts(moe.experts, BLK, ensure, sel)]
    with torch.no_grad():
        moe(x)
    for h in handles:
        h.remove()
    return acc, sel


def test_counts_are_exact_topk_slots():
    moe = _moe()
    x = torch.randn(T, H)
    acc, sel = _run(moe, x)

    gate = acc[f"{BLK}.ffn_gate_exps.weight"]
    # every token contributes to exactly TOPK experts -> total slots = T * TOPK
    assert gate["counts"].sum() == T * TOPK, (gate["counts"].sum(), T * TOPK)
    # and per-expert counts must equal the router's own top-k tally
    idx = sel[BLK]
    expect = np.bincount(idx.reshape(-1).numpy(), minlength=NE)
    assert np.array_equal(gate["counts"], expect), (gate["counts"], expect)
    # all three routed tensors share the same row count
    for t in ("ffn_up_exps", "ffn_down_exps"):
        assert np.array_equal(acc[f"{BLK}.{t}.weight"]["counts"], expect), t


def test_zero_activation_row_does_not_lose_a_count():
    """The old nonzero-row heuristic undercounts here; the top-k mask does not."""
    moe = _moe()
    x = torch.randn(T, H)
    x[3] = 0.0                      # a genuinely zero hidden row, still routed
    acc, sel = _run(moe, x)

    idx = sel[BLK]
    expect = np.bincount(idx.reshape(-1).numpy(), minlength=NE)
    got = acc[f"{BLK}.ffn_gate_exps.weight"]["counts"]
    assert got.sum() == T * TOPK, got.sum()
    assert np.array_equal(got, expect)
    # demonstrate the old heuristic really would have been wrong on this input
    with torch.no_grad():
        scores, _ = moe.router(x)
        X = (x.repeat(scores.shape[1], 1) * scores.t().reshape(-1, 1)).view(NE, T, H)
        old = (X.abs().sum(-1) > 0).sum(1).numpy()
    assert old.sum() < expect.sum(), (old.sum(), expect.sum())


def test_shapes_and_router_weighting_is_preserved():
    moe = _moe()
    x = torch.randn(T, H)
    acc, sel = _run(moe, x)

    gate = acc[f"{BLK}.ffn_gate_exps.weight"]
    down = acc[f"{BLK}.ffn_down_exps.weight"]
    assert gate["sums"].shape == (NE, H), gate["sums"].shape
    assert down["sums"].shape == (NE, INTERM), down["sums"].shape
    # gate and up share one input stat (same MUL_MAT_ID input in the graph)
    assert np.array_equal(gate["sums"], acc[f"{BLK}.ffn_up_exps.weight"]["sums"])

    # The recorded gate/up stat must be the ROUTER-WEIGHTED second moment, matching
    # llama.cpp's weight_before_ffn path. Compare against an explicit recomputation.
    with torch.no_grad():
        scores, _ = moe.router(x)
        weighted = (x.repeat(scores.shape[1], 1) * scores.t().reshape(-1, 1)).view(NE, T, H)
        idx = sel[BLK]
        keep = torch.zeros(T, NE, dtype=torch.bool).scatter_(1, idx, True)
        expect = ((weighted.float() * keep.t().unsqueeze(-1)) ** 2).sum(1).double().numpy()
    np.testing.assert_allclose(gate["sums"], expect, rtol=1e-6, atol=1e-9)

    # sanity: the unweighted stat is materially different, so this really is pinning
    # the weighted convention rather than passing trivially
    with torch.no_grad():
        unweighted = ((x.repeat(NE, 1).view(NE, T, H).float() * keep.t().unsqueeze(-1)) ** 2
                      ).sum(1).double().numpy()
    assert not np.allclose(gate["sums"], unweighted, rtol=1e-3)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
