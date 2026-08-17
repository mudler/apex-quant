"""Synthetic checks for the GLM-4.5 routed-expert statistic. No checkpoint needed.

The bug this locks down: mapping mlp.experts to all three of
ffn_{gate,up,down}_exps and feeding them one hidden-dim Sum(x^2). ffn_down_exps takes
moe_intermediate_size inputs (the gated intermediate), so that entry was wrong in
both shape and value.

Run: python3 test_glm4_expert_stats.py   (or under pytest)
"""
import torch
import torch.nn.functional as F

from glm4moe_serialized_gen import apply_gate_up, expert_stats


HIDDEN, INTERM, NE, K, T = 16, 6, 4, 2, 32


def _fixture(layout, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, HIDDEN, generator=g, dtype=torch.float64)
    topk = torch.stack([torch.randperm(NE, generator=g)[:K] for _ in range(T)])
    w = torch.randn(NE, 2 * INTERM, HIDDEN, generator=g, dtype=torch.float64)
    return x, topk, (w if layout == "linear" else w.transpose(1, 2).contiguous())


def _reference(x, topk, w_linear, act):
    """Independent per-expert loop: what each expert's two matmuls actually see."""
    s_in = torch.zeros(NE, HIDDEN, dtype=torch.float64)
    s_inter = torch.zeros(NE, INTERM, dtype=torch.float64)
    cnt = torch.zeros(NE, dtype=torch.float64)
    for e in range(NE):
        rows = [t for t in range(T) if e in topk[t].tolist()]
        for t in rows:
            xt = x[t]
            s_in[e] += xt * xt
            g, u = F.linear(xt, w_linear[e]).chunk(2, dim=-1)
            inter = act(g) * u
            s_inter[e] += inter * inter
        cnt[e] = len(rows)
    return s_in, s_inter, cnt


def test_shapes_and_values_match_reference():
    """down stat has INTERM entries and equals the gated intermediate's Sum(x^2)."""
    for layout in ("linear", "bmm"):
        x, topk, w = _fixture(layout)
        w_linear = w if layout == "linear" else w.transpose(1, 2)
        s_in, s_inter, cnt = expert_stats(x, topk, w, F.silu, NE, HIDDEN)
        r_in, r_inter, r_cnt = _reference(x, topk, w_linear, F.silu)

        assert s_in.shape == (NE, HIDDEN), (layout, s_in.shape)
        assert s_inter.shape == (NE, INTERM), (layout, s_inter.shape)
        assert s_inter.shape[-1] != s_in.shape[-1], "down stat must not be hidden-dim"
        # expert_stats computes in float32 (matching the real activation dtype) and
        # accumulates in float64, so it tracks this float64 reference to ~float32 eps.
        torch.testing.assert_close(s_in, r_in, rtol=1e-6, atol=1e-5)
        torch.testing.assert_close(s_inter, r_inter, rtol=1e-6, atol=1e-5)
        torch.testing.assert_close(cnt, r_cnt)
        # every (token, selected-expert) slot counted exactly once
        assert cnt.sum().item() == T * K


def test_both_fused_layouts_agree():
    """Layout is derived from the shape, so the two conventions give one answer."""
    a = expert_stats(*_fixture("linear"), F.silu, NE, HIDDEN)
    b = expert_stats(*_fixture("bmm"), F.silu, NE, HIDDEN)
    for ta, tb in zip(a, b):
        torch.testing.assert_close(ta, tb, rtol=1e-9, atol=1e-9)


def test_ambiguous_and_bad_layouts_hard_fail():
    """A layout we cannot resolve must raise, never silently pick a side."""
    x = torch.zeros(4, HIDDEN, dtype=torch.float64)
    for w, why in [(torch.zeros(HIDDEN, HIDDEN, dtype=torch.float64), "square"),
                   (torch.zeros(7, 9, dtype=torch.float64), "no hidden axis")]:
        try:
            apply_gate_up(x, w, HIDDEN)
        except ValueError:
            continue
        raise AssertionError(f"apply_gate_up accepted {why} weight {tuple(w.shape)}")


def test_wrong_input_dim_hard_fails():
    x, topk, w = _fixture("linear")
    try:
        expert_stats(x[:, :HIDDEN - 1], topk, w, F.silu, NE, HIDDEN)
    except ValueError:
        return
    raise AssertionError("expert_stats accepted a mismatched hidden dim")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
