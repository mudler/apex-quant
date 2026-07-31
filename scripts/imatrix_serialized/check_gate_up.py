"""Diagnostic: confirm tensors that share a graph input share an imatrix stat.

gate/up (routed and shared) are fed the same activation by the ggml graph, so their
in_sum2 vectors should be identical; the router sees that same hidden state too. A
divergence here means a hook is attached to the wrong module.

This inspects a real imatrix file, so the path is an argument rather than a hardcoded
dev path. It is a diagnostic, not a unit test -- the assertion-based tests are
test_roundtrip.py / test_calib.py / test_glm4_expert_stats.py /
test_llama4_expert_counts.py, none of which need external files.

Usage:
    python3 check_gate_up.py path/to/some.imatrix [layer ...]
"""
import sys

import numpy as np

from imatrix_io import read_imatrix


def rel(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.shape != b.shape:
        return float("nan"), float("nan")
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return np.abs(a - b).max() / (np.abs(a).max() + 1e-30), corr


def compare(entries, a_name, b_name, label):
    if a_name not in entries or b_name not in entries:
        print(f"  {label}: skipped (missing {a_name if a_name not in entries else b_name})")
        return
    r, c = rel(entries[a_name]["in_sum2"], entries[b_name]["in_sum2"])
    print(f"  {label}: maxrel={r:.2e} corr={c:.6f}  (identical if input shared)")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]
    layers = [int(x) for x in argv[2:]] if len(argv) > 2 else None
    _, e = read_imatrix(path)

    if layers is None:
        # derive the layers actually present instead of assuming a fixed model
        found = sorted({int(n.split(".")[1]) for n in e
                        if n.startswith("blk.") and n.split(".")[1].isdigit()})
        layers = found[:: max(1, len(found) // 5)][:5] if found else []
    print(f"{path}: {len(e)} entries; checking layers {layers}")

    for L in layers:
        print(f"blk.{L}")
        compare(e, f"blk.{L}.ffn_gate_exps.weight", f"blk.{L}.ffn_up_exps.weight", "gate vs up exps")
        compare(e, f"blk.{L}.ffn_gate_shexp.weight", f"blk.{L}.ffn_up_shexp.weight", "gate vs up shexp")
        compare(e, f"blk.{L}.ffn_gate_shexp.weight", f"blk.{L}.ffn_gate_inp.weight", "shexp-gate vs router")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
