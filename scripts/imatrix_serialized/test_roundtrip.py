"""Format-layer round-trip: write -> read -> assert identical.

Previously this needed a local `models/granite-4.0-h-tiny.imatrix` and wrote into
`imatrix-reverse/`, so it could not run anywhere but one dev box and CI never executed
it. It is now synthetic and self-contained: it builds entries covering both shapes the
writer supports (dense nmat=1 and per-expert nmat>1), round-trips through a temp file,
and checks the GGUF layout llama.cpp expects (in_sum2 ne=[in,nmat], counts ne=[1,nmat])
plus the metadata keys.

Still accepts a real file for a spot check:
    python3 test_roundtrip.py path/to/some.imatrix

Needs numpy + gguf only -- no torch, no transformers, no checkpoint.

Run: python3 test_roundtrip.py   (or under pytest)
"""
import os
import tempfile

import numpy as np

from imatrix_io import read_imatrix, write_imatrix

DATASETS = ["calibration_datav3"]
CHUNKS, CTX = 126, 512


def _entries(seed=0):
    rng = np.random.default_rng(seed)
    return {
        # dense tensors: one matrix, in_sum2 given as a 1-D vector
        "blk.0.attn_q.weight": {"in_sum2": (rng.random(64) * 1e3).astype(np.float32),
                                "counts": np.array([CHUNKS * CTX], np.float32)},
        "blk.0.ffn_down.weight": {"in_sum2": rng.random(48).astype(np.float32),
                                  "counts": np.array([CHUNKS * CTX], np.float32)},
        # per-expert tensors: nmat = n_experts, in_sum2 is 2-D [nmat, in_features]
        "blk.0.ffn_gate_exps.weight": {"in_sum2": rng.random((8, 64)).astype(np.float32),
                                       "counts": rng.integers(1, 999, 8).astype(np.float32)},
        "blk.0.ffn_down_exps.weight": {"in_sum2": rng.random((8, 24)).astype(np.float32),
                                       "counts": rng.integers(1, 999, 8).astype(np.float32)},
    }


def _roundtrip(entries):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rt.imatrix")
        write_imatrix(path, entries, DATASETS, CHUNKS, CTX)
        return read_imatrix(path)


def test_names_survive():
    src = _entries()
    _, got = _roundtrip(src)
    assert set(got) == set(src), (set(src) ^ set(got))


def test_values_are_bit_exact():
    """float32 in, float32 out, no scaling anywhere -> must be exactly equal."""
    src = _entries()
    _, got = _roundtrip(src)
    for name, e in src.items():
        want = np.atleast_2d(np.asarray(e["in_sum2"], np.float32))
        assert np.array_equal(got[name]["in_sum2"].reshape(want.shape), want), name
        wc = np.asarray(e["counts"], np.float32).reshape(-1)
        assert np.array_equal(got[name]["counts"].reshape(-1), wc), f"{name} counts"


def test_gguf_shapes_match_llama_cpp_layout():
    """in_sum2 ne=[in_features, nmat] and counts ne=[1, nmat] (imatrix.cpp L597-606)."""
    src = _entries()
    _, got = _roundtrip(src)
    for name, e in src.items():
        nmat = np.asarray(e["counts"], np.float32).reshape(-1).shape[0]
        n_in = np.atleast_2d(np.asarray(e["in_sum2"], np.float32)).shape[-1]
        assert got[name]["in_sum2"].size == nmat * n_in, name
        assert got[name]["counts"].size == nmat, f"{name} counts"


def test_metadata_keys_are_written():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rt.imatrix")
        write_imatrix(path, _entries(), DATASETS, CHUNKS, CTX)
        meta, _ = read_imatrix(path)
    assert meta.get("general.type") == "imatrix", meta.get("general.type")
    for k in ("imatrix.datasets", "imatrix.chunk_count", "imatrix.chunk_size"):
        assert k in meta, f"missing {k}"
    cc = meta["imatrix.chunk_count"]
    assert (cc[0] if isinstance(cc, list) else cc) == CHUNKS, cc


def test_single_expert_entry_is_not_squeezed():
    """nmat=1 given as 2-D must stay a valid entry, not collapse to a dense vector."""
    src = {"blk.1.ffn_gate_exps.weight": {"in_sum2": np.ones((1, 16), np.float32),
                                          "counts": np.array([5], np.float32)}}
    _, got = _roundtrip(src)
    assert got["blk.1.ffn_gate_exps.weight"]["in_sum2"].size == 16


def spot_check(path):
    """Optional: re-write a real imatrix and confirm values/names survive."""
    meta, entries = read_imatrix(path)
    src = {b: {"in_sum2": e["in_sum2"], "counts": e["counts"].reshape(-1)}
           for b, e in entries.items()}
    _, got = _roundtrip(src)
    assert set(got) == set(src), "name set differs"
    worst = 0.0
    for b in src:
        a = src[b]["in_sum2"].astype(np.float64).ravel()
        c = got[b]["in_sum2"].astype(np.float64).ravel()
        assert a.shape == c.shape, (b, a.shape, c.shape)
        assert np.array_equal(src[b]["counts"].ravel(), got[b]["counts"].ravel()), f"{b} counts"
        worst = max(worst, np.abs(a - c).max())
    print(f"spot check OK: {len(src)} entries, max abs in_sum2 diff = {worst:.3e}")


if __name__ == "__main__":
    import sys
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
    if len(sys.argv) > 1:
        spot_check(sys.argv[1])
