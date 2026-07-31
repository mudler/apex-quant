"""Synthetic checks for calibration windowing + llama.cpp BOS parity. No model needed.

Run: python3 test_calib.py   (or under pytest)
"""
import torch

from calib import load_calibration_chunks


class FakeTok:
    """Minimal stand-in: one token per character, optional BOS id."""
    def __init__(self, bos_token_id=None):
        self.bos_token_id = bos_token_id

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        assert add_special_tokens is False, "corpus must be tokenized as a clean stream"
        ids = torch.tensor([[ord(c) for c in text]])
        return type("Enc", (), {"input_ids": ids})()


def _corpus(tmp, n):
    p = tmp / "calib.txt"
    p.write_text("".join(chr(65 + (i % 26)) for i in range(n)))
    return str(p)


def test_windowing_shape_and_contiguity(tmp_path):
    path = _corpus(tmp_path, 100)
    chunks, info = load_calibration_chunks(FakeTok(), path, n_chunks=99, ctx=10, add_bos=False)
    assert chunks.shape == (10, 10), chunks.shape
    assert chunks.dtype == torch.int64
    # windows must tile the stream in order with no gaps
    flat = chunks.reshape(-1)
    assert flat.tolist() == [ord(chr(65 + (i % 26))) for i in range(100)]
    assert "chunks=10" in info


def test_chunks_cap_is_respected(tmp_path):
    path = _corpus(tmp_path, 100)
    chunks, _ = load_calibration_chunks(FakeTok(), path, n_chunks=3, ctx=10, add_bos=False)
    assert chunks.shape == (3, 10)


def test_bos_forced_at_position_zero_of_every_chunk(tmp_path):
    """llama.cpp imatrix.cpp L865-866 overwrites token 0 of each chunk with BOS."""
    path = _corpus(tmp_path, 100)
    chunks, info = load_calibration_chunks(FakeTok(bos_token_id=7), path,
                                           n_chunks=99, ctx=10, add_bos=True)
    assert (chunks[:, 0] == 7).all(), chunks[:, 0]
    assert "llama.cpp parity" in info
    # only position 0 is touched; the rest of each window is untouched stream
    plain, _ = load_calibration_chunks(FakeTok(bos_token_id=7), path,
                                        n_chunks=99, ctx=10, add_bos=False)
    assert torch.equal(chunks[:, 1:], plain[:, 1:])
    assert not torch.equal(chunks[:, 0], plain[:, 0]), "BOS should differ from the stream"


def test_no_bos_id_is_not_fatal(tmp_path):
    path = _corpus(tmp_path, 30)
    chunks, info = load_calibration_chunks(FakeTok(bos_token_id=None), path,
                                           n_chunks=99, ctx=10, add_bos=True)
    assert chunks.shape == (3, 10)
    assert "no bos_token_id" in info


def test_corpus_too_short_hard_fails(tmp_path):
    path = _corpus(tmp_path, 5)
    try:
        load_calibration_chunks(FakeTok(), path, n_chunks=10, ctx=512)
    except ValueError as e:
        assert "too short" in str(e)
        return
    raise AssertionError("expected ValueError on a corpus shorter than one chunk")


if __name__ == "__main__":
    import pathlib, tempfile
    with tempfile.TemporaryDirectory() as d:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_"):
                sub = pathlib.Path(d) / name
                sub.mkdir()
                fn(sub)
                print(f"ok  {name}")
    print("all passed")
