"""Shared calibration tokenization/windowing for the imatrix generators.

One place, because every generator needs the identical thing and because the BOS
detail below is easy to get silently wrong.

llama.cpp's reference implementation (tools/imatrix/imatrix.cpp) tokenizes the whole
corpus once and then, for every chunk, OVERWRITES the first token with BOS when the
vocab has add_bos:

    const bool add_bos = llama_vocab_get_add_bos(vocab);      // L776
    ...
    if (add_bos && j == 0) {                                  // L865
        tokens[seq_start] = llama_vocab_bos(vocab);           // L866
    }

Windowing the tokenized stream without that leaves every chunk after the first
starting mid-sentence, so position 0 of each chunk sees a different distribution than
the reference run. It is ~1 token in `ctx` (0.2% at ctx=512), but it is a real and
avoidable divergence, so `add_bos=True` is the default here.
"""
import torch


def load_calibration_chunks(tokenizer, path, n_chunks, ctx, add_bos=True):
    """Tokenize `path` and window it into an int64 tensor of shape [nch, ctx].

    add_bos=True reproduces llama.cpp by forcing chunk[:, 0] = BOS (skipped when the
    tokenizer has no BOS id). Pass add_bos=False to reproduce a plain stream-window,
    i.e. the behaviour of these generators before the BOS fix.

    Returns (chunks, info) where info is a short human-readable summary string.
    """
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    # add_special_tokens=False so the corpus is a clean stream; per-chunk BOS is
    # applied below exactly where llama.cpp applies it.
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    nch = min(n_chunks, ids.shape[0] // ctx)
    if nch < 1:
        raise ValueError(
            f"{path}: {ids.shape[0]} tokens is too short for one chunk of ctx={ctx}")
    chunks = torch.stack([ids[c * ctx:(c + 1) * ctx] for c in range(nch)]).long()

    bos = getattr(tokenizer, "bos_token_id", None)
    if add_bos and bos is not None:
        chunks = chunks.clone()
        chunks[:, 0] = bos
        bos_note = f"bos={bos} forced at chunk[:,0] (llama.cpp parity)"
    elif add_bos:
        bos_note = "no bos_token_id on this tokenizer; chunks left as-is"
    else:
        bos_note = "per-chunk BOS disabled"
    return chunks, f"tokens={ids.shape[0]} chunks={nch} ctx={ctx} | {bos_note}"


def add_calib_args(ap):
    """Register the calibration flags shared by every generator."""
    ap.add_argument("--chunks", type=int, default=126)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--no-bos-per-chunk", dest="bos_per_chunk", action="store_false",
                    help="do NOT force BOS at position 0 of every chunk; llama.cpp does "
                         "force it, so this only exists to reproduce pre-fix runs")
    ap.set_defaults(bos_per_chunk=True)
    return ap
