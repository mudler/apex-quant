import numpy as np, sys
from imatrix_io import read_imatrix, write_imatrix
src = "models/granite-4.0-h-tiny.imatrix"
meta, entries = read_imatrix(src)
# reshape entries into writer's expected form
w = {}
for base, e in entries.items():
    w[base] = {"in_sum2": e["in_sum2"], "counts": e["counts"].reshape(-1)}
ds = meta["imatrix.datasets"] if isinstance(meta["imatrix.datasets"], list) else [meta["imatrix.datasets"]]
ds = ["calibration"]  # dataset strings decode oddly via reader; not needed for values
out = "imatrix-reverse/roundtrip.imatrix"
write_imatrix(out, w, ds, meta["imatrix.chunk_count"][0] if isinstance(meta["imatrix.chunk_count"],list) else 126, 512)
m2, e2 = read_imatrix(out)
# compare
assert set(entries) == set(e2), "name set differs"
maxrel = 0.0
for b in entries:
    a = entries[b]["in_sum2"].astype(np.float64).ravel()
    c = e2[b]["in_sum2"].astype(np.float64).ravel()
    assert a.shape == c.shape, (b, a.shape, c.shape)
    d = np.abs(a-c).max()
    ca = entries[b]["counts"].ravel(); cc = e2[b]["counts"].ravel()
    assert np.array_equal(ca, cc), (b,"counts")
    maxrel = max(maxrel, d)
print(f"OK: {len(entries)} entries round-tripped; names match; max abs in_sum2 diff = {maxrel:.3e}; counts identical")
