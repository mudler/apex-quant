"""Read/write llama.cpp GGUF-format imatrix files.

Format (from tools/imatrix/imatrix.cpp save_imatrix + common/imatrix-loader.cpp):
  KV: general.type="imatrix", imatrix.datasets(str[]),
      imatrix.chunk_count(u32), imatrix.chunk_size(u32)
  Per weight tensor <name>:
    <name>.in_sum2 : F32 [in_features, nmat]  = sum over tokens of x_j^2  (raw)
    <name>.counts  : F32 [1, nmat]            = token count per matrix/expert
"""
import numpy as np
import gguf


def read_imatrix(path):
    r = gguf.GGUFReader(path)
    meta = {}
    for k, f in r.fields.items():
        try:
            if f.types and f.types[0] == gguf.GGUFValueType.STRING:
                meta[k] = [str(bytes(f.parts[idx]), "utf-8") for idx in f.data] \
                    if len(f.data) > 1 else str(bytes(f.parts[f.data[0]]), "utf-8")
            else:
                meta[k] = f.parts[f.data[0]].tolist() if f.data else None
        except Exception:
            meta[k] = "<unparsed>"
    entries = {}
    for t in r.tensors:
        name = t.name
        for suf in (".in_sum2", ".counts"):
            if name.endswith(suf):
                base = name[: -len(suf)]
                entries.setdefault(base, {})[suf[1:]] = np.array(t.data, dtype=np.float32).reshape(tuple(reversed(t.shape.tolist())))
    return meta, entries


def write_imatrix(path, entries, datasets, chunk_count, chunk_size):
    """entries: {base_name: {"in_sum2": np.ndarray[nmat, in_feat] or [in_feat],
                             "counts": np.ndarray[nmat] or scalar}}"""
    w = gguf.GGUFWriter(path, arch="imatrix")  # arch unused; general.type set below
    w.add_type("imatrix")
    w.add_array("imatrix.datasets", datasets)
    w.add_uint32("imatrix.chunk_count", int(chunk_count))
    w.add_uint32("imatrix.chunk_size", int(chunk_size))
    for base in sorted(entries):
        e = entries[base]
        sums = np.asarray(e["in_sum2"], dtype=np.float32)
        counts = np.asarray(e["counts"], dtype=np.float32)
        if sums.ndim == 1:
            sums = sums[None, :]           # [1, in_feat]
        counts = counts.reshape(-1)        # [nmat]
        # GGUF tensor dims are stored reversed; add_tensor takes numpy with
        # shape [nmat, in_feat] -> ggml ne = [in_feat, nmat]
        w.add_tensor(base + ".in_sum2", np.ascontiguousarray(sums, dtype=np.float32))
        w.add_tensor(base + ".counts", np.ascontiguousarray(counts.reshape(nmat := counts.shape[0], 1), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    import sys
    meta, entries = read_imatrix(sys.argv[1])
    print("=== metadata ===")
    for k, v in meta.items():
        vs = v if not isinstance(v, list) else f"[{len(v)} items] {v[:2]}"
        print(f"  {k}: {vs}")
    print(f"=== {len(entries)} tensor entries ===")
    for i, (base, e) in enumerate(sorted(entries.items())):
        s = e.get("in_sum2"); c = e.get("counts")
        if i < 12 or i >= len(entries) - 4:
            print(f"  {base:42s} in_sum2{tuple(s.shape) if s is not None else None} "
                  f"counts{tuple(c.shape) if c is not None else None} "
                  f"count0={c.flatten()[0] if c is not None else '?':.0f}")
        elif i == 12:
            print("   ...")
    # sanity: any entry missing a half?
    bad = [b for b, e in entries.items() if "in_sum2" not in e or "counts" not in e]
    print("incomplete entries:", bad if bad else "none")
