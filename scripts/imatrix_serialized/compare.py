import sys, numpy as np
from imatrix_io import read_imatrix
_, A = read_imatrix(sys.argv[1])  # mine
_, B = read_imatrix(sys.argv[2])  # real
na, nb = set(A), set(B)
print(f"mine={len(na)} real={len(nb)} | only-mine={sorted(na-nb)[:5]} only-real={sorted(nb-na)[:5]}")
def norm(e):  # per-column mean = sum/count
    s = e["in_sum2"].astype(np.float64); c = e["counts"].astype(np.float64).reshape(-1)
    if s.ndim==1: return s/max(c[0],1)
    return s / np.maximum(c[:,None],1)
corrs, rels = [], []
worst=[]
for n in sorted(na & nb):
    a = norm(A[n]).ravel(); b = norm(B[n]).ravel()
    if a.shape!=b.shape:
        print(f"  SHAPE MISMATCH {n}: {a.shape} vs {b.shape}"); continue
    c = np.corrcoef(a,b)[0,1] if a.std()>0 and b.std()>0 else float('nan')
    rel = np.abs(a-b).sum()/(np.abs(b).sum()+1e-30)
    corrs.append(c); rels.append(rel); worst.append((c,rel,n))
corrs=np.array(corrs); rels=np.array(rels)
print(f"\nper-tensor correlation: mean={np.nanmean(corrs):.5f} min={np.nanmin(corrs):.5f} median={np.nanmedian(corrs):.5f}")
print(f"per-tensor L1 rel err:  mean={np.mean(rels):.4f} max={np.max(rels):.4f}")
print("\nlowest-correlation tensors:")
for c,r,n in sorted(worst)[:6]:
    print(f"  corr={c:.4f} relerr={r:.3f}  {n}")
