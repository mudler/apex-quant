import numpy as np
from imatrix_io import read_imatrix
_, e = read_imatrix("models/granite-4.0-h-tiny.imatrix")
def rel(a,b):
    a=a.astype(np.float64).ravel(); b=b.astype(np.float64).ravel()
    return np.abs(a-b).max()/(np.abs(a).max()+1e-30), float(np.corrcoef(a,b)[0,1])
for L in [0,1,5,20,35]:
    g=e[f"blk.{L}.ffn_gate_exps.weight"]["in_sum2"]
    u=e[f"blk.{L}.ffn_up_exps.weight"]["in_sum2"]
    r,c=rel(g,u)
    print(f"L{L:2d} gate vs up exps: maxrel={r:.2e} corr={c:.6f}  (identical if input shared)")
# shared gate vs up
for L in [0,5]:
    g=e[f"blk.{L}.ffn_gate_shexp.weight"]["in_sum2"]
    u=e[f"blk.{L}.ffn_up_shexp.weight"]["in_sum2"]
    r,c=rel(g,u); print(f"L{L} gate vs up SHEXP: maxrel={r:.2e} corr={c:.6f}")
# router vs shexp gate (both take same hidden input)
for L in [0]:
    g=e[f"blk.{L}.ffn_gate_shexp.weight"]["in_sum2"]
    r_=e[f"blk.{L}.ffn_gate_inp.weight"]["in_sum2"]
    rr,cc=rel(g,r_); print(f"L{L} shexp-gate vs router: maxrel={rr:.2e} corr={cc:.6f}")
