"""Paired table-bootstrap of view-C' AP between pair models. Usage: python scripts/boot.py <base> <cand> [reps]"""
import sys
import numpy as np
import polars as pl
sys.path.insert(0, "src")
import common as C

base, cand = sys.argv[1], sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 400
spies = pl.read_parquet(C.DER / "spies_exp005.parquet").select("pA", "pB")
def load(e):
    d = pl.read_parquet(C.OOF_DIR / f"{e}_oof_w2a.parquet")
    if "y_orig" in d.columns:      # spy-promoted runs must be scored on the ORIGINAL labels
        d = d.drop("y").rename({"y_orig": "y"})
    return d.join(spies, on=["pA", "pB"], how="anti").sort("pA", "pB")
a, b = load(base), load(cand)
assert a.height == b.height and (a["pA"] == b["pA"]).all()
tabs = a["table_idx"].to_numpy()
y = a["y"].to_numpy()
sa, sb = a["oof"].to_numpy(), b["oof"].to_numpy()
ut = np.unique(tabs)
idx_by_tab = {t: np.flatnonzero(tabs == t) for t in ut}
rng = np.random.default_rng(42)
d = []
for _ in range(reps):
    pick = rng.choice(ut, size=len(ut), replace=True)
    idx = np.concatenate([idx_by_tab[t] for t in pick])
    d.append(C._ap(y[idx], sb[idx]) - C._ap(y[idx], sa[idx]))
d = np.array(d)
print(f"{cand} vs {base}: AP {C._ap(y, sb):.4f} vs {C._ap(y, sa):.4f} | mean delta {d.mean():+.4f} "
      f"| P(better) {(d > 0).mean():.3f} | 90% CI [{np.quantile(d, .05):+.4f}, {np.quantile(d, .95):+.4f}]")
