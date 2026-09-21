"""Paired table-bootstrap on view C2 (BOTH frozen spy lists removed from the negatives).
Same protocol as scripts/boot.py; use it when a candidate changes how the spy pairs are TRAINED,
because view C' still scores spies2_exp022 as negatives and would reward a model for missing them.
Usage: python scripts/boot2.py <base> <cand> [reps]"""
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import common as C  # noqa: E402

base, cand = sys.argv[1], sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 400
spies = pl.read_parquet(C.DER / "spies_exp005.parquet").select("pA", "pB")
p2 = C.DER / "spies2_exp022.parquet"
if p2.exists():
    spies = pl.concat([spies, pl.read_parquet(p2).select("pA", "pB")]).unique()


def load(e):
    d = pl.read_parquet(C.OOF_DIR / f"{e}_oof_w2a.parquet")
    if "y_orig" in d.columns:
        d = d.drop("y").rename({"y_orig": "y"})
    return d.join(spies, on=["pA", "pB"], how="anti").sort("pA", "pB")


a, b = load(base), load(cand)
assert a.height == b.height and (a["pA"] == b["pA"]).all()
tabs, y = a["table_idx"].to_numpy(), a["y"].to_numpy()
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
print(f"[C2] {cand} vs {base}: AP {C._ap(y, sb):.4f} vs {C._ap(y, sa):.4f} | mean delta {d.mean():+.4f} "
      f"| P(better) {(d > 0).mean():.3f} | 90% CI [{np.quantile(d, .05):+.4f}, {np.quantile(d, .95):+.4f}]")
