"""View C' = view C with a FROZEN spy list (likely hidden dev positives) removed from the negatives.
Spy list built once from exp005 dev_full OOF (unlabelled pairs, no labelled-positive player, oof > 0.3).
Usage: python scripts/eval_views.py exp005 exp005_wu0.03 ...
"""
import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, "src")
import common as C

spy_path = C.DER / "spies_exp005.parquet"
if not spy_path.exists():
    o = pl.read_parquet(C.OOF_DIR / "exp005_oof_full.parquet")
    spies = o.filter(~pl.col("is_lab") & ~pl.col("touch_pos") & (pl.col("oof") > 0.3)).select("pA", "pB", "oof")
    spies.write_parquet(spy_path)
    print("frozen spy list:", spies.height)
spies = pl.read_parquet(spy_path).select("pA", "pB")
sp2 = C.DER / "spies2_exp022.parquet"
spies_all = pl.concat([spies, pl.read_parquet(sp2).select("pA", "pB")]).unique() if sp2.exists() else spies
for exp in sys.argv[1:]:
    w = pl.read_parquet(C.OOF_DIR / f"{exp}_oof_w2a.parquet")
    if "y_orig" in w.columns:      # spy-promoted runs: score against the ORIGINAL labels only
        w = w.drop("y").rename({"y_orig": "y"})
    wc = w.join(spies, on=["pA", "pB"], how="anti")
    wc2 = w.join(spies_all, on=["pA", "pB"], how="anti")
    res = {"C": C._ap(w["y"].to_numpy(), w["oof"].to_numpy()), "Cprime": C._ap(wc["y"].to_numpy(), wc["oof"].to_numpy()),
           "C2": C._ap(wc2["y"].to_numpy(), wc2["oof"].to_numpy())}
    for f in C.FAMILIES:
        s = wc.filter((pl.col("behavior_family") == f) | (pl.col("y") == 0))
        res[f"Cp_{f[:2]}"] = C._ap(s["y"].to_numpy(), s["oof"].to_numpy())
    pos = wc.filter(pl.col("y") == 1)
    thr = np.sort(wc.filter(pl.col("y") == 0)["oof"].to_numpy())[::-1]
    res["neg_above_median_pos"] = int((thr > np.median(pos["oof"].to_numpy())).sum())
    print(exp, {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
