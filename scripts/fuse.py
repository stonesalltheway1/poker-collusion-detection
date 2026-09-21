"""Rank-average fusion of pair models. Usage: python scripts/fuse.py <out_exp> <exp1> <exp2> ...
Evaluates view C/C' on the fused OOF and writes oof/<out_exp>_eval_scores.parquet (+ _oof_w2a for eval_views)."""
import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, "src")
import common as C

out, exps = sys.argv[1], sys.argv[2:]
w2a, ev = None, None
for e in exps:
    a = pl.read_parquet(C.OOF_DIR / f"{e}_oof_w2a.parquet").sort("pA", "pB")
    r = (a["oof"].rank() / a.height).alias(f"r_{e}")
    keep_y = ["y", "y_orig"] if "y_orig" in a.columns else ["y"]
    w2a = a.select("pA", "pB", "pair_id", "table_idx", "fold", "n", *keep_y, "is_lab", "behavior_family",
                   *[f"P_{f}" for f in C.FAMILIES]).with_columns(r) if w2a is None else w2a.with_columns(r)
    b = pl.read_parquet(C.OOF_DIR / f"{e}_eval_scores.parquet").sort("pair_id")
    rb = (b["risk_raw"].rank() / b.height).alias(f"r_{e}")
    bb = b.select("pair_id", *[pl.col(f"P_{f}").alias(f"P_{f}_{e}") for f in C.FAMILIES]).with_columns(rb)
    ev = bb if ev is None else ev.join(bb, on="pair_id")
w2a = w2a.with_columns(pl.mean_horizontal([f"r_{e}" for e in exps]).alias("oof"))
ev = ev.with_columns(pl.mean_horizontal([f"r_{e}" for e in exps]).alias("risk_raw"),
                     *[pl.mean_horizontal([f"P_{f}_{e}" for e in exps]).alias(f"P_{f}") for f in C.FAMILIES])
w2a.write_parquet(C.OOF_DIR / f"{out}_oof_w2a.parquet")
ev.select("pair_id", "risk_raw", *[f"P_{f}" for f in C.FAMILIES]).write_parquet(C.OOF_DIR / f"{out}_eval_scores.parquet")
print("fused", out, "from", exps, "->", ev.height, "eval rows")
