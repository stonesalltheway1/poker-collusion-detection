"""Weighted rank fusion of pair models + view-C' evaluation, for sweeping a blend weight cheaply.

  python scripts/fuse_w.py sweep <base> <cand> [w1 w2 ...]      report pooled C' and per-family AP
  python scripts/fuse_w.py write <out> <base> <cand> <w>        write oof/<out>_{oof_w2a,eval_scores}.parquet

Fusion is on within-file ranks (the project convention, scripts/fuse.py): oof = (1-w)*rank(base) + w*rank(cand).
Behaviour posteriors P_* are taken from BASE unchanged, so a fusion changes the RISK component only.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import common as C  # noqa: E402

SPY = pl.read_parquet(C.DER / "spies_exp005.parquet").select("pA", "pB")


def load_w2a(e):
    d = pl.read_parquet(C.OOF_DIR / f"{e}_oof_w2a.parquet")
    if "y_orig" in d.columns:
        d = d.drop("y").rename({"y_orig": "y"})
    return d.sort("pA", "pB")


def rank01(x):
    return pl.Series(x).rank().to_numpy() / len(x)


def report(tag, y, s, fam):
    out = {"pooled": C._ap(y, s)}
    for f in C.FAMILIES:
        m = (y == 0) | ((y == 1) & (fam == f))
        out[f[:2]] = C._ap(y[m], s[m])
    print(f"{tag:28s} " + "  ".join(f"{k}={v:.4f}" for k, v in out.items()))
    return out


def main():
    mode = sys.argv[1]
    if mode == "sweep":
        base, cand = sys.argv[2], sys.argv[3]
        ws = [float(x) for x in sys.argv[4:]] or [0.0, 0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 1.0]
        a, b = load_w2a(base), load_w2a(cand)
        assert a.height == b.height and (a["pA"] == b["pA"]).all() and (a["pB"] == b["pB"]).all()
        a = a.join(SPY, on=["pA", "pB"], how="anti")
        b = b.join(SPY, on=["pA", "pB"], how="anti")
        y, fam = a["y"].to_numpy(), a["behavior_family"].fill_null("none").to_numpy()
        ra, rb = rank01(a["oof"].to_numpy()), rank01(b["oof"].to_numpy())
        for w in ws:
            report(f"w={w:.3f}", y, (1 - w) * ra + w * rb, fam)
        report("max(rank)", y, np.maximum(ra, rb), fam)
        return
    out, base, cand, w = sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
    a, b = load_w2a(base), load_w2a(cand)
    assert a.height == b.height and (a["pA"] == b["pA"]).all()
    fused = (1 - w) * rank01(a["oof"].to_numpy()) + w * rank01(b["oof"].to_numpy())
    a.with_columns(pl.Series("oof", fused), pl.col("y").alias("y_orig")).write_parquet(
        C.OOF_DIR / f"{out}_oof_w2a.parquet")
    ea = pl.read_parquet(C.OOF_DIR / f"{base}_eval_scores.parquet").sort("pair_id")
    eb = pl.read_parquet(C.OOF_DIR / f"{cand}_eval_scores.parquet").sort("pair_id")
    assert (ea["pair_id"] == eb["pair_id"]).all()
    r = (1 - w) * rank01(ea["risk_raw"].to_numpy()) + w * rank01(eb["risk_raw"].to_numpy())
    ea.with_columns(pl.Series("risk_raw", r)).select(
        "pair_id", "risk_raw", *[f"P_{f}" for f in C.FAMILIES]).write_parquet(
        C.OOF_DIR / f"{out}_eval_scores.parquet")
    print("wrote", out, "w=", w)


if __name__ == "__main__":
    main()
