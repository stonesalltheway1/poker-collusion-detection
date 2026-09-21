"""Explicit mixture log-likelihood-ratio pair score from hand-level detector scores (exp025).

For a pair with hands h and per-hand planted-probability score s_h, the likelihood ratio of
"this pair is coordinated (a fraction pi of its hands are planted)" vs "clean" is, under independence,
    LLR = sum_h log(1 + pi * (LR(s_h) - 1)),   LR(s) = f1(s) / f0(s)
with f1 estimated on dev EVIDENCE hands (out-of-fold scores) and f0 on confirmed-negative-pair hands.
This is the statistically right aggregator for low-exposure pairs: a handful of very strong hands dominates,
and exposure enters through the sum, not a rate. Outputs one column per family + 'any' per window.

Usage: python scripts/mixllr.py [hs_dir] [out_tag]     (default: hs_v2, v2)
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import common as C  # noqa: E402

HSDIR = C.DER / (sys.argv[1] if len(sys.argv) > 1 else "hs_v2")
TAG = sys.argv[2] if len(sys.argv) > 2 else "v2"
SC = {"s_dt": "directed_transfer", "s_sp": "soft_play", "s_ci": "coordinated_isolation", "s_any": None}
PI = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
WINDOWS = {"dev_full": (0, 0, 3000), "dev_w2000a": (0, 0, 2000), "dev_w2000b": (0, 1000, 3000), "eval": (1, 3000, 5000)}
col = pl.col


def calibrate():
    """bin edges on the negative-hand score distribution + log-LR per bin (per score column)."""
    lab = C.load("labels").select(col("p1").alias("pA"), col("p2").alias("pB"), "pair_id", "label", "behavior_family")
    ev = C.load("evidence").select("pair_id", "hand_idx", pl.lit(1).alias("is_ev"))
    rows = (pl.scan_parquet(HSDIR / "phase0" / "*.parquet")
            .join(lab.lazy(), on=["pA", "pB"]).join(ev.lazy(), on=["pair_id", "hand_idx"], how="left")
            .with_columns(col("is_ev").fill_null(0)).collect())
    out = {}
    for s, fam in SC.items():
        neg = rows.filter(col("label") == 0)[s].to_numpy()
        pos = rows.filter((col("is_ev") == 1) & ((col("behavior_family") == fam) if fam else True))[s].to_numpy()
        qs = np.concatenate([np.linspace(0, 0.99, 100), 1 - np.logspace(-2, -5, 40)])
        edges = np.unique(np.quantile(neg, qs))
        b0 = np.clip(np.searchsorted(edges, neg, side="right"), 0, len(edges))
        b1 = np.clip(np.searchsorted(edges, pos, side="right"), 0, len(edges))
        nb = len(edges) + 1
        f0 = (np.bincount(b0, minlength=nb) + 0.5) / (len(neg) + 0.5 * nb)
        f1 = (np.bincount(b1, minlength=nb) + 0.5) / (len(pos) + 0.5 * nb)
        out[s] = dict(edges=edges.tolist(), logterm=np.log1p(PI * (f1 / f0 - 1.0)).tolist(),
                      n_pos=int(len(pos)), n_neg=int(len(neg)))
    (C.DER / f"mixllr_cal_{TAG}.json").write_text(json.dumps(out))
    print({s: (v["n_pos"], round(max(v["logterm"]), 2)) for s, v in out.items()})
    return out


def build(cal):
    for window, (phase, lo, hi) in WINDOWS.items():
        lf = (pl.scan_parquet(HSDIR / f"phase{phase}" / "*.parquet")
              .filter((col("hand_seq") >= lo) & (col("hand_seq") < hi)))
        ex = []
        for s, v in cal.items():
            edges, lt = np.array(v["edges"]), np.array(v["logterm"])
            ex.append(col(s).map_batches(
                lambda x, e=edges, l=lt: pl.Series(l[np.clip(np.searchsorted(e, x.to_numpy(), side="right"), 0, len(e))]),
                return_dtype=pl.Float64).alias(f"t_{s}"))
        g = (lf.with_columns(ex).group_by("pA", "pB")
             .agg([col(f"t_{s}").sum().alias(f"mllr_{s}") for s in cal]
                  + [col(f"t_{s}").sort(descending=True).head(5).sum().alias(f"mllr5_{s}") for s in cal]
                  + [col(f"t_{s}").max().alias(f"mllrmax_{s}") for s in cal])
             .collect(engine="streaming"))
        g.write_parquet(C.DER / f"pfm{TAG}_{window}.parquet")
        print(f"[{window}] {g.height:,} pairs -> pfm{TAG}_{window}.parquet", flush=True)


if __name__ == "__main__":
    build(calibrate())
