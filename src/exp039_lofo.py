"""exp039: leave-one-family-out test of the PAIR model, and a generic-only branch.

Why: the evidence work showed ~14% of eval positives are not served by 3-family routing (the undisclosed family
plus variants). If the pair model also ranks such pairs poorly, a family-agnostic branch fused in would lift them
without disturbing the 86%. LOFO simulates it: train with the positives of two families only, then measure the AP
of the held-out family's positives against the same negative pool.

Variants: ALL = every feature; GEN = generic (family-agnostic) features only.
Usage: python src/exp039_lofo.py [rounds]
Env: same feature switches as exp005_pairmodel (PF_TAG/PFH_TAG/USE_OC/USE_MIX/ADD_SPIES_POS/CFG).
"""
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

sys.argv = [sys.argv[0], "exp039", os.environ.get("W_U", "0.1"), os.environ.get("ROUNDS", "600"), "all3"]
import exp005_pairmodel as M  # noqa: E402

FAM = list(C.FAMILIES)
# family-specific markers: anything naming a single family, or a per-family hand/score column
FAM_PAT = ("ci_", "dt_", "sp_", "_ci", "_dt", "_sp", "s_dt", "s_sp", "s_ci", "f_DT", "f_SP", "f_CI")


def is_generic(col):
    return not any(p in col for p in FAM_PAT)


def main():
    t0 = time.time()
    lab, full, w2a, ev, w2b = M.load()
    feats_all = M.feature_cols(full)
    feats_gen = [f for f in feats_all if is_generic(f)]
    print(f"features: all {len(feats_all)}, generic {len(feats_gen)}")
    keep_full = pl.col("is_lab") | ((~pl.col("touch_pos")) & (pl.col("n") >= 57))
    keep_w = pl.col("is_lab") | (~pl.col("touch_pos"))
    parts = [full.filter(keep_full), w2a.filter(keep_w)] + ([w2b.filter(keep_w)] if w2b is not None else [])
    train_all = pl.concat(parts, how="diagonal_relaxed")
    fam_col = pl.col("behavior_family").fill_null("none")
    out = {}
    for held in FAM:
        rows = train_all.filter(~((pl.col("y") == 1) & (fam_col == held)))   # drop held-out family positives
        w = np.where(rows["is_lab"].fill_null(False).to_numpy(), 1.0, M.W_U)
        if "is_spy" in rows.columns:
            w = np.where(rows["is_spy"].fill_null(False).to_numpy(), 0.5, w)
        y = rows["y"].to_numpy()
        fold = rows["fold"].to_numpy()
        # evaluation pool: held-out family positives + all negatives (other families' positives removed)
        pool = w2a.filter(keep_w).filter(~((pl.col("y") == 1) & (fam_col != held)))
        yy = ((pool["y"] == 1) & (pool["behavior_family"].fill_null("none") == held)).to_numpy().astype(int)
        pfold = pool["fold"].to_numpy()
        res = {}
        for name, feats in (("ALL", feats_all), ("GEN", feats_gen)):
            X = rows.select(feats).to_numpy().astype(np.float32)
            Xp = pool.select(feats).to_numpy().astype(np.float32)
            pred = np.zeros(pool.height)
            for k in range(5):
                tr = fold != k
                m = lgb.train(M.PARAMS, lgb.Dataset(X[tr], y[tr], weight=w[tr], feature_name=feats), M.ROUNDS)
                pred[pfold == k] = m.predict(Xp[pfold == k])
            res[name] = pred
            print(f"  [{held}] {name}: AP {C._ap(yy, pred):.4f}  ({time.time() - t0:.0f}s)", flush=True)
        ra = (-res["ALL"]).argsort().argsort() / len(yy)
        rg = (-res["GEN"]).argsort().argsort() / len(yy)
        fuse = -(0.5 * ra + 0.5 * rg)
        fuse7 = -(0.7 * ra + 0.3 * rg)
        out[held] = {"n_pos": int(yy.sum()), "AP_all": round(C._ap(yy, res["ALL"]), 4),
                     "AP_gen": round(C._ap(yy, res["GEN"]), 4),
                     "AP_fuse50": round(C._ap(yy, fuse), 4), "AP_fuse70": round(C._ap(yy, fuse7), 4)}
        print(f"  [{held}] {out[held]}", flush=True)
    (C.OOF_DIR / "exp039_lofo.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("mean:", {k: round(float(np.mean([out[f][k] for f in FAM])), 4) for k in ("AP_all", "AP_gen", "AP_fuse50", "AP_fuse70")})


if __name__ == "__main__":
    main()
