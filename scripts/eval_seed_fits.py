"""Extra full-data eval fits at NEW seeds, for widening the eval-side bag of a pair risk model.

Why this exists
---------------
src/exp005_pairmodel.py and src/exp_xgb_pairmodel.py bag their FINAL eval prediction over seeds
{42,43,44}, hardcoded.  Before the EVAL_SEED_BASE patch the SEED env var reached only the 5 OOF
fold models, so runs that differed only in SEED wrote BIT-IDENTICAL eval_scores (verified:
exp034_v4sf / exp035_sf101 / exp035_sf202 agree to 0 ulp).  The shipped "3-seed LightGBM bag"
exp035_bagsf is therefore a 3-FIT bag at fixed seeds, not a 3-seed bag, on the side that ships.

This script re-fits ONLY the final full-data model, once per requested seed, and writes one eval
risk vector per seed.  It skips the 5-fold OOF entirely (already on disk and deterministic) and
skips the behaviour head (taken from the incumbent), so it is ~5x cheaper and has a much smaller
peak RSS than a full re-run -- it never materialises an X[fold] copy, which is what OOMs on a
shared 32 GB host.

Seeds 42,43,44 reproduce the incumbent's eval risk exactly (control).

Usage: python scripts/eval_seed_fits.py <lgb|xgb> <out_tag> <seed[,seed...]>
Env: the usual E_PAIR block (PH_DIR, CFG, PF_TAG, PFH_TAG, MIX_TAG, USE_*, ADD_SPIES_POS, W_SPY).
"""
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import common as C  # noqa: E402

BACKEND = sys.argv[1]
TAG = sys.argv[2]
SEEDS = [int(s) for s in sys.argv[3].split(",")]

# exp005_pairmodel parses sys.argv at import time (EXP / W_U / ROUNDS / TRAIN); hide ours from it.
_argv, sys.argv = sys.argv, [sys.argv[0]]
import exp005_pairmodel as pm  # noqa: E402
sys.argv = _argv
ROUNDS = int(os.environ.get("ROUNDS", "600" if BACKEND == "lgb" else "500"))
W_U = float(os.environ.get("W_U", "0.1"))
pm.TRAIN = os.environ.get("TRAIN", "all3")


def to_f32(df, cols):
    """Column-wise fill of a preallocated float32 array.

    df.select(cols).to_numpy() materialises a float64 copy first (2x the memory) and then a float32
    copy on top of it; on a contended host that is what raises ArrayMemoryError.  This keeps the
    peak at one float32 array.
    """
    out = np.empty((df.height, len(cols)), dtype=np.float32)
    for i, c in enumerate(cols):
        out[:, i] = df[c].to_numpy().astype(np.float32, copy=False)
    return out


def main():
    t0 = time.time()
    lab, full, w2a, ev, w2b = pm.load()
    feats = pm.feature_cols(full)
    print(f"[{TAG}] loaded: {len(feats)} feats, full {full.height}, w2a {w2a.height}", flush=True)

    if os.environ.get("ADD_SPIES_POS") == "1":
        sp = pl.read_parquet(C.DER / "spies_exp005.parquet").select("pA", "pB")
        if (C.DER / "spies2_exp022.parquet").exists():
            sp = pl.concat([sp, pl.read_parquet(C.DER / "spies2_exp022.parquet").select("pA", "pB")]).unique()
        sp = sp.with_columns(pl.lit(True).alias("is_spy"))

        def promote(df):
            if df is None:
                return None
            d = df.join(sp, on=["pA", "pB"], how="left").with_columns(pl.col("is_spy").fill_null(False))
            return d.with_columns(pl.col("y").alias("y_orig"),
                                  pl.when(pl.col("is_spy")).then(1).otherwise(pl.col("y")).alias("y"))
        full, w2a, w2b = promote(full), promote(w2a), promote(w2b)

    keep_full = pl.col("is_lab") | ((~pl.col("touch_pos")) & (pl.col("n") >= 57))
    keep_w = pl.col("is_lab") | (~pl.col("touch_pos"))
    if pm.TRAIN == "full":
        train = full.filter(keep_full)
    elif pm.TRAIN == "w2000ab":
        train = pl.concat([w2a.filter(keep_w), w2b.filter(keep_w)], how="diagonal_relaxed")
    else:
        train = pl.concat([full.filter(keep_full), w2a.filter(keep_w), w2b.filter(keep_w)], how="diagonal_relaxed")

    w = np.where(train["is_lab"].to_numpy(), 1.0, W_U)
    if "is_spy" in train.columns:
        w = np.where(train["is_spy"].to_numpy(), float(os.environ.get("W_SPY", "0.5")), w)
    X = to_f32(train, feats)
    y = train["y"].to_numpy().astype(np.float32)

    ep = C.load("eval_pairs").select("pair_id", pl.col("p1").alias("pA"), pl.col("p2").alias("pB"))
    evp = ep.join(ev, on=["pA", "pB"], how="left")
    assert evp["n"].null_count() == 0
    Xe = to_f32(evp, feats)
    pair_id = evp["pair_id"].to_numpy()
    del train, full, w2a, w2b, ev, evp, lab
    gc.collect()
    print(f"[{TAG}] X {X.shape} Xe {Xe.shape}  ({time.time()-t0:.0f}s)", flush=True)

    out = {"pair_id": pair_id}
    if BACKEND == "lgb":
        import lightgbm as lgb
        PARAMS = pm.PARAMS
        ds = lgb.Dataset(X, y, weight=w, feature_name=feats)
        for s in SEEDS:
            t = time.time()
            m = lgb.train({**PARAMS, "seed": s}, ds, ROUNDS)
            out[f"r{s}"] = m.predict(Xe).astype(np.float32)
            del m
            gc.collect()
            print(f"[{TAG}] lgb seed {s} done {time.time()-t:.0f}s (total {time.time()-t0:.0f}s)", flush=True)
    else:
        import xgboost as xgb
        params = {"tree_method": "hist", "device": os.environ.get("XGB_DEVICE", "cuda"),
                  "objective": "binary:logistic", "eval_metric": "logloss", "learning_rate": 0.05,
                  "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.4, "reg_lambda": 1.0}
        dtrain = xgb.DMatrix(X, label=y, weight=w)
        deval = xgb.DMatrix(Xe)
        for s in SEEDS:
            t = time.time()
            bst = xgb.train(dict(params, seed=s), dtrain, num_boost_round=ROUNDS)
            out[f"r{s}"] = bst.predict(deval).astype(np.float32)
            del bst
            gc.collect()
            print(f"[{TAG}] xgb seed {s} done {time.time()-t:.0f}s (total {time.time()-t0:.0f}s)", flush=True)

    C.OOF_DIR.mkdir(exist_ok=True)
    p = C.OOF_DIR / f"{TAG}_evalseeds.parquet"
    pl.DataFrame(out).write_parquet(p)
    print(f"[{TAG}] wrote {p} ({len(SEEDS)} seeds) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
