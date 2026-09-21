"""exp053 — per-family SPECIALIST pair risk models (Lane B2: soft play carries the pair-AP loss).

Diagnosis that motivates this file (2026-09-20): on view C' the champion fusion exp048_fuse_bags scores
directed_transfer .9602 / soft_play .8817 / coordinated_isolation .9660.  The soft-play loss is concentrated
in 15 of 117 SP positives that carry >20 negatives above them; they are the LOW-EXPOSURE tail
(median shared hands 57 vs 90 for the well-ranked SP positives) with 3-4 planted hands instead of 5.
Rescuing the 15 worst is worth +0.023 pooled C' (oracle).

A single pooled binary model has to spend its capacity on whichever family dominates the gradient.  A
family-restricted objective lets the tree ensemble specialise on one signature at a time; the specialists are
then recombined (rank-mean, or rank-max = an OR detector over the mixture components).

Env:
  FAM_TARGET    directed_transfer | soft_play | coordinated_isolation   (required)
  OTHER_MODE    drop (default) | neg     what to do with the other families' labelled positives
  SPY_MODE      drop (default) | neg | pos   what to do with the frozen likely-hidden-positive list
  W_POS         extra weight multiplier on the target-family positives (default 1.0)
  W_U           positional arg 2 (default 0.1); everything else matches src/run_all.py E_PAIR.
  DONOR         experiment whose behaviour posteriors P_* are copied verbatim (default exp048_fuse_bags),
                so that a fusion with this member leaves the BEHAVIOUR component bit-identical.

Usage: python src/exp053_famspec.py <exp_id> [W_U] [rounds] [train_window]
"""
import json
import os
import sys
import time
from pathlib import Path

import gc

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import exp005_pairmodel as pm  # noqa: E402

EXP = sys.argv[1] if len(sys.argv) > 1 else "exp053_sp"
W_U = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 600
TRAIN = sys.argv[4] if len(sys.argv) > 4 else "all3"
FAM = list(C.FAMILIES)
TARGET = os.environ.get("FAM_TARGET", "soft_play")
OTHER_MODE = os.environ.get("OTHER_MODE", "drop")
SPY_MODE = os.environ.get("SPY_MODE", "drop")
W_POS = float(os.environ.get("W_POS", "1.0"))
DONOR = os.environ.get("DONOR", "exp048_fuse_bags")
assert TARGET in FAM, TARGET


def spy_list():
    sp = pl.read_parquet(C.DER / "spies_exp005.parquet").select("pA", "pB")
    p2 = C.DER / "spies2_exp022.parquet"
    if p2.exists():
        sp = pl.concat([sp, pl.read_parquet(p2).select("pA", "pB")]).unique()
    return sp.with_columns(pl.lit(True).alias("is_spy"))


def f32(df, feats):
    """float32 design matrix filled COLUMN BY COLUMN.

    The box is shared with ~8 other agents; a whole-frame `.to_numpy()` on 557 columns materialises a
    float64 intermediate (350k x 557 x 8 B = 1.6 GB) and this script was OOM-killed doing exactly that.
    Column-at-a-time costs one 1.4 MB temporary.
    """
    out = np.empty((df.height, len(feats)), dtype=np.float32)
    for j, c in enumerate(feats):
        out[:, j] = df[c].cast(pl.Float32).to_numpy(allow_copy=True)
    return out


def main():
    t0 = time.time()
    pm.TRAIN = TRAIN
    lab, full, w2a, ev, w2b = pm.load()
    if os.environ.get("USE_OB2") == "1":               # count-aware co-player block (src/pairfeat_ob2.py)
        def _ob2(df, win):
            if df is None:
                return None
            o = pl.read_parquet(C.DER / f"pfob2_{win}.parquet")
            return df.join(o, on=["pA", "pB"], how="left")
        full = _ob2(full, "dev_full")
        w2a = _ob2(w2a, "dev_w2000a")
        w2b = _ob2(w2b, "dev_w2000b")
        ev = _ob2(ev, "eval")
        print(f"[{EXP}] ob2 block joined", flush=True)
    feats = pm.feature_cols(full)
    del lab
    if os.environ.get("SKIP_EVAL") == "1":
        ev = None                                       # 112k x 557 not needed for an OOF-only screen
    gc.collect()
    sp = spy_list()

    def mark(df):
        if df is None:
            return None
        return df.join(sp, on=["pA", "pB"], how="left").with_columns(pl.col("is_spy").fill_null(False))

    full, w2a, w2b = mark(full), mark(w2a), mark(w2b)
    print(f"[{EXP}] target={TARGET} other={OTHER_MODE} spy={SPY_MODE} feats={len(feats)}", flush=True)

    keep_full = pl.col("is_lab") | ((~pl.col("touch_pos")) & (pl.col("n") >= 57))
    keep_w = pl.col("is_lab") | (~pl.col("touch_pos"))
    parts = [full.filter(keep_full), w2a.filter(keep_w), w2b.filter(keep_w)] if TRAIN == "all3" \
        else [full.filter(keep_full)]
    train = pl.concat(parts, how="diagonal_relaxed")

    is_pos_other = (pl.col("is_lab") & (pl.col("y") == 1) & (pl.col("behavior_family") != TARGET))
    if OTHER_MODE == "drop":
        train = train.filter(~is_pos_other)
    if SPY_MODE == "drop":
        train = train.filter(~pl.col("is_spy"))
    del w2b
    gc.collect()
    # ytr: 1 for the target family's labelled positives; also for the other families when OTHER_MODE=keep
    # (that variant is the champion objective with the target family merely UP-WEIGHTED), and for the
    # frozen spy list when SPY_MODE=pos (the champion's ADD_SPIES_POS=1 W_SPY=0.5 setting).
    pos_expr = pl.col("is_lab") & (pl.col("y") == 1) & (pl.col("behavior_family") == TARGET)
    if OTHER_MODE == "keep":
        pos_expr = pos_expr | is_pos_other
    if SPY_MODE == "pos":
        pos_expr = pos_expr | pl.col("is_spy")
    train = train.with_columns(pl.when(pos_expr).then(1).otherwise(0).alias("ytr"))

    y = train["ytr"].to_numpy()
    w = np.where(train["is_lab"].to_numpy(), 1.0, W_U)
    if SPY_MODE in ("pos", "neg"):
        w = np.where(train["is_spy"].to_numpy(), float(os.environ.get("W_SPY", "0.5")), w)
    tgt = train.select((pl.col("is_lab") & (pl.col("y") == 1)
                        & (pl.col("behavior_family") == TARGET)).fill_null(False)).to_series().to_numpy()
    w = np.where(tgt, w * W_POS, w)
    print(f"[{EXP}] train rows {train.height} positives {int(y.sum())} (target family only)", flush=True)

    X = f32(train, feats)
    fold = train["fold"].to_numpy()
    train = train.select("fold")            # release the 557-column frame
    del full
    gc.collect()
    params = dict(pm.PARAMS, num_threads=int(os.environ.get("NTHR", "4")))
    oof_w2a = np.zeros(w2a.height)
    wfold = w2a["fold"].to_numpy()
    for k in ([] if os.environ.get("EVAL_ONLY") == "1" else range(5)):
        tr = np.flatnonzero(fold != k)
        ds = lgb.Dataset(X[tr], y[tr], weight=w[tr], feature_name=feats, free_raw_data=True)
        m = lgb.train(params, ds, ROUNDS)
        del ds
        gc.collect()
        vi = np.flatnonzero(wfold == k)     # score the held-out fold only: 23k x 557, not 114k x 557
        oof_w2a[vi] = m.predict(f32(w2a[vi], feats))
        del m
        gc.collect()
        print(f"  fold {k} {time.time() - t0:.0f}s", flush=True)

    EVAL_ONLY = os.environ.get("EVAL_ONLY") == "1"
    w2a = w2a.with_columns(pl.lit(0.0).alias("oof") if EVAL_ONLY else pl.Series("oof", oof_w2a))
    poolC = w2a.filter(pl.col("is_lab") | ~pl.col("touch_pos"))
    if EVAL_ONLY:                                # reuse the OOF the screening run already wrote
        prev = pl.read_parquet(C.OOF_DIR / f"{EXP}_oof_w2a.parquet").select("pA", "pB", "oof")
        poolC = poolC.drop("oof").join(prev, on=["pA", "pB"], how="left")
        assert poolC["oof"].null_count() == 0, "EVAL_ONLY needs the screening run's _oof_w2a"
    spies1 = pl.read_parquet(C.DER / "spies_exp005.parquet").select("pA", "pB")
    wc = poolC.join(spies1, on=["pA", "pB"], how="anti")
    yv, sv = wc["y"].to_numpy(), wc["oof"].to_numpy()
    fv = wc["behavior_family"].fill_null("none").to_numpy()
    views = {"Cprime": C._ap(yv, sv)}
    for f in FAM:
        m = (yv == 0) | ((yv == 1) & (fv == f))
        views[f"Cp_{f[:2]}"] = C._ap(yv[m], sv[m])
    print(json.dumps({k: round(v, 4) for k, v in views.items()}))

    # ---- eval predictions (3 seeds, full-data refit), behaviour posteriors copied from DONOR
    if os.environ.get("SKIP_EVAL") == "1":           # screening run: OOF only, no eval refit
        o = (poolC.with_columns(pl.col("y").alias("y_orig"))
             .select("pA", "pB", "pair_id", "table_idx", "fold", "n", "y", "y_orig", "is_lab",
                     "behavior_family", "oof")
             .join(pl.read_parquet(C.OOF_DIR / f"{DONOR}_oof_w2a.parquet")
                   .select("pA", "pB", *[f"P_{f}" for f in FAM]), on=["pA", "pB"], how="left"))
        o.write_parquet(C.OOF_DIR / f"{EXP}_oof_w2a.parquet")
        views.update(exp=EXP, target=TARGET, other_mode=OTHER_MODE, spy_mode=SPY_MODE, w_pos=W_POS,
                     skip_eval=1, rounds=ROUNDS, runtime_s=round(time.time() - t0))
        (C.OOF_DIR / f"{EXP}_views.json").write_text(json.dumps(views, indent=1))
        print(f"done (OOF only) {time.time() - t0:.0f}s")
        return
    ep = C.load("eval_pairs").select("pair_id", pl.col("p1").alias("pA"), pl.col("p2").alias("pB"))
    evp = ep.join(ev, on=["pA", "pB"], how="left")
    assert evp["n"].null_count() == 0
    nseed = int(os.environ.get("NSEED", "3"))
    risk = np.zeros(evp.height)
    for sd in range(nseed):
        m = lgb.train({**params, "seed": 42 + sd},
                      lgb.Dataset(X, y, weight=w, feature_name=feats, free_raw_data=True), ROUNDS)
        for lo in range(0, evp.height, 20000):          # chunked scoring keeps the design matrix small
            hi = min(lo + 20000, evp.height)
            risk[lo:hi] += m.predict(f32(evp[lo:hi], feats)) / nseed
        del m
        gc.collect()
    donor_ev = pl.read_parquet(C.OOF_DIR / f"{DONOR}_eval_scores.parquet").select("pair_id", *[f"P_{f}" for f in FAM])
    out = evp.select("pair_id").with_columns(pl.Series("risk_raw", risk)).join(donor_ev, on="pair_id", how="left")
    assert out[f"P_{FAM[0]}"].null_count() == 0
    out.write_parquet(C.OOF_DIR / f"{EXP}_eval_scores.parquet")

    donor_w = pl.read_parquet(C.OOF_DIR / f"{DONOR}_oof_w2a.parquet").select("pA", "pB", *[f"P_{f}" for f in FAM])
    o = (poolC.with_columns(pl.col("y").alias("y_orig"))
         .select("pA", "pB", "pair_id", "table_idx", "fold", "n", "y", "y_orig", "is_lab", "behavior_family", "oof")
         .join(donor_w, on=["pA", "pB"], how="left"))
    assert o[f"P_{FAM[0]}"].null_count() == 0 and o.height == poolC.height
    if not EVAL_ONLY:
        o.write_parquet(C.OOF_DIR / f"{EXP}_oof_w2a.parquet")
    views.update(exp=EXP, target=TARGET, other_mode=OTHER_MODE, spy_mode=SPY_MODE, w_pos=W_POS, W_U=W_U,
                 rounds=ROUNDS, train=TRAIN, n_feats=len(feats), n_train=train.height, n_pos=int(y.sum()),
                 runtime_s=round(time.time() - t0))
    if not EVAL_ONLY:                            # never clobber the screening run's views
        (C.OOF_DIR / f"{EXP}_views.json").write_text(json.dumps(views, indent=1))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
