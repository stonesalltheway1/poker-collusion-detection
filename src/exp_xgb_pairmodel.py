"""exp_xgb_pairmodel: XGBoost GPU pair risk model on engine pair features.
Mirrors src/exp005_pairmodel.py with XGBoost backend (CUDA hist).
Usage: python src/exp_xgb_pairmodel.py [exp_id] [W_U] [rounds] [train_window]
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import exp005_pairmodel as pm  # noqa: E402

EXP = sys.argv[1] if len(sys.argv) > 1 else "exp043_xgb"
W_U = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 500
TRAIN = sys.argv[4] if len(sys.argv) > 4 else "all3"
FAM = list(C.FAMILIES)


def ap(y, s):
    return C._ap(np.asarray(y), np.asarray(s))


def main():
    t0 = time.time()
    print(f"[{EXP}] Starting XGBoost pair model training (ROUNDS={ROUNDS}, W_U={W_U}, TRAIN={TRAIN})...", flush=True)

    pm.TRAIN = TRAIN
    lab, full, w2a, ev, w2b = pm.load()
    feats = pm.feature_cols(full)
    print(f"[{EXP}] Data loaded: {len(feats)} features. full: {full.height}, w2a: {w2a.height}", flush=True)

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
        print(f"[{EXP}] Spy pairs promoted to positives: {int(full['is_spy'].sum())}", flush=True)

    keep_full = pl.col("is_lab") | ((~pl.col("touch_pos")) & (pl.col("n") >= 57))
    keep_w = pl.col("is_lab") | (~pl.col("touch_pos"))
    if TRAIN == "full":
        train = full.filter(keep_full)
    elif TRAIN == "w2000ab":
        train = pl.concat([w2a.filter(keep_w), w2b.filter(keep_w)], how="diagonal_relaxed")
    else:
        parts = [full.filter(keep_full), w2a.filter(keep_w), w2b.filter(keep_w)]
        train = pl.concat(parts, how="diagonal_relaxed")

    w = np.where(train["is_lab"].to_numpy(), 1.0, W_U)
    if "is_spy" in train.columns:
        w = np.where(train["is_spy"].to_numpy(), float(os.environ.get("W_SPY", "0.5")), w)

    X = train.select(feats).to_numpy().astype(np.float32)
    y = train["y"].to_numpy().astype(np.float32)
    fold = train["fold"].to_numpy()

    oof_full = np.zeros(full.height, dtype=np.float32)
    oof_w2a = np.zeros(w2a.height, dtype=np.float32)
    Xf = full.select(feats).to_numpy().astype(np.float32)
    Xw = w2a.select(feats).to_numpy().astype(np.float32)
    ffold, wfold = full["fold"].to_numpy(), w2a["fold"].to_numpy()

    xgb_params = {
        "tree_method": "hist",
        "device": os.environ.get("XGB_DEVICE", "cuda"),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.4,
        "reg_lambda": 1.0,
        "seed": 42,
    }
    if os.environ.get("SEED"):
        xgb_params["seed"] = int(os.environ["SEED"])

    for k in range(5):
        t_fold = time.time()
        tr = fold != k
        dtrain = xgb.DMatrix(X[tr], label=y[tr], weight=w[tr])

        idx_f = np.flatnonzero(ffold == k)
        idx_w = np.flatnonzero(wfold == k)
        dfull_val = xgb.DMatrix(Xf[idx_f])
        dw2a_val = xgb.DMatrix(Xw[idx_w])

        bst = xgb.train(xgb_params, dtrain, num_boost_round=ROUNDS)
        oof_full[idx_f] = bst.predict(dfull_val)
        oof_w2a[idx_w] = bst.predict(dw2a_val)
        print(f"[{EXP}] Fold {k} done in {time.time() - t_fold:.1f}s (total elapsed {time.time() - t0:.0f}s)", flush=True)
        # release the per-fold DMatrix/booster before building the next one: on a shared 32 GB host the
        # 5 folds otherwise accumulate and xgboost raises "bad allocation".  Numerically inert.
        del dtrain, dfull_val, dw2a_val, bst
        gc.collect()

    full = full.with_columns(pl.Series("oof", oof_full))
    w2a = w2a.with_columns(pl.Series("oof", oof_w2a))

    labf = full.filter(pl.col("is_lab"))
    poolB = full.filter(pl.col("is_lab") | ((~pl.col("touch_pos")) & (pl.col("n") >= 57)))
    poolC = w2a.filter(pl.col("is_lab") | ~pl.col("touch_pos"))

    views = {
        "A_ap": ap(labf["y"], labf["oof"]), "A_auc": roc_auc_score(labf["y"], labf["oof"]),
        "B_ap": ap(poolB["y"], poolB["oof"]), "B_n": poolB.height,
        "C_ap": ap(poolC["y"], poolC["oof"]), "C_n": poolC.height, "C_pos": int(poolC["y"].sum()),
    }
    for f in FAM:
        for nm, pool in (("B", poolB), ("C", poolC)):
            sub = pool.filter((pl.col("behavior_family") == f) | (pl.col("y") == 0))
            views[f"{nm}_ap_{f}"] = ap(sub["y"], sub["oof"])
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in views.items()}, indent=1), flush=True)

    # ---- behaviour (positives only, OOF), posteriors for pool C and eval
    posf = full.filter((pl.col("y") == 1) & pl.col("behavior_family").is_not_null()
                       & (pl.col("behavior_family") != "none"))
    Xp = posf.select(feats).to_numpy().astype(np.float32)
    yp = posf["behavior_family"].replace_strict({f: i for i, f in enumerate(FAM)}).to_numpy()
    pfold = posf["fold"].to_numpy()
    bparams = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=7, min_data_in_leaf=10,
                   feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=6,
                   verbose=-1, seed=42)
    post_C = np.zeros((w2a.height, 3))
    post_pos = np.zeros((posf.height, 3))
    for k in range(5):
        tr = pfold != k
        cw = np.bincount(yp[tr], minlength=3)
        wt = (len(yp[tr]) / (3 * cw))[yp[tr]]
        m = lgb.train(bparams, lgb.Dataset(Xp[tr], yp[tr], weight=wt, feature_name=feats), 300)
        post_pos[pfold == k] = m.predict(Xp[pfold == k])
        post_C[wfold == k] = m.predict(Xw[wfold == k])
    views["beh_acc_pos"] = float((post_pos.argmax(1) == yp).mean())
    poolC = w2a.with_columns([pl.Series(f"P_{f}", post_C[:, i]) for i, f in enumerate(FAM)]).filter(
        pl.col("is_lab") | ~pl.col("touch_pos"))
    predfam = np.array(FAM)[poolC.select([f"P_{f}" for f in FAM]).to_numpy().argmax(1)]
    bap = []
    for f in FAM:
        yt = ((poolC["y"] == 1) & (poolC["behavior_family"].fill_null("none") == f)).to_numpy().astype(int)
        bap.append(ap(yt, np.where(predfam == f, poolC["oof"].to_numpy(), 0.0)))
    views["C_behAP"] = float(np.mean(bap))
    views["composite_noev"] = 0.7 * views["C_ap"] + 0.1 * views["C_behAP"]
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in views.items() if k.startswith(("beh", "C_beh", "comp"))}), flush=True)

    # ---- final fit on all training rows, eval predictions
    ep = C.load("eval_pairs").select("pair_id", pl.col("p1").alias("pA"), pl.col("p2").alias("pB"))
    evp = ep.join(ev, on=["pA", "pB"], how="left")
    assert evp["n"].null_count() == 0
    Xe = evp.select(feats).to_numpy().astype(np.float32)
    deval = xgb.DMatrix(Xe)
    gc.collect()
    dtrain_all = xgb.DMatrix(X, label=y, weight=w)

    # See src/exp005_pairmodel.py: defaults 42/3 reproduce the shipped eval_scores bit for bit.
    ESB = int(os.environ.get("EVAL_SEED_BASE", "42"))
    NES = int(os.environ.get("EVAL_SEEDS", "3"))
    risk = np.zeros(evp.height, dtype=np.float32)
    for seed in range(NES):
        t_fit = time.time()
        p = dict(xgb_params, seed=ESB + seed)
        bst_all = xgb.train(p, dtrain_all, num_boost_round=ROUNDS)
        risk += bst_all.predict(deval) / float(NES)
        del bst_all
        gc.collect()
        print(f"[{EXP}] Eval seed {seed} trained in {time.time() - t_fit:.1f}s", flush=True)

    cwall = np.bincount(yp, minlength=3)
    mb = lgb.train(bparams, lgb.Dataset(Xp, yp, weight=(len(yp) / (3 * cwall))[yp], feature_name=feats), 300)
    post_e = mb.predict(Xe)
    out = evp.select("pair_id").with_columns(pl.Series("risk_raw", risk),
                                             *[pl.Series(f"P_{f}", post_e[:, i]) for i, f in enumerate(FAM)])

    C.OOF_DIR.mkdir(exist_ok=True)
    out.write_parquet(C.OOF_DIR / f"{EXP}_eval_scores.parquet")
    full.with_columns(pl.col("y").alias("y_orig") if "y_orig" not in full.columns else pl.col("y_orig")).select(
        "pA", "pB", "pair_id", "table_idx", "fold", "n", "y", "y_orig", "is_lab", "touch_pos", "behavior_family", "oof").write_parquet(
        C.OOF_DIR / f"{EXP}_oof_full.parquet")
    if "y_orig" not in poolC.columns:
        poolC = poolC.with_columns(pl.col("y").alias("y_orig"))
    poolC.select("pA", "pB", "pair_id", "table_idx", "fold", "n", "y", "y_orig", "is_lab", "behavior_family", "oof",
                 *[f"P_{f}" for f in FAM]).write_parquet(C.OOF_DIR / f"{EXP}_oof_w2a.parquet")
    posf.select("pair_id", "behavior_family").with_columns(*[pl.Series(f"P_{f}", post_pos[:, i]) for i, f in enumerate(FAM)]).write_parquet(
        C.OOF_DIR / f"{EXP}_oof_behavior_pos.parquet")

    views.update(exp=EXP, W_U=W_U, rounds=ROUNDS, train=TRAIN, n_feats=len(feats), runtime_s=round(time.time() - t0))
    (C.OOF_DIR / f"{EXP}_views.json").write_text(json.dumps(views, indent=1))
    print(f"[{EXP}] Completed successfully in {time.time() - t0:.0f}s! All artifacts written.", flush=True)


if __name__ == "__main__":
    main()
