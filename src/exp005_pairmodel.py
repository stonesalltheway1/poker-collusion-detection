"""exp005: pair risk model on engine pair features (src/pairfeat.py) + behaviour multinomial.

Training (dev_full window): 372 positives + 1488 labelled negatives (w=1) + unknown pairs with n>=57 and no
labelled-positive player (w=W_U). LightGBM, frozen table folds, fixed rounds (no early stopping on the held-out fold).
Views (all OOF on held-out tables):
  A  labelled-only AP (dev_full rows)
  B  pool AP: positives vs negatives + unknown (n>=57, no labelled-positive player), dev_full rows
  C  pool AP on dev_w2000a rows (n>=38; eval-like exposure), same exclusions
Behaviour: LightGBM multiclass on positives (OOF), argmax for every pair; BehAP reported on view C.
Usage: python src/exp005_pairmodel.py [exp_id] [W_U] [rounds]
"""
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

EXP = sys.argv[1] if len(sys.argv) > 1 else "exp005"
W_U = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 600
TRAIN = sys.argv[4] if len(sys.argv) > 4 else "full"      # full | w2000ab | all3
FAM = list(C.FAMILIES)
DROP_PREFIX = ("c_", "s_")
DROP = {"pA", "pB", "table_idx", "n", "hn", "is_spy", "is_lab_or_spy", "giftmax_ab", "giftmax_ba", "pair_id", "label", "behavior_family", "y", "is_lab", "touch_pos", "fold", "oof"}
_EVNEG = None
CFG = os.environ.get("CFG", "A")
_BASE = dict(objective="binary", num_threads=6, verbose=-1, seed=42)
PARAMS = {
    "A": dict(_BASE, learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.5,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0),
    "B": dict(_BASE, learning_rate=0.02, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.3,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, seed=7),
    "C": dict(_BASE, boosting="goss", learning_rate=0.05, num_leaves=63, min_data_in_leaf=10,
              feature_fraction=0.4, lambda_l2=0.5, seed=99),
}[CFG]
if os.environ.get("SEED"):
    PARAMS = dict(PARAMS, seed=int(os.environ["SEED"]))


TZ_FEATS = ['h_s_any_mean', 'h_s_any_top5_dcmin', 'h_s_any_top3', 'h_s_any_q999_r', 'h_s_any_q999_z', 'h_s_any_top5_dcmean', 'fam_unexplained', 'G_limp_first', 'h_s_any_q99_r', 'h_s_sp_mean', 'h_s_any_top1', 'G_pre_both_vol', 'oc_score', 'G_pf_overcall_3rd_Yin', 'h_s_any_q999', 'ob_llift_max_x_trash', 'mtop_limp_first', 'ob_rk_worst_sp_strongcheck', 'r_sp_raiseback', 'z_x_face_max', 'ob_llift_min_sp_strongcheck', 'h_s_any_q99_z', 'ob_llift_max_dt_pcall15', 'ob_exc_min_sp_g1', 'r_x_foldface_min']


def add_table_z(df):
    """within-table standardisation of the strongest features (tables differ in looseness/exposure)."""
    ex = []
    for f in TZ_FEATS:
        if f in df.columns:
            m = pl.col(f).mean().over("table_idx")
            sd = pl.col(f).std().over("table_idx")
            ex.append(((pl.col(f) - m) / (sd + 1e-9)).alias(f"tz_{f}"))
    return df.with_columns(ex) if ex else df


def feature_cols(df):
    return [c for c in df.columns if c not in DROP and not c.startswith(DROP_PREFIX)]


def load():
    lab = C.load("labels").select(col_pA := pl.col("p1").alias("pA"), pl.col("p2").alias("pB"), "pair_id", "label",
                                  "behavior_family")
    pos_players = set(lab.filter(pl.col("label") == 1).select(pl.concat_list("pA", "pB").explode())["pA"].to_list())
    folds = C.get_table_folds()

    def prep(name, min_n):
        df = pl.read_parquet(C.DER / f"pf{os.environ.get('PF_TAG', '')}_{name}.parquet").filter(pl.col("n") >= min_n)
        if os.environ.get("USE_HANDDET") == "1":
            df = df.join(pl.read_parquet(C.DER / f"pfh{os.environ.get('PFH_TAG', '')}_{name}.parquet"), on=["pA", "pB"], how="left")
        if os.environ.get("TABLE_Z") == "1":
            df = add_table_z(df)
        if os.environ.get("USE_MIX") == "1":
            df = df.join(pl.read_parquet(C.DER / f"pfm{os.environ.get('MIX_TAG', 'v2')}_{name}.parquet"),
                         on=["pA", "pB"], how="left")
        if os.environ.get("USE_OC") == "1":
            oc = pl.read_parquet(C.DER / f"oc_pair_scores_{name}.parquet").drop(
                [c for c in ("table_idx", "n", "n_right", "pair_id") if c], strict=False)
            df = df.join(oc, on=["pA", "pB"], how="left")
        df = df.join(lab, on=["pA", "pB"], how="left").join(folds, on="table_idx")
        touch = pl.col("pA").is_in(list(pos_players)) | pl.col("pB").is_in(list(pos_players))
        df = df.with_columns(
            pl.when(pl.col("label").is_not_null()).then(pl.col("label")).otherwise(0).alias("y"),
            pl.col("label").is_not_null().alias("is_lab"),
            (pl.col("label").is_null() & touch).alias("touch_pos"))
        return df
    extra = prep("dev_w2000b", 38) if TRAIN != "full" else None
    evneg = None
    if TRAIN.endswith("evneg"):
        evneg = prep("eval", 38).filter(pl.col("is_lab")).with_columns(
            pl.lit(0).alias("y"), pl.lit(True).alias("is_lab"), pl.lit(False).alias("touch_pos"),
            pl.lit("none").alias("behavior_family"))
    ev = pl.read_parquet(C.DER / f"pf{os.environ.get('PF_TAG', '')}_eval.parquet")
    globals()["_EVNEG"] = evneg
    if os.environ.get("USE_HANDDET") == "1":
        ev = ev.join(pl.read_parquet(C.DER / f"pfh{os.environ.get('PFH_TAG', '')}_eval.parquet"), on=["pA", "pB"], how="left")
    if os.environ.get("TABLE_Z") == "1":
        ev = add_table_z(ev)
    if os.environ.get("USE_MIX") == "1":
        ev = ev.join(pl.read_parquet(C.DER / f"pfm{os.environ.get('MIX_TAG', 'v2')}_eval.parquet"), on=["pA", "pB"], how="left")
    if os.environ.get("USE_OC") == "1":
        oc = pl.read_parquet(C.DER / "oc_pair_scores_eval.parquet").drop(
            [c for c in ("table_idx", "n", "n_right", "pair_id") if c], strict=False)
        ev = ev.join(oc, on=["pA", "pB"], how="left")
    return lab, prep("dev_full", 38), prep("dev_w2000a", 38), ev, extra


def ap(y, s):
    return C._ap(np.asarray(y), np.asarray(s))


def main():
    t0 = time.time()
    lab, full, w2a, ev, w2b = load()
    feats = feature_cols(full)
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
        print("spy pairs promoted to positives:", int(full["is_spy"].sum()))
    if os.environ.get("DROP_SPIES") == "1" and os.environ.get("ADD_SPIES_POS") != "1":
        sp = pl.read_parquet(C.DER / "spies_exp005.parquet").select("pA", "pB")
        full, w2a = full.join(sp, on=["pA", "pB"], how="anti"), w2a
        if w2b is not None:
            w2b = w2b.join(sp, on=["pA", "pB"], how="anti")
        print("dropped spies from training pools")
    keep_full = pl.col("is_lab") | ((~pl.col("touch_pos")) & (pl.col("n") >= 57))
    keep_w = pl.col("is_lab") | (~pl.col("touch_pos"))
    if TRAIN == "full":
        train = full.filter(keep_full)
    elif TRAIN == "w2000ab":
        train = pl.concat([w2a.filter(keep_w), w2b.filter(keep_w)], how="diagonal_relaxed")
    else:
        parts = [full.filter(keep_full), w2a.filter(keep_w), w2b.filter(keep_w)]
        if _EVNEG is not None:
            parts.append(_EVNEG)
            print("added eval-window labelled-pair rows as clean negatives:", _EVNEG.height)
        train = pl.concat(parts, how="diagonal_relaxed")
    w = np.where(train["is_lab"].to_numpy(), 1.0, W_U)
    if "is_spy" in train.columns:
        w = np.where(train["is_spy"].to_numpy(), float(os.environ.get("W_SPY", "0.5")), w)
    X, y, fold = train.select(feats).to_numpy().astype(np.float32), train["y"].to_numpy(), train["fold"].to_numpy()
    oof_full = np.zeros(full.height)
    oof_w2a = np.zeros(w2a.height)
    Xf, Xw = full.select(feats).to_numpy().astype(np.float32), w2a.select(feats).to_numpy().astype(np.float32)
    ffold, wfold = full["fold"].to_numpy(), w2a["fold"].to_numpy()
    imp = np.zeros(len(feats))
    for k in range(5):
        tr = fold != k
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], weight=w[tr], feature_name=feats), ROUNDS)
        oof_full[ffold == k] = m.predict(Xf[ffold == k])
        oof_w2a[wfold == k] = m.predict(Xw[wfold == k])
        imp += m.feature_importance("gain")
        print(f"fold {k} done {time.time() - t0:.0f}s", flush=True)
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
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in views.items()}, indent=1))

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
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in views.items() if k.startswith(("beh", "C_beh", "comp"))}))

    # ---- final fit on all training rows, eval predictions
    ep = C.load("eval_pairs").select("pair_id", pl.col("p1").alias("pA"), pl.col("p2").alias("pB"))
    evp = ep.join(ev, on=["pA", "pB"], how="left")
    assert evp["n"].null_count() == 0
    Xe = evp.select(feats).to_numpy().astype(np.float32)
    # EVAL_SEED_BASE/EVAL_SEEDS: the final full-data fits are bagged over seeds [base, base+n).
    # Defaults 42/3 reproduce every shipped eval_scores file BIT FOR BIT.  Without an explicit base
    # the SEED env var (which reseeds the OOF fold models) does NOT reach these fits, so two runs
    # that differ only in SEED write identical eval predictions.
    ESB = int(os.environ.get("EVAL_SEED_BASE", "42"))
    NES = int(os.environ.get("EVAL_SEEDS", "3"))
    risk = np.zeros(evp.height)
    for seed in range(NES):
        m = lgb.train({**PARAMS, "seed": ESB + seed}, lgb.Dataset(X, y, weight=w, feature_name=feats), ROUNDS)
        risk += m.predict(Xe) / NES
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
    top = np.argsort(-imp)[:30]
    views["top_features"] = [(feats[i], round(float(imp[i]), 1)) for i in top]
    views.update(exp=EXP, W_U=W_U, rounds=ROUNDS, train=TRAIN, n_feats=len(feats), runtime_s=round(time.time() - t0))
    (C.OOF_DIR / f"{EXP}_views.json").write_text(json.dumps(views, indent=1))
    print("top features:", views["top_features"][:15])
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
