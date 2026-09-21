"""exp004 -- first baseline submission (sub001).

Pair model : generic pair_generic rate features -> PU LightGBM (positives + labelled negatives w1 +
             sampled unknown dev pairs w0.3), OOF by frozen table folds, fixed rounds.
Behavior   : LightGBM multiclass on the 372 positives (class-balanced), argmax family for every pair.
Risk       : rank-normalised LGBM score, ties broken by a PU LogReg score (then a feature z-sum).
Evidence   : family-routed rule scores over shared phase-local hands (polars), tie-break pot size in bb.

Usage: python src/exp004_baseline.py [model|evidence|all]   (evidence cached in oof/exp004_ev_*.parquet)
"""
import os
import sys
import time

os.environ.setdefault("POLARS_MAX_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb

import common as C

T0 = time.time()
SEED = 42
NTHR = 4
FAMS = list(C.FAMILIES)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ============================================================================ features
SYM = ["both_vpip", "both_saw_flop", "sd_together", "hu_final", "hu_flop", "checkdown_hu", "both_aggr",
       "seat_opposite", "cojoin", "coleave"]
DIR = ["off1", "off2", "acts_v", "aggr_v", "acts_v_post", "aggr_v_post", "acts_hu", "aggr_hu", "faced",
       "fold_faced", "reraise_faced", "faced_post", "fold_faced_post", "sd_win", "faced_pre_strong",
       "fold_faced_pre_strong"]
FLOW = ["flow", "flow_hu", "flow_sd"]
DRATE = [("aggr_rate_1v2", "aggr_rate_2v1"), ("aggr_rate_1_not2", "aggr_rate_2_not1"),
         ("aggr_rate_hu_1v2", "aggr_rate_hu_2v1"), ("fold_rate_1_to_2", "fold_rate_2_to_1")]


def _mma(e12, e21, name):
    return [pl.max_horizontal(e12, e21).alias(f"{name}_max"), pl.min_horizontal(e12, e21).alias(f"{name}_min"),
            (e12 - e21).abs().alias(f"{name}_adiff")]


def make_features(df):
    c = pl.col
    S = c("shared_hands").cast(pl.Float64).clip(lower_bound=1)
    ex = [S.log().alias("log_shared"), c("copresence_lift"), c("copresence_min"), c("seat_adjacent_rate")]
    ex += [(c(k) / S).alias(f"r_{k}") for k in SYM]
    for k in DIR:
        ex += _mma(c(f"{k}_12") / S, c(f"{k}_21") / S, f"r_{k}")
    for k in FLOW:
        ex += _mma(c(f"{k}_12_bb") * 100 / S, c(f"{k}_21_bb") * 100 / S, f"f100_{k}")
    for k in ["net_flow_12_bb", "net_flow_hu_12_bb", "net_flow_sd_12_bb"]:
        ex.append((c(k).abs() * 100 / S).alias(f"abs_{k}_100"))
    for a, b in DRATE:
        ex += _mma(c(a), c(b), a.replace("1v2", "").replace("1_not2", "not").replace("1_to_2", "to"))
    # conditional rates with pseudo-counts (shrunk)
    for num, den in [("fold_faced", "faced"), ("reraise_faced", "faced"), ("fold_faced_post", "faced_post"),
                     ("fold_faced_pre_strong", "faced_pre_strong"), ("aggr_v", "acts_v"), ("aggr_hu", "acts_hu"),
                     ("sd_win", "sd_together")]:
        d21 = c(den) if den == "sd_together" else c(f"{den}_21")
        d12 = c(den) if den == "sd_together" else c(f"{den}_12")
        ex += _mma((c(f"{num}_12") + 1) / (d12 + 3), (c(f"{num}_21") + 1) / (d21 + 3), f"cr_{num}")
    # aggression toward partner minus aggression when partner is out
    ex += _mma(c("aggr_rate_1v2") - c("aggr_rate_1_not2"), c("aggr_rate_2v1") - c("aggr_rate_2_not1"), "aggr_lift")
    out = df.select("table_idx", "p1", "p2", "shared_hands", *ex)
    return out.with_columns(pl.selectors.float().fill_nan(None))


def feat_cols(df):
    return [k for k in df.columns if k not in ("table_idx", "p1", "p2", "shared_hands")]


# ============================================================================ model stage
LGB_PAIR = dict(objective="binary", learning_rate=0.03, num_leaves=15, feature_fraction=0.5, bagging_fraction=0.8,
                bagging_freq=1, min_data_in_leaf=20, lambda_l2=1.0, num_threads=NTHR, verbose=-1, seed=SEED)
N_ROUNDS = 500
LGB_BEH = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=7, feature_fraction=0.5,
               bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=8, lambda_l2=2.0, num_threads=NTHR,
               verbose=-1, seed=SEED)
N_ROUNDS_BEH = 250


def logreg_fit_predict(Xtr, ytr, wtr, Xte_list):
    from sklearn.linear_model import LogisticRegression
    med = np.nanmedian(Xtr, axis=0)
    fill = lambda X: np.where(np.isnan(X), med, X)
    A = fill(Xtr)
    mu, sd = A.mean(0), A.std(0) + 1e-9
    m = LogisticRegression(C=0.3, max_iter=2000)
    m.fit((A - mu) / sd, ytr, sample_weight=wtr)
    return [m.decision_function((fill(X) - mu) / sd) for X in Xte_list]


def ap_simple(y, s):
    return C._ap(y, s)


def stage_model():
    folds = C.get_table_folds()
    lab = C.load("labels").select("pair_id", "p1", "p2", "label", "behavior_family")
    pg_dev = C.load("pair_generic_dev")
    pg_w2k = C.load("pair_generic_dev_w2000")
    pg_eval = C.load("pair_generic_eval")
    ep = C.load("eval_pairs").select("pair_id", "p1", "p2", "shared_hands")

    F_dev = make_features(pg_dev).join(folds, on="table_idx")
    F_w2k = make_features(pg_w2k).join(folds, on="table_idx")
    F_eval = ep.select("pair_id", "p1", "p2").join(make_features(pg_eval), on=["p1", "p2"], how="inner")
    assert F_eval.height == ep.height, (F_eval.height, ep.height)
    chk = F_eval.join(ep.select("p1", "p2", pl.col("shared_hands").alias("sh_ep")), on=["p1", "p2"])
    assert (chk["shared_hands"] == chk["sh_ep"]).all()
    FC = feat_cols(make_features(pg_dev.head(5)))
    log(f"features: {len(FC)}")

    # ---- PU training set
    F_dev = F_dev.join(lab, on=["p1", "p2"], how="left")
    pos_players = pl.concat([lab.filter(pl.col("label") == 1)["p1"], lab.filter(pl.col("label") == 1)["p2"]]).unique()
    unk = F_dev.filter(pl.col("label").is_null() & (pl.col("shared_hands") >= 57)
                       & ~pl.col("p1").is_in(pos_players.implode()) & ~pl.col("p2").is_in(pos_players.implode()))
    rng = np.random.default_rng(SEED)
    unk = unk.with_columns(pl.Series("_r", rng.random(unk.height))).sort(["table_idx", "_r"])
    unk_s = unk.filter(pl.int_range(pl.len()).over("table_idx") < 50).drop("_r")
    labd = F_dev.filter(pl.col("label").is_not_null())
    train = pl.concat([labd.with_columns(pl.lit(1.0).alias("w"), pl.col("label").cast(pl.Int8).alias("y")),
                       unk_s.with_columns(pl.lit(0.3).alias("w"), pl.lit(0, pl.Int8).alias("y"))], how="diagonal")
    log(f"train rows: pos {int((labd['label'] == 1).sum())} neg {int((labd['label'] == 0).sum())} unk {unk_s.height}")

    X = train.select(FC).to_numpy().astype(np.float32)
    y = train["y"].to_numpy()
    w = train["w"].to_numpy()
    tf = train["fold"].to_numpy()
    is_lab = train["label"].is_not_null().to_numpy()

    # held-out pools
    pool_B = F_dev.filter(pl.col("label").is_not_null() | (pl.col("shared_hands") >= 57))
    pool_C = (F_w2k.join(lab.select("p1", "p2", "label"), on=["p1", "p2"], how="left")
              .filter(pl.col("shared_hands") >= 38))
    XB, XC = pool_B.select(FC).to_numpy().astype(np.float32), pool_C.select(FC).to_numpy().astype(np.float32)
    fB, fC = pool_B["fold"].to_numpy(), pool_C["fold"].to_numpy()
    oof = np.full(len(y), np.nan)
    oof_lr = np.full(len(y), np.nan)
    pB, pC = np.full(len(XB), np.nan), np.full(len(XC), np.nan)
    pB_lr = np.full(len(XB), np.nan)
    for k in range(5):
        tr, te = tf != k, tf == k
        bst = lgb.train(LGB_PAIR, lgb.Dataset(X[tr], y[tr], weight=w[tr]), N_ROUNDS)
        oof[te] = bst.predict(X[te], raw_score=True)
        pB[fB == k] = bst.predict(XB[fB == k], raw_score=True)
        pC[fC == k] = bst.predict(XC[fC == k], raw_score=True)
        lr_te, lr_B = logreg_fit_predict(X[tr], y[tr], w[tr], [X[te], XB[fB == k]])
        oof_lr[te], pB_lr[fB == k] = lr_te, lr_B
        ml = te & is_lab
        log(f"fold {k}: lab AP lgb {ap_simple(y[ml], oof[ml]):.4f} lr {ap_simple(y[ml], oof_lr[ml]):.4f}")

    yl, ol = y[is_lab], oof[is_lab]
    view = {"A_lab_ap_lgb": ap_simple(yl, ol), "A_lab_ap_lr": ap_simple(yl, oof_lr[is_lab])}
    yB = pool_B["label"].fill_null(0).to_numpy()
    view["B_pool_ap"] = ap_simple(yB, pB)
    view["B_pool_ap_lr"] = ap_simple(yB, pB_lr)
    view["B_n"] = [int(yB.sum()), int(len(yB))]
    yC = pool_C["label"].fill_null(0).to_numpy()
    view["C_pool_ap_w2000"] = ap_simple(yC, pC)
    labC = pool_C["label"].is_not_null().to_numpy()
    view["C_lab_ap_w2000"] = ap_simple(yC[labC], pC[labC])
    view["C_n"] = [int(yC.sum()), int(labC.sum()), int(len(yC))]
    view = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in view.items()}
    log("views:", view)

    # ---- final pair model on all training rows -> eval
    XE = F_eval.select(FC).to_numpy().astype(np.float32)
    bst = lgb.train(LGB_PAIR, lgb.Dataset(X, y, weight=w), N_ROUNDS)
    pe = bst.predict(XE, raw_score=True)
    (pe_lr,) = logreg_fit_predict(X, y, w, [XE])
    imp = sorted(zip(FC, bst.feature_importance("gain")), key=lambda t: -t[1])[:15]
    log("top gain:", [(a, int(b)) for a, b in imp])

    # ---- behavior multiclass on positives (OOF for all labelled dev pairs)
    fam_idx = {f: i for i, f in enumerate(FAMS)}
    posm = (y == 1)
    Xp = X[posm]
    yp = np.array([fam_idx[f] for f in train.filter(pl.col("y") == 1)["behavior_family"].to_list()])
    fp = tf[posm]
    cw = len(yp) / (3 * np.bincount(yp, minlength=3))
    beh_oof = np.full((len(y), 3), np.nan)
    for k in range(5):
        tr = fp != k
        bb = lgb.train(LGB_BEH, lgb.Dataset(Xp[tr], yp[tr], weight=cw[yp[tr]]), N_ROUNDS_BEH)
        te = (tf == k) & is_lab
        beh_oof[te] = bb.predict(X[te])
    acc = float((beh_oof[posm & is_lab].argmax(1) == yp).mean())
    log(f"behavior OOF accuracy on positives: {acc:.4f}")
    bb = lgb.train(LGB_BEH, lgb.Dataset(Xp, yp, weight=cw[yp]), N_ROUNDS_BEH)
    beh_eval = bb.predict(XE)

    # ---- assemble frames
    dev = (train.filter(pl.col("label").is_not_null()).select("pair_id", "p1", "p2", "table_idx")
           .with_columns(pl.Series("s_lgb", oof[is_lab]), pl.Series("s_lr", oof_lr[is_lab]),
                         pl.Series("beh", [FAMS[i] for i in beh_oof[is_lab].argmax(1)])))
    ev = (F_eval.select("pair_id", "p1", "p2")
          .with_columns(pl.Series("s_lgb", pe), pl.Series("s_lr", pe_lr),
                        pl.Series("beh", [FAMS[i] for i in beh_eval.argmax(1)]),
                        *[pl.Series(f"pb_{f}", beh_eval[:, i]) for i, f in enumerate(FAMS)]))
    zsum = F_eval.select(FC).fill_null(0).to_numpy()
    zsum = ((zsum - zsum.mean(0)) / (zsum.std(0) + 1e-9)).sum(1)
    ev = ev.with_columns(pl.Series("s_z", zsum))
    dev.write_parquet(C.OOF_DIR / "exp004_model_dev.parquet")
    ev.write_parquet(C.OOF_DIR / "exp004_model_eval.parquet")
    import json
    (C.OOF_DIR / "exp004_views.json").write_text(json.dumps({**view, "beh_acc_oof": round(acc, 4),
                                                             "n_features": len(FC), "top_gain": [a for a, _ in imp]},
                                                            indent=2))
    log("eval behavior dist:", ev["beh"].value_counts().to_dicts())


def rank_risk(df, cols):
    """(0,1) risk from lexicographic ranking on cols (primary first). Distinct by construction."""
    arrs = [df[c].to_numpy() for c in cols]
    order = np.lexsort(tuple(arrs[::-1]))  # last key in lexsort is primary
    r = np.empty(len(order))
    r[order] = (np.arange(len(order)) + 1) / (len(order) + 1)
    # count ties unresolved by all keys (resolved arbitrarily only there)
    key = np.stack(arrs, 1)[order]
    unresolved = int((np.all(key[1:] == key[:-1], axis=1)).sum()) if len(key) > 1 else 0
    return r, unresolved


# ============================================================================ evidence stage
def action_frame(hand_filter):
    """Actions (filtered hands) with prior aggressor on street and actor weak-hand flag."""
    c = pl.col
    pfe = C.load("preflop_equity_169").select(pl.col("class_id").alias("preflop_class"),
                                              (pl.col("top_pct_vs1") > 0.70).alias("weak"))
    a = (C.scan("actions").select("hand_idx", "action_no", "street", "player_idx", "action", "amount", "to_call",
                                  "players_active")
         .join(hand_filter.lazy(), on="hand_idx", how="semi")
         .sort("hand_idx", "action_no")
         .with_columns(((c("action") == 3) | (c("action") == 4) | ((c("action") == 5) & (c("amount") > c("to_call"))))
                       .alias("aggr"))
         .with_columns(pl.when(c("aggr")).then(c("player_idx")).otherwise(None).alias("_ap"))
         .with_columns(c("_ap").shift(1).forward_fill().over(["hand_idx", "street"]).alias("last_aggr"))
         .drop("_ap", "amount")
         .join(C.scan("seat_strength").select("hand_idx", "player_idx", "preflop_class")
               .join(hand_filter.lazy(), on="hand_idx", how="semi")
               .join(pfe.lazy(), on="preflop_class", how="left"), on=["hand_idx", "player_idx"], how="left")
         .drop("preflop_class")
         .collect())
    return a


def pair_hands(pairs, seats):
    """pairs: pair_id,p1,p2 ; seats: hand_idx,player_idx,net_chips (phase-filtered) -> shared hands."""
    s1 = seats.rename({"player_idx": "p1", "net_chips": "net1"})
    s2 = seats.rename({"player_idx": "p2", "net_chips": "net2"})
    return pairs.join(s1, on="p1").join(s2, on=["hand_idx", "p2"])


def family_scores(ph, acts, hands):
    """ph: pair_id,p1,p2,hand_idx,net1,net2 ; returns ph + sc_dt, sc_sp, sc_ci, pot_bb."""
    c = pl.col
    j = ph.select("pair_id", "hand_idx", "p1", "p2").join(acts, on="hand_idx")
    j = j.with_columns((c("player_idx") == c("p1")).alias("i1"), (c("player_idx") == c("p2")).alias("i2"))
    j = j.with_columns((c("i1") | c("i2")).alias("ip"),
                       ((c("i1") & (c("last_aggr") == c("p2"))) | (c("i2") & (c("last_aggr") == c("p1")))).fill_null(False)
                       .alias("face_partner"))
    # CI anchor: first weak pair preflop raise, else first pair preflop raise
    pre_pair_aggr = c("ip") & (c("street") == 0) & c("aggr")
    j = j.with_columns(
        c("action_no").filter(pre_pair_aggr & c("weak").fill_null(False)).min().over(["pair_id", "hand_idx"]).alias("wr_no"),
        c("action_no").filter(pre_pair_aggr).min().over(["pair_id", "hand_idx"]).alias("r_no"))
    j = j.with_columns(pl.coalesce("wr_no", "r_no").alias("anc_no"))
    j = j.with_columns(c("player_idx").filter(c("ip") & (c("action_no") == c("anc_no"))).first()
                       .over(["pair_id", "hand_idx"]).alias("anc_player"))
    after = (c("street") == 0) & (c("action_no") > c("anc_no"))
    st = (j.group_by("pair_id", "hand_idx", "street")
          .agg(c("i1").any().alias("a1"), c("i2").any().alias("a2"), c("aggr").sum().alias("n_aggr"),
               c("players_active").max().alias("pa_max")))
    st = (st.group_by("pair_id", "hand_idx")
          .agg((c("a1") & c("a2")).sum().alias("streets_both"),
               (c("a1") & c("a2") & (c("street") >= 1) & (c("pa_max") <= 2) & (c("n_aggr") == 0)).any().alias("hu_nobet")))
    g = (j.group_by("pair_id", "hand_idx")
         .agg((c("i1") & (c("action") == 0) & c("face_partner")).any().alias("f1_face2"),
              (c("i2") & (c("action") == 0) & c("face_partner")).any().alias("f2_face1"),
              (c("face_partner") & ((c("action") == 0) | (c("action") == 2) | ((c("action") == 5) & ~c("aggr")))).sum()
              .alias("passive_resp"),
              (c("face_partner") & c("aggr")).sum().alias("reraise_resp"),
              c("action").filter(c("i1") & (c("street") == 0)).sort_by(c("action_no").filter(c("i1") & (c("street") == 0)))
              .first().alias("first1"),
              c("action").filter(c("i2") & (c("street") == 0)).sort_by(c("action_no").filter(c("i2") & (c("street") == 0)))
              .first().alias("first2"),
              c("wr_no").first().is_not_null().alias("weak_raise"),
              (after & c("ip") & (c("player_idx") != c("anc_player")) & ((c("action") == 0) | c("aggr"))).any()
              .fill_null(False).alias("other_resp"),
              (after & ~c("ip") & (c("action") == 0)).sum().fill_null(0).alias("third_folds")))
    out = (ph.join(g, on=["pair_id", "hand_idx"], how="left").join(st, on=["pair_id", "hand_idx"], how="left")
           .join(hands, on="hand_idx", how="left"))
    bb = c("bb").cast(pl.Float64)
    oneway = ((c("net1") < 0) & (c("net2") > 0)) | ((c("net2") < 0) & (c("net1") > 0))
    loser_fold = pl.when(c("net1") < 0).then(c("f1_face2")).otherwise(c("f2_face1")).fill_null(False)
    sc_dt = pl.when(oneway).then(pl.min_horizontal(c("net1").abs(), c("net2").abs()) / bb
                                 + 5 * loser_fold.cast(pl.Float64)).otherwise(-1.0)
    gate_sp = (c("first1").fill_null(-1) != 0) & (c("first2").fill_null(-1) != 0)
    sp_raw = (10 * ((c("passive_resp").fill_null(0) > 0) & (c("reraise_resp").fill_null(0) == 0)).cast(pl.Float64)
              + c("streets_both").fill_null(0).cast(pl.Float64) + 2 * c("hu_nobet").fill_null(False).cast(pl.Float64))
    sc_sp = pl.when(gate_sp).then(sp_raw).otherwise(-1.0)
    sc_ci = (10 * c("weak_raise").fill_null(False).cast(pl.Float64)
             + 5 * c("other_resp").fill_null(False).cast(pl.Float64) + c("third_folds").fill_null(0).cast(pl.Float64))
    return out.select("pair_id", "hand_idx", sc_dt.alias("sc_dt"), sc_sp.alias("sc_sp"), sc_ci.alias("sc_ci"),
                      (c("final_pot") / bb).alias("pot_bb"))


def top5(sc, col):
    return (sc.sort(["pair_id", col, "pot_bb"], descending=[False, True, True])
            .group_by("pair_id", maintain_order=True).head(5)
            .with_columns(pl.int_range(pl.len()).over("pair_id").alias("k")))


def evidence_for(pairs, phase, table_batches, tag):
    """pairs: pair_id,p1,p2,table_idx. Returns long frame pair_id, fam, k, hand_idx."""
    hands_all = C.scan("hands").select("hand_idx", "table_idx", "phase", "bb", "final_pot")
    outs = []
    for bi, tabs in enumerate(table_batches):
        pb = pairs.filter(pl.col("table_idx").is_in(tabs))
        if pb.height == 0:
            continue
        hb = hands_all.filter((pl.col("phase") == phase) & pl.col("table_idx").is_in(tabs)).collect()
        seats = (C.scan("seats").select("hand_idx", "player_idx", "net_chips")
                 .join(hb.lazy().select("hand_idx"), on="hand_idx", how="semi").collect())
        ph = pair_hands(pb.select("pair_id", "p1", "p2"), seats)
        acts = action_frame(ph.select("hand_idx").unique())
        sc = family_scores(ph, acts, hb.select("hand_idx", "bb", "final_pot"))
        for fam, col in zip(FAMS, ["sc_dt", "sc_sp", "sc_ci"]):
            outs.append(top5(sc, col).select("pair_id", pl.lit(fam).alias("fam"), "k", "hand_idx"))
        log(f"evidence {tag} batch {bi + 1}/{len(table_batches)}: pairs {pb.height} pair-hands {ph.height} acts {acts.height}")
    return pl.concat(outs)


def stage_evidence(which=("dev", "eval")):
    pt = C.player_table()
    if "dev" in which:
        lab = C.load("labels").select("pair_id", "p1", "p2").join(pt.rename({"player_idx": "p1"}), on="p1")
        tabs = sorted(lab["table_idx"].unique().to_list())
        batches = [tabs[i:i + 100] for i in range(0, len(tabs), 100)]
        evidence_for(lab, 0, batches, "dev").write_parquet(C.OOF_DIR / "exp004_ev_dev.parquet")
    if "eval" in which:
        ep = C.load("eval_pairs").select("pair_id", "p1", "p2").join(pt.rename({"player_idx": "p1"}), on="p1")
        tabs = sorted(ep["table_idx"].unique().to_list())
        batches = [tabs[i:i + 25] for i in range(0, len(tabs), 25)]
        evidence_for(ep, 1, batches, "eval").write_parquet(C.OOF_DIR / "exp004_ev_eval.parquet")


def wide_evidence(long, route):
    """long: pair_id,fam,k,hand_idx ; route: pair_id,beh -> pair_id + evidence_hand_1..5 (hand_id strings)."""
    hid = C.scan("hands").select("hand_idx", "hand_id").join(long.lazy().select("hand_idx").unique(), on="hand_idx",
                                                             how="semi").collect()
    sel = (long.join(route.select("pair_id", pl.col("beh").alias("fam")), on=["pair_id", "fam"], how="inner")
           .join(hid, on="hand_idx").with_columns((pl.lit("evidence_hand_") + (pl.col("k") + 1).cast(pl.Utf8)).alias("col")))
    wide = sel.pivot(on="col", index="pair_id", values="hand_id")
    for c_ in C.EV_COLS:
        if c_ not in wide.columns:
            wide = wide.with_columns(pl.lit(None, pl.Utf8).alias(c_))
    return wide.select("pair_id", *C.EV_COLS)


# ============================================================================ assemble
def stage_assemble():
    import json
    dev = pl.read_parquet(C.OOF_DIR / "exp004_model_dev.parquet")
    ev = pl.read_parquet(C.OOF_DIR / "exp004_model_eval.parquet")
    ev_dev = pl.read_parquet(C.OOF_DIR / "exp004_ev_dev.parquet")
    ev_eval = pl.read_parquet(C.OOF_DIR / "exp004_ev_eval.parquet")
    views = json.loads((C.OOF_DIR / "exp004_views.json").read_text())

    r_dev, _ = rank_risk(dev, ["s_lgb", "s_lr"])
    dev_pred = (dev.with_columns(pl.Series("risk_score", r_dev)).rename({"beh": "predicted_behavior"})
                .join(wide_evidence(ev_dev, dev), on="pair_id", how="left")
                .select(C.SUB_COLS).to_pandas())

    # per-family evidence MAP@5 on dev (oracle routing = true family; predicted routing)
    sol = C.dev_solution()
    lab = C.load("labels").filter(pl.col("label") == 1).select("pair_id", pl.col("behavior_family").alias("beh"))
    evmap = {}
    for route_name, route in [("oracle", lab), ("pred", dev.select("pair_id", "beh"))]:
        w = wide_evidence(ev_dev, route).to_pandas().set_index("pair_id")
        per = {}
        for fam in FAMS:
            ids = lab.filter(pl.col("beh") == fam)["pair_id"].to_list()
            scores = []
            for pid in ids:
                rel = set(sol.set_index("pair_id").loc[pid, C.EV_COLS]) - {C.NO_EV}
                sub = [h for h in (w.loc[pid, C.EV_COLS].tolist() if pid in w.index else []) if isinstance(h, str)][:5]
                hits, ps = 0, 0.0
                for k, h in enumerate(sub, 1):
                    if h in rel:
                        hits += 1
                        ps += hits / k
                scores.append(ps / min(len(rel), 5) if rel else 0.0)
            per[fam] = round(float(np.mean(scores)), 4)
        evmap[route_name] = per
    log("dev evidence MAP@5 per family:", evmap)

    r_ev, unresolved = rank_risk(ev, ["s_lgb", "s_lr", "s_z"])
    eval_pred = (ev.with_columns(pl.Series("risk_score", r_ev)).rename({"beh": "predicted_behavior"})
                 .join(wide_evidence(ev_eval, ev), on="pair_id", how="left")
                 .select(C.SUB_COLS).to_pandas())
    ties_lgb = int(ev["s_lgb"].is_duplicated().sum())
    meta = {"desc": "generic pair_generic rate feats, PU LGBM + LGBM behavior + family rule evidence",
            "views": views, "evidence_map5_dev_by_family": evmap,
            "eval_ties_primary": ties_lgb, "eval_ties_unresolved_all_keys": unresolved,
            "runtime_s": round(time.time() - T0, 1)}
    parts = C.save_experiment("exp004", dev_pred, eval_pred, meta)
    log("CV parts:", parts)
    path, stats = C.write_submission("exp004", eval_pred)
    log("submission:", path, stats, "ties primary:", ties_lgb, "unresolved:", unresolved)
    log("behavior dist eval:", eval_pred.predicted_behavior.value_counts().to_dict())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("model", "all"):
        stage_model()
    if what in ("evidence", "all"):
        stage_evidence()
    if what == "evidence_dev":
        stage_evidence(("dev",))
    if what == "evidence_eval":
        stage_evidence(("eval",))
    if what in ("assemble", "all"):
        stage_assemble()
    log("done")
