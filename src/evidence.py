"""exp007 / exp008 -- in-pair EVIDENCE RANKERS on the pair-hand engine output
(src/engine.py; data/derived/ph_v2 by default, override with EVID_PH; EVID_REUSE=1 reuses the cached exp007 grid).

exp007 = NO chronology: hand features (orientation-free, donor-oriented, family gates, within-pair percentiles, pair
         context), single-stage, or a second stage on ORDERLESS stage-1 summaries when that scores better on OOF.
         (Measured: the orderless second stage is worth ~0, so exp008's gain is chronology, not stacking.)
exp008 = exp007 + WITHIN-PHASE CHRONOLOGY: relative timeline position q, prior counts of earlier gate/sub-type
         signature hands, and a two-stage model-based chronology (stage-1 = no-chronology binary model score ->
         K = summed score of earlier hands, # earlier hands above thresholds / in the pair's top-5/10, same
         sub-type counts). Nested cross-fit: stage-1 scores for stage-2 TRAINING rows come from inner folds that
         exclude the outer validation fold.
Chronology works because of how the dev evidence lists were built (early-biased, two chronologically sorted
sub-type segments); whether eval lists follow the same rule is an LB A/B (research/EVIDENCE.md).

Targets: in-pair objective (candidates = all phase shared hands of positive pairs, target = listed evidence).
CV: frozen table folds; OOF MAP@5 per family with the host AP@5 (denominator min(|relevant|, 5)).

Stages:  python src/evidence.py [cache|cv|eval|compose|all]
  cache    dev positive-pair hand rows -> data/derived/evidence_cache/dev_pos.parquet
  cv       exp007 grid (per-family vs pooled, binary vs lambdarank, rounds) + exp008 nested two-stage grid
           -> data/derived/evidence_cache/cv_results.json (all grids/configs/calibration), dev OOF score frames
  eval     fit chosen configs on all dev; score every eval-phase shared hand of the 112,540 eval pairs
  compose  compose_evidence() on dev OOF with true-family one-hot and uniform posteriors (+ host scorer check)

Score files (data/derived/evidence_scores_{dev,eval}_{exp007,exp008}.parquet):
  pair_id, hand_id, hand_idx, s_directed_transfer, s_soft_play, s_coordinated_isolation
  s_f = CALIBRATED P(hand is listed evidence | hand, pair is family f): Platt scaling of the family model's OOF
  margin fit on dev family-f pairs (dev in-pair base rate ~4%; eval pools are ~0.64x so true rates are higher, but
  equally for all families, so mixture rankings are unaffected). Kept rows = union over families of each pair's
  top-15 hands. Dev file = OOF scores for the 372 labelled positive pairs (every family model scores every pair).
compose_evidence(scores_df, posteriors_df): mixture score(h) = sum_f P(f|pair) * s_f(h) -> top-5 distinct hands.
"""
import json
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
SHORT = {"directed_transfer": "dt", "soft_play": "sp", "coordinated_isolation": "ci"}
PH = C.DER / os.environ.get("EVID_PH", "ph_v2" if (C.DER / "ph_v2").exists() else "ph")
CACHE = C.DER / "evidence_cache"
TOPK_KEEP = 15
GRID = [100, 200, 350, 500, 750, 1000]
c = pl.col


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ================================================================================================ features
ID_COLS = ("hand_idx", "table_idx", "hand_seq", "pA", "pB", "seatA", "seatB")


def ph_layout():
    sch = pl.read_parquet_schema(next(iter(sorted((PH / "phase0").glob("*.parquet")))))
    dirs = [k[:-3] for k in sch if k.endswith("_ab")]
    syms = [k for k in sch if k not in ID_COLS and not k.endswith("_ab") and not k.endswith("_ba")]
    return syms, dirs


SYMS, DIRS = ph_layout()
# pair-level donor orientation (donor = larger summed EV gift over the pair's phase hands; 97% right on DT)
PDON = ["net_bb", "gift", "gift_max", "imp", "max_hueq_fold_post", "max_eq_fold_face", "min_hueq_call_post",
        "min_hueq_call_tr", "min_pcall_pre", "max_pfeq_call_pre", "fold_pre", "fold_post", "call_pre", "call_post",
        "raise_pre", "raise_post", "bluffinto", "face"]
FPA_AGG = ["trash", "weak10", "minpa", "aggpre", "aggpost", "vol"]            # first pair aggressor's own
FPA_PART = ["fold_after_aggr", "raise_pre", "call_pre", "fold_pre", "vol", "enter", "face", "minpa", "trash"]  # partner
FLAGS = ["flow", "dt_strong", "dt_gift5", "dt_pfweak", "dt_union", "dt_cand", "dt_fold", "dt_call", "dt_bluff",
         "don_pair", "sp_g0", "sp_g1", "sp_pl", "sp_sc", "sp_foldresp", "sp_callresp", "sp_soft", "ci_s1", "ci_s14",
         "ci_s15", "ci_s7", "ci_sandw", "gate_union"]
PCT = {"transfer": 1, "gift_don": 1, "gift_max_mx": 1, "imp_don": 1, "imp_mx": 1, "max_hueq_fold_post_don": 1,
       "max_eq_fold_face_don": 1, "min_hueq_call_tr_don": -1, "min_hueq_call_post_don": -1, "min_pcall_pre_don": -1,
       "pot_bb": 1, "max_hs_check_mx": 1, "max_hs_call_post_mx": 1, "max_hs_fold_post_mx": 1, "minpa_mn": -1,
       "fpa_p": -1, "tf_pre": 1, "face_sum": 1, "strongcheck_sum": 1}
CHRONO_FLAGS = ["flow", "dt_strong", "dt_gift5", "dt_pfweak", "dt_cand", "dt_fold", "dt_call", "dt_bluff", "sp_g1",
                "sp_pl", "sp_sc", "sp_foldresp", "sp_callresp", "sp_soft", "ci_s1", "ci_s14", "ci_s15", "ci_s7"]


def hand_features(df):
    """df: ph rows (+ pair_id), SORTED by pair_id, hand_seq. Orientation-free features (canonical A/B order carries
    no signal: every directional feature enters as max/min or oriented by a symmetric rule)."""
    fwd = (c("net_bb_ab") < 0) & (c("net_bb_ba") > 0)          # A gives to B
    bwd = (c("net_bb_ba") < 0) & (c("net_bb_ab") > 0)
    donA = fwd | (~bwd & ((c("gift_ab") > c("gift_ba")) |
                          ((c("gift_ab") == c("gift_ba")) & (c("net_bb_ab") < c("net_bb_ba"))) |
                          ((c("gift_ab") == c("gift_ba")) & (c("net_bb_ab") == c("net_bb_ba")) & (c("imp_ab") >= c("imp_ba")))))
    df = df.with_columns(flow=fwd | bwd, donA=donA, n_shared=pl.len().over("pair_id").cast(pl.Float32),
                         pdonA=c("gift_ab").sum().over("pair_id") >= c("gift_ba").sum().over("pair_id"),
                         pg_ab=c("gift_ab").sum().over("pair_id"), pg_ba=c("gift_ba").sum().over("pair_id"))
    ex = []
    for f in DIRS:
        a, b = c(f + "_ab"), c(f + "_ba")
        ex += [pl.max_horizontal(a, b).alias(f + "_mx"), pl.min_horizontal(a, b).alias(f + "_mn"),
               pl.when(c("donA")).then(a).otherwise(b).alias(f + "_don"),
               pl.when(c("donA")).then(b).otherwise(a).alias(f + "_rec")]
    for f in PDON:
        a, b = c(f + "_ab"), c(f + "_ba")
        ex += [pl.when(c("pdonA")).then(a).otherwise(b).alias(f + "_pdon"),
               pl.when(c("pdonA")).then(b).otherwise(a).alias(f + "_prec")]
    aggA, aggB = c("fpa_role") == 1, c("fpa_role") == 2
    for f in FPA_AGG:
        ex.append(pl.when(aggA).then(c(f + "_ab")).when(aggB).then(c(f + "_ba")).otherwise(None).alias(f + "_fpa"))
    for f in FPA_PART:
        ex.append(pl.when(aggA).then(c(f + "_ba")).when(aggB).then(c(f + "_ab")).otherwise(None).alias(f + "_fpp"))
    ex += [(c("face_ab") + c("face_ba")).alias("face_sum"), (c("strongcheck_ab") + c("strongcheck_ba")).alias("strongcheck_sum"),
           (c("raise_pre_ab") + c("raise_post_ab") + c("raise_pre_ba") + c("raise_post_ba")).alias("raises_vs"),
           pl.when(fwd | bwd).then(pl.min_horizontal(c("net_bb_ab").abs(), c("net_bb_ba").abs())).otherwise(0.0).alias("transfer"),
           (pl.max_horizontal(c("pg_ab"), c("pg_ba")) / (c("pg_ab") + c("pg_ba") + 1e-3)).alias("pair_gift_share"),
           (pl.max_horizontal(c("pg_ab"), c("pg_ba")) / c("n_shared")).alias("pair_gift_rate"),
           c("pot_bb").log1p().alias("log_pot"), c("n_shared").log().alias("log_shared")]
    df = df.with_columns(ex)
    g = df.with_columns(
        dt_strong=c("flow") & ((c("max_hueq_fold_post_don") >= 0.7) | (c("min_hueq_call_tr_don") <= 0.15)),
        dt_gift5=c("flow") & (c("gift_don") > 5),
        dt_pfweak=c("flow") & (c("min_pcall_pre_don") <= 0.15),
        dt_union=c("flow") & ((c("max_hueq_fold_post_don") >= 0.5) | (c("max_eq_fold_face_don") >= 0.5) |
                              (c("min_hueq_call_post_don") <= 0.25) | (c("bluffinto_don") > 0)),
        dt_fold=c("flow") & ((c("fold_pre_don") + c("fold_post_don")) > 0),
        dt_bluff=c("flow") & (c("bluffinto_don") > 0),
        don_pair=c("donA") == c("pdonA"),
        sp_g0=(c("first_fold_nonfacing_ab") == 0) & (c("first_fold_nonfacing_ba") == 0),
        sp_g1=(c("enter_ab") == 1) & (c("enter_ba") == 1),
        sp_sc=c("strongcheck_sum") > 0,
        ci_s1=(c("trash_ab") + c("trash_ba")) > 0,
        ci_s14=(c("fpa_ord") == 0) & (c("fpa_p") < 0.3),
        ci_resp=(c("fold_after_aggr_fpp") == 1) | (c("raise_pre_fpp") > 0),
        ci_sandw=c("sandwich") > 0,
    )
    g = g.with_columns(
        dt_cand=c("dt_union") & c("don_pair"),
        dt_call=c("flow") & ~c("dt_fold") & ((c("call_post_don") > 0) | (c("sd_both") > 0)),
        sp_pl=c("sp_g1") & (c("face_sum") > 0) & (c("raises_vs") == 0),
        sp_foldresp=c("sp_g0") & ((c("fold_pre_mx") + c("fold_post_mx")) > 0),
        sp_callresp=c("sp_g0") & ((c("call_pre_mx") + c("call_post_mx")) > 0),
        ci_s15=(c("fpa_ord") <= 2) & (c("fpa_partner_behind") == 1) & c("ci_resp").fill_null(False),
        ci_s7=c("ci_s1") & c("ci_resp").fill_null(False),
    )
    g = g.with_columns(
        sp_soft=c("sp_pl") & (c("sp_sc") | (c("fold_post_mx") > 0) | (c("call_post_mx") > 0) |
                              ((c("call_pre_mx") > 0) & (c("max_pfeq_call_pre_mx") >= 0.55))),
        gate_union=c("sp_g0") | c("ci_s15") | (c("face_sum") > 0),
    )
    ctx = []
    for f in FLAGS:
        n = c(f).cast(pl.Float32).sum().over("pair_id")
        ctx += [n.alias("n_" + f), (n / c("n_shared")).alias("r_" + f)]
    for k, sgn in PCT.items():
        v = c(k) * sgn
        ctx += [(v.rank("average").over("pair_id") / c("n_shared")).alias("pct_" + k),
                (v.rank("min", descending=True).over("pair_id") - 1).cast(pl.Float32).alias("top_" + k)]
    return g.with_columns(ctx)


def base_feature_names(F):
    drop = set(ID_COLS) | {"pair_id", "family", "is_ev", "fold", "evidence_rank", "n_rel", "donA", "pdonA", "pg_ab",
                           "pg_ba", "ci_resp", "hand_id"}
    return [k for k in F.columns if k not in drop and not k.endswith("_ab") and not k.endswith("_ba")]


def sub_type(fam):
    if fam == "directed_transfer":
        return pl.when(c("dt_fold")).then(1).when(c("dt_call")).then(2).otherwise(0)
    if fam == "soft_play":
        return pl.when(c("sp_foldresp")).then(1).when(c("sp_g0")).then(2).otherwise(0)
    return pl.when(c("ci_s14")).then(1).when(c("ci_s15")).then(2).otherwise(0)


def chrono_features(F, s1, fam, order=True):
    """Stage-2 features for family model `fam`. F sorted by pair_id, hand_seq; s1 = stage-1 (no-chronology)
    probability for this family, aligned to F rows.
    order=False (exp007 stacked): ORDERLESS only -- s1, its within-pair rank/percentile, # pair hands above thresholds.
    order=True  (exp008): adds within-phase chronology -- earlier/later ORDER inside the pair's phase."""
    G = F.select("pair_id", "n_shared", "sp_g0", *CHRONO_FLAGS, *[f"n_{f}" for f in CHRONO_FLAGS]).with_columns(
        s1=pl.Series(s1.astype(np.float32)), sub=sub_type(fam))
    s = c("s1")
    if not order:
        ex = [s.alias("s1"), (s.rank("min", descending=True).over("pair_id") - 1).cast(pl.Float32).alias("s1_top"),
              (s.rank("average").over("pair_id") / c("n_shared")).alias("s1_pct")]
        for t in (0.1, 0.25, 0.5):
            ex.append((s > t).cast(pl.Float32).sum().over("pair_id").alias(f"s1_n_gt{str(t).replace('0.', '')}"))
        return G.with_columns(ex).select([e.meta.output_name() for e in ex])
    ex = [((pl.int_range(pl.len()).over("pair_id") + 0.5) / c("n_shared")).cast(pl.Float32).alias("q")]
    for f in CHRONO_FLAGS:
        x = c(f).cast(pl.Float32)
        prior = x.cum_sum().over("pair_id") - x
        ex += [prior.alias("prior_" + f),
               pl.when(c("n_" + f) > 0).then(prior / c("n_" + f)).otherwise(None).alias("prel_" + f)]
    ex += [s.alias("s1"), (s.cum_sum().over("pair_id") - s).alias("s1_K"),
           (s.rank("min", descending=True).over("pair_id") - 1).cast(pl.Float32).alias("s1_top"),
           (s.rank("average").over("pair_id") / c("n_shared")).alias("s1_pct"), c("sub").cast(pl.Float32).alias("sub")]
    for t in (0.1, 0.25, 0.5):
        x = (s > t).cast(pl.Float32)
        tag = str(t).replace("0.", "")
        n = x.sum().over("pair_id")
        prior = x.cum_sum().over("pair_id") - x
        ex += [prior.alias(f"s1_prior_gt{tag}"), n.alias(f"s1_n_gt{tag}"),
               pl.when(n > 0).then(prior / n).otherwise(None).alias(f"s1_prel_gt{tag}"),
               ((x.cum_sum().over(["pair_id", "sub"]) - x)).alias(f"s1_prior_sub_gt{tag}")]
    ex.append((s.cum_sum().over(["pair_id", "sub"]) - s).alias("s1_K_sub"))
    for k in (5, 10):
        top = (s.rank("ordinal", descending=True).over("pair_id") <= k).cast(pl.Float32)
        ex.append((top.cum_sum().over("pair_id") - top).alias(f"s1_prior_top{k}"))
    out = G.with_columns(ex)
    names = [e.meta.output_name() for e in ex]
    return out.select(names)


# ================================================================================================ data
def cache_dev():
    t = time.time()
    lab = C.labelled_pairs().filter(c("label") == 1).select(
        "pair_id", c("behavior_family").alias("family"), c("p1").alias("pA"), c("p2").alias("pB"), "fold")
    ev = C.load("evidence").select("pair_id", "hand_idx", "evidence_rank")
    D = (pl.scan_parquet(PH / "phase0" / "*.parquet").join(lab.lazy(), on=["pA", "pB"]).collect()
         .join(ev, on=["pair_id", "hand_idx"], how="left")
         .with_columns(is_ev=c("evidence_rank").is_not_null().cast(pl.Int8))
         .with_columns(n_rel=c("is_ev").cast(pl.Int32).sum().over("pair_id"))
         .sort("pair_id", "hand_seq"))
    assert D["is_ev"].sum() == ev.height, (D["is_ev"].sum(), ev.height)
    CACHE.mkdir(parents=True, exist_ok=True)
    D.write_parquet(CACHE / "dev_pos.parquet")
    stamp = {f.name: f.stat().st_mtime for f in sorted((PH / "phase1").glob("*.parquet"))}
    stamp.update({"phase0/" + f.name: f.stat().st_mtime for f in sorted((PH / "phase0").glob("*.parquet"))})
    stamp["_dir"] = PH.name
    (CACHE / "ph_stamp.json").write_text(json.dumps(stamp))
    log(f"cache: {D.height:,} dev positive pair-hand rows, {D['pair_id'].n_unique()} pairs, "
        f"{D['is_ev'].sum()} evidence, {time.time() - t:.1f}s")


def load_dev():
    D = pl.read_parquet(CACHE / "dev_pos.parquet")
    F = hand_features(D)
    return F


# ================================================================================================ metric
def map5_frame(pair_id, fam, is_ev, n_rel, score, tb):
    """Host AP@5 per pair (top-5 by score, ties by tb desc), denominator min(|relevant|, 5)."""
    t = (pl.DataFrame({"pair_id": pair_id, "family": fam, "is_ev": is_ev, "n_rel": n_rel,
                       "s": np.asarray(score, dtype=np.float64), "tb": tb})
         .sort(["pair_id", "s", "tb"], descending=[False, True, True])
         .with_columns(k=pl.int_range(pl.len()).over("pair_id") + 1)
         .filter(c("k") <= 5)
         .with_columns(hits=c("is_ev").cast(pl.Int32).cum_sum().over("pair_id")))
    return t.group_by("pair_id", "family").agg(
        ((c("hits") / c("k")) * c("is_ev")).sum().alias("num"), c("n_rel").first().alias("n_rel")).with_columns(
        ap=pl.when(c("n_rel") > 0).then(c("num") / c("n_rel").clip(upper_bound=5)).otherwise(0.0))


class Dev:
    def __init__(self, F):
        self.F = F
        self.pair = F["pair_id"].to_numpy()
        self.fam = F["family"].to_numpy()
        self.fold = F["fold"].to_numpy()
        self.y = F["is_ev"].to_numpy().astype(np.int32)
        self.n_rel = F["n_rel"].to_numpy()
        self.tb = F["pot_bb"].to_numpy()

    def map5(self, score, fam=None):
        m = map5_frame(self.pair, self.fam, self.y, self.n_rel, score, self.tb)
        if fam is not None:
            return float(m.filter(c("family") == fam)["ap"].mean())
        return {f: float(m.filter(c("family") == f)["ap"].mean()) for f in FAMS}

    def groups(self, mask):
        p = self.pair[mask]
        brk = np.flatnonzero(p[1:] != p[:-1]) + 1
        return np.diff(np.concatenate([[0], brk, [len(p)]]))


# ================================================================================================ models
def params(obj, seed=SEED):
    p = dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.5, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=1.0, num_threads=NTHR, verbose=-1, seed=seed, max_bin=255)
    if obj == "binary":
        p["objective"] = "binary"
    else:
        p.update(objective="lambdarank", lambdarank_truncation_level=20, eval_at=[5])
    return p


def onehot(fams_arr):
    return np.stack([(fams_arr == f).astype(np.float32) for f in FAMS], axis=1)


def fit(dev, X, mask, obj, rounds, pooled):
    Xtr = X[mask]
    if pooled:
        Xtr = np.hstack([Xtr, onehot(dev.fam[mask])])
    ds = lgb.Dataset(Xtr, dev.y[mask], group=dev.groups(mask) if obj != "binary" else None, free_raw_data=True)
    return lgb.train(params(obj), ds, num_boost_round=rounds)


def predict(model, X, obj, pooled, fam, rounds=None):
    if pooled:
        X = np.hstack([X, np.tile(onehot(np.array([fam])), (len(X), 1))])
    return model.predict(X, num_iteration=rounds, raw_score=True)   # margin (logit for binary)


def cv_grid(dev, X, obj, pooled, grid=GRID, fams=FAMS):
    """OOF margins for every (family model, rounds): model trained on folds != k (family rows, or all rows if
    pooled), predicting ALL fold-k rows. Returns {fam: {rounds: array}}."""
    out = {f: {r: np.zeros(len(dev.y)) for r in grid} for f in fams}
    for k in range(5):
        va = dev.fold == k
        if pooled:
            m = fit(dev, X, dev.fold != k, obj, max(grid), True)
        for f in fams:
            if not pooled:
                m = fit(dev, X, (dev.fold != k) & (dev.fam == f), obj, max(grid), False)
            for r in grid:
                out[f][r][va] = predict(m, X[va], obj, pooled, f, r)
    return out


def platt(margin, y):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(margin.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


# ================================================================================================ CV
def run_cv():
    t_start = time.time()
    F = load_dev()
    dev = Dev(F)
    feats = base_feature_names(F)
    X0 = F.select([c(k).cast(pl.Float32) for k in feats]).to_numpy()
    keep = np.nanstd(X0, axis=0) > 0
    feats = [k for k, kk in zip(feats, keep) if kk]
    X = X0[:, keep]
    log(f"dev rows {len(X):,}, features {len(feats)} (dropped {int((~keep).sum())} constant)")
    res = {"exp007": {}, "exp008": {}, "features_007": feats, "ph_dir": PH.name}

    # ---- sanity: rule recipes vs forensics numbers
    ci_rule = (F["ci_s14"].cast(pl.Float64) * 10000 - F["hand_seq"].cast(pl.Float64)).to_numpy()
    res["rule_ci_first5_s14_chrono"] = dev.map5(ci_rule, "coordinated_isolation")
    log("sanity rule CI first-5 S14 (forensics 0.586):", round(res["rule_ci_first5_s14_chrono"], 4))

    # ---- exp007 single-stage grid (per-family vs pooled, binary vs lambdarank, rounds)
    gpath = CACHE / "grid007_single.npz"
    reuse = os.environ.get("EVID_REUSE") == "1" and gpath.exists()
    grids = {}
    if reuse:
        z = np.load(gpath)
        for key in z.files:
            tag, f, r = key.split("|")
            grids.setdefault(tag, {}).setdefault(f, {})[int(r)] = z[key]
    for obj in ("binary", "lambdarank"):
        for pooled in (False, True):
            tag = f"{'pooled' if pooled else 'fam'}_{obj}"
            t = time.time()
            if not reuse:
                grids[tag] = cv_grid(dev, X, obj, pooled)
            tab = {f: {r: round(dev.map5(grids[tag][f][r], f), 4) for r in GRID} for f in FAMS}
            res["exp007"][tag] = tab
            log(f"exp007 single {tag} ({time.time() - t:.0f}s):", json.dumps(tab))
    if not reuse:
        np.savez(gpath, **{f"{tag}|{f}|{r}": grids[tag][f][r] for tag in grids for f in FAMS for r in GRID})
    best_single = {}
    for f in FAMS:
        sc, tag, r = max((res["exp007"][tag][f][r], tag, r) for tag in grids for r in GRID)
        bsc, btag, br = max((res["exp007"][tag][f][r], tag, r) for tag in grids if tag.endswith("binary") for r in GRID)
        best_single[f] = {"map5": sc, "tag": tag, "rounds": r, "stage1_tag": btag, "stage1_rounds": br, "stage1_map5": bsc}
    res["best007_single"] = best_single
    log("best007 single-stage:", json.dumps(best_single))
    oof_single = {f: grids[best_single[f]["tag"]][f][best_single[f]["rounds"]] for f in FAMS}
    s1_oof = {f: sigmoid(grids[best_single[f]["stage1_tag"]][f][best_single[f]["stage1_rounds"]]) for f in FAMS}
    del grids

    # ---- nested two-stage: stage-1 = best single-stage BINARY config; stage-2 variants
    #      exp007s = orderless stacking (no chronology), exp008 = + within-phase chronology
    def stage1_inner(k):
        """s1 for all rows under outer fold k: fold-k rows from s1_oof (trained on folds != k); rows of fold j != k
        from a stage-1 model trained on folds not in {k, j} (so stage-2 training rows never saw fold k)."""
        s1 = {f: s1_oof[f].copy() for f in FAMS}
        for j in range(5):
            if j == k:
                continue
            vj = dev.fold == j
            trm = (dev.fold != k) & (dev.fold != j)
            pooled_models = {}
            for f in FAMS:
                tag, r = best_single[f]["stage1_tag"], best_single[f]["stage1_rounds"]
                pooled = tag.startswith("pooled")
                if pooled:
                    if r not in pooled_models:
                        pooled_models[r] = fit(dev, X, trm, "binary", r, True)
                    m = pooled_models[r]
                else:
                    m = fit(dev, X, trm & (dev.fam == f), "binary", r, False)
                s1[f][vj] = sigmoid(predict(m, X[vj], "binary", pooled, f))
        return s1

    VARIANTS = {"exp007s": False, "exp008": True}
    t = time.time()
    g2 = {v: {tag: {f: {r: np.zeros(len(dev.y)) for r in GRID} for f in FAMS} for tag in ("fam_binary", "fam_lambdarank")}
          for v in VARIANTS}
    names2 = {}
    for k in range(5):
        s1 = stage1_inner(k)
        va = dev.fold == k
        for f in FAMS:
            trm = (dev.fold != k) & (dev.fam == f)
            for v, order in VARIANTS.items():
                Cf = chrono_features(F, s1[f], f, order)
                names2[v] = list(Cf.columns)
                X2 = np.hstack([X, Cf.to_numpy().astype(np.float32)])
                for obj in ("binary", "lambdarank"):
                    m = fit(dev, X2, trm, obj, max(GRID), False)
                    for r in GRID:
                        g2[v][f"fam_{obj}"][f][r][va] = predict(m, X2[va], obj, False, f, r)
        log(f"nested outer fold {k} done ({time.time() - t:.0f}s)")
    res["features_007s_stage2"] = names2["exp007s"]
    res["features_008_chrono"] = names2["exp008"]
    best2, oof2 = {}, {}
    for v in VARIANTS:
        res[v] = {}
        for tag in g2[v]:
            tab = {f: {r: round(dev.map5(g2[v][tag][f][r], f), 4) for r in GRID} for f in FAMS}
            res[v][tag] = tab
            log(f"{v} {tag}:", json.dumps(tab))
        best2[v] = {}
        for f in FAMS:
            sc, tag, r = max((res[v][tag][f][r], tag, r) for tag in g2[v] for r in GRID)
            best2[v][f] = {"map5": sc, "tag": tag, "rounds": r}
        oof2[v] = {f: g2[v][best2[v][f]["tag"]][f][best2[v][f]["rounds"]] for f in FAMS}
        log(f"best {v}:", json.dumps(best2[v]))
    res["best007s"] = best2["exp007s"]
    res["best008"] = best2["exp008"]

    # final exp007 (no chronology) = better of single-stage and orderless stacked, per family
    best007, oof007 = {}, {}
    for f in FAMS:
        if best2["exp007s"][f]["map5"] > best_single[f]["map5"]:
            best007[f] = {"kind": "stacked", **best2["exp007s"][f]}
            oof007[f] = oof2["exp007s"][f]
        else:
            best007[f] = {"kind": "single", **{k: best_single[f][k] for k in ("map5", "tag", "rounds")}}
            oof007[f] = oof_single[f]
    res["best007"] = best007
    log("FINAL exp007:", json.dumps(best007))
    res["cal007"] = {f: platt(oof007[f][dev.fam == f], dev.y[dev.fam == f]) for f in FAMS}
    res["cal008"] = {f: platt(oof2["exp008"][f][dev.fam == f], dev.y[dev.fam == f]) for f in FAMS}

    # ---- DT decay recipe sanity (forensics p*exp(-K/2) on stage-1): chronology without a stage-2 model
    s1dt = s1_oof["directed_transfer"]
    K = F.select(pl.Series("s", s1dt)).with_columns(pair_id=F["pair_id"]).with_columns(
        K=c("s").cum_sum().over("pair_id") - c("s"))["K"].to_numpy()
    res["rule_dt_decay_K2"] = dev.map5(s1dt * np.exp(-K / 2), "directed_transfer")
    log("sanity DT stage-1 p*exp(-K/2) (forensics 0.554):", round(res["rule_dt_decay_K2"], 4))

    # ---- dev score frames (OOF, calibrated) for all 372 positive pairs
    for exp, oof in (("exp007", oof007), ("exp008", oof2["exp008"])):
        cal = res[f"cal{exp[3:]}"]
        S = {f: sigmoid(cal[f][0] * oof[f] + cal[f][1]) for f in FAMS}
        write_scores(F.select("pair_id", "hand_idx", "pot_bb"), S, C.DER / f"evidence_scores_dev_{exp}.parquet",
                     keep_all=True)
    np.save(CACHE / "dev_s1_oof.npy", np.stack([s1_oof[f] for f in FAMS], axis=1))
    res["runtime_cv_s"] = round(time.time() - t_start, 1)
    (CACHE / "cv_results.json").write_text(json.dumps(res, indent=1))
    log(f"cv done in {res['runtime_cv_s']}s")
    return res


def write_scores(keys, S, path, keep_all=False, topk=TOPK_KEEP):
    """keys: pair_id, hand_idx, pot_bb aligned with S arrays. Adds a 1e-7 pot-size tie-break (never order/ID)."""
    df = keys.with_columns([pl.Series(f"s_{f}", S[f].astype(np.float64)) for f in FAMS])
    df = df.with_columns(tbk=(c("pot_bb").rank("average").over("pair_id") / pl.len().over("pair_id")) * 1e-7)
    df = df.with_columns([(c(f"s_{f}") + c("tbk")).alias(f"s_{f}") for f in FAMS])
    if not keep_all:
        rk = [c(f"s_{f}").rank("ordinal", descending=True).over("pair_id") for f in FAMS]
        df = df.filter(pl.min_horizontal(rk) <= topk)
    hid = C.load("hands").select("hand_idx", "hand_id")
    out = df.join(hid, on="hand_idx", how="left").select("pair_id", "hand_id", "hand_idx", *[f"s_{f}" for f in FAMS])
    if path is not None:
        out.write_parquet(path)
    return out


# ================================================================================================ eval
def run_eval(res=None, tables_per_batch=20):
    t_start = time.time()
    res = res or json.loads((CACHE / "cv_results.json").read_text())
    stamp = json.loads((CACHE / "ph_stamp.json").read_text())
    now = {f.name: f.stat().st_mtime for f in sorted((PH / "phase1").glob("*.parquet"))}
    now.update({"phase0/" + f.name: f.stat().st_mtime for f in sorted((PH / "phase0").glob("*.parquet"))})
    now["_dir"] = PH.name
    assert now == stamp, "engine output changed since `cache` -- rerun: python src/evidence.py all"
    F = load_dev()
    dev = Dev(F)
    feats = res["features_007"]
    X = F.select([c(k).cast(pl.Float32) for k in feats]).to_numpy()
    best007, best008 = res["best007"], res["best008"]
    s1_oof_arr = np.load(CACHE / "dev_s1_oof.npy")
    s1_oof = {f: s1_oof_arr[:, i] for i, f in enumerate(FAMS)}
    allrows = np.ones(len(X), dtype=bool)

    # full-dev fits
    models007, models_s1, models008 = {}, {}, {}
    pooled_cache = {}

    def get(tag, r, f, Xm):
        obj = tag.split("_", 1)[1]
        pooled = tag.startswith("pooled")
        if pooled:
            key = (tag, r)
            if key not in pooled_cache:
                pooled_cache[key] = fit(dev, Xm, allrows, obj, r, True)
            return pooled_cache[key]
        return fit(dev, Xm, allrows & (dev.fam == f), obj, r, False)

    bs = res["best007_single"]
    for f in FAMS:
        models_s1[f] = (get(bs[f]["stage1_tag"], bs[f]["stage1_rounds"], f, X), bs[f]["stage1_tag"])
        b = best007[f]
        if b["kind"] == "single":
            models007[f] = (get(b["tag"], b["rounds"], f, X), b["tag"], "single")
        else:   # orderless stacked: stage-2 trained on dev rows with OOF stage-1 scores
            Cf = chrono_features(F, s1_oof[f], f, order=False)
            assert list(Cf.columns) == res["features_007s_stage2"]
            X2 = np.hstack([X, Cf.to_numpy().astype(np.float32)])
            models007[f] = (fit(dev, X2, dev.fam == f, b["tag"].split("_", 1)[1], b["rounds"], False), b["tag"], "stacked")
        Cf = chrono_features(F, s1_oof[f], f)
        assert list(Cf.columns) == res["features_008_chrono"]
        X2 = np.hstack([X, Cf.to_numpy().astype(np.float32)])
        b8 = best008[f]
        models008[f] = (fit(dev, X2, allrows & (dev.fam == f), b8["tag"].split("_", 1)[1], b8["rounds"], False), b8["tag"])
    log(f"eval: full-dev models fitted ({time.time() - t_start:.0f}s)")

    ep = C.load("eval_pairs").select("pair_id", c("p1").alias("pA"), c("p2").alias("pB"), c("shared_hands"))
    files = sorted((PH / "phase1").glob("*.parquet"))
    outs = {"exp007": [], "exp008": []}
    n_rows = 0
    n_pairs_seen = 0
    for tb in range(0, 400, tables_per_batch):
        t = time.time()
        R = (pl.scan_parquet(files).filter(c("table_idx").is_between(tb, tb + tables_per_batch - 1))
             .join(ep.lazy(), on=["pA", "pB"]).collect().sort("pair_id", "hand_seq"))
        chk = R.group_by("pair_id").agg(pl.len(), c("shared_hands").first())
        assert (chk["len"] == chk["shared_hands"]).all(), "shared-hand count mismatch vs eval_pairs"
        n_pairs_seen += chk.height
        E = hand_features(R.drop("shared_hands"))
        del R
        XE = E.select([c(k).cast(pl.Float32) for k in feats]).to_numpy()
        keys = E.select("pair_id", "hand_idx", "pot_bb")
        S7, S8 = {}, {}
        for f in FAMS:
            m1, tag1 = models_s1[f]
            s1 = sigmoid(predict(m1, XE, "binary", tag1.startswith("pooled"), f))
            m, tag, kind = models007[f]
            a, b_ = res["cal007"][f]
            if kind == "single":
                S7[f] = sigmoid(a * predict(m, XE, tag.split("_", 1)[1], tag.startswith("pooled"), f) + b_)
            else:
                CE = chrono_features(E, s1, f, order=False).to_numpy().astype(np.float32)
                S7[f] = sigmoid(a * predict(m, np.hstack([XE, CE]), tag.split("_", 1)[1], False, f) + b_)
            CE = chrono_features(E, s1, f).to_numpy().astype(np.float32)
            m8, tag8 = models008[f]
            a, b_ = res["cal008"][f]
            S8[f] = sigmoid(a * predict(m8, np.hstack([XE, CE]), tag8.split("_", 1)[1], False, f) + b_)
            del CE
        outs["exp007"].append(write_scores(keys, S7, None))
        outs["exp008"].append(write_scores(keys, S8, None))
        n_rows += len(XE)
        del E, XE
        log(f"eval tables {tb}-{tb + tables_per_batch - 1}: {len(keys):,} rows, {time.time() - t:.0f}s")
    assert n_pairs_seen == ep.height, (n_pairs_seen, ep.height)
    for exp in outs:
        out = pl.concat(outs[exp]).sort("pair_id", "hand_idx")
        out.write_parquet(C.DER / f"evidence_scores_eval_{exp}.parquet")
        log(f"wrote evidence_scores_eval_{exp}.parquet: {out.height:,} rows, {out['pair_id'].n_unique():,} pairs")
    res["runtime_eval_s"] = round(time.time() - t_start, 1)
    res["eval_rows_scored"] = n_rows
    (CACHE / "cv_results.json").write_text(json.dumps(res, indent=1))
    log(f"eval done in {res['runtime_eval_s']}s ({n_rows:,} hand rows)")


# ================================================================================================ compose
def compose_evidence(scores_df, posteriors_df, k=5):
    """Mixture evidence: score(h) = sum_f P(f|pair) * s_f(h); top-k distinct hands per pair.
    scores_df: pair_id, hand_id, s_directed_transfer, s_soft_play, s_coordinated_isolation (polars or pandas).
    posteriors_df: pair_id, P_directed_transfer, P_soft_play, P_coordinated_isolation (missing pairs -> uniform).
    Returns pandas DataFrame[pair_id, evidence_hand_1..5] for every pair in posteriors_df (plus any pair only in
    scores_df), padded with NO_EVIDENCE only when a pair has fewer than k candidate hands."""
    S = scores_df if isinstance(scores_df, pl.DataFrame) else pl.from_pandas(scores_df)
    P = posteriors_df if isinstance(posteriors_df, pl.DataFrame) else pl.from_pandas(posteriors_df)
    pcols = [f"P_{f}" for f in FAMS]
    P = P.select("pair_id", *pcols).with_columns([c(p).cast(pl.Float64) for p in pcols])
    tot = pl.sum_horizontal(pcols)
    P = P.with_columns([pl.when(tot > 0).then(c(p) / tot).otherwise(1.0 / len(FAMS)).alias(p) for p in pcols])
    M = (S.select("pair_id", "hand_id", *[f"s_{f}" for f in FAMS]).unique(["pair_id", "hand_id"])
         .join(P, on="pair_id", how="left")
         .with_columns([c(p).fill_null(1.0 / len(FAMS)) for p in pcols])
         .with_columns(mix=pl.sum_horizontal([c(f"P_{f}") * c(f"s_{f}") for f in FAMS]),
                       smax=pl.max_horizontal([c(f"s_{f}") for f in FAMS]))
         .sort(["pair_id", "mix", "smax", "hand_id"], descending=[False, True, True, False])
         .with_columns(r=pl.int_range(pl.len()).over("pair_id"))
         .filter(c("r") < k))
    wide = (M.with_columns(col=pl.format("evidence_hand_{}", c("r") + 1))
            .pivot(on="col", index="pair_id", values="hand_id"))
    ids = pl.concat([P.select("pair_id"), S.select("pair_id")]).unique()
    wide = ids.join(wide, on="pair_id", how="left")
    for e in C.EV_COLS:
        if e not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.String).alias(e))
    wide = wide.with_columns([c(e).fill_null(C.NO_EV) for e in C.EV_COLS]).select("pair_id", *C.EV_COLS)
    return wide.sort("pair_id").to_pandas()


def run_compose_test():
    lab = C.load("labels").filter(c("label") == 1).select("pair_id", c("behavior_family").alias("family"))
    out = {}
    sol = C.dev_solution(lab["pair_id"].to_list())
    for exp in ("exp007", "exp008"):
        S = pl.read_parquet(C.DER / f"evidence_scores_dev_{exp}.parquet")
        onehot_P = lab.with_columns([(c("family") == f).cast(pl.Float64).alias(f"P_{f}") for f in FAMS])
        unif_P = lab.with_columns([pl.lit(1.0 / 3).alias(f"P_{f}") for f in FAMS])
        for name, P in (("true_family", onehot_P), ("uniform", unif_P)):
            ev = compose_evidence(S, P)
            sub = sol[["pair_id", "risk_score", "predicted_behavior"]].merge(ev, on="pair_id")
            parts = C.score_parts(sol, sub)
            # per-family breakdown with the same AP@5 definition
            rel = sol.set_index("pair_id")[C.EV_COLS]
            per = {}
            fam_of = dict(zip(lab["pair_id"].to_list(), lab["family"].to_list()))
            evi = ev.set_index("pair_id")
            for f in FAMS:
                aps = []
                for p in [q for q in sol.pair_id if fam_of[q] == f]:
                    R = {h for h in rel.loc[p] if h != C.NO_EV}
                    hits, ps = 0, 0.0
                    for kk, h in enumerate([h for h in evi.loc[p] if h != C.NO_EV][:5], 1):
                        if h in R:
                            hits += 1
                            ps += hits / kk
                    aps.append(ps / min(len(R), 5))
                per[f] = round(float(np.mean(aps)), 4)
            # also top-15-union restriction (as in the eval file) vs all hands
            S15 = write_scores(S.with_columns(pot_bb=pl.lit(0.0)).select("pair_id", "hand_idx", "pot_bb"),
                               {f: S[f"s_{f}"].to_numpy() for f in FAMS}, None)
            ev15 = compose_evidence(S15, P)
            parts15 = C.score_parts(sol, sol[["pair_id", "risk_score", "predicted_behavior"]].merge(ev15, on="pair_id"))
            out[f"{exp}_{name}"] = {"evidence_map5_host": parts["evidence_map5"], "per_family": per,
                                    "evidence_map5_top15_union": parts15["evidence_map5"]}
            log(f"compose {exp} {name}: host MAP@5 {parts['evidence_map5']} (top15-union {parts15['evidence_map5']}) per family {per}")
    # uniform-posterior variant with within-pair percentile normalisation (for the doc)
    for exp in ("exp007", "exp008"):
        S = pl.read_parquet(C.DER / f"evidence_scores_dev_{exp}.parquet").with_columns(
            [(c(f"s_{f}").rank("average").over("pair_id") / pl.len().over("pair_id")).alias(f"s_{f}") for f in FAMS])
        unif_P = lab.with_columns([pl.lit(1.0 / 3).alias(f"P_{f}") for f in FAMS])
        ev = compose_evidence(S, unif_P)
        parts = C.score_parts(sol, sol[["pair_id", "risk_score", "predicted_behavior"]].merge(ev, on="pair_id"))
        out[f"{exp}_uniform_pct_norm"] = {"evidence_map5_host": parts["evidence_map5"]}
        log(f"compose {exp} uniform with within-pair percentile normalisation: {parts['evidence_map5']}")
    res = json.loads((CACHE / "cv_results.json").read_text())
    res["compose"] = out
    (CACHE / "cv_results.json").write_text(json.dumps(res, indent=1))
    return out


if __name__ == "__main__":
    stages = sys.argv[1:] or ["all"]
    if "all" in stages:
        stages = ["cache", "cv", "eval", "compose"]
    for st in stages:
        log(f"=== stage {st}")
        {"cache": cache_dev, "cv": run_cv, "eval": run_eval, "compose": run_compose_test}[st]()
