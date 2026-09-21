"""exp016ev -- EVIDENCE RANKER v2 (sub-type/segment chronology + hand-detector features + routing).

Builds on src/evidence.py (exp007/exp008 stay intact as the fallback). Three v2 levers, each measured separately:

1. SUB-TYPE (SEGMENT) CHRONOLOGY -- Critic addendum C1/C3. A dev evidence list is two chronologically sorted
   segments concatenated (exactly one time inversion: DT 91/91, SP 93/96, CI 11/11). We recover the segment of every
   LISTED hand from that inversion, fit a segment-1-vs-segment-2 hand classifier per family (cross-fit by table fold,
   nested inside the ranker's folds), and give the ranker per-segment chronology: stage-1-weighted earlier mass
   inside the hand's own segment, # earlier same-segment candidates, rank within segment by time and by score.
   Also tested as an explicit two-segment COMPOSITION (n1 hands from segment 1 + 5-n1 from segment 2).
2. HAND-DETECTOR FEATURES -- exp015 per-pair-hand scores (data/derived/hs/phase{0,1}: s_dt/s_sp/s_ci/s_any; dev OOF
   by the same frozen table folds, eval = 5-model mean) plus their within-pair percentile / top-rank.
3. ROUTING -- real behaviour posteriors (oof/exp016_full_oof_behavior_pos.parquet dev, exp016_full_eval_scores eval),
   sharpening (temperature / argmax one-hot) and the exp013 "fam_unexplained -> SP first" rule.

Stages:  python src/evidence_v2.py [cv|report|eval|all]
Outputs: data/derived/evidence_scores_{dev,eval}_exp016ev.parquet (schema as exp007/exp008),
         data/derived/evidence_cache/cv_results_v2.json (grids, selections, routing + composition tables).
"""
import json
import os
import sys
import time

os.environ.setdefault("POLARS_MAX_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import polars as pl
import lightgbm as lgb

import common as C
import evidence as EV
from evidence import FAMS, CACHE, PH, chrono_features, fit, predict, platt, sigmoid, log

c = pl.col
HS = C.DER / "hs"
HS_RAW = ["s_dt", "s_sp", "s_ci", "s_any"]
HS_COLS = HS_RAW + [f"{p}_{k}" for k in HS_RAW for p in ("pct", "top")]
GRID = [100, 200, 350, 500, 750]
VARIANTS = {"A_base": (False, False), "B_hs": (True, False), "C_seg": (False, True), "D_hs_seg": (True, True)}
OBJS = {"A_base": ("binary", "lambdarank"), "B_hs": ("binary",), "C_seg": ("binary",),
        "D_hs_seg": ("binary", "lambdarank")}
SEGP = dict(objective="binary", learning_rate=0.05, num_leaves=7, min_data_in_leaf=15, feature_fraction=0.5,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=EV.NTHR, verbose=-1, seed=11)
SEG_ROUNDS = 120


# ============================================================================================ data
def hs_join(df, phase):
    """attach the exp015 per-pair-hand detector scores (keys hand_idx, pA, pB)."""
    keys = df.select("pA", "pB").unique()
    hs = (pl.scan_parquet(HS / f"phase{phase}" / "*.parquet")
          .filter(c("table_idx").is_in(df["table_idx"].unique()))
          .select("hand_idx", "pA", "pB", *HS_RAW)
          .join(keys.lazy(), on=["pA", "pB"], how="semi").collect())
    out = df.join(hs, on=["hand_idx", "pA", "pB"], how="left")
    assert out.height == df.height and out["s_dt"].null_count() == 0, "hs join failed"
    return out


def add_hs_feats(F):
    ex = []
    for k in HS_RAW:
        ex += [(c(k).rank("average").over("pair_id") / c("n_shared")).alias("pct_" + k),
               (c(k).rank("min", descending=True).over("pair_id") - 1).cast(pl.Float32).alias("top_" + k)]
    return F.with_columns(ex)


def segment_labels():
    """Critic C1: an evidence list is two chronologically sorted segments -> exactly one time inversion.
    Returns (pair_id, hand_idx, seg in {1,2}) for pairs with exactly one inversion, plus per-family counts."""
    ev = C.load("evidence").select("pair_id", "evidence_rank", "hand_idx", "behavior_family")
    hd = C.load("hands").select("hand_idx", "hand_seq")
    e = (ev.join(hd, on="hand_idx").sort("pair_id", "evidence_rank")
         .with_columns(prev=c("hand_seq").shift(1).over("pair_id"))
         .with_columns(inv=((c("hand_seq") < c("prev")).fill_null(False)).cast(pl.Int32))
         .with_columns(n_inv=c("inv").sum().over("pair_id"))
         .with_columns(seg=c("inv").cum_sum().over("pair_id") + 1))
    stats = e.group_by("behavior_family").agg(pairs=c("pair_id").n_unique(),
                                              one_inv=c("pair_id").filter(c("n_inv") == 1).n_unique(),
                                              zero_inv=c("pair_id").filter(c("n_inv") == 0).n_unique(),
                                              multi_inv=c("pair_id").filter(c("n_inv") > 1).n_unique()).sort("behavior_family")
    return e.filter(c("n_inv") == 1).select("pair_id", "hand_idx", "seg"), stats


def build_dev():
    D = pl.read_parquet(CACHE / "dev_pos.parquet")
    D = hs_join(D, 0)
    seg, stats = segment_labels()
    D = D.join(seg, on=["pair_id", "hand_idx"], how="left").sort("pair_id", "hand_seq")
    return add_hs_feats(EV.hand_features(D)), stats


# ============================================================================================ features
def seg_features(F, s1, p2):
    """per-SUB-TYPE chronology: how much evidence-like mass of the hand's OWN segment came earlier."""
    G = F.select("pair_id", "n_shared").with_columns(s1=pl.Series(s1.astype(np.float32)),
                                                     p2=pl.Series(p2.astype(np.float32)))
    G = G.with_columns(hard=(c("p2") > 0.5).cast(pl.Int8), w2=c("s1") * c("p2"), w1=c("s1") * (1 - c("p2")))
    pr2 = c("w2").cum_sum().over("pair_id") - c("w2")
    pr1 = c("w1").cum_sum().over("pair_id") - c("w1")
    n2, n1 = c("w2").sum().over("pair_id"), c("w1").sum().over("pair_id")
    same_prior = c("p2") * pr2 + (1 - c("p2")) * pr1
    same_n = c("p2") * n2 + (1 - c("p2")) * n1
    hi = (c("s1") > 0.25).cast(pl.Float32)
    ex = [c("p2").alias("seg_p2"), c("hard").cast(pl.Float32).alias("seg_hard"),
          pr2.alias("seg_prior_w2"), pr1.alias("seg_prior_w1"), n2.alias("seg_n_w2"), n1.alias("seg_n_w1"),
          same_prior.alias("seg_prior_same"), same_n.alias("seg_n_same"),
          (same_prior / (same_n + 1e-6)).alias("seg_prel_same"),
          (hi.cum_sum().over(["pair_id", "hard"]) - hi).alias("seg_prior_hi_same"),
          hi.sum().over(["pair_id", "hard"]).alias("seg_n_hi_same"),
          pl.int_range(pl.len()).over(["pair_id", "hard"]).cast(pl.Float32).alias("seg_time_rank"),
          (c("s1").rank("min", descending=True).over(["pair_id", "hard"]) - 1).cast(pl.Float32).alias("seg_score_rank"),
          pl.len().over(["pair_id", "hard"]).cast(pl.Float32).alias("seg_size"),
          (c("s1").cum_sum().over(["pair_id", "hard"]) - c("s1")).alias("seg_K_hard")]
    return G.with_columns(ex).select([e.meta.output_name() for e in ex])


def fit_seg(X, y, mask):
    return lgb.train(SEGP, lgb.Dataset(X[mask], y[mask], free_raw_data=True), num_boost_round=SEG_ROUNDS)


def stage2_matrix(F, X, s1, p2, fam, use_seg):
    blocks = [X]
    cf = chrono_features(F, s1, fam)
    blocks.append(cf.to_numpy().astype(np.float32))
    names = list(cf.columns)
    if use_seg:
        sf = seg_features(F, s1, p2)
        blocks.append(sf.to_numpy().astype(np.float32))
        names += list(sf.columns)
    return np.hstack(blocks), names


# ============================================================================================ routing / metrics
def mix_scores(S, post):
    return sum(post[:, i] * S[f] for i, f in enumerate(FAMS))


def true_family_mix(dev, S):
    mix = np.zeros(len(dev.y))
    for f in FAMS:
        m = dev.fam == f
        mix[m] = S[f][m]
    return mix


def map5_of(dev, score, per_family=True):
    m = EV.map5_frame(dev.pair, dev.fam, dev.y, dev.n_rel, score, dev.tb)
    out = {"all": round(float(m["ap"].mean()), 4)}
    if per_family:
        for f in FAMS:
            out[EV.SHORT[f]] = round(float(m.filter(c("family") == f)["ap"].mean()), 4)
    return out


def dev_posteriors(pairs, temp=1.0, argmax=False, sp_first_unexplained=False):
    b = pl.read_parquet("oof/exp016_full_oof_behavior_pos.parquet")
    P = (pl.DataFrame({"pair_id": pairs}).join(b, on="pair_id", how="left")
         .select([c(f"P_{f}").fill_null(1 / 3) for f in FAMS]).to_numpy())
    if sp_first_unexplained:
        oc = pl.read_parquet(C.DER / "oc_pair_scores_dev_full.parquet", columns=["pA", "pB", "fam_unexplained"])
        lab = C.load("labels").select("pair_id", c("p1").alias("pA"), c("p2").alias("pB"))
        u = set(lab.join(oc, on=["pA", "pB"]).filter(c("fam_unexplained"))["pair_id"].to_list())
        P = P.copy()
        P[np.array([p in u for p in pairs])] = np.array([0.0, 1.0, 0.0])
    if argmax:
        Q = np.zeros_like(P)
        Q[np.arange(len(P)), P.argmax(1)] = 1.0
        return Q
    P = P ** temp
    return P / P.sum(1, keepdims=True)


def _rankpct(pairs, s):
    return (pl.DataFrame({"p": pairs, "s": s})
            .with_columns(r=c("s").rank("average").over("p") / pl.len().over("p"))["r"].to_numpy())


# ============================================================================================ CV
def run_cv():
    t0 = time.time()
    F, segstats = build_dev()
    dev = EV.Dev(F)
    res0 = json.loads((CACHE / "cv_results.json").read_text())
    base_feats = [k for k in EV.base_feature_names(F) if k not in HS_COLS and k != "seg"]
    X0 = F.select([c(k).cast(pl.Float32) for k in base_feats]).to_numpy()
    keep = np.nanstd(X0, axis=0) > 0
    base_feats = [k for k, kk in zip(base_feats, keep) if kk]
    Xb = X0[:, keep]
    Xh = np.hstack([Xb, F.select([c(k).cast(pl.Float32) for k in HS_COLS]).to_numpy()])
    XS = {"b": Xb, "h": Xh}
    seg_lab = F["seg"].is_not_null().to_numpy()
    seg_t = (F["seg"].fill_null(1).to_numpy().astype(np.float64) - 1.0)
    res = {"ph_dir": PH.name, "features_base": base_feats, "features_hs": HS_COLS,
           "segment_label_stats": segstats.to_dicts(), "seg_labeled_hands": int(seg_lab.sum())}
    log(f"dev rows {len(Xb):,}; base feats {len(base_feats)}, +hs {Xh.shape[1]}; "
        f"segment-labelled evidence hands {int(seg_lab.sum())}")
    log("segment labels:", json.dumps(segstats.to_dicts()))

    s1cfg = {f: (res0["best007_single"][f]["stage1_tag"], res0["best007_single"][f]["stage1_rounds"]) for f in FAMS}

    def fit_s1(Xm, trm, f):
        tag, r = s1cfg[f]
        pooled = tag.startswith("pooled")
        return fit(dev, Xm, trm if pooled else (trm & (dev.fam == f)), "binary", r, pooled), pooled

    # ---- outer OOF stage-1 + segment probabilities (models trained on folds != k)
    s1_oof = {s: {f: np.zeros(len(dev.y)) for f in FAMS} for s in XS}
    p2_oof = {s: {f: np.zeros(len(dev.y)) for f in FAMS} for s in XS}
    for k in range(5):
        va = dev.fold == k
        for s, Xm in XS.items():
            for f in FAMS:
                m, pooled = fit_s1(Xm, dev.fold != k, f)
                s1_oof[s][f][va] = sigmoid(predict(m, Xm[va], "binary", pooled, f))
                p2_oof[s][f][va] = fit_seg(Xm, seg_t, (dev.fold != k) & seg_lab & (dev.fam == f)).predict(Xm[va])
    log(f"outer stage-1 + segment OOF done ({time.time() - t0:.0f}s)")
    res["stage1_oof_map5"] = {EV.SHORT[f]: round(dev.map5(s1_oof["b"][f], f), 4) for f in FAMS}
    res["stage1_hs_oof_map5"] = {EV.SHORT[f]: round(dev.map5(s1_oof["h"][f], f), 4) for f in FAMS}
    log("stage-1 (no chronology) OOF MAP5  base:", json.dumps(res["stage1_oof_map5"]),
        " +hs:", json.dumps(res["stage1_hs_oof_map5"]))
    from sklearn.metrics import roc_auc_score
    res["segment_clf_oof_auc"] = {}
    for f in FAMS:
        m = seg_lab & (dev.fam == f)
        if m.sum() > 20 and len(np.unique(seg_t[m])) == 2:
            res["segment_clf_oof_auc"][EV.SHORT[f]] = {
                "n": int(m.sum()), "auc_base": round(float(roc_auc_score(seg_t[m], p2_oof["b"][f][m])), 4),
                "auc_hs": round(float(roc_auc_score(seg_t[m], p2_oof["h"][f][m])), 4)}
    log("segment classifier OOF AUC (seg2 vs seg1 on listed hands):", json.dumps(res["segment_clf_oof_auc"]))

    # ---- nested stage-2 grid
    g2 = {v: {o: {f: {r: np.zeros(len(dev.y)) for r in GRID} for f in FAMS} for o in OBJS[v]} for v in VARIANTS}
    names = {}
    for k in range(5):
        tk = time.time()
        va = dev.fold == k
        s1 = {s: {f: s1_oof[s][f].copy() for f in FAMS} for s in XS}
        p2 = {s: {f: p2_oof[s][f].copy() for f in FAMS} for s in XS}
        for j in range(5):                      # inner cross-fit: never touches fold k
            if j == k:
                continue
            vj = dev.fold == j
            trm = (dev.fold != k) & (dev.fold != j)
            for s, Xm in XS.items():
                for f in FAMS:
                    m, pooled = fit_s1(Xm, trm, f)
                    s1[s][f][vj] = sigmoid(predict(m, Xm[vj], "binary", pooled, f))
                    p2[s][f][vj] = fit_seg(Xm, seg_t, trm & seg_lab & (dev.fam == f)).predict(Xm[vj])
        for v, (use_hs, use_seg) in VARIANTS.items():
            s = "h" if use_hs else "b"
            for f in FAMS:
                X2, nm = stage2_matrix(F, XS[s], s1[s][f], p2[s][f], f, use_seg)
                names[v] = list(base_feats) + (HS_COLS if use_hs else []) + nm
                trm = (dev.fold != k) & (dev.fam == f)
                for o in OBJS[v]:
                    mm = fit(dev, X2, trm, o, max(GRID), False)
                    for r in GRID:
                        g2[v][o][f][r][va] = predict(mm, X2[va], o, False, f, r)
        log(f"v2 nested outer fold {k} done ({time.time() - tk:.0f}s; total {time.time() - t0:.0f}s)")
    res["n_features"] = {v: len(names[v]) for v in names}

    # ---- MAP tables (true-family routing) + selection
    tab = {}
    for v in VARIANTS:
        for o in OBJS[v]:
            tab[f"{v}|{o}"] = {EV.SHORT[f]: {r: round(dev.map5(g2[v][o][f][r], f), 4) for r in GRID} for f in FAMS}
            log(f"{v}|{o}:", json.dumps(tab[f"{v}|{o}"]))
    res["grid"] = tab
    best = {}
    for f in FAMS:
        sc, v, o, r = max((tab[f"{v}|{o}"][EV.SHORT[f]][r], v, o, r) for v in VARIANTS for o in OBJS[v] for r in GRID)
        best[f] = {"map5": sc, "variant": v, "obj": o, "rounds": r}
    res["best_v2"] = best
    log("best v2 per family (true-family routing):", json.dumps(best))

    # per-variant best-rounds OOF (calibrated) for the routing report
    store = {}
    for v in VARIANTS:
        for o in OBJS[v]:
            S = {}
            for f in FAMS:
                r = max(GRID, key=lambda rr: tab[f"{v}|{o}"][EV.SHORT[f]][rr])
                a, b_ = platt(g2[v][o][f][r][dev.fam == f], dev.y[dev.fam == f])
                S[f] = sigmoid(a * g2[v][o][f][r] + b_)
                store[f"{v}|{o}|{f}"] = S[f]
    # rank-average fusion (binary + lambdarank) where both exist
    for v in ("A_base", "D_hs_seg"):
        ra = {}
        for f in FAMS:
            rb = max(GRID, key=lambda r: tab[f"{v}|binary"][EV.SHORT[f]][r])
            rl = max(GRID, key=lambda r: tab[f"{v}|lambdarank"][EV.SHORT[f]][r])
            s = (_rankpct(dev.pair, g2[v]["binary"][f][rb]) + _rankpct(dev.pair, g2[v]["lambdarank"][f][rl])) / 2
            ra[EV.SHORT[f]] = round(dev.map5(s, f), 4)
            store[f"{v}|rankavg|{f}"] = s
        res[f"rankavg_{v}"] = ra
        log(f"rank-average {v} (binary + lambdarank):", json.dumps(ra))

    # ---- chosen configuration -> dev OOF score file
    oof = {f: g2[best[f]["variant"]][best[f]["obj"]][f][best[f]["rounds"]] for f in FAMS}
    cal = {f: platt(oof[f][dev.fam == f], dev.y[dev.fam == f]) for f in FAMS}
    res["cal"] = cal
    S = {f: sigmoid(cal[f][0] * oof[f] + cal[f][1]) for f in FAMS}
    EV.write_scores(F.select("pair_id", "hand_idx", "pot_bb"), S,
                    C.DER / "evidence_scores_dev_exp016ev.parquet", keep_all=True)
    np.savez(CACHE / "v2_oof_store.npz", **store, **{f"chosen|{f}": S[f] for f in FAMS},
             **{f"s1|{s}|{f}": s1_oof[s][f] for s in XS for f in FAMS},
             **{f"p2|{s}|{f}": p2_oof[s][f] for s in XS for f in FAMS})
    F.select("pair_id", "hand_idx", "hand_seq", "table_idx", "fold", "family", "is_ev", "n_rel", "pot_bb",
             "seg").write_parquet(CACHE / "v2_dev_keys.parquet")
    res["runtime_cv_s"] = round(time.time() - t0, 1)
    (CACHE / "cv_results_v2.json").write_text(json.dumps(res, indent=1))
    log(f"v2 cv done in {res['runtime_cv_s']}s")
    return res


# ============================================================================================ report
class Rows:
    """minimal row bundle for the metric helpers (subset-able)."""
    def __init__(self, pair, fam, y, n_rel, tb):
        self.pair, self.fam, self.y, self.n_rel, self.tb = pair, fam, y, n_rel, tb

    def sub(self, m):
        return Rows(self.pair[m], self.fam[m], self.y[m], self.n_rel[m], self.tb[m])


def _dev_frame():
    K = pl.read_parquet(CACHE / "v2_dev_keys.parquet")
    return K, Rows(K["pair_id"].to_numpy(), K["family"].to_numpy(), K["is_ev"].to_numpy().astype(np.int32),
                   K["n_rel"].to_numpy(), K["pot_bb"].to_numpy())


def composition_map5(dev, score, p2, n1, order="score"):
    """explicit two-segment composition: n1 hands from segment 1 + (5-n1) from segment 2."""
    t = pl.DataFrame({"pair_id": dev.pair, "family": dev.fam, "is_ev": dev.y, "n_rel": dev.n_rel,
                      "s": score, "tb": dev.tb, "seq": np.arange(len(score)), "hard": (p2 > 0.5).astype(np.int8)})
    t = (t.sort(["pair_id", "hard", "s", "tb"], descending=[False, False, True, True])
         .with_columns(r=pl.int_range(pl.len()).over(["pair_id", "hard"]) + 1))
    t = t.filter(((c("hard") == 0) & (c("r") <= n1)) | ((c("hard") == 1) & (c("r") <= 5 - n1)))
    if order == "score":
        t = t.sort(["pair_id", "s", "tb"], descending=[False, True, True])
    else:                                        # segment 1 (chronological) then segment 2 (chronological)
        t = t.sort(["pair_id", "hard", "seq"])
    t = (t.with_columns(k=pl.int_range(pl.len()).over("pair_id") + 1).filter(c("k") <= 5)
         .with_columns(hits=c("is_ev").cast(pl.Int32).cum_sum().over("pair_id")))
    m = t.group_by("pair_id", "family").agg(num=((c("hits") / c("k")) * c("is_ev")).sum(),
                                            n_rel=c("n_rel").first())
    m = m.with_columns(ap=c("num") / c("n_rel").clip(upper_bound=5))
    out = {"all": round(float(m["ap"].mean()), 4)}
    for f in FAMS:
        v = m.filter(c("family") == f)["ap"].mean()
        if v is not None:
            out[EV.SHORT[f]] = round(float(v), 4)
    return out


def run_report():
    t0 = time.time()
    res = json.loads((CACHE / "cv_results_v2.json").read_text())
    K, dev = _dev_frame()
    z = np.load(CACHE / "v2_oof_store.npz")
    rep = {}

    # exp008 (fallback) scores on the same rows
    e8 = (K.select("pair_id", "hand_idx")
          .join(pl.read_parquet(C.DER / "evidence_scores_dev_exp008.parquet"), on=["pair_id", "hand_idx"], how="left"))
    S8 = {f: e8[f"s_{f}"].to_numpy() for f in FAMS}
    cand = {"exp008": S8, "exp016ev_chosen": {f: z[f"chosen|{f}"] for f in FAMS}}
    for v in VARIANTS:
        for o in list(OBJS[v]) + (["rankavg"] if v in ("A_base", "D_hs_seg") else []):
            key = f"{v}|{o}"
            if f"{key}|{FAMS[0]}" in z:
                cand[key] = {f: z[f"{key}|{f}"] for f in FAMS}

    P1 = dev_posteriors(dev.pair)
    routing = {"true_family": None, "posterior_T1": P1, "posterior_T2": dev_posteriors(dev.pair, temp=2.0),
               "posterior_T4": dev_posteriors(dev.pair, temp=4.0),
               "posterior_argmax": dev_posteriors(dev.pair, argmax=True),
               "posterior_T1_spfirst_unexpl": dev_posteriors(dev.pair, sp_first_unexplained=True),
               "uniform": np.full((len(dev.y), 3), 1 / 3)}
    table = {}
    for name, S in cand.items():
        row = {}
        for rname, post in routing.items():
            mix = true_family_mix(dev, S) if post is None else mix_scores(S, post)
            row[rname] = map5_of(dev, mix)
        table[name] = row
        log(f"{name}: true-family {row['true_family']} | posterior {row['posterior_T1']} | "
            f"argmax {row['posterior_argmax']['all']} | uniform {row['uniform']['all']}")
    rep["routing_table"] = table

    # ---- two-segment composition on the chosen scores (true-family routing)
    best = res["best_v2"]
    flat = map5_of(dev, true_family_mix(dev, {g: z[f"chosen|{g}"] for g in FAMS}))
    comp = {}
    for f in FAMS:
        s = "h" if VARIANTS[best[f]["variant"]][0] else "b"
        m = dev.fam == f
        sub, sc, p2 = dev.sub(m), z[f"chosen|{f}"][m], z[f"p2|{s}|{f}"][m]
        comp[EV.SHORT[f]] = {"flat": flat[EV.SHORT[f]],
                             "seg2_share": round(float((p2 > 0.5).mean()), 4)}
        for n1 in range(6):
            for order in ("score", "segtime"):
                comp[EV.SHORT[f]][f"n1={n1}|{order}"] = composition_map5(sub, sc, p2, n1, order)[EV.SHORT[f]]
    rep["two_segment_composition"] = comp
    log("two-segment composition (true-family):", json.dumps(comp))
    res["report"] = rep
    res["runtime_report_s"] = round(time.time() - t0, 1)
    (CACHE / "cv_results_v2.json").write_text(json.dumps(res, indent=1))
    return rep


# ============================================================================================ eval
def run_eval(tables_per_batch=20):
    t0 = time.time()
    res = json.loads((CACHE / "cv_results_v2.json").read_text())
    res0 = json.loads((CACHE / "cv_results.json").read_text())
    F, _ = build_dev()
    dev = EV.Dev(F)
    base_feats = res["features_base"]
    Xb = F.select([c(k).cast(pl.Float32) for k in base_feats]).to_numpy()
    Xh = np.hstack([Xb, F.select([c(k).cast(pl.Float32) for k in HS_COLS]).to_numpy()])
    XS = {"b": Xb, "h": Xh}
    seg_lab = F["seg"].is_not_null().to_numpy()
    seg_t = (F["seg"].fill_null(1).to_numpy().astype(np.float64) - 1.0)
    z = np.load(CACHE / "v2_oof_store.npz")
    best, cal = res["best_v2"], res["cal"]
    s1cfg = {f: (res0["best007_single"][f]["stage1_tag"], res0["best007_single"][f]["stage1_rounds"]) for f in FAMS}
    allrows = np.ones(len(Xb), dtype=bool)

    M = {}
    for f in FAMS:
        v = best[f]["variant"]
        use_hs, use_seg = VARIANTS[v]
        s = "h" if use_hs else "b"
        tag, r = s1cfg[f]
        pooled = tag.startswith("pooled")
        m1 = fit(dev, XS[s], allrows if pooled else (dev.fam == f), "binary", r, pooled)
        mseg = fit_seg(XS[s], seg_t, seg_lab & (dev.fam == f)) if use_seg else None
        X2, _ = stage2_matrix(F, XS[s], z[f"s1|{s}|{f}"], z[f"p2|{s}|{f}"], f, use_seg)
        m2 = fit(dev, X2, dev.fam == f, best[f]["obj"], best[f]["rounds"], False)
        M[f] = dict(s=s, pooled=pooled, m1=m1, mseg=mseg, m2=m2, use_seg=use_seg, obj=best[f]["obj"])
    log(f"eval: full-dev models fitted ({time.time() - t0:.0f}s)")

    ep = C.load("eval_pairs").select("pair_id", c("p1").alias("pA"), c("p2").alias("pB"), "shared_hands")
    files = sorted((PH / "phase1").glob("*.parquet"))
    outs, n_rows, n_pairs = [], 0, 0
    for tb in range(0, 400, tables_per_batch):
        t = time.time()
        R = (pl.scan_parquet(files).filter(c("table_idx").is_between(tb, tb + tables_per_batch - 1))
             .join(ep.lazy(), on=["pA", "pB"]).collect())
        chk = R.group_by("pair_id").agg(pl.len(), c("shared_hands").first())
        assert (chk["len"] == chk["shared_hands"]).all()
        n_pairs += chk.height
        R = hs_join(R.drop("shared_hands"), 1).sort("pair_id", "hand_seq")
        E = add_hs_feats(EV.hand_features(R))
        del R
        XE = {"b": E.select([c(k).cast(pl.Float32) for k in base_feats]).to_numpy()}
        XE["h"] = np.hstack([XE["b"], E.select([c(k).cast(pl.Float32) for k in HS_COLS]).to_numpy()])
        keys = E.select("pair_id", "hand_idx", "pot_bb")
        SE = {}
        for f in FAMS:
            d = M[f]
            Xm = XE[d["s"]]
            s1 = sigmoid(predict(d["m1"], Xm, "binary", d["pooled"], f))
            p2 = d["mseg"].predict(Xm) if d["use_seg"] else np.zeros(len(Xm))
            X2, _ = stage2_matrix(E, Xm, s1, p2, f, d["use_seg"])
            a, b_ = cal[f]
            SE[f] = sigmoid(a * predict(d["m2"], X2, d["obj"], False, f) + b_)
            del X2
        outs.append(EV.write_scores(keys, SE, None))
        n_rows += len(keys)
        del E, XE
        log(f"eval tables {tb}-{tb + tables_per_batch - 1}: {len(keys):,} rows, {time.time() - t:.0f}s")
    assert n_pairs == ep.height, (n_pairs, ep.height)
    out = pl.concat(outs).sort("pair_id", "hand_idx")
    out.write_parquet(C.DER / "evidence_scores_eval_exp016ev.parquet")
    log(f"wrote evidence_scores_eval_exp016ev.parquet: {out.height:,} rows, {out['pair_id'].n_unique():,} pairs")
    res["runtime_eval_s"] = round(time.time() - t0, 1)
    res["eval_rows_scored"] = n_rows
    (CACHE / "cv_results_v2.json").write_text(json.dumps(res, indent=1))
    log(f"v2 eval done in {res['runtime_eval_s']}s ({n_rows:,} rows)")


# ============================================================================================ top-K re-ranker
# Diagnostic that motivates it: 77-78% of listed hands are already in the flat top-5, 96% in the top-10, and
# ORDERING the existing top-5 perfectly would give MAP 0.772 (+0.054). So the remaining loss is list COMPOSITION
# inside the shortlist, which is exactly where the two-segment rule (Critic C1/C3) should apply -- the segment
# classifier is meaningful on evidence-like hands, not on the 100+ junk candidates (why the pool-wide quota failed).
RR_BASE = ["log_pot", "transfer", "board_n", "sd_both", "flow", "dt_fold", "dt_call", "dt_strong", "dt_gift5",
           "dt_pfweak", "sp_foldresp", "sp_callresp", "sp_sc", "sp_pl", "ci_s14", "ci_s15", "face_sum", "gift_don",
           "gift_max_mx", "strongcheck_sum", "max_hueq_fold_post_don", "min_hueq_call_tr_don", "min_pcall_pre_don",
           "log_shared", "q_glob", "s_dt", "s_sp", "s_ci"]


def rr_frame(keys, F, s_flat, p2, K):
    """shortlist frame: top-K hands per pair by the flat score + within-shortlist structure features."""
    T = (F.select("pair_id", "hand_idx", "hand_seq", "n_shared", *[k for k in RR_BASE if k != "q_glob"])
         .with_columns(s=pl.Series(s_flat.astype(np.float64)), p2=pl.Series(p2.astype(np.float64)),
                       y=pl.Series(keys["is_ev"].to_numpy().astype(np.int32)),
                       n_rel=pl.Series(keys["n_rel"].to_numpy()), fam=pl.Series(keys["family"].to_numpy()),
                       fold=pl.Series(keys["fold"].to_numpy()), tb=pl.Series(keys["pot_bb"].to_numpy()))
         .with_columns(q_glob=(c("hand_seq").rank("ordinal").over("pair_id") / c("n_shared")).cast(pl.Float64))
         .with_columns(rk=c("s").rank("ordinal", descending=True).over("pair_id"))
         .filter(c("rk") <= K)
         .with_columns(hard=(c("p2") > 0.5).cast(pl.Int8))
         .sort("pair_id", "hand_seq"))
    T = T.with_columns(
        s_max=c("s").max().over("pair_id"), s_min=c("s").min().over("pair_id"),
        n_seg2=c("hard").sum().over("pair_id"),
        t_rank=pl.int_range(pl.len()).over("pair_id").cast(pl.Float64),                 # time order inside shortlist
        t_rank_seg=pl.int_range(pl.len()).over(["pair_id", "hard"]).cast(pl.Float64),   # time order inside segment
        seg_size=pl.len().over(["pair_id", "hard"]).cast(pl.Float64),
        s_rank_seg=c("s").rank("ordinal", descending=True).over(["pair_id", "hard"]).cast(pl.Float64),
        p2_sum=c("p2").sum().over("pair_id"))
    T = T.with_columns(s_gap=c("s_max") - c("s"), s_rel=c("s") / (c("s_max") + 1e-9),
                       first_in_seg=(c("t_rank_seg") == 0).cast(pl.Int8),
                       s_pct=c("s").rank("average").over("pair_id") / pl.len().over("pair_id"),
                       t_pct=c("t_rank") / pl.len().over("pair_id"))
    return T


RR_FEATS = RR_BASE + ["s", "p2", "rk", "hard", "n_seg2", "t_rank", "t_rank_seg", "seg_size", "s_rank_seg", "p2_sum",
                      "s_gap", "s_rel", "first_in_seg", "s_pct", "t_pct"]
RRP = dict(learning_rate=0.05, num_leaves=7, min_data_in_leaf=20, feature_fraction=0.6, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=2.0, num_threads=EV.NTHR, verbose=-1, seed=5)


def rr_map5(T, score):
    t = (T.select("pair_id", "family", "is_ev", "n_rel", "tb").with_columns(s=pl.Series(score))
         .sort(["pair_id", "s", "tb"], descending=[False, True, True])
         .with_columns(k=pl.int_range(pl.len()).over("pair_id") + 1).filter(c("k") <= 5)
         .with_columns(hits=c("is_ev").cast(pl.Int32).cum_sum().over("pair_id")))
    m = (t.group_by("pair_id", "family").agg(num=((c("hits") / c("k")) * c("is_ev")).sum(), n_rel=c("n_rel").first())
         .with_columns(ap=c("num") / c("n_rel").clip(upper_bound=5)))
    out = {"all": round(float(m["ap"].mean()), 4)}
    for f in FAMS:
        v = m.filter(c("family") == f)["ap"].mean()
        if v is not None:
            out[EV.SHORT[f]] = round(float(v), 4)
    return out


def run_rerank():
    """exp016ev-R: listwise re-ranking of each pair's top-K shortlist (cross-fit by the frozen table folds)."""
    t0 = time.time()
    res = json.loads((CACHE / "cv_results_v2.json").read_text())
    F, _ = build_dev()
    keys = F.select("pair_id", "hand_idx", "family", "fold", "is_ev", "n_rel", "pot_bb")
    z = np.load(CACHE / "v2_oof_store.npz")
    out = {}
    best = {}
    for f in FAMS:
        s = "h" if VARIANTS[res["best_v2"][f]["variant"]][0] else "b"
        s_flat, p2 = z[f"chosen|{f}"], z[f"p2|{s}|{f}"]
        for K in (8, 10, 12, 16):
            T = rr_frame(keys, F, s_flat, p2, K)
            T = T.rename({"y": "is_ev_", "fam": "family_"}).with_columns(
                is_ev=c("is_ev_"), family=c("family_"))
            Tf = T.filter(c("family") == f).sort("pair_id", "hand_seq")
            X = Tf.select([c(k).cast(pl.Float32) for k in RR_FEATS]).to_numpy()
            y = Tf["is_ev"].to_numpy().astype(np.int32)
            fold = Tf["fold"].to_numpy()
            pair = Tf["pair_id"].to_numpy()
            grp_all = np.diff(np.concatenate([[0], np.flatnonzero(pair[1:] != pair[:-1]) + 1, [len(pair)]]))
            base = rr_map5(Tf, Tf["s"].to_numpy())          # flat ranker restricted to the shortlist == flat MAP
            for obj in ("binary", "lambdarank"):
                for rounds in (50, 100, 200, 400):
                    oof = np.zeros(len(y))
                    for k in range(5):
                        tr, va = fold != k, fold == k
                        p = dict(RRP, objective=obj) if obj == "binary" else dict(
                            RRP, objective="lambdarank", lambdarank_truncation_level=5, eval_at=[5])
                        ptr = pair[tr]
                        g = np.diff(np.concatenate([[0], np.flatnonzero(ptr[1:] != ptr[:-1]) + 1, [len(ptr)]]))
                        ds = lgb.Dataset(X[tr], y[tr], group=None if obj == "binary" else g, free_raw_data=True)
                        m = lgb.train(p, ds, num_boost_round=rounds)
                        oof[va] = m.predict(X[va], raw_score=True)
                    out[f"{EV.SHORT[f]}|K{K}|{obj}|{rounds}"] = rr_map5(Tf, oof)[EV.SHORT[f]]
            out[f"{EV.SHORT[f]}|flat"] = base[EV.SHORT[f]]
        cands = [(v, k) for k, v in out.items() if k.startswith(EV.SHORT[f] + "|K")]
        sc, key = max(cands)
        best[f] = {"map5": sc, "cfg": key, "flat": out[f"{EV.SHORT[f]}|flat"]}
        log(f"rerank {EV.SHORT[f]}: flat {best[f]['flat']} -> best {sc} ({key})")
    res["rerank"] = {"grid": out, "best": best, "runtime_s": round(time.time() - t0, 1)}
    (CACHE / "cv_results_v2.json").write_text(json.dumps(res, indent=1))
    log("rerank grid:", json.dumps({k: v for k, v in sorted(out.items())}))
    return res["rerank"]


def finalize_selection():
    """Robust per-family pick: prefer the configuration without exp015 hs features when it is statistically tied
    (dev hs scores are single-model OOF while eval hs is a 5-model mean -> a small distribution shift), and keep the
    raw grid maximum on record. Rebuilds the chosen dev OOF scores / calibration from the stored per-variant OOF."""
    res = json.loads((CACHE / "cv_results_v2.json").read_text())
    tab = res["grid"]
    PICK = {"directed_transfer": ("C_seg", "binary"),        # hs-free, ties D_hs_seg (.7261 vs .7262)
            "soft_play": ("B_hs", "binary"),                 # +hs is the only SP gain (+0.008) -> keep, flagged
            "coordinated_isolation": ("A_base", "lambdarank")}  # = exp008 config; hs and seg both neutral/negative
    res.setdefault("best_v2_raw", res["best_v2"])
    z = dict(np.load(CACHE / "v2_oof_store.npz"))
    K, dev = _dev_frame()
    best, S, cal = {}, {}, {}
    for f, (v, o) in PICK.items():
        row = tab[f"{v}|{o}"][EV.SHORT[f]]
        r = max(GRID, key=lambda rr: row[str(rr)])
        best[f] = {"map5": row[str(r)], "variant": v, "obj": o, "rounds": r}
        S[f] = z[f"{v}|{o}|{f}"]                 # already Platt-calibrated per family in run_cv
        cal[f] = [1.0, 0.0]
    res["best_v2"], res["cal"] = best, cal
    z.update({f"chosen|{f}": S[f] for f in FAMS})
    np.savez(CACHE / "v2_oof_store.npz", **z)
    EV.write_scores(K.select("pair_id", "hand_idx", "pot_bb"), S,
                    C.DER / "evidence_scores_dev_exp016ev.parquet", keep_all=True)
    res["final_selection_map5_true_family"] = map5_of(dev, true_family_mix(dev, S))
    res["final_selection_map5_posterior"] = map5_of(dev, mix_scores(S, dev_posteriors(dev.pair)))
    (CACHE / "cv_results_v2.json").write_text(json.dumps(res, indent=1))
    log("final selection:", json.dumps(best))
    log("final MAP@5 true-family:", json.dumps(res["final_selection_map5_true_family"]),
        "| predicted posteriors:", json.dumps(res["final_selection_map5_posterior"]))
    return res


if __name__ == "__main__":
    stages = sys.argv[1:] or ["all"]
    if "all" in stages:
        stages = ["cv", "report", "rerank", "final", "eval"]
    for st in stages:
        log(f"=== v2 stage {st}")
        {"cv": run_cv, "report": run_report, "rerank": run_rerank, "final": finalize_selection,
         "eval": run_eval}[st]()
