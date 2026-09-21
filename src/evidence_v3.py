"""exp023ev -- calibrated SEGMENT composition of the evidence list (retry of the quota lane).

Why the v2 quota failed: the segment classifier was 2-class, trained on LISTED hands only, so on the full candidate
pool it is an unconditional model with the wrong prior (~90% of candidates labelled "segment 2"). Fixes here:

  (A) 3-CLASS model per family, cross-fit on the frozen table folds:
      0 = not listed, 1 = listed segment-1 hand, 2 = listed segment-2 hand
      -> calibrated P(seg1|h), P(seg2|h), P(not listed|h) on the REAL candidate pool.
      Two definitions of class 0 are measured: "in-pair" (non-listed hands of positive pairs) and
      "in-pair + confirmed-negative-pair hands".
  (B) FACTORED model: P(seg_s|h) = P(listed|h) * P(seg2|listed,h), where P(listed|h) is the exp016ev flat OOF
      probability and P(seg2|listed,h) is the 2-class model trained on listed hands (OOF AUC .959 DT / .946 SP).
      Same information, correct prior by construction.

Segment sizes are modelled per pair (k1 = # listed segment-1 hands) instead of fixed, from the calibrated mass
sum_h P(seg1|h), and compared with the dev truth (segment-2 start index: DT 1->11, 2->31, 3->33, 4->16;
SP 1->18, 2->40, 3->25, 4->10; CI "rank 5 appended").

Composition policies (all scored with the host AP@5, denominator min(|rel|,5)):
  flat          : top-5 by P(listed)                                   [= exp016ev, the baseline to beat]
  quota_k1      : top-k1 by P(seg1) then top-(5-k1) by P(seg2), each chronological (Critic C1 order) or by score
  quota_est     : same, k1 = clamp(round(sum_h P(seg1|h)), 1, 4) per pair
  greedy_eap    : sequential maximisation of E[AP@5] with segment-slot discounting (no hard quota)
Stages: python src/evidence_v3.py [fit|diag|compose|eval|all]
Outputs (only if it beats exp016ev): data/derived/evidence_scores_{dev,eval}_exp023ev.parquet
Always: data/derived/evidence_cache/cv_results_v3.json
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
import evidence_v2 as V2
from evidence import FAMS, CACHE, PH, log, sigmoid

c = pl.col
MC = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=15, min_data_in_leaf=20,
          feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=EV.NTHR,
          verbose=-1, seed=23)
MC_ROUNDS = 250
B2 = dict(V2.SEGP)


# ===================================================================================== fit (3-class + factored)
def run_fit():
    """cross-fit (frozen table folds) 3-class segment/listing models + the factored decomposition."""
    t0 = time.time()
    res2 = json.loads((CACHE / "cv_results_v2.json").read_text())
    _BASE = res2["features_base"]
    F, _ = V2.build_dev()
    dev = EV.Dev(F)
    z = np.load(CACHE / "v2_oof_store.npz")
    seg_lab = F["seg"].is_not_null().to_numpy()
    seg2 = (F["seg"].fill_null(1).to_numpy().astype(np.float64) - 1.0)      # 1 = listed segment-2 hand
    y_ev = dev.y.astype(bool)
    out = {"n_seg_labelled": int(seg_lab.sum())}
    Xb = F.select([c(k).cast(pl.Float32) for k in _BASE]).to_numpy()
    Xh = np.hstack([Xb, F.select([c(k).cast(pl.Float32) for k in V2.HS_COLS]).to_numpy()])
    XS = {"b": Xb, "h": Xh}

    P3 = {}          # (mode, fam) -> (n_rows, 3) calibrated class probabilities  [not listed, seg1, seg2]
    PF = {}          # fam -> (P_listed, P_seg2_given_listed)
    for f in FAMS:
        v = res2["best_v2"][f]["variant"]
        s = "h" if V2.VARIANTS[v][0] else "b"
        s1, p2v = z[f"s1|{s}|{f}"], z[f"p2|{s}|{f}"]
        X2, _ = V2.stage2_matrix(F, XS[s], s1, p2v, f, True)
        fam_rows = dev.fam == f
        # 3-class target on this family's pairs: 0 not listed, 1 listed seg1, 2 listed seg2
        t3 = np.zeros(len(dev.y), dtype=int)
        t3[y_ev & (seg2 == 0) & seg_lab] = 1
        t3[y_ev & (seg2 == 1) & seg_lab] = 2
        usable = fam_rows & (~y_ev | seg_lab)          # drop listed hands whose segment is unknown (0-inversion pairs)
        p3 = np.zeros((len(dev.y), 3))
        for k in range(5):
            tr, va = usable & (dev.fold != k), dev.fold == k
            m = lgb.train(MC, lgb.Dataset(X2[tr], t3[tr], free_raw_data=True), num_boost_round=MC_ROUNDS)
            p3[va] = m.predict(X2[va])
        P3[("inpair", f)] = p3
        # factored: P(listed) = exp016ev flat OOF (calibrated), P(seg2|listed) = 2-class on listed hands
        p_listed = z[f"chosen|{f}"]
        p2l = np.zeros(len(dev.y))
        for k in range(5):
            tr, va = seg_lab & y_ev & fam_rows & (dev.fold != k), dev.fold == k
            m = lgb.train(dict(B2), lgb.Dataset(X2[tr], seg2[tr], free_raw_data=True), num_boost_round=V2.SEG_ROUNDS)
            p2l[va] = m.predict(X2[va])
        PF[f] = (p_listed, p2l)
        log(f"fit {EV.SHORT[f]}: 3-class + factored done ({time.time() - t0:.0f}s)")
    np.savez(CACHE / "v3_store.npz",
             **{f"p3|inpair|{f}": P3[("inpair", f)] for f in FAMS},
             **{f"pl|{f}": PF[f][0] for f in FAMS}, **{f"p2l|{f}": PF[f][1] for f in FAMS},
             seg_lab=seg_lab, seg2=seg2)
    F.select("pair_id", "hand_idx", "hand_seq", "table_idx", "fold", "family", "is_ev", "n_rel", "pot_bb",
             "seg").write_parquet(CACHE / "v3_dev_keys.parquet")
    out["runtime_fit_s"] = round(time.time() - t0, 1)
    p = CACHE / "cv_results_v3.json"
    r = json.loads(p.read_text()) if p.exists() else {}
    r.update({"fit": out})
    p.write_text(json.dumps(r, indent=1))
    log(f"fit done in {out['runtime_fit_s']}s")


# ===================================================================================== policies
def _pair_frame(K, p_seg1, p_seg2, fam):
    """per-row frame for one family's pairs with calibrated segment probabilities."""
    m = (K["family"] == fam).to_numpy()
    T = K.filter(pl.Series(m)).with_columns(
        p1=pl.Series(p_seg1[m]), p2=pl.Series(p_seg2[m])).with_columns(
        pl_=c("p1") + c("p2")).sort("pair_id", "hand_seq")
    return T.with_columns(t_rank=pl.int_range(pl.len()).over("pair_id"))


def _ap5(T, order_col_sets):
    """order_col_sets: list of (expr list, descending list) applied in order -> AP@5 per pair."""
    t = T.sort(*order_col_sets[0], descending=order_col_sets[1])
    t = (t.with_columns(k=pl.int_range(pl.len()).over("pair_id") + 1).filter(c("k") <= 5)
         .with_columns(hits=c("is_ev").cast(pl.Int32).cum_sum().over("pair_id")))
    m = (t.group_by("pair_id").agg(num=((c("hits") / c("k")) * c("is_ev")).sum(), n_rel=c("n_rel").first())
         .with_columns(ap=c("num") / c("n_rel").clip(upper_bound=5)))
    return round(float(m["ap"].mean()), 4)


def policy_flat(T):
    return _ap5(T, ([["pair_id", "pl_", "pot_bb"]], [False, True, True]))


def policy_quota(T, k1, order="chrono"):
    """top-k1 by P(seg1) + top-(5-k1) by P(seg2); k1 int or 'est' (per-pair calibrated mass)."""
    if k1 == "est":
        T = T.with_columns(k1=c("p1").sum().over("pair_id").round().clip(1, 4))
    else:
        T = T.with_columns(k1=pl.lit(k1))
    a = (T.sort(["pair_id", "p1"], descending=[False, True])
         .with_columns(r1=pl.int_range(pl.len()).over("pair_id") + 1))
    b = (T.sort(["pair_id", "p2"], descending=[False, True])
         .with_columns(r2=pl.int_range(pl.len()).over("pair_id") + 1)).select("pair_id", "hand_idx", "r2")
    t = a.join(b, on=["pair_id", "hand_idx"])
    pick1 = t.filter(c("r1") <= c("k1")).with_columns(seg_pick=pl.lit(1), key=c("p1"))
    pick2 = (t.join(pick1.select("pair_id", "hand_idx"), on=["pair_id", "hand_idx"], how="anti")
             .sort(["pair_id", "p2"], descending=[False, True])
             .with_columns(r2b=pl.int_range(pl.len()).over("pair_id") + 1)
             .filter(c("r2b") <= 5 - c("k1")).with_columns(seg_pick=pl.lit(2), key=c("p2")).drop("r2b"))
    sel = pl.concat([pick1, pick2], how="diagonal")
    if order == "chrono":     # Critic C1: segment 1 chronologically, then segment 2 chronologically
        keys, desc = [["pair_id", "seg_pick", "t_rank"]], [False, False, False]
    else:                     # by probability
        keys, desc = [["pair_id", "key", "pot_bb"]], [False, True, True]
    return _ap5(sel, (keys, desc))


def policy_greedy_eap(T, n_max=5, listed_counts=(3, 4, 5), count_w=(11 / 372, 21 / 372, 340 / 372)):
    """sequential maximisation of E[AP@5] with segment-slot discounting.

    Under independence, ranking by P(listed) already maximises E[AP@5] (ordering high-p first is optimal), so the
    only source of gain is the NEGATIVE dependence inside a segment: a pair's list holds about k_s listed hands of
    segment s, so the (k_s+1)-th hand of that segment is much less likely to be listed than its marginal p implies.
    Effective probability of the next candidate from segment s:  p_h * max(0, k_s - n_s) / max(k_s, 1) clipped [0,1].
    """
    rows = T.select("pair_id", "hand_idx", "is_ev", "n_rel", "p1", "p2", "pl_", "t_rank").to_dicts()
    by = {}
    for r in rows:
        by.setdefault(r["pair_id"], []).append(r)
    aps = []
    for pid, hs in by.items():
        k1 = float(np.clip(sum(h["p1"] for h in hs), 0.5, 4.5))
        k2 = float(np.clip(sum(h["p2"] for h in hs), 0.5, 4.5))
        chosen, taken, n1, n2 = [], set(), 0, 0
        pool = sorted(hs, key=lambda h: -h["pl_"])[:40]
        for _ in range(n_max):
            best, bkey = None, -1.0
            for h in pool:
                if h["hand_idx"] in taken:
                    continue
                s1w = max(0.0, k1 - n1) / max(k1, 1e-6)
                s2w = max(0.0, k2 - n2) / max(k2, 1e-6)
                eff = h["p1"] * min(1.0, s1w) + h["p2"] * min(1.0, s2w)
                if eff > bkey:
                    best, bkey = h, eff
            if best is None:
                break
            chosen.append(best)
            taken.add(best["hand_idx"])
            if best["p2"] > best["p1"]:
                n2 += 1
            else:
                n1 += 1
        hits, ps = 0, 0.0
        for k, h in enumerate(chosen, 1):
            if h["is_ev"]:
                hits += 1
                ps += hits / k
        aps.append(ps / min(hs[0]["n_rel"], 5))
    return round(float(np.mean(aps)), 4)


# ===================================================================================== diag + compose
def run_diag():
    t0 = time.time()
    K = pl.read_parquet(CACHE / "v3_dev_keys.parquet")
    z = np.load(CACHE / "v3_store.npz")
    seg_lab, seg2 = z["seg_lab"], z["seg2"]
    out = {}
    # true k1 distribution (one-inversion pairs)
    tr = (K.with_columns(seg=pl.Series(np.where(seg_lab, seg2 + 1, np.nan)), lab=pl.Series(seg_lab))
          .filter((c("is_ev") == 1) & c("lab"))          # one-inversion pairs only (segment truth known)
          .group_by("pair_id", "family").agg(k1=(c("seg") == 1).sum(), k2=(c("seg") == 2).sum()))
    out["true_k1_distribution"] = {EV.SHORT[f]: tr.filter(c("family") == f)["k1"].value_counts().sort("k1").to_dicts()
                                   for f in FAMS}
    log("true k1 distribution:", json.dumps(out["true_k1_distribution"]))
    for f in FAMS:
        for mode, (p1, p2) in {"3class": (z[f"p3|inpair|{f}"][:, 1], z[f"p3|inpair|{f}"][:, 2]),
                               "factored": (z[f"pl|{f}"] * (1 - z[f"p2l|{f}"]), z[f"pl|{f}"] * z[f"p2l|{f}"])}.items():
            T = _pair_frame(K, p1, p2, f)
            agg = T.group_by("pair_id").agg(k1_hat=c("p1").sum(), k2_hat=c("p2").sum(),
                                            pl_sum=c("pl_").sum()).join(tr, on="pair_id", how="left")
            d = agg.drop_nulls("k1")
            out[f"calibration_{EV.SHORT[f]}_{mode}"] = {
                "mean_k1_hat": round(float(agg["k1_hat"].mean()), 3), "mean_k1_true": round(float(d["k1"].mean()), 3),
                "mean_k2_hat": round(float(agg["k2_hat"].mean()), 3), "mean_k2_true": round(float(d["k2"].mean()), 3),
                "mean_listed_mass": round(float(agg["pl_sum"].mean()), 3),
                "corr_k1": round(float(np.corrcoef(d["k1_hat"].to_numpy(), d["k1"].to_numpy())[0, 1]), 3)}
            # segment composition of our own flat top-5 vs the true lists
            top5 = (T.sort(["pair_id", "pl_"], descending=[False, True])
                    .with_columns(k=pl.int_range(pl.len()).over("pair_id") + 1).filter(c("k") <= 5)
                    .with_columns(is2=(c("p2") > c("p1")).cast(pl.Int32))
                    .group_by("pair_id").agg(n2_pick=c("is2").sum()))
            out[f"top5_seg2_count_{EV.SHORT[f]}_{mode}"] = round(float(top5["n2_pick"].mean()), 3)
            log(f"{EV.SHORT[f]}/{mode}:", json.dumps(out[f"calibration_{EV.SHORT[f]}_{mode}"]),
                "mean #seg2 in our top-5:", out[f"top5_seg2_count_{EV.SHORT[f]}_{mode}"])
    p = CACHE / "cv_results_v3.json"
    r = json.loads(p.read_text())
    r["diag"] = out
    r["diag"]["runtime_s"] = round(time.time() - t0, 1)
    p.write_text(json.dumps(r, indent=1))
    return out


def run_compose():
    t0 = time.time()
    K = pl.read_parquet(CACHE / "v3_dev_keys.parquet")
    z = np.load(CACHE / "v3_store.npz")
    res2 = json.loads((CACHE / "cv_results_v2.json").read_text())
    zz = np.load(CACHE / "v2_oof_store.npz")
    tab = {}
    for f in FAMS:
        ref = _pair_frame(K, zz[f"chosen|{f}"], np.zeros(len(K)), f)   # exp016ev flat baseline
        tab[f"{EV.SHORT[f]}|exp016ev_flat"] = policy_flat(ref)
        for mode, (p1, p2) in {"3class": (z[f"p3|inpair|{f}"][:, 1], z[f"p3|inpair|{f}"][:, 2]),
                               "factored": (z[f"pl|{f}"] * (1 - z[f"p2l|{f}"]), z[f"pl|{f}"] * z[f"p2l|{f}"])}.items():
            T = _pair_frame(K, p1, p2, f)
            tab[f"{EV.SHORT[f]}|{mode}|flat"] = policy_flat(T)
            for k1 in (1, 2, 3, 4, "est"):
                for order in ("chrono", "score"):
                    tab[f"{EV.SHORT[f]}|{mode}|quota{k1}|{order}"] = policy_quota(T, k1, order)
            tab[f"{EV.SHORT[f]}|{mode}|greedy_eap"] = policy_greedy_eap(T)
        log(f"{EV.SHORT[f]} policies:", json.dumps({k.split('|', 1)[1]: v for k, v in tab.items()
                                                    if k.startswith(EV.SHORT[f] + "|")}))
    p = CACHE / "cv_results_v3.json"
    r = json.loads(p.read_text())
    r["compose"] = {"table": tab, "runtime_s": round(time.time() - t0, 1)}
    # verdict: best policy per family vs the exp016ev flat baseline
    verdict = {}
    for f in FAMS:
        base = tab[f"{EV.SHORT[f]}|exp016ev_flat"]
        cands = [(v, k) for k, v in tab.items() if k.startswith(EV.SHORT[f] + "|") and "exp016ev_flat" not in k]
        sc, key = max(cands)
        verdict[EV.SHORT[f]] = {"exp016ev_flat": base, "best_v3": sc, "best_key": key, "delta": round(sc - base, 4)}
    r["compose"]["verdict"] = verdict
    p.write_text(json.dumps(r, indent=1))
    log("VERDICT:", json.dumps(verdict))
    return verdict


if __name__ == "__main__":
    stages = sys.argv[1:] or ["all"]
    if "all" in stages:
        stages = ["fit", "diag", "compose"]
    for st in stages:
        log(f"=== v3 stage {st}")
        {"fit": run_fit, "diag": run_diag, "compose": run_compose}[st]()
