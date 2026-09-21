"""exp027ev -- evidence rankers rebuilt on the self-trained hand detector (data/derived/hs_v2) + OC/LOFO probe
              + a train/serve protocol fix for the two-stage ranker.

Inputs that changed: src/handdet.py now self-trains (unlisted hands of positive pairs with round-1 s_family > .5 as
weak positives, w=8). Hand OOF AUC vs negative-pair hands DT .9963->.9983, SP .9931->.9957, CI .9926->.9964;
recall of listed evidence at the negative q999 threshold DT .65->.82, SP .41->.59, CI .62->.83.

Stages (python src/evidence_v4.py [skew|stage1|cv|lofo|eval|all]):
  skew    dev(OOF single model) vs eval(mean of 5 folds) distribution shift of the hs_v2 scores.
  stage1  no-chronology ranker quality for 4 feature sets:
            b   engine features only (321)                          [exp008/exp016ev base]
            h2  engine + hs_v2 raw + within-pair rank features
            h2r engine + hs_v2 RANK-ONLY (skew-immune)
            e   hs_v2 block + pair context ONLY (the "sole hand-quality block" test)
  cv      nested two-stage grid over those sets (chronology + segment blocks as in exp016ev).
  lofo    family-agnostic "planted-ness" head: train on TWO families, score the held-out one -- the honest proxy for
          the hidden other_coordination family -- against what the system does today (uniform mixture of the two
          seen family heads).
  eval    fit the chosen configs on all dev and score every eval-phase shared hand.

PROTOCOL NOTE (measured here, applies to exp008/exp016ev too): the stage-2 ranker generalises ~0.014 MAP better when
it is TRAINED on noisy stage-1 inputs (models fitted on 3 folds) and SERVED sharper ones, than when it is trained on
the 80% outer-OOF stage-1. `inner_s1` reproduces the training condition; the OOF loop in run_eval reproduces the
serving condition exactly (train on inner stage-1, score rows carrying the sharper stage-1), so its MAP and its Platt
coefficients both describe the shipped pipeline.
"""
import json
import os
import sys
import time

os.environ.setdefault("POLARS_MAX_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import polars as pl

import common as C
import evidence as EV
import evidence_v2 as V2
from evidence import FAMS, CACHE, PH, chrono_features, fit, predict, platt, sigmoid, log

c = pl.col
V2.HS = C.DER / "hs_v2"                      # <- the only input switch; column names are unchanged
HS_RANK = [k for k in V2.HS_COLS if k not in V2.HS_RAW]
CTX = ["n_shared", "log_shared", "pot_bb", "log_pot", "board_n", "players_vol"]
GRID = [100, 200, 350, 500, 750]
RES = CACHE / "cv_results_v4.json"

# Chosen on the nested dev OOF (true-family), preferring configs whose whole rounds-row is high, not a single cell:
#   DT  D2_hsv2_seg | lambdarank | 100   (row mean .7278; exp016ev DT .7261)
#   SP  D2_hsv2_seg | binary     | 200   (row mean .7208; exp016ev SP .7156; hs_v1->hs_v2 is +.012 in this variant)
#   CI  A_base      | lambdarank | 350   (= exp016ev; hs_v2 hurts CI at every setting)
CHOSEN = {"directed_transfer": dict(s="h2", use_seg=True, obj="lambdarank", rounds=100),
          "soft_play": dict(s="h2", use_seg=True, obj="binary", rounds=200),
          "coordinated_isolation": dict(s="b", use_seg=False, obj="lambdarank", rounds=350)}


def _res():
    return json.loads(RES.read_text()) if RES.exists() else {}


def _save(d):
    RES.write_text(json.dumps(d, indent=1))


def feature_sets(F, base_feats):
    Xb = F.select([c(k).cast(pl.Float32) for k in base_feats]).to_numpy()
    Xraw = F.select([c(k).cast(pl.Float32) for k in V2.HS_COLS]).to_numpy()
    Xrank = F.select([c(k).cast(pl.Float32) for k in HS_RANK]).to_numpy()
    Xe = F.select([c(k).cast(pl.Float32) for k in V2.HS_COLS + CTX]).to_numpy()
    return {"b": Xb, "h2": np.hstack([Xb, Xraw]), "h2r": np.hstack([Xb, Xrank]), "e": Xe}


def _chrono_cols(fam, F, s1, model_only):
    cf = chrono_features(F, s1, fam)
    if model_only:      # drop engine-gate priors: keep q + stage-1-derived chronology only
        cf = cf.select([k for k in cf.columns if k == "q" or k.startswith("s1")])
    return cf


# ============================================================================================ skew
def run_skew():
    t0 = time.time()
    F, _ = V2.build_dev()
    tabs = list(range(0, 40))
    ep = C.load("eval_pairs").select(c("p1").alias("pA"), c("p2").alias("pB"))
    ev = (pl.scan_parquet(V2.HS / "phase1" / "*.parquet").filter(c("table_idx").is_in(tabs))
          .select("pA", "pB", *V2.HS_RAW).join(ep.lazy(), on=["pA", "pB"], how="semi").collect())
    out = {}
    for k in V2.HS_RAW:
        d, e = F[k].to_numpy(), ev[k].to_numpy()
        out[k] = {"dev_mean": round(float(d.mean()), 5), "eval_mean": round(float(e.mean()), 5),
                  "dev_p99": round(float(np.quantile(d, 0.99)), 4), "eval_p99": round(float(np.quantile(e, 0.99)), 4),
                  "dev_frac>0.5": round(float((d > 0.5).mean()), 5), "eval_frac>0.5": round(float((e > 0.5).mean()), 5)}
    log("hs_v2 dev(positive-pair candidates) vs eval(all pairs):", json.dumps(out))
    r = _res()
    r["skew"] = {"note": "populations differ (dev rows are positive-pair candidates); see "
                         "skew_clean_negpairs_vs_eval for the like-for-like comparison",
                 "stats": out, "runtime_s": round(time.time() - t0, 1)}
    _save(r)
    return out


# ============================================================================================ stage-1
def run_stage1():
    t0 = time.time()
    res2 = json.loads((CACHE / "cv_results_v2.json").read_text())
    res0 = json.loads((CACHE / "cv_results.json").read_text())
    F, _ = V2.build_dev()
    dev = EV.Dev(F)
    XS = feature_sets(F, res2["features_base"])
    out, store = {}, {}
    for s, Xm in XS.items():
        for f in FAMS:
            tag, r = res0["best007_single"][f]["stage1_tag"], res0["best007_single"][f]["stage1_rounds"]
            pooled = tag.startswith("pooled")
            p = np.zeros(len(dev.y))
            for k in range(5):
                m = fit(dev, Xm, (dev.fold != k) if pooled else ((dev.fold != k) & (dev.fam == f)), "binary", r, pooled)
                p[dev.fold == k] = sigmoid(predict(m, Xm[dev.fold == k], "binary", pooled, f))
            store[f"{s}|{f}"] = p
            out[f"{s}|{EV.SHORT[f]}"] = round(dev.map5(p, f), 4)
        log(f"stage-1 {s}:", json.dumps({k.split('|')[1]: v for k, v in out.items() if k.startswith(s + "|")}),
            f"({time.time() - t0:.0f}s)")
    np.savez(CACHE / "v4_s1.npz", **store)
    F.select("pair_id", "hand_idx", "hand_seq", "table_idx", "fold", "family", "is_ev", "n_rel", "pot_bb",
             "seg").write_parquet(CACHE / "v4_dev_keys.parquet")
    r = _res()
    r["stage1"] = {"map5": out, "runtime_s": round(time.time() - t0, 1),
                   "reference_hs_v1_stage1": {"b": {"dt": 0.5983, "sp": 0.5836, "ci": 0.486},
                                              "h": {"dt": 0.5827, "sp": 0.5913, "ci": 0.4502}}}
    _save(r)
    return out


# ============================================================================================ nested stage-2
def run_cv(sets=("b", "h2", "h2r", "e")):
    t0 = time.time()
    res0 = json.loads((CACHE / "cv_results.json").read_text())
    res2 = json.loads((CACHE / "cv_results_v2.json").read_text())
    F, _ = V2.build_dev()
    dev = EV.Dev(F)
    XS = {k: v for k, v in feature_sets(F, res2["features_base"]).items() if k in sets}
    seg_lab = F["seg"].is_not_null().to_numpy()
    seg_t = (F["seg"].fill_null(1).to_numpy().astype(np.float64) - 1.0)
    s1cfg = {f: (res0["best007_single"][f]["stage1_tag"], res0["best007_single"][f]["stage1_rounds"]) for f in FAMS}
    VAR = {"A_base": ("b", False, False, ("binary",)),
           "B2_hsv2": ("h2", False, False, ("binary",)),
           "B2r_rankonly": ("h2r", False, False, ("binary",)),
           "D2_hsv2_seg": ("h2", True, False, ("binary", "lambdarank")),
           "E_hsonly": ("e", False, True, ("binary",))}
    VAR = {k: v for k, v in VAR.items() if v[0] in XS}

    def fit_s1(Xm, trm, f):
        tag, r = s1cfg[f]
        pooled = tag.startswith("pooled")
        return fit(dev, Xm, trm if pooled else (trm & (dev.fam == f)), "binary", r, pooled), pooled

    s1_oof = {s: {f: np.zeros(len(dev.y)) for f in FAMS} for s in XS}
    p2_oof = {s: {f: np.zeros(len(dev.y)) for f in FAMS} for s in XS}
    z1 = np.load(CACHE / "v4_s1.npz") if (CACHE / "v4_s1.npz").exists() else None
    for k in range(5):
        va = dev.fold == k
        for s, Xm in XS.items():
            for f in FAMS:
                if z1 is not None and f"{s}|{f}" in z1.files:
                    s1_oof[s][f] = z1[f"{s}|{f}"]
                else:
                    m, pooled = fit_s1(Xm, dev.fold != k, f)
                    s1_oof[s][f][va] = sigmoid(predict(m, Xm[va], "binary", pooled, f))
                p2_oof[s][f][va] = V2.fit_seg(Xm, seg_t, (dev.fold != k) & seg_lab & (dev.fam == f)).predict(Xm[va])
    log(f"outer stage-1/segment OOF ready ({time.time() - t0:.0f}s)")

    g2 = {v: {o: {f: {r: np.zeros(len(dev.y)) for r in GRID} for f in FAMS} for o in VAR[v][3]} for v in VAR}
    for k in range(5):
        tk = time.time()
        va = dev.fold == k
        s1 = {s: {f: s1_oof[s][f].copy() for f in FAMS} for s in XS}
        p2 = {s: {f: p2_oof[s][f].copy() for f in FAMS} for s in XS}
        for j in range(5):
            if j == k:
                continue
            vj, trm = dev.fold == j, (dev.fold != k) & (dev.fold != j)
            for s, Xm in XS.items():
                for f in FAMS:
                    m, pooled = fit_s1(Xm, trm, f)
                    s1[s][f][vj] = sigmoid(predict(m, Xm[vj], "binary", pooled, f))
                    p2[s][f][vj] = V2.fit_seg(Xm, seg_t, trm & seg_lab & (dev.fam == f)).predict(Xm[vj])
        for v, (s, use_seg, model_only, objs) in VAR.items():
            for f in FAMS:
                blocks = [XS[s], _chrono_cols(f, F, s1[s][f], model_only).to_numpy().astype(np.float32)]
                if use_seg:
                    blocks.append(V2.seg_features(F, s1[s][f], p2[s][f]).to_numpy().astype(np.float32))
                X2 = np.hstack(blocks)
                trm = (dev.fold != k) & (dev.fam == f)
                for o in objs:
                    mm = fit(dev, X2, trm, o, max(GRID), False)
                    for r in GRID:
                        g2[v][o][f][r][va] = predict(mm, X2[va], o, False, f, r)
        log(f"v4 nested fold {k} done ({time.time() - tk:.0f}s; total {time.time() - t0:.0f}s)")

    tab, store = {}, {}
    for v in VAR:
        for o in VAR[v][3]:
            tab[f"{v}|{o}"] = {EV.SHORT[f]: {r: round(dev.map5(g2[v][o][f][r], f), 4) for r in GRID} for f in FAMS}
            log(f"{v}|{o}:", json.dumps(tab[f"{v}|{o}"]))
            for f in FAMS:
                r = max(GRID, key=lambda rr: tab[f"{v}|{o}"][EV.SHORT[f]][rr])
                a, b_ = platt(g2[v][o][f][r][dev.fam == f], dev.y[dev.fam == f])
                store[f"{v}|{o}|{f}"] = sigmoid(a * g2[v][o][f][r] + b_)
    np.savez(CACHE / "v4_store.npz", **store, **{f"s1|{s}|{f}": s1_oof[s][f] for s in XS for f in FAMS},
             **{f"p2|{s}|{f}": p2_oof[s][f] for s in XS for f in FAMS})
    r = _res()
    r["cv"] = {"grid": tab, "runtime_s": round(time.time() - t0, 1), "variants": {k: list(v) for k, v in VAR.items()}}
    _save(r)
    return tab


# ============================================================================================ LOFO / OC head
def run_lofo():
    """family-agnostic planted-ness head: train on two families, score the third (proxy for other_coordination)."""
    t0 = time.time()
    res2 = json.loads((CACHE / "cv_results_v2.json").read_text())
    F, _ = V2.build_dev()
    dev = EV.Dev(F)
    XS = feature_sets(F, res2["features_base"])
    zv2 = np.load(CACHE / "v2_oof_store.npz")
    out = {}
    for s in ("b", "h2"):
        Xm = XS[s]
        for held in FAMS:
            seen = [g for g in FAMS if g != held]
            p = np.zeros(len(dev.y))
            for k in range(5):
                m = fit(dev, Xm, (dev.fold != k) & np.isin(dev.fam, seen), "binary", 300, False)
                p[dev.fold == k] = predict(m, Xm[dev.fold == k], "binary", False, held)
            out[f"{s}|{EV.SHORT[held]}|agnostic"] = round(dev.map5(p, held), 4)
            prod = {g: zv2[f"chosen|{g}"] for g in seen}          # exp016ev production heads
            out[f"{s}|{EV.SHORT[held]}|uniform_mix_seen"] = round(dev.map5(sum(prod.values()) / 2, held), 4)
            out[f"{s}|{EV.SHORT[held]}|max_seen"] = round(dev.map5(np.maximum(*list(prod.values())), held), 4)
            out[f"{s}|{EV.SHORT[held]}|own_family_head"] = round(dev.map5(zv2[f"chosen|{held}"], held), 4)
            log(f"LOFO {s} held-out {EV.SHORT[held]}:",
                json.dumps({k.split('|', 2)[2]: v for k, v in out.items() if k.startswith(f"{s}|{EV.SHORT[held]}|")}))
    r = _res()
    r["lofo"] = {"map5": out, "runtime_s": round(time.time() - t0, 1)}
    _save(r)
    return out


# ============================================================================================ eval (exp027ev)
def inner_s1(dev, Xm, f, tag, rounds):
    """stage-1 OOF in the NESTED-CV TRAINING condition: each fold predicted by a model trained on 3 folds (60%),
    not 4 (80%). Training stage-2 on these noisier inputs is worth ~+0.014 MAP versus training on the outer OOF."""
    pooled = tag.startswith("pooled")
    out = np.zeros(len(dev.y))
    for j in range(5):
        trm = ~np.isin(dev.fold, [j, (j + 1) % 5])
        m = fit(dev, Xm, trm if pooled else (trm & (dev.fam == f)), "binary", rounds, pooled)
        out[dev.fold == j] = sigmoid(predict(m, Xm[dev.fold == j], "binary", pooled, f))
    return out


def run_eval(tables_per_batch=20, s1_mode=None):
    # PROVENANCE (measured 2026-09-20, scratch provenance.py): the SHIPPED
    # data/derived/evidence_scores_{dev,eval}_exp027ev.parquet -- the partner behind sub024/sub025/sub027 --
    # were produced with s1_mode="outer" (stage-2 trained AND served on the cached 80% outer-OOF stage-1).
    # Rebuilding them with the "inner" default gives a DIFFERENT model: dev OOF dt .7236 / sp .7072 / ci .6837
    # vs the shipped .7222 / .7191 / .6907, and -0.0039 dev MAP inside the exp042_ramp blend.
    # An "outer" rebuild reproduces the shipped file EXACTLY (max |diff| = 0 on all 45,129 rows).
    # So: to reproduce the submitted pipeline set EVID_S1MODE=outer. The default is left at "inner"
    # (the honest train-noisy/serve-sharp protocol described in the module docstring) so nothing silently
    # changes for callers that do not ask.
    s1_mode = s1_mode or os.environ.get("EVID_S1MODE", "inner")
    t0 = time.time()
    res0 = json.loads((CACHE / "cv_results.json").read_text())
    res2 = json.loads((CACHE / "cv_results_v2.json").read_text())
    F, _ = V2.build_dev()
    dev = EV.Dev(F)
    XS = feature_sets(F, res2["features_base"])
    seg_lab = F["seg"].is_not_null().to_numpy()
    seg_t = (F["seg"].fill_null(1).to_numpy().astype(np.float64) - 1.0)
    z = np.load(CACHE / "v4_store.npz")
    s1cfg = {f: (res0["best007_single"][f]["stage1_tag"], res0["best007_single"][f]["stage1_rounds"]) for f in FAMS}
    allrows = np.ones(len(dev.y), dtype=bool)
    M, cal, Sdev, oof_map = {}, {}, {}, {}
    for f, cfg in CHOSEN.items():
        s, use_seg, obj, rounds = cfg["s"], cfg["use_seg"], cfg["obj"], cfg["rounds"]
        tag, r1 = s1cfg[f]
        pooled = tag.startswith("pooled")
        s1_tr = inner_s1(dev, XS[s], f, tag, r1) if s1_mode == "inner" else z[f"s1|{s}|{f}"]
        s1_va = z[f"s1|{s}|{f}"]                       # 80%-model OOF = the serving-like (sharper) stage-1
        p2o = z[f"p2|{s}|{f}"]

        def _mat(s1v):
            bl = [XS[s], chrono_features(F, s1v, f).to_numpy().astype(np.float32)]
            if use_seg:
                bl.append(V2.seg_features(F, s1v, p2o).to_numpy().astype(np.float32))
            return np.hstack(bl)

        X2_tr, X2_va = _mat(s1_tr), _mat(s1_va)
        # OOF in the SERVING condition: stage-2 trained on noisy stage-1, scored on rows carrying sharp stage-1
        oof = np.zeros(len(dev.y))
        for k in range(5):
            mm = fit(dev, X2_tr, (dev.fold != k) & (dev.fam == f), obj, rounds, False)
            oof[dev.fold == k] = predict(mm, X2_va[dev.fold == k], obj, False, f)
        a, b_ = platt(oof[dev.fam == f], dev.y[dev.fam == f])
        cal[f] = [a, b_]
        Sdev[f] = sigmoid(a * oof + b_)
        oof_map[EV.SHORT[f]] = round(dev.map5(oof, f), 4)
        M[f] = dict(s=s, use_seg=use_seg, obj=obj, pooled=pooled,
                    m1=fit(dev, XS[s], allrows if pooled else (dev.fam == f), "binary", r1, pooled),
                    mseg=V2.fit_seg(XS[s], seg_t, seg_lab & (dev.fam == f)) if use_seg else None,
                    m2=fit(dev, X2_tr, dev.fam == f, obj, rounds, False))
        log(f"eval fit {EV.SHORT[f]} {cfg} serving-condition OOF MAP5={oof_map[EV.SHORT[f]]} ({time.time() - t0:.0f}s)")
    EV.write_scores(F.select("pair_id", "hand_idx", "pot_bb"), Sdev,
                    C.DER / "evidence_scores_dev_exp027ev.parquet", keep_all=True)
    r = _res()
    r["eval"] = {"chosen": CHOSEN, "cal": cal, "s1_mode": s1_mode, "serving_condition_oof_map5": oof_map}
    _save(r)

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
        R = V2.hs_join(R.drop("shared_hands"), 1).sort("pair_id", "hand_seq")
        E = V2.add_hs_feats(EV.hand_features(R))
        del R
        XE = feature_sets(E, res2["features_base"])
        keys = E.select("pair_id", "hand_idx", "pot_bb")
        SE = {}
        for f, d in M.items():
            Xm = XE[d["s"]]
            s1 = sigmoid(predict(d["m1"], Xm, "binary", d["pooled"], f))
            bl = [Xm, chrono_features(E, s1, f).to_numpy().astype(np.float32)]
            if d["use_seg"]:
                bl.append(V2.seg_features(E, s1, d["mseg"].predict(Xm)).to_numpy().astype(np.float32))
            a, b_ = cal[f]
            SE[f] = sigmoid(a * predict(d["m2"], np.hstack(bl), d["obj"], False, f) + b_)
        outs.append(EV.write_scores(keys, SE, None))
        n_rows += len(keys)
        del E, XE
        log(f"eval tables {tb}-{tb + tables_per_batch - 1}: {len(keys):,} rows, {time.time() - t:.0f}s")
    assert n_pairs == ep.height, (n_pairs, ep.height)
    out = pl.concat(outs).sort("pair_id", "hand_idx")
    out.write_parquet(C.DER / "evidence_scores_eval_exp027ev.parquet")
    r = _res()
    r["eval"].update(rows=out.height, pairs=out["pair_id"].n_unique(), runtime_s=round(time.time() - t0, 1))
    _save(r)
    log(f"wrote evidence_scores_eval_exp027ev.parquet: {out.height:,} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    stages = sys.argv[1:] or ["all"]
    if "all" in stages:
        stages = ["skew", "stage1", "cv", "lofo", "eval"]
    for st in stages:
        log(f"=== v4 stage {st}")
        {"skew": run_skew, "stage1": run_stage1, "cv": run_cv, "lofo": run_lofo, "eval": run_eval}[st]()
