"""exp050 -- SEED-BAG the exp027ev in-pair evidence partner ranker.

Why this is NOT the closed "partner averaging" lane (research/CLOSED_LANES.md):
  that lane mixed TWO DIFFERENT architectures (exp008 + exp027ev). The listing-rule blend
  s = (1-w)*partner + w*DP needs the partner to be a calibrated P(listed | family) on the DP's
  probability scale; mixing two differently-calibrated models breaks that and the blend lost
  0.0024-0.0105 even though the mixture was the best standalone ranker (+0.0047).
  Seed-bagging ONE architecture with its own Platt step per seed is pure variance reduction and
  leaves the calibration scale intact. Never measured before this run.

What is reseeded (everything exp027ev fits):
  * the pooled stage-1 event model  (evidence.params seed)
  * the segment model V2.fit_seg    (evidence_v2.SEGP seed)
  * the stage-2 chronology model    (evidence.params seed)
  * hence also the per-seed Platt calibration fitted on that seed's serving-condition OOF.
  The hs_v2 hand-detector inputs and the DP event/type heads of scripts/listing_rule.py are NOT
  reseeded here (exp041 already measured bagging the DP heads: +0.0008, variance-neutral).

Aggregation choices measured (dev, true family, 372 pairs):
  probmean  mean of the per-seed calibrated probabilities   <- the stated default
  margmean  mean of the per-seed stage-2 margins, one Platt on the bagged margin
  replatt   probmean -> logit -> Platt refit                <- restores the DP scale exactly

Stages:  python src/exp050_bag027ev.py seed <sd>          fit ONE seed (OOF + full models), persist both
         python src/exp050_bag027ev.py combine <seeds>    write the bagged dev score files
         python src/exp050_bag027ev.py eval <agg> <seeds> score the eval phase with the bag
Outputs: data/derived/evidence_scores_{dev,eval}_exp050bag_<agg>.parquet
         data/derived/evidence_cache/exp050_seed<sd>.npz  + exp050_seed<sd>/ (LightGBM full-dev models)
Seeds are independent processes on purpose -- the box is shared, so they run concurrently at EVID_NTHR threads.
"""
import gc
import json
import os
import sys
import time

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import polars as pl
import lightgbm as lgb

import common as C
import evidence as EV
import evidence_v2 as V2
import evidence_v4 as V4
from evidence import FAMS, CACHE, PH, chrono_features, fit, predict, platt, sigmoid, log

c = pl.col
CUR_SEED = [42]
if os.environ.get("EVID_NTHR"):
    EV.NTHR = int(os.environ["EVID_NTHR"])
    V2.SEGP["num_threads"] = EV.NTHR

# ---------------------------------------------------------------- seed plumbing (monkeypatch, no project edits)
_orig_params = EV.params


def _params(obj, seed=None):
    return _orig_params(obj, CUR_SEED[0] if seed is None else seed)


def _fit_seg(X, y, mask):
    p = dict(V2.SEGP)
    p["seed"] = 11 if CUR_SEED[0] == 42 else 11 + 7919 * CUR_SEED[0]
    return _retry(lgb.train, p, lgb.Dataset(X[mask], y[mask], free_raw_data=True), num_boost_round=V2.SEG_ROUNDS)


EV.params = _params
V2.fit_seg = _fit_seg
_raw_fit = EV.fit


def _retry(fn, *a, **k):
    for att in range(8):
        try:
            return fn(*a, **k)
        except (MemoryError, OSError) as e:          # native LightGBM alloc failure surfaces as OSError
            gc.collect()
            log(f"  fit failed ({type(e).__name__}: {str(e)[:70]}), retry {att + 1}/8 in 45s")
            time.sleep(45)
    raise SystemExit("repeated allocation failures -- box is out of memory")


def fit(*a, **k):
    return _retry(_raw_fit, *a, **k)


# ---------------------------------------------------------------- pooled stage-1 with a fit cache
class S1Cache:
    """The stage-1 models are POOLED (one model over all families, family one-hot appended), so the dt/sp fits on
    feature set 'h2' differ only in num_boost_round (500 vs 1000). LightGBM boosts sequentially, so the first 500
    trees of the 1000-round model ARE the 500-round model: fit once at max rounds, predict with num_iteration."""

    def __init__(self, dev, XS, need):
        self.dev, self.XS = dev, XS
        self.maxr = {}
        for f, (s, tag, r) in need.items():
            self.maxr[s] = max(self.maxr.get(s, 0), r)
        self.store = {}

    def get(self, s, mask, key):
        k = (s, key)
        if k not in self.store:
            self.store[k] = fit(self.dev, self.XS[s], mask, "binary", self.maxr[s], True)
        return self.store[k]

    def clear(self):
        self.store = {}


def _fsets(df, base_feats):
    """only 'b' and 'h2' -- the feature sets exp027ev's CHOSEN configs use (memory-light vs V4.feature_sets)."""
    need = {V4.CHOSEN[f]["s"] for f in FAMS}
    Xb = df.select([c(k).cast(pl.Float32) for k in base_feats]).to_numpy()
    out = {"b": Xb}
    if "h2" in need:
        out["h2"] = np.hstack([Xb, df.select([c(k).cast(pl.Float32) for k in V2.HS_COLS]).to_numpy()])
    assert need <= set(out), need
    return out


def _build():
    F, _ = V2.build_dev()
    dev = EV.Dev(F)
    res0 = json.loads((CACHE / "cv_results.json").read_text())
    res2 = json.loads((CACHE / "cv_results_v2.json").read_text())
    XS = _fsets(F, res2["features_base"])
    s1cfg = {f: (res0["best007_single"][f]["stage1_tag"], res0["best007_single"][f]["stage1_rounds"]) for f in FAMS}
    for f, (tag, r) in s1cfg.items():
        assert tag.startswith("pooled"), (f, tag)          # the cache assumes pooled stage-1
    return F, dev, XS, s1cfg


def fit_seed(F, dev, XS, s1cfg, seed, fit_full=True):
    """One seed of the whole exp027ev partner. Returns per-family OOF margins, Platt coefs, dev probs, models."""
    t0 = time.time()
    CUR_SEED[0] = seed
    seg_lab = F["seg"].is_not_null().to_numpy()
    seg_t = (F["seg"].fill_null(1).to_numpy().astype(np.float64) - 1.0)
    allrows = np.ones(len(dev.y), dtype=bool)
    need = {f: (V4.CHOSEN[f]["s"], s1cfg[f][0], s1cfg[f][1]) for f in FAMS}
    cache = S1Cache(dev, XS, need)

    # --- outer stage-1 OOF (80% models = the SERVING-like sharper stage-1) + segment OOF
    s1_va, p2o = {}, {}
    for f in FAMS:
        s1_va[f] = np.zeros(len(dev.y))
        p2o[f] = np.zeros(len(dev.y))
    for k in range(5):
        va = dev.fold == k
        for f in FAMS:
            s = V4.CHOSEN[f]["s"]
            m = cache.get(s, dev.fold != k, f"outer{k}")
            s1_va[f][va] = sigmoid(predict(m, XS[s][va], "binary", True, f, s1cfg[f][1]))
            p2o[f][va] = V2.fit_seg(XS[s], seg_t, (dev.fold != k) & seg_lab & (dev.fam == f)).predict(XS[s][va])
    cache.clear()
    gc.collect()
    log(f"seed {seed}: outer stage-1 + segment OOF ({time.time() - t0:.0f}s)")

    # --- inner stage-1 (60% models = the TRAINING condition, worth ~+0.014 MAP; see evidence_v4 PROTOCOL NOTE)
    s1_tr = {f: np.zeros(len(dev.y)) for f in FAMS}
    for j in range(5):
        trm = ~np.isin(dev.fold, [j, (j + 1) % 5])
        for f in FAMS:
            s = V4.CHOSEN[f]["s"]
            m = cache.get(s, trm, f"inner{j}")
            s1_tr[f][dev.fold == j] = sigmoid(predict(m, XS[s][dev.fold == j], "binary", True, f, s1cfg[f][1]))
    cache.clear()
    gc.collect()
    log(f"seed {seed}: inner stage-1 ({time.time() - t0:.0f}s)")

    oof, cal, Sdev, M, oof_map = {}, {}, {}, {}, {}
    for f in FAMS:
        cfg = V4.CHOSEN[f]
        s, use_seg, obj, rounds = cfg["s"], cfg["use_seg"], cfg["obj"], cfg["rounds"]

        def _mat(s1v):
            bl = [XS[s], chrono_features(F, s1v, f).to_numpy().astype(np.float32)]
            if use_seg:
                bl.append(V2.seg_features(F, s1v, p2o[f]).to_numpy().astype(np.float32))
            return _retry(np.hstack, bl)

        X2_tr, X2_va = _mat(s1_tr[f]), _mat(s1_va[f])
        o = np.zeros(len(dev.y))
        for k in range(5):
            mm = fit(dev, X2_tr, (dev.fold != k) & (dev.fam == f), obj, rounds, False)
            o[dev.fold == k] = predict(mm, X2_va[dev.fold == k], obj, False, f)
        a, b_ = platt(o[dev.fam == f], dev.y[dev.fam == f])
        oof[f], cal[f], Sdev[f] = o, [a, b_], sigmoid(a * o + b_)
        oof_map[EV.SHORT[f]] = round(dev.map5(o, f), 4)
        if fit_full:
            M[f] = dict(s=s, use_seg=use_seg, obj=obj, pooled=True,
                        m1=fit(dev, XS[s], allrows, "binary", s1cfg[f][1], True),
                        mseg=V2.fit_seg(XS[s], seg_t, seg_lab & (dev.fam == f)) if use_seg else None,
                        m2=fit(dev, X2_tr, dev.fam == f, obj, rounds, False))
        del X2_tr, X2_va
        gc.collect()
    log(f"seed {seed}: stage-2 OOF + full fits {oof_map} ({time.time() - t0:.0f}s)")
    return dict(oof=oof, cal=cal, Sdev=Sdev, M=M, oof_map=oof_map)


# ---------------------------------------------------------------- aggregation
def aggregate(dev, per_seed, agg):
    """per_seed: list of dicts from fit_seed. Returns {fam: dev prob array}, and the parameters needed at eval."""
    prm = {}
    S = {}
    for f in FAMS:
        if agg == "probmean":
            S[f] = np.mean([r["Sdev"][f] for r in per_seed], axis=0)
            prm[f] = dict(mode="probmean", cal=[r["cal"][f] for r in per_seed])
        elif agg == "margmean":
            mm = np.mean([r["oof"][f] for r in per_seed], axis=0)
            a, b_ = platt(mm[dev.fam == f], dev.y[dev.fam == f])
            S[f] = sigmoid(a * mm + b_)
            prm[f] = dict(mode="margmean", cal=[a, b_])
        elif agg == "replatt":
            pm = np.mean([r["Sdev"][f] for r in per_seed], axis=0)
            z = np.log(np.clip(pm, 1e-9, 1 - 1e-9) / np.clip(1 - pm, 1e-9, 1))
            a, b_ = platt(z[dev.fam == f], dev.y[dev.fam == f])
            S[f] = sigmoid(a * z + b_)
            prm[f] = dict(mode="replatt", cal=[r["cal"][f] for r in per_seed], post=[a, b_])
        else:
            raise SystemExit(f"unknown agg {agg}")
    return S, prm


def _sdir(sd):
    return CACHE / f"exp050_seed{sd}"


def run_seed(sd):
    """Fit ONE seed end to end and persist: OOF margins + Platt (npz) and the full-dev models (LightGBM text)."""
    t0 = time.time()
    F, dev, XS, s1cfg = _build()
    r = fit_seed(F, dev, XS, s1cfg, sd, fit_full=True)
    d = _sdir(sd)
    d.mkdir(parents=True, exist_ok=True)
    store = {}
    for f in FAMS:
        store[f"oof|{f}"] = r["oof"][f]
        store[f"cal|{f}"] = np.array(r["cal"][f])
        M = r["M"][f]
        M["m1"].save_model(str(d / f"m1_{f}.txt"))
        M["m2"].save_model(str(d / f"m2_{f}.txt"))
        if M["mseg"] is not None:
            M["mseg"].save_model(str(d / f"mseg_{f}.txt"))
    np.savez(CACHE / f"exp050_seed{sd}.npz", **store)
    (d / "meta.json").write_text(json.dumps(dict(seed=sd, oof_map5=r["oof_map"], cal=r["cal"],
                                                 chosen=V4.CHOSEN, s1cfg=s1cfg), indent=1))
    EV.write_scores(F.select("pair_id", "hand_idx", "pot_bb"), r["Sdev"],
                    C.DER / f"evidence_scores_dev_exp050seed{sd}.parquet", keep_all=True)
    log(f"seed {sd} complete in {time.time() - t0:.0f}s: {r['oof_map']}")


def _load_seed(sd):
    z = np.load(CACHE / f"exp050_seed{sd}.npz")
    return dict(oof={f: z[f"oof|{f}"] for f in FAMS}, cal={f: list(z[f"cal|{f}"]) for f in FAMS})


def _load_models(sd):
    d = _sdir(sd)
    M = {}
    for f in FAMS:
        cfg = V4.CHOSEN[f]
        M[f] = dict(s=cfg["s"], use_seg=cfg["use_seg"], obj=cfg["obj"], pooled=True,
                    m1=lgb.Booster(model_file=str(d / f"m1_{f}.txt")),
                    m2=lgb.Booster(model_file=str(d / f"m2_{f}.txt")),
                    mseg=lgb.Booster(model_file=str(d / f"mseg_{f}.txt")) if cfg["use_seg"] else None)
    return M


def run_combine(seeds):
    t0 = time.time()
    F, _ = V2.build_dev()
    dev = EV.Dev(F)
    per_seed = []
    for sd in seeds:
        r = _load_seed(sd)
        r["Sdev"] = {f: sigmoid(r["cal"][f][0] * r["oof"][f] + r["cal"][f][1]) for f in FAMS}
        per_seed.append(r)
    keys = F.select("pair_id", "hand_idx", "pot_bb")
    report = {"seeds": list(seeds),
              "per_seed_oof_map5": {str(sd): {EV.SHORT[f]: round(dev.map5(per_seed[i]["oof"][f], f), 4) for f in FAMS}
                                    for i, sd in enumerate(seeds)}}
    for agg in ("probmean", "margmean", "replatt"):
        S, _ = aggregate(dev, per_seed, agg)
        EV.write_scores(keys, S, C.DER / f"evidence_scores_dev_exp050bag_{agg}.parquet", keep_all=True)
        report[f"bag_{agg}_oof_map5"] = {EV.SHORT[f]: round(dev.map5(S[f], f), 4) for f in FAMS}
    (CACHE / "exp050_bag_dev.json").write_text(json.dumps(report, indent=1))
    log(f"combine done in {time.time() - t0:.0f}s: {json.dumps(report)}")


def run_eval(agg, seeds, tables_per_batch=20):
    t0 = time.time()
    F, _ = V2.build_dev()
    dev = EV.Dev(F)
    res0 = json.loads((CACHE / "cv_results.json").read_text())
    s1cfg = {f: (res0["best007_single"][f]["stage1_tag"], res0["best007_single"][f]["stage1_rounds"]) for f in FAMS}
    per_seed = []
    for sd in seeds:
        r = _load_seed(sd)
        r["Sdev"] = {f: sigmoid(r["cal"][f][0] * r["oof"][f] + r["cal"][f][1]) for f in FAMS}
        r["M"] = _load_models(sd)
        per_seed.append(r)
    Sdev, prm = aggregate(dev, per_seed, agg)
    log(f"bag dev MAP5 {{{', '.join(f'{EV.SHORT[f]}: {dev.map5(Sdev[f], f):.4f}' for f in FAMS)}}}")
    del F, dev

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
        res2 = json.loads((CACHE / "cv_results_v2.json").read_text())
        XE = _fsets(E, res2["features_base"])
        kk = E.select("pair_id", "hand_idx", "pot_bb")
        SE = {}
        for f in FAMS:
            per = []
            for i, sd in enumerate(seeds):
                d = per_seed[i]["M"][f]
                Xm = XE[d["s"]]
                s1 = sigmoid(predict(d["m1"], Xm, "binary", True, f, s1cfg[f][1]))
                bl = [Xm, chrono_features(E, s1, f).to_numpy().astype(np.float32)]
                if d["use_seg"]:
                    bl.append(V2.seg_features(E, s1, d["mseg"].predict(Xm)).to_numpy().astype(np.float32))
                mg = predict(d["m2"], np.hstack(bl), d["obj"], False, f)
                if prm[f]["mode"] == "probmean" or prm[f]["mode"] == "replatt":
                    a, b_ = prm[f]["cal"][i]
                    per.append(sigmoid(a * mg + b_))
                else:
                    per.append(mg)
            m = np.mean(per, axis=0)
            if prm[f]["mode"] == "margmean":
                a, b_ = prm[f]["cal"]
                SE[f] = sigmoid(a * m + b_)
            elif prm[f]["mode"] == "replatt":
                z = np.log(np.clip(m, 1e-9, 1 - 1e-9) / np.clip(1 - m, 1e-9, 1))
                a, b_ = prm[f]["post"]
                SE[f] = sigmoid(a * z + b_)
            else:
                SE[f] = m
        outs.append(EV.write_scores(kk, SE, None))
        n_rows += len(kk)
        del E, XE
        log(f"eval tables {tb}-{tb + tables_per_batch - 1}: {len(kk):,} rows, {time.time() - t:.0f}s")
    assert n_pairs == ep.height, (n_pairs, ep.height)
    out = pl.concat(outs).sort("pair_id", "hand_idx")
    out.write_parquet(C.DER / f"evidence_scores_eval_exp050bag_{agg}.parquet")
    log(f"wrote evidence_scores_eval_exp050bag_{agg}.parquet: {out.height:,} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "seed":
        run_seed(int(sys.argv[2]))
    elif stage == "combine":
        run_combine([int(x) for x in sys.argv[2].split(",")])
    elif stage == "eval":
        run_eval(sys.argv[2], [int(x) for x in sys.argv[3].split(",")])
    else:
        raise SystemExit(__doc__)
