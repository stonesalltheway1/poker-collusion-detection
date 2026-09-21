"""exp015: hand-level planted-behaviour detector + top-k pair aggregation.

Model: LightGBM 4-class on dev pair-hand rows of labelled pairs: class 0 = shared hands of confirmed-negative pairs,
class 1/2/3 = listed evidence hands of DT/SP/CI pairs. Non-listed hands of positive pairs are unknown -> excluded.
Features: orientation-free transforms of the engine row (ph_v2): symmetric columns, max/min over directions, and
donor/receiver-oriented copies (donor = the player with net<0 when the other has net>0, else larger gift).
Cross-fitting: model k is trained on tables with fold != k and scores ALL pair-hand rows of fold-k tables (dev).
Eval rows are scored with the mean of the 5 fold models.
Output: data/derived/hs/phase{0,1}/part_XX.parquet  (hand_idx, pA, pB, table_idx, hand_seq, s_dt, s_sp, s_ci, s_any)
Then `agg` builds pair features per window -> data/derived/pfh_<window>.parquet
Usage: python src/handdet.py train | score | agg | all
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
import engine as E  # noqa: E402

PH = C.DER / "ph_v2"
HS = C.DER / os.environ.get("HS_IN", "hs")
HS_OUT = C.DER / os.environ.get("HS_OUT", os.environ.get("HS_IN", "hs"))
MOD = C.DER / "handdet_models"
FAMS = ["directed_transfer", "soft_play", "coordinated_isolation"]
SC = ["s_dt", "s_sp", "s_ci"]
NTHR = int(os.environ.get("NTHR", "5"))
col = pl.col


def hand_features(lf):
    """orientation-free feature expressions over an engine-row frame."""
    flow_ab = (col("net_bb_ab") < 0) & (col("net_bb_ba") > 0)
    flow_ba = (col("net_bb_ba") < 0) & (col("net_bb_ab") > 0)
    donor_is_a = flow_ab | (~flow_ba & (col("gift_ab") >= col("gift_ba")))
    ex = [col(c).alias(c) for c in E.SYM]
    for f in E.DIR:
        a, b = col(f"{f}_ab"), col(f"{f}_ba")
        ex += [pl.max_horizontal(a, b).alias(f"mx_{f}"), pl.min_horizontal(a, b).alias(f"mn_{f}"),
               pl.when(donor_is_a).then(a).otherwise(b).alias(f"dn_{f}"),
               pl.when(donor_is_a).then(b).otherwise(a).alias(f"rc_{f}")]
    ex += [(flow_ab | flow_ba).cast(pl.Int8).alias("flow"),
           pl.when(flow_ab).then(pl.min_horizontal(-col("net_bb_ab"), col("net_bb_ba")))
           .when(flow_ba).then(pl.min_horizontal(-col("net_bb_ba"), col("net_bb_ab"))).otherwise(0.0).alias("transfer")]
    return lf.select("hand_idx", "pA", "pB", "table_idx", "hand_seq", *ex)


def feat_names():
    return list(E.SYM) + [f"{p}_{f}" for f in E.DIR for p in ("mx", "mn", "dn", "rc")] + ["flow", "transfer"]


def train():
    t0 = time.time()
    MOD.mkdir(parents=True, exist_ok=True)
    lab = C.load("labels").select(col("p1").alias("pA"), col("p2").alias("pB"), "pair_id", "label", "behavior_family")
    ev = C.load("evidence").select("pair_id", "hand_idx", pl.lit(1).alias("is_ev"))
    rows = (hand_features(pl.scan_parquet(PH / "phase0" / "*.parquet"))
            .join(lab.lazy(), on=["pA", "pB"]).join(ev.lazy(), on=["pair_id", "hand_idx"], how="left")
            .filter((col("label") == 0) | (col("label") == 1)).collect())
    folds = C.get_table_folds()
    rows = rows.join(folds, on="table_idx")
    rows = rows.with_columns(pl.col("is_ev").fill_null(0))
    # self-training (SELFTRAIN=1): non-listed hands of positive pairs whose previous-round OOF score exceeds the
    # negative-hand q999 are very likely UNLISTED PLANTED hands (~2x the listed ones per forensics) -> weak positives.
    keep = (col("label") == 0) | (col("is_ev") == 1)
    wt_extra = None
    if os.environ.get("SELFTRAIN") == "1":
        prev = pl.scan_parquet(HS / "phase0" / "*.parquet").select("hand_idx", "pA", "pB", "s_any", *SC)
        thr0 = json.loads((MOD / "thresholds.json").read_text())
        rows = rows.join(prev.collect(), on=["hand_idx", "pA", "pB"], how="left")
        fam_col = pl.coalesce([pl.when(col("behavior_family") == f).then(col(s)) for f, s in zip(FAMS, SC)])
        rows = rows.with_columns(fam_col.alias("s_fam"))
        extra = (col("label") == 1) & (col("is_ev") == 0) & (col("s_fam") > pl.lit(0.5))
        rows = rows.filter(keep | extra).with_columns(extra.cast(pl.Int8).alias("is_extra"))
        print("self-training extra positives:", int(rows["is_extra"].sum()), "listed:", int(rows["is_ev"].sum()))
    else:
        rows = rows.filter(keep).with_columns(pl.lit(0).cast(pl.Int8).alias("is_extra"))
    y = rows.select(pl.when(col("label") == 0).then(0).otherwise(
        col("behavior_family").replace_strict({f: i + 1 for i, f in enumerate(FAMS)}, default=0))).to_series().to_numpy()
    fn = feat_names()
    X = rows.select(fn).to_numpy().astype(np.float32)
    fold = rows["fold"].to_numpy()
    w = np.where(y == 0, 1.0, 25.0)
    w = np.where(rows["is_extra"].to_numpy() == 1, float(os.environ.get("W_EXTRA", "8")), w)
    params = dict(objective="multiclass", num_class=4, learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
                  feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=NTHR,
                  verbose=-1, seed=7)
    oof = np.zeros((len(y), 4))
    for k in range(5):
        tr = fold != k
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr], weight=w[tr], feature_name=fn), 400)
        m.save_model(str(MOD / f"fold{k}.txt"))
        oof[fold == k] = m.predict(X[fold == k])
        print(f"fold {k} {time.time() - t0:.0f}s", flush=True)
    res = {}
    for i, f in enumerate(FAMS, 1):
        sel = (y == 0) | (y == i)
        res[f"auc_{f}_vs_neg"] = roc_auc_score(y[sel] == i, oof[sel, i])
        other = (y > 0)
        res[f"auc_{f}_vs_otherfam"] = roc_auc_score(y[other] == i, oof[other, i])
    res["auc_any_vs_neg"] = roc_auc_score(y > 0, 1 - oof[:, 0])
    # thresholds on negative-hand OOF score quantiles (used for pair counts)
    thr = {}
    for i, s in enumerate(SC, 1):
        neg = oof[y == 0, i]
        thr[s] = {q: float(np.quantile(neg, q)) for q in (0.99, 0.999)}
        res[f"recall_ev_{s}_at_q999"] = float((oof[y == i, i] > thr[s][0.999]).mean())
    negany = 1 - oof[y == 0, 0]
    thr["s_any"] = {q: float(np.quantile(negany, q)) for q in (0.99, 0.999)}
    (MOD / "thresholds.json").write_text(json.dumps(thr, indent=1))
    (MOD / "oof_metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: round(v, 4) for k, v in res.items()}, indent=1))


def score():
    t0 = time.time()
    folds = C.get_table_folds()
    tab_fold = np.zeros(400, dtype=np.int64)
    tab_fold[folds["table_idx"].to_numpy()] = folds["fold"].to_numpy()
    models = [lgb.Booster(model_file=str(MOD / f"fold{k}.txt")) for k in range(5)]
    fn = feat_names()
    for phase in [int(x) for x in os.environ.get("PHASES", "0,1").split(",")]:
        outdir = HS_OUT / f"phase{phase}"
        outdir.mkdir(parents=True, exist_ok=True)
        for part in sorted((PH / f"phase{phase}").glob("part_*.parquet")):
            df = hand_features(pl.scan_parquet(part)).collect()
            X = df.select(fn).to_numpy().astype(np.float32)
            P = np.zeros((df.height, 4))
            if phase == 0:
                fk = tab_fold[df["table_idx"].to_numpy()]
                for k in range(5):
                    m = fk == k
                    if m.any():
                        P[m] = models[k].predict(X[m], num_threads=NTHR)
            elif os.environ.get("SAMEFOLD_EVAL") == "1":
                # score each eval table with the fold model that held that table out (same convention as dev OOF);
                # a 5-model mean thins the score tail vs the single-model dev scores the pair model learned from
                fk = tab_fold[df["table_idx"].to_numpy()]
                for k in range(5):
                    m = fk == k
                    if m.any():
                        P[m] = models[k].predict(X[m], num_threads=NTHR)
            else:
                for k in range(5):
                    P += models[k].predict(X, num_threads=NTHR) / 5
            out = df.select("hand_idx", "pA", "pB", "table_idx", "hand_seq").with_columns(
                pl.Series("s_dt", P[:, 1].astype(np.float32)), pl.Series("s_sp", P[:, 2].astype(np.float32)),
                pl.Series("s_ci", P[:, 3].astype(np.float32)), pl.Series("s_any", (1 - P[:, 0]).astype(np.float32)))
            out.write_parquet(outdir / part.name)
            print(f"phase {phase} {part.name} {df.height:,} rows {time.time() - t0:.0f}s", flush=True)


def _pop_quantiles(phase, cols, nq=201):
    """population quantile grid of each hand score (for exceedance p-values)."""
    sample = pl.scan_parquet(HS_OUT / f"phase{phase}" / "part_00.parquet").select(cols).collect()
    qs = np.linspace(0, 1, nq)
    return {c: np.quantile(sample[c].to_numpy(), qs) for c in cols}, qs


def agg():
    thr = json.loads((MOD / "thresholds.json").read_text())
    for window, (phase, lo, hi) in {"dev_full": (0, 0, 3000), "dev_w2000a": (0, 0, 2000),
                                    "dev_w2000b": (0, 1000, 3000), "eval": (1, 3000, 5000)}.items():
        t0 = time.time()
        lf = pl.scan_parquet(HS_OUT / f"phase{phase}" / "*.parquet").filter((col("hand_seq") >= lo) & (col("hand_seq") < hi))
        flags = []
        for s in SC + ["s_any"]:
            for q in ("0.99", "0.999"):
                flags.append((col(s) > thr[s][q]).cast(pl.Int32).alias(f"h_{s}_q{q[2:]}"))
        grid, qs = _pop_quantiles(phase, SC + ["s_any"])
        def _logit(c):
            return (col(c).clip(1e-6, 1 - 1e-6) / (1 - col(c).clip(1e-6, 1 - 1e-6))).log()
        def _mlogp(c):
            g, q = grid[c], qs
            return col(c).map_batches(
                lambda x, g=g, q=q: pl.Series(-np.log(np.clip(1.0 - np.interp(x.to_numpy(), g, q), 1e-6, 1.0))),
                return_dtype=pl.Float64)
        lf = lf.with_columns(flags + [_logit(c).alias(f"lg_{c}") for c in SC + ["s_any"]]
                             + [_mlogp(c).alias(f"mlp_{c}") for c in SC + ["s_any"]])
        aggs = [pl.len().alias("hn")]
        for c in SC + ["s_any"]:
            lg, mlp = col(f"lg_{c}"), col(f"mlp_{c}")
            aggs += [lg.clip(-6, 6).sum().alias(f"h_{c}_sumlogit"),
                     lg.clip(-6, 6).clip(lower_bound=0).sum().alias(f"h_{c}_sumposlogit"),
                     lg.sort(descending=True).head(3).sum().alias(f"h_{c}_top3sum"),
                     lg.sort(descending=True).slice(1, 1).sum().alias(f"h_{c}_second"),
                     mlp.sum().alias(f"h_{c}_fisher"), mlp.max().alias(f"h_{c}_mlpmax")]
        for s in SC + ["s_any"]:
            srt = col(s).sort(descending=True)
            aggs += [srt.first().alias(f"h_{s}_top1"), srt.head(3).mean().alias(f"h_{s}_top3"),
                     srt.head(5).mean().alias(f"h_{s}_top5"), srt.head(10).mean().alias(f"h_{s}_top10"),
                     col(s).mean().alias(f"h_{s}_mean")]
        aggs += [col(c.meta.output_name()).sum() for c in flags]
        pf = lf.group_by("pA", "pB").agg(aggs).collect(engine="streaming")
        n = col("hn").cast(pl.Float64)
        ex = [((col(f"h_{c}_fisher") - n) / (n.sqrt() + 1e-9)).alias(f"h_{c}_fisher_z") for c in SC + ["s_any"]]
        for c in flags:
            name = c.meta.output_name()
            p0 = pf[name].sum() / pf["hn"].sum()
            ex += [(col(name) / n).alias(f"{name}_r"),
                   ((col(name) - n * p0) / (n * p0 * (1 - p0) + 1e-9).sqrt()).alias(f"{name}_z")]
        pf = pf.with_columns(ex)
        # double-centring: pair top5 minus the mean top5 of each player's other pairs
        for s in SC + ["s_any"]:
            t5 = f"h_{s}_top5"
            long = pl.concat([pf.select(col("pA").alias("X"), col("pB").alias("Y"), col(t5)),
                              pf.select(col("pB").alias("X"), col("pA").alias("Y"), col(t5))])
            long = long.with_columns(((col(t5).sum().over("X") - col(t5)) / (pl.len().over("X") - 1)).alias("mo"))
            long = long.with_columns(pl.min_horizontal("X", "Y").alias("pA"), pl.max_horizontal("X", "Y").alias("pB"))
            dc = long.group_by("pA", "pB").agg((col(t5).first() - col("mo").max()).alias(f"{t5}_dcmin"),
                                               (col(t5).first() - col("mo").mean()).alias(f"{t5}_dcmean"))
            pf = pf.join(dc, on=["pA", "pB"], how="left")
        pf.write_parquet(C.DER / f"pfh{os.environ.get('PFH_TAG', '')}_{window}.parquet")
        print(f"[{window}] {pf.height:,} pairs {pf.width} cols {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st in ("train", "all"):
        train()
    if st in ("score", "all"):
        score()
    if st in ("agg", "all"):
        agg()
