"""exp031: transductive self-training of the pair model.

Idea: the model is trained on development-period rows but must rank EVALUATION-period rows, where positives sit at
lower exposure. Take the base model's top-K target-window pairs as pseudo-positives, add the target window's rows
to training (pseudo-positives at weight W_PL, the rest as low-weight negatives) and refit. No true label of a
target row is ever used, so evaluating against true labels stays valid (standard transductive protocol).

Modes:
  dev   target = dev_w2000a  -> validates the PROCEDURE against true labels (view C'/C2)
  eval  target = eval window -> produces oof/<exp>_eval_scores.parquet for submission

Usage: python src/exp031_transductive.py <mode> <base_exp> <out_exp> [K] [W_PL] [W_U2]
Env: same feature switches as exp005_pairmodel (PF_TAG, PFH_TAG, USE_OC, USE_MIX, ADD_SPIES_POS, CFG...).
"""
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MODE, BASE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 400
W_PL = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
W_U2 = float(sys.argv[6]) if len(sys.argv) > 6 else 0.05
# exp005_pairmodel parses argv at import time -> hand it the shape it expects
sys.argv = [sys.argv[0], OUT, os.environ.get("W_U", "0.1"), os.environ.get("ROUNDS", "600"), "all3"]
import exp005_pairmodel as M  # noqa: E402
FAM = list(C.FAMILIES)


def main():
    t0 = time.time()
    lab, full, w2a, ev, w2b = M.load()
    feats = M.feature_cols(full)
    keep_full = pl.col("is_lab") | ((~pl.col("touch_pos")) & (pl.col("n") >= 57))
    keep_w = pl.col("is_lab") | (~pl.col("touch_pos"))
    dev_parts = [full.filter(keep_full), w2b.filter(keep_w)] if w2b is not None else [full.filter(keep_full)]

    # ---- target window + base scores -> pseudo-labels
    if MODE == "dev":
        tgt = w2a
        base = (pl.read_parquet(C.OOF_DIR / f"{BASE}_oof_w2a.parquet").select("pA", "pB", pl.col("oof").alias("base")))
    else:
        ep = C.load("eval_pairs").select("pair_id", pl.col("p1").alias("pA"), pl.col("p2").alias("pB"))
        tgt = ep.join(ev, on=["pA", "pB"], how="left").drop("pair_id_right", strict=False)
        base = (pl.read_parquet(C.OOF_DIR / f"{BASE}_eval_scores.parquet")
                .join(ep, on="pair_id").select("pA", "pB", pl.col("risk_raw").alias("base")))
    tgt = tgt.join(base, on=["pA", "pB"], how="inner")   # base scores exist for the evaluated pool only
    assert tgt["base"].null_count() == 0, tgt["base"].null_count()
    thr = tgt["base"].sort(descending=True)[K - 1]
    tgt = tgt.with_columns((pl.col("base") >= thr).alias("pl_pos"))
    print(f"[{MODE}] target rows {tgt.height:,}; pseudo-positives {int(tgt['pl_pos'].sum())} (threshold {thr:.4g})")
    if MODE == "dev":
        print("  of which truly positive:", int(tgt.filter(pl.col("pl_pos"))["y"].sum()),
              "| true positives in window:", int(tgt["y"].sum()))

    tgt_train = tgt.with_columns(pl.when(pl.col("pl_pos")).then(1).otherwise(0).alias("y_pl"))
    train = pl.concat([p.with_columns(pl.lit(False).alias("pl_pos"), pl.col("y").alias("y_pl")) for p in dev_parts]
                      + [tgt_train], how="diagonal_relaxed")
    isl = train["is_lab"].fill_null(False).to_numpy()
    ispl = train["pl_pos"].fill_null(False).to_numpy()
    w = np.where(isl, 1.0, np.where(ispl, W_PL, W_U2))
    if "is_spy" in train.columns:
        w = np.where(train["is_spy"].fill_null(False).to_numpy(), 0.5, w)
    X = train.select(feats).to_numpy().astype(np.float32)
    y = train["y_pl"].to_numpy()
    fold = train["fold"].to_numpy()
    Xt = tgt.select(feats).to_numpy().astype(np.float32)
    tfold = tgt["fold"].to_numpy()
    pred = np.zeros(tgt.height)
    for k in range(5):
        tr = fold != k
        m = lgb.train(M.PARAMS, lgb.Dataset(X[tr], y[tr], weight=w[tr], feature_name=feats), M.ROUNDS)
        pred[tfold == k] = m.predict(Xt[tfold == k])
        print(f"  fold {k} {time.time() - t0:.0f}s", flush=True)
    tgt = tgt.with_columns(pl.Series("oof", pred))

    if MODE == "dev":
        sp = pl.read_parquet(C.DER / "spies_exp005.parquet").select("pA", "pB")
        keep = ["pA", "pB", "pair_id", "table_idx", "fold", "n", "y", "is_lab", "behavior_family", "oof"]
        if "y_orig" in tgt.columns:
            keep.insert(7, "y_orig")
        pool = tgt.filter(pl.col("is_lab") | ~pl.col("touch_pos"))
        for f in FAM:
            if f"P_{f}" not in pool.columns:
                pool = pool.with_columns(pl.lit(1 / 3).alias(f"P_{f}"))
        pool.select(*keep, *[f"P_{f}" for f in FAM]).write_parquet(C.OOF_DIR / f"{OUT}_oof_w2a.parquet")
        yy = pool["y_orig"].to_numpy() if "y_orig" in pool.columns else pool["y"].to_numpy()
        print(json.dumps({"C_ap": round(C._ap(yy, pool["oof"].to_numpy()), 4),
                          "n": pool.height, "pos": int(yy.sum())}))
    else:
        base_scores = pl.read_parquet(C.OOF_DIR / f"{BASE}_eval_scores.parquet")
        outp = (tgt.select("pA", "pB", pl.col("oof").alias("risk_raw"))
                .join(C.load("eval_pairs").select("pair_id", pl.col("p1").alias("pA"), pl.col("p2").alias("pB")),
                      on=["pA", "pB"]).join(base_scores.select("pair_id", *[f"P_{f}" for f in FAM]), on="pair_id"))
        outp.select("pair_id", "risk_raw", *[f"P_{f}" for f in FAM]).write_parquet(C.OOF_DIR / f"{OUT}_eval_scores.parquet")
        print("wrote", f"{OUT}_eval_scores.parquet", outp.height,
              "| rank corr with base:", round(float(np.corrcoef(
                  outp["risk_raw"].rank().to_numpy(), base_scores.join(outp.select("pair_id"), on="pair_id")["risk_raw"].rank().to_numpy())[0, 1]), 4))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
