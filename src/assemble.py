"""Assemble a submission from component files.

risk      <exp>_eval_scores.parquet: pair_id, risk_raw, P_<family>   (behaviour = argmax family, never 'none')
evidence  one of:
  rule:<path>     long format pair_id, fam, k, hand_idx (per-family ranked lists; routed by predicted family)
  scores:<path>   pair_id, hand_idx, s_<family> (mixture by posteriors via src/evidence.compose_evidence)
Ties in risk are broken by a secondary legitimate model score (never pair_id).
Usage: python src/assemble.py <sub_id> <risk_exp> <evidence_spec> [tiebreak_exp004]
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

FAM = list(C.FAMILIES)


def risk_frame(risk_exp):
    r = pl.read_parquet(C.OOF_DIR / f"{risk_exp}_eval_scores.parquet")
    sec = pl.read_parquet(C.OOF_DIR / "exp004_model_eval.parquet").select("pair_id", pl.col("s_z").alias("sec"))
    r = r.join(sec, on="pair_id", how="left").with_columns(pl.col("sec").fill_null(0.0))
    r = r.sort(["risk_raw", "sec"]).with_row_index("rk").with_columns(
        ((pl.col("rk") + 1) / (pl.len() + 1)).alias("risk_score"))
    post = r.select([f"P_{f}" for f in FAM]).to_numpy()
    return r.with_columns(pl.Series("predicted_behavior", np.array(FAM)[post.argmax(1)]))


def evidence_rule(path, r):
    ev = pl.read_parquet(path)
    hid = C.load("hands").select("hand_idx", "hand_id")
    ev = (ev.join(r.select("pair_id", pl.col("predicted_behavior").alias("fam")), on=["pair_id", "fam"])
          .join(hid, on="hand_idx").sort("pair_id", "k"))
    wide = ev.group_by("pair_id", maintain_order=True).agg(pl.col("hand_id").head(5))
    return wide.with_columns([pl.col("hand_id").list.get(i, null_on_oob=True).fill_null(C.NO_EV).alias(c)
                              for i, c in enumerate(C.EV_COLS)]).drop("hand_id")


def evidence_scores(path, r):
    import evidence as E
    sc = pl.read_parquet(path)
    post = r.select("pair_id", *[pl.col(f"P_{f}") for f in FAM])
    return E.compose_evidence(sc, post)


def main():
    sub_id, risk_exp, ev_spec = sys.argv[1], sys.argv[2], sys.argv[3]
    r = risk_frame(risk_exp)
    kind, path = ev_spec.split(":", 1)
    wide = evidence_rule(path, r) if kind == "rule" else evidence_scores(path, r)
    if isinstance(wide, pl.DataFrame):
        wide = wide.to_pandas()
    sub = (r.select("pair_id", "risk_score", "predicted_behavior").to_pandas()
           .merge(wide, on="pair_id", how="left"))
    sub = C.finalize(sub)
    ep = C.load("sample_submission") if (C.DER / "sample_submission.parquet").exists() else None
    path_out, stats = C.write_submission(sub_id, sub)
    print(path_out, stats)
    print(sub.predicted_behavior.value_counts().to_dict())


if __name__ == "__main__":
    main()
