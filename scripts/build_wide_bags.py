"""Build wide-eval-bag pair risk files from scripts/eval_seed_fits.py outputs.

The dev-side OOF (view C'/C2, boot.py) is carried over VERBATIM from the incumbent bag, because
this change is eval-side only: the 5-fold OOF models are unchanged and deterministic, so the dev
protocol cannot see the change.  What changes is how many independent full-data fits the shipped
eval risk averages over.

Usage: python scripts/build_wide_bags.py
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import common as C  # noqa: E402

FAM = list(C.FAMILIES)


def rank01(v):
    return (pl.Series(v).rank() / len(v)).to_numpy()


def widen(out_tag, incumbent, seedfile, seeds):
    """eval risk = rank-mean over `seeds`; everything else inherited from `incumbent`."""
    sf = pl.read_parquet(C.OOF_DIR / seedfile).sort("pair_id")
    inc = pl.read_parquet(C.OOF_DIR / f"{incumbent}_eval_scores.parquet").sort("pair_id")
    assert (sf["pair_id"].to_numpy() == inc["pair_id"].to_numpy()).all()
    cols = [f"r{s}" for s in seeds]
    missing = [c for c in cols if c not in sf.columns]
    assert not missing, f"missing seeds {missing} in {seedfile}"
    risk = np.mean([rank01(sf[c].to_numpy()) for c in cols], axis=0)
    ev = inc.select("pair_id", *[f"P_{f}" for f in FAM]).with_columns(pl.Series("risk_raw", risk))
    ev.select("pair_id", "risk_raw", *[f"P_{f}" for f in FAM]).write_parquet(
        C.OOF_DIR / f"{out_tag}_eval_scores.parquet")
    # dev side: unchanged, copied so eval_views.py / boot.py / fuse.py work on the new tag
    pl.read_parquet(C.OOF_DIR / f"{incumbent}_oof_w2a.parquet").write_parquet(
        C.OOF_DIR / f"{out_tag}_oof_w2a.parquet")
    print(f"{out_tag}: eval risk = rank-mean of {len(cols)} fits {cols}; "
          f"dev OOF copied from {incumbent}")
    return risk


def main():
    seeds = list(range(42, 51))
    x = widen("exp054_xgbag9", "exp047_xgbag", "exp054_xgb_evalseeds.parquet", seeds)
    lgb_file = C.OOF_DIR / "exp053_lgb_evalseeds.parquet"
    if lgb_file.exists():
        l = widen("exp053_bagsf9", "exp035_bagsf", "exp053_lgb_evalseeds.parquet", seeds)
    else:
        print("NOTE: no LightGBM seed file; LGB half kept at the incumbent exp035_bagsf (3 fits)")
        inc = pl.read_parquet(C.OOF_DIR / "exp035_bagsf_eval_scores.parquet").sort("pair_id")
        l = rank01(inc["risk_raw"].to_numpy())
        pl.read_parquet(C.OOF_DIR / "exp035_bagsf_oof_w2a.parquet").write_parquet(
            C.OOF_DIR / "exp053_bagsf9_oof_w2a.parquet")
        inc.select("pair_id", "risk_raw", *[f"P_{f}" for f in FAM]).write_parquet(
            C.OOF_DIR / "exp053_bagsf9_eval_scores.parquet")

    # nested 50/50, exactly the exp048 recipe
    inc48 = pl.read_parquet(C.OOF_DIR / "exp048_fuse_bags_eval_scores.parquet").sort("pair_id")
    fused = (rank01(l) + rank01(x)) / 2
    inc48.select("pair_id", *[f"P_{f}" for f in FAM]).with_columns(
        pl.Series("risk_raw", fused)).select(
        "pair_id", "risk_raw", *[f"P_{f}" for f in FAM]).write_parquet(
        C.OOF_DIR / "exp055_fuse_bags9_eval_scores.parquet")
    pl.read_parquet(C.OOF_DIR / "exp048_fuse_bags_oof_w2a.parquet").write_parquet(
        C.OOF_DIR / "exp055_fuse_bags9_oof_w2a.parquet")
    r48 = rank01(inc48["risk_raw"].to_numpy())
    print(f"exp055_fuse_bags9: nested 50/50; rank corr vs shipped exp048 = "
          f"{np.corrcoef(fused, r48)[0, 1]:.8f}")
    for K in (100, 500, 1000):
        a = set(np.argsort(-fused)[:K].tolist())
        b = set(np.argsort(-r48)[:K].tolist())
        print(f"  top-{K} overlap with exp048: {len(a & b)}/{K}")


if __name__ == "__main__":
    main()
