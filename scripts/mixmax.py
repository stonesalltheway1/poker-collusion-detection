"""Combine a pooled pair model with per-family SPECIALIST models on a common probability scale.

Two combiners, both written as a new oof/<out>_oof_w2a.parquet so scripts/eval_views.py and scripts/boot.py
can score them exactly like any other experiment:

  max   s = max( qmap(base), lam * p_spec )
        The base is a RANK vector (scripts/fuse.py output), so it is first quantile-mapped onto the
        probability scale of `--ref` (a single LightGBM member) before the max; that mapping is monotone,
        so with lam = 0 the candidate reproduces the base ranking exactly.
  sum   s = sum_f p_f  over the three family specialists -- the mixture identity
        P(pair is a positive) = P(DT) + P(SP) + P(CI); no cross-model scale assumption.

Usage:
  python scripts/mixmax.py max <out> <base> <ref> <spec> <lam>
  python scripts/mixmax.py sum <out> <spec_dt> <spec_sp> <spec_ci>
  python scripts/mixmax.py rmix <out> <base> <spec_dt> <spec_sp> <spec_ci> <w>   rank-blend base with the sum
"""
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import common as C  # noqa: E402


def load(e):
    d = pl.read_parquet(C.OOF_DIR / f"{e}_oof_w2a.parquet")
    if "y_orig" in d.columns:
        d = d.drop("y").rename({"y_orig": "y"})
    return d.sort("pA", "pB")


def rk(v):
    return np.argsort(np.argsort(v)) / (len(v) - 1)


def write(out, frame, s):
    keep = ["pA", "pB", "pair_id", "table_idx", "fold", "n", "y", "is_lab", "behavior_family",
            *[f"P_{f}" for f in C.FAMILIES]]
    frame.select([c for c in keep if c in frame.columns]).with_columns(
        pl.Series("oof", s), pl.col("y").alias("y_orig")).write_parquet(C.OOF_DIR / f"{out}_oof_w2a.parquet")
    print("wrote", C.OOF_DIR / f"{out}_oof_w2a.parquet")


def load_ev(e):
    return pl.read_parquet(C.OOF_DIR / f"{e}_eval_scores.parquet").sort("pair_id")


def write_ev(out, base_ev, risk):
    """Eval-side twin of write(); behaviour posteriors are taken from base_ev unchanged."""
    base_ev.with_columns(pl.Series("risk_raw", risk)).select(
        "pair_id", "risk_raw", *[f"P_{f}" for f in C.FAMILIES]).write_parquet(
        C.OOF_DIR / f"{out}_eval_scores.parquet")
    print("wrote", C.OOF_DIR / f"{out}_eval_scores.parquet")


def main():
    mode, out = sys.argv[1], sys.argv[2]
    if mode == "max":
        base, ref, spec, lam = sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6])
        b, r, p = load(base), load(ref), load(spec)
        assert (b["pA"] == r["pA"]).all() and (b["pA"] == p["pA"]).all()
        v = b["oof"].to_numpy()
        n = len(v)
        qs = np.sort(r["oof"].to_numpy())
        pb = qs[np.clip((rk(v) * (n - 1)).astype(int), 0, n - 1)]
        write(out, b, np.maximum(pb, lam * p["oof"].to_numpy()))
        return
    if mode == "sum":
        specs = [load(e) for e in sys.argv[3:6]]
        t = sum(d["oof"].to_numpy() for d in specs)
        write(out, specs[0], t)
        if "--eval" in sys.argv:
            ev = [load_ev(e) for e in sys.argv[3:6]]
            assert all((e["pair_id"] == ev[0]["pair_id"]).all() for e in ev)
            write_ev(out, ev[0], sum(e["risk_raw"].to_numpy() for e in ev))
        return
    # rmix <out> <base> <spec_dt> <spec_sp> <spec_ci> <w>
    base, w = sys.argv[3], float(sys.argv[7])
    specs = [load(e) for e in sys.argv[4:7]]
    t = sum(d["oof"].to_numpy() for d in specs)
    b = load(base)
    assert (b["pA"] == specs[0]["pA"]).all()
    write(out, b, (1 - w) * rk(b["oof"].to_numpy()) + w * rk(t))
    if "--eval" in sys.argv:
        ev = [load_ev(e) for e in sys.argv[4:7]]
        be = load_ev(base)
        assert all((e["pair_id"] == be["pair_id"]).all() for e in ev)
        se = sum(e["risk_raw"].to_numpy() for e in ev)
        write_ev(out, be, (1 - w) * rk(be["risk_raw"].to_numpy()) + w * rk(se))


if __name__ == "__main__":
    main()
