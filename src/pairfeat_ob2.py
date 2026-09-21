"""ob2 — COUNT-AWARE co-player comparison features for the soft-play signature set.

Motivation (Lane B2 diagnosis, 2026-09-20).  On view C' the champion scores soft_play .8817 against
directed_transfer .9602 / coordinated_isolation .9660, and the loss is concentrated in 15 of 117 SP
positives whose median shared-hand count is 57 (vs 90 for the well-ranked ones).  Ranking those 15 by
feature percentile against exposure-matched negatives, the single largest discriminator is the
`ob_top_any_sp_*` block — "is the partner the top co-player of this player on signature k" — on which the
badly-ranked pairs sit at the 6th-20th percentile while the well-ranked ones sit at the 60th-92nd.

`ob_*` (src/pairfeat.own_baseline) compares co-players by a smoothed log-lift log((c+1)/(n*rate+1)) and by
its ORDINAL RANK.  Both are count-blind: at 40-60 shared hands one extra hit flips the rank.  This module
re-does the same comparison in binomial-z space and adds the two count-aware quantities own_baseline has
no form of: the GAP to the best rival co-player, and a soft "probability Y is X's top partner".

It reads the existing data/derived/pf<PF_TAG>_<window>.parquet (the c_* count columns are already there,
they are merely dropped at model time) — no engine re-run.

Output: data/derived/pfob2_<window>.parquet  (pA, pB, ob2_*)
Usage:  python src/pairfeat_ob2.py [window ...]        default: all four windows
"""
import os
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

WINDOWS = ["dev_full", "dev_w2000a", "dev_w2000b", "eval"]
PF_TAG = os.environ.get("PF_TAG", "v2")
# the eleven symmetric soft-play signature flags (src/pairfeat.hand_flags); only four of them
# (g1, pl, pl_soft, strongcheck) ever received an own_baseline co-player comparison.
KEYS = ["sp_g0", "sp_g1", "sp_pl", "sp_strongcheck", "sp_foldpost_hs60", "sp_callpost_hs80",
        "sp_callpre_eq55", "sp_hu_checked", "sp_u1", "sp_pl_soft", "sp_raiseback"]


def build(window):
    t0 = time.time()
    src = C.DER / f"pf{PF_TAG}_{window}.parquet"
    cols = ["pA", "pB", "n"] + [f"c_{k}" for k in KEYS]
    agg = pl.read_parquet(src, columns=cols)
    L = pl.concat([
        agg.select(pl.col("pA").alias("X"), pl.col("pB").alias("Y"), "n",
                   *[pl.col(f"c_{k}").alias(k) for k in KEYS]),
        agg.select(pl.col("pB").alias("X"), pl.col("pA").alias("Y"), "n",
                   *[pl.col(f"c_{k}").alias(k) for k in KEYS])])
    L = L.unpivot(index=["X", "Y", "n"], on=KEYS, variable_name="k", value_name="c")
    # leave-one-out baseline rate of signature k for player X over ALL his other co-players
    L = L.with_columns(pl.col("c").sum().over("X", "k").alias("CX"),
                       pl.col("n").sum().over("X", "k").alias("NX"))
    rate = ((pl.col("CX") - pl.col("c")) / (pl.col("NX") - pl.col("n")).clip(lower_bound=1)).clip(1e-6, 1 - 1e-6)
    L = L.with_columns(
        ((pl.col("c") - pl.col("n") * rate) / (pl.col("n") * rate * (1 - rate)).sqrt()).alias("z"))
    # rank, gap to the best RIVAL co-player, and a softmax share -- all inside (X, k)
    top1 = pl.col("z").max().over("X", "k")
    top2 = pl.col("z").top_k(2).min().over("X", "k")      # second largest z among X's co-players
    L = L.with_columns(
        pl.col("z").rank("ordinal", descending=True).over("X", "k").alias("zrk"),
        pl.when(pl.col("z") >= top1).then(pl.col("z") - top2).otherwise(pl.col("z") - top1).alias("zgap"),
        (pl.col("z").exp() / pl.col("z").exp().sum().over("X", "k")).alias("zsh"))
    L = L.with_columns(pl.min_horizontal("X", "Y").alias("pA"), pl.max_horizontal("X", "Y").alias("pB"))
    W = L.group_by("pA", "pB", "k").agg(
        pl.col("z").min().alias("zmin"), pl.col("z").max().alias("zmax"), pl.col("z").sum().alias("zsum"),
        pl.col("zgap").min().alias("zgapmin"), pl.col("zgap").max().alias("zgapmax"),
        pl.col("zsh").min().alias("zshmin"), pl.col("zsh").max().alias("zshmax"),
        pl.col("zrk").max().alias("zrkworst"))
    wide = W.pivot(on="k", index=["pA", "pB"],
                   values=["zmin", "zmax", "zsum", "zgapmin", "zgapmax", "zshmin", "zshmax", "zrkworst"])
    wide = wide.rename({c: "ob2_" + c for c in wide.columns if c not in ("pA", "pB")})
    out = C.DER / f"pfob2_{window}.parquet"
    wide.write_parquet(out)
    print(f"[{window}] pairs {wide.height:,} cols {wide.width} in {time.time() - t0:.1f}s -> {out.name}", flush=True)


if __name__ == "__main__":
    for w in (sys.argv[1:] or WINDOWS):
        build(w)
