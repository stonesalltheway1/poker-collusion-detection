"""Build data/derived/seat_strength.parquet (12M seat rows) + data/derived/preflop169.parquet.

Run:  python src/build_seat_strength.py            (~1-2 min, <2 GB RAM, 3 threads)

seat_strength.parquet  (sorted by hand_idx, seat_no; one row per dealt seat)
  hand_idx i32, player_idx i32, seat_no i8
  preflop_class i16          0..168 13x13 grid index (see handeval.preflop_class)
  pf_eq_vs1, pf_eq_vs5 f32   class-level MC all-in equity vs 1 / 5 random hands (from preflop169)
  rank_flop, rank_turn, final_rank_7   i16  dense rank 0..7461 (higher better) of hole + board prefix
                                            (3/4/5 cards); null when the hand's board never reached that street
  cat_flop, cat_turn, cat_river        i8   made-hand category 0..8 of the above; null likewise
  n_opp_better_river i8      # of the other 5 dealt players (folded included) with a strictly better 7-card hand
  n_opp_tied_river i8        # of other dealt players with an identical rank
  best_hand_at_table bool    n_opp_better_river == 0 ("would win/chop at showdown had nobody folded"); null w/o river
  last_board_n i8            cards on the final board of the hand (0,3,4,5)
  rank_last i16              rank on the final board (= rank_flop/turn/final_rank_7 for that n); null if no board
  n_opp_better_last i8, best_at_last_board bool   same comparison on the final board; null if no board
  fold_street i8             street (0..3) where this player folded; null if never folded
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "3")
os.environ.setdefault("HANDEVAL_THREADS", "3")

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import handeval as he  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
D = BASE / "data" / "derived"
CHUNK = 250_000


def preflop_table(path: Path) -> pl.DataFrame:
    if path.exists():
        return pl.read_parquet(path)
    t = time.perf_counter()
    eq1, eq5 = he.preflop_equity_table(n_samples=400_000, seed=12345)
    df = pl.DataFrame({
        "preflop_class": np.arange(169, dtype=np.int16),
        "name": he.PREFLOP_NAMES,
        "combos": he.PREFLOP_COMBOS,
        "eq_vs1": eq1.astype(np.float32),
        "eq_vs5": eq5.astype(np.float32),
    }).with_columns(
        pl.col("eq_vs1").rank("ordinal", descending=True).cast(pl.Int16).alias("order_vs1"),
        pl.col("eq_vs5").rank("ordinal", descending=True).cast(pl.Int16).alias("order_vs5"),
    )
    df.write_parquet(path, compression="zstd")
    print(f"preflop169: {time.perf_counter() - t:.1f}s", flush=True)
    return df


def null_neg(cols):
    return [pl.when(pl.col(c) >= 0).then(pl.col(c)).otherwise(None).alias(c) for c in cols]


def main():
    t0 = time.perf_counter()
    pf = preflop_table(D / "preflop169.parquet")
    pf_eq1 = pf.sort("preflop_class")["eq_vs1"].to_numpy()
    pf_eq5 = pf.sort("preflop_class")["eq_vs5"].to_numpy()

    n_hands = pl.scan_parquet(D / "hands.parquet").select(pl.len()).collect().item()
    parts = []
    chk_tot = chk_ok = 0
    for lo in range(0, n_hands, CHUNK):
        hi = min(lo + CHUNK, n_hands)
        h = (pl.scan_parquet(D / "hands.parquet")
             .filter(pl.col("hand_idx").is_between(lo, hi - 1))
             .select("hand_idx", pl.col("board").list.len().cast(pl.Int8).alias("nb"),
                     *[pl.col("board").list.get(i, null_on_oob=True).fill_null(-1).alias(f"b{i}") for i in range(5)])
             .sort("hand_idx").collect())
        s = (pl.scan_parquet(D / "seats.parquet")
             .filter(pl.col("hand_idx").is_between(lo, hi - 1))
             .select("hand_idx", "player_idx", "seat_no", "c1", "c2", "went_to_showdown", "won_share")
             .sort("hand_idx", "seat_no").collect())
        n = h.height
        assert s.height == 6 * n, (lo, s.height, n)
        hid = s["hand_idx"].to_numpy().reshape(n, 6)
        assert (hid == h["hand_idx"].to_numpy()[:, None]).all()

        holes = np.stack([s["c1"].to_numpy(), s["c2"].to_numpy()], axis=1).reshape(n, 6, 2).astype(np.int8)
        board5 = h.select([f"b{i}" for i in range(5)]).to_numpy().astype(np.int8)
        nb = h["nb"].to_numpy().astype(np.int8)

        r3 = he.table_ranks(holes, board5, nb, 3)
        r4 = he.table_ranks(holes, board5, nb, 4)
        r5 = he.table_ranks(holes, board5, nb, 5)
        rl = np.where((nb == 5)[:, None], r5, np.where((nb == 4)[:, None], r4, r3))

        def cmp(r):
            ri = r.astype(np.int32)
            better = (ri[:, None, :] > ri[:, :, None]).sum(axis=2) - 0  # [i,p] = #q with r[q] > r[p]
            tied = (ri[:, None, :] == ri[:, :, None]).sum(axis=2) - 1
            valid = r >= 0
            return np.where(valid, better, -1).astype(np.int8), np.where(valid, tied, -1).astype(np.int8)

        b5, t5 = cmp(r5)
        bl, _ = cmp(rl)

        # sanity: among showdown players on a 5-card board, the best-ranked one must receive a share
        sd = s["went_to_showdown"].to_numpy().reshape(n, 6)
        ws = s["won_share"].to_numpy().reshape(n, 6)
        m = (nb == 5) & (sd.sum(1) >= 2)
        if m.any():
            rr = np.where(sd[m], r5[m].astype(np.int32), -10)
            top = rr == rr.max(1, keepdims=True)
            chk_tot += int(m.sum())
            chk_ok += int(((ws[m] > 0) | ~top).all(1).sum())

        pc = he.preflop_class(holes[:, :, 0].ravel(), holes[:, :, 1].ravel()).astype(np.int16)
        df = pl.DataFrame({
            "hand_idx": s["hand_idx"], "player_idx": s["player_idx"], "seat_no": s["seat_no"],
            "preflop_class": pc,
            "pf_eq_vs1": pf_eq1[pc].astype(np.float32), "pf_eq_vs5": pf_eq5[pc].astype(np.float32),
            "rank_flop": r3.ravel(), "rank_turn": r4.ravel(), "final_rank_7": r5.ravel(),
            "cat_flop": he.hand_category(r3.ravel()), "cat_turn": he.hand_category(r4.ravel()),
            "cat_river": he.hand_category(r5.ravel()),
            "n_opp_better_river": b5.ravel(), "n_opp_tied_river": t5.ravel(),
            "last_board_n": np.repeat(nb, 6),
            "rank_last": rl.ravel(), "n_opp_better_last": bl.ravel(),
        })
        df = df.with_columns(
            null_neg(["rank_flop", "rank_turn", "final_rank_7", "cat_flop", "cat_turn", "cat_river",
                      "n_opp_better_river", "n_opp_tied_river", "rank_last", "n_opp_better_last"])
        ).with_columns(
            (pl.col("n_opp_better_river") == 0).alias("best_hand_at_table"),
            (pl.col("n_opp_better_last") == 0).alias("best_at_last_board"),
        )
        folds = (pl.scan_parquet(D / "actions.parquet")
                 .filter(pl.col("hand_idx").is_between(lo, hi - 1) & (pl.col("action") == 0))
                 .group_by("hand_idx", "player_idx").agg(pl.col("street").min().alias("fold_street"))
                 .collect())
        df = df.join(folds, on=["hand_idx", "player_idx"], how="left").sort("hand_idx", "seat_no")
        parts.append(df)
        print(f"  hands {lo:>9,}-{hi - 1:>9,}  {time.perf_counter() - t0:6.1f}s", flush=True)

    out = pl.concat(parts).select(
        "hand_idx", "player_idx", "seat_no", "preflop_class", "pf_eq_vs1", "pf_eq_vs5",
        "rank_flop", "rank_turn", "final_rank_7", "cat_flop", "cat_turn", "cat_river",
        "n_opp_better_river", "n_opp_tied_river", "best_hand_at_table",
        "last_board_n", "rank_last", "n_opp_better_last", "best_at_last_board", "fold_street")
    out.write_parquet(D / "seat_strength.parquet", compression="zstd", row_group_size=240_000, statistics=True)
    print(f"seat_strength.parquet: {out.height:,} rows, {out.width} cols, "
          f"{(D / 'seat_strength.parquet').stat().st_size / 1e6:.1f} MB, total {time.perf_counter() - t0:.1f}s")
    print(f"showdown sanity: best-ranked showdown player received a share in {chk_ok:,}/{chk_tot:,} hands "
          f"({chk_ok / max(chk_tot, 1):.5%})")


if __name__ == "__main__":
    main()
