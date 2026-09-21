"""Pair-level features (exp005) from the pair-hand engine output (data/derived/ph/phase*/).

For one phase window it aggregates every within-table pair's shared hands into count-aware statistics:
  c_<f>   count of hands with signature f          r_<f>  rate per shared hand
  z_<f>   binomial z vs the population base rate of f in the same window (all pair-hand rows)
Directional signatures (X acting toward Y) are kept per direction and summarised as max/min over the pair.
Own-baseline directedness (label-free, gameplay only): for X's events with Y seated vs X's rate with the other
co-players in the window -> excess counts, shrunk log-lift, rank of Y among X's co-players, mutual-top flags.

Windows: dev_full (phase 0), dev_w2000a (hand_seq < 2000), dev_w2000b (1000 <= hand_seq < 3000), eval (phase 1).
Output: data/derived/pf_<window>.parquet   (pA, pB, table_idx, n, features...)
Usage: python src/pairfeat.py [window ...]
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

BASE = Path(__file__).resolve().parent.parent
DER = BASE / "data" / "derived"
WINDOWS = {
    "dev_full": (0, 0, 3000),
    "dev_w2000a": (0, 0, 2000),
    "dev_w2000b": (0, 1000, 3000),
    "eval": (1, 3000, 5000),
}
col = pl.col


def _flow(d, o):
    return (col(f"net_bb_{d}") < 0) & (col(f"net_bb_{o}") > 0)


def hand_flags():
    """Expressions computed per pair-hand row. Returns (symmetric flags, directional flag builders, sums)."""
    face_any = (col("face_ab") + col("face_ba")) > 0
    raiseback = (col("raise_pre_ab") + col("raise_post_ab") + col("raise_pre_ba") + col("raise_post_ba")) > 0
    g0 = ~((col("first_fold_nonfacing_ab") == 1) | (col("first_fold_nonfacing_ba") == 1))
    g1 = (col("enter_ab") == 1) & (col("enter_ba") == 1)
    strongcheck = (col("strongcheck_ab") + col("strongcheck_ba")) > 0
    foldpost = (col("fold_post_ab") + col("fold_post_ba")) > 0
    foldpost_hs60 = pl.max_horizontal("max_hs_fold_post_ab", "max_hs_fold_post_ba") >= 0.6
    callpost_hs80 = pl.max_horizontal("max_hs_call_post_ab", "max_hs_call_post_ba") >= 0.8
    callpre_eq55 = (((col("call_pre_ab") > 0) & (col("max_pfeq_call_pre_ab") >= 0.55))
                    | ((col("call_pre_ba") > 0) & (col("max_pfeq_call_pre_ba") >= 0.55)))
    s_pl = g1 & face_any & ~raiseback
    u1 = strongcheck | foldpost | callpost_hs80
    trash_any = (col("trash_ab") + col("trash_ba")) > 0
    fold_after = (col("fold_after_aggr_ab") == 1) | (col("fold_after_aggr_ba") == 1)
    sym = {
        "both_vol": (col("vol_ab") == 1) & (col("vol_ba") == 1),
        "both_flop": col("both_streets") >= 2,
        "sd_both": col("sd_both") == 1,
        "hu_final": col("hu_final") == 1,
        "sp_g0": g0,
        "sp_g1": g1,
        "sp_pl": s_pl,
        "sp_strongcheck": strongcheck,
        "sp_foldpost_hs60": foldpost & foldpost_hs60,
        "sp_callpost_hs80": callpost_hs80,
        "sp_callpre_eq55": callpre_eq55,
        "sp_hu_checked": col("hu_checked_streets") > 0,
        "sp_u1": u1,
        "sp_pl_soft": s_pl & (u1 | callpre_eq55),
        "sp_raiseback": raiseback,
        "ci_trash": trash_any,
        "ci_weak10": (col("weak10_ab") + col("weak10_ba")) > 0,
        "ci_s14": (col("fpa_ord") == 0) & (col("fpa_p") < 0.3),
        "ci_s7": trash_any & (fold_after | raiseback),
        "ci_s10": trash_any & (col("tf_pre") >= 3),
        "ci_s16": (col("fpa_ord") <= 2) & (col("fpa_partner_behind") == 1) & (fold_after | raiseback) & (col("fpa_p") < 0.3),
        "ci_sandwich": col("sandwich") == 1,
        "ci_tf3": col("tf_both_active") >= 3,
    }
    dirf = {
        "dt_flow": lambda d, o: _flow(d, o),
        "dt_strong": lambda d, o: _flow(d, o) & ((col(f"max_hueq_fold_post_{d}") >= 0.7) | (col(f"min_hueq_call_tr_{d}") <= 0.15)),
        "dt_union": lambda d, o: _flow(d, o) & ((col(f"max_hueq_fold_post_{d}") >= 0.5) | (col(f"min_hueq_call_post_{d}") <= 0.25)
                                                | (col(f"bluffinto_{d}") > 0) | ((col(f"fold_pre_{d}") > 0) & (col(f"max_eq_fold_face_{d}") >= 0.5))),
        "dt_gift5": lambda d, o: _flow(d, o) & (col(f"gift_{d}") > 5),
        "dt_pcall15": lambda d, o: col(f"min_pcall_pre_{d}") <= 0.15,
        "dt_pcall05": lambda d, o: col(f"min_pcall_pre_{d}") <= 0.05,
        "dt_callpre": lambda d, o: col(f"call_pre_{d}") > 0,
        "dt_raisepre": lambda d, o: col(f"raise_pre_{d}") > 0,
        "dt_bluffinto": lambda d, o: col(f"bluffinto_{d}") > 0,
        "x_trash": lambda d, o: col(f"trash_{d}") > 0,
        "x_face": lambda d, o: col(f"face_{d}") > 0,
        "x_foldface": lambda d, o: (col(f"fold_pre_{d}") + col(f"fold_post_{d}")) > 0,
        "x_strongcheck": lambda d, o: col(f"strongcheck_{d}") > 0,
        "x_polcall10": lambda d, o: col(f"min_polcall_pre_{d}") <= 0.10,
        "x_polagg50": lambda d, o: col(f"max_polagg_resp_{d}") >= 0.5,
        "x_polcall15": lambda d, o: col(f"min_polcall_pre_{d}") <= 0.15,
        "x_polcall05": lambda d, o: col(f"min_polcall_pre_{d}") <= 0.05,
        "x_polagg30": lambda d, o: col(f"max_polagg_resp_{d}") >= 0.3,
        "x_polfold20": lambda d, o: col(f"min_polfold_face_{d}") <= 0.2,
        "x_polfold05": lambda d, o: col(f"min_polfold_face_{d}") <= 0.05,
        "dt_polfold20f": lambda d, o: _flow(d, o) & (col(f"min_polfold_face_{d}") <= 0.2),
    }
    sums = {
        "gift": lambda d, o: col(f"gift_{d}"),
        "imp": lambda d, o: col(f"imp_{d}"),
        "transfer": lambda d, o: pl.when(_flow(d, o)).then(pl.min_horizontal(-col(f"net_bb_{d}"), col(f"net_bb_{o}"))).otherwise(0.0),
        "surp_face": lambda d, o: col(f"surp_face_{d}"),
    }
    return sym, dirf, sums


def build(window):
    t0 = time.time()
    phase, lo, hi = WINDOWS[window]
    sym, dirf, sums = hand_flags()
    lf = (pl.scan_parquet(DER / os.environ.get("PH_DIR", "ph") / f"phase{phase}" / "*.parquet")
          .filter((col("hand_seq") >= lo) & (col("hand_seq") < hi)))
    exprs = [e.cast(pl.Int8).alias(k) for k, e in sym.items()]
    for k, fn in dirf.items():
        exprs += [fn("ab", "ba").cast(pl.Int8).alias(f"{k}_ab"), fn("ba", "ab").cast(pl.Int8).alias(f"{k}_ba")]
    for k, fn in sums.items():
        exprs += [fn("ab", "ba").cast(pl.Float32).alias(f"{k}_ab"), fn("ba", "ab").cast(pl.Float32).alias(f"{k}_ba")]
    exprs += [col("gift_max_ab").alias("giftmax_ab"), col("gift_max_ba").alias("giftmax_ba")]
    rows = lf.select("pA", "pB", "table_idx", *exprs)
    flag_names = list(sym) + [f"{k}_{d}" for k in dirf for d in ("ab", "ba")]
    sum_names = [f"{k}_{d}" for k in sums for d in ("ab", "ba")]
    agg = (rows.group_by("pA", "pB", "table_idx")
           .agg([pl.len().alias("n")] + [col(f).sum().cast(pl.Int32).alias(f"c_{f}") for f in flag_names]
                + [col(f).sum().alias(f"s_{f}") for f in sum_names]
                + [col("giftmax_ab").max(), col("giftmax_ba").max()])
           .collect(engine="streaming"))
    # population base rates per flag (row-weighted, i.e. per pair-hand)
    ntot = agg["n"].sum()
    p0 = {f: agg[f"c_{f}"].sum() / ntot for f in flag_names}
    n = col("n").cast(pl.Float64)
    feats = [n.log().alias("log_n")]
    for f in sym:
        c = col(f"c_{f}")
        feats += [(c / n).alias(f"r_{f}"), ((c - n * p0[f]) / (n * p0[f] * (1 - p0[f]) + 1e-9).sqrt()).alias(f"z_{f}")]
    for k in dirf:
        cab, cba = col(f"c_{k}_ab"), col(f"c_{k}_ba")
        pk = 0.5 * (p0[f"{k}_ab"] + p0[f"{k}_ba"])
        cmax, cmin = pl.max_horizontal(cab, cba), pl.min_horizontal(cab, cba)
        feats += [(cmax / n).alias(f"r_{k}_max"), (cmin / n).alias(f"r_{k}_min"),
                  ((cmax - n * pk) / (n * pk * (1 - pk) + 1e-9).sqrt()).alias(f"z_{k}_max"),
                  ((cmin - n * pk) / (n * pk * (1 - pk) + 1e-9).sqrt()).alias(f"z_{k}_min"),
                  ((cab - cba).abs() / n).alias(f"r_{k}_asym")]
    for k in sums:
        sab, sba = col(f"s_{k}_ab"), col(f"s_{k}_ba")
        feats += [(pl.max_horizontal(sab, sba) / n).alias(f"m_{k}_max"), (pl.min_horizontal(sab, sba) / n).alias(f"m_{k}_min"),
                  ((sab - sba).abs() / n).alias(f"m_{k}_asym")]
    feats += [pl.max_horizontal("giftmax_ab", "giftmax_ba").alias("m_giftmax")]
    pf = agg.with_columns(feats)
    wide = own_baseline(agg, [k for k in dirf], sym_for_lift=["sp_g1", "sp_pl", "sp_pl_soft", "sp_strongcheck",
                                                              "both_vol", "ci_s7", "ci_s16", "sd_both", "hu_final"])
    pf = pf.join(wide, on=["pA", "pB"], how="left")
    out = DER / f"pf{os.environ.get('PF_TAG', '')}_{window}.parquet"
    pf.write_parquet(out)
    print(f"[{window}] pairs {pf.height:,} cols {pf.width} rows {ntot:,} in {time.time() - t0:.1f}s -> {out.name}", flush=True)
    return pf


def own_baseline(agg, dir_keys, sym_for_lift):
    """Directedness vs each player's rate with the other co-players in the same window.
    Returns expressions evaluated on `agg` (joined back by pA,pB)."""
    longs = []
    for k in dir_keys:
        longs.append(pl.concat([
            agg.select(col("pA").alias("X"), col("pB").alias("Y"), col("n"), col(f"c_{k}_ab").alias("c")),
            agg.select(col("pB").alias("X"), col("pA").alias("Y"), col("n"), col(f"c_{k}_ba").alias("c"))]).with_columns(pl.lit(k).alias("k")))
    for k in sym_for_lift:
        longs.append(pl.concat([
            agg.select(col("pA").alias("X"), col("pB").alias("Y"), col("n"), col(f"c_{k}").alias("c")),
            agg.select(col("pB").alias("X"), col("pA").alias("Y"), col("n"), col(f"c_{k}").alias("c"))]).with_columns(pl.lit(k).alias("k")))
    L = pl.concat(longs)
    L = L.with_columns(col("c").sum().over("X", "k").alias("CX"), col("n").sum().over("X", "k").alias("NX"))
    rate_wo = (col("CX") - col("c")) / (col("NX") - col("n")).clip(lower_bound=1)
    L = L.with_columns(
        (col("c") - col("n") * rate_wo).alias("exc"),
        (((col("c") + 1.0) / (col("n") * rate_wo + 1.0)).log()).alias("llift"))
    L = L.with_columns(col("llift").rank("ordinal", descending=True).over("X", "k").alias("rk"))
    # back to unordered pairs
    L = L.with_columns(pl.min_horizontal("X", "Y").alias("pA"), pl.max_horizontal("X", "Y").alias("pB"))
    W = (L.group_by("pA", "pB", "k").agg(
        col("exc").sum().alias("exc_sum"), col("exc").min().alias("exc_min"),
        col("llift").sum().alias("llift_sum"), col("llift").min().alias("llift_min"), col("llift").max().alias("llift_max"),
        (col("rk") == 1).all().cast(pl.Int8).alias("mtop"), (col("rk") == 1).any().cast(pl.Int8).alias("top_any"),
        col("rk").max().alias("rk_worst")))
    wide = W.pivot(on="k", index=["pA", "pB"], values=["exc_sum", "exc_min", "llift_sum", "llift_min", "llift_max",
                                                       "mtop", "top_any", "rk_worst"])
    return wide.rename({c: "ob_" + c for c in wide.columns if c not in ("pA", "pB")})


def build_window(window):
    return build(window)


if __name__ == "__main__":
    ws = sys.argv[1:] or list(WINDOWS)
    for w in ws:
        build_window(w)
