"""exp001 acceptance: engine signature coverage on dev evidence vs confirmed-negative shared hands (forensics numbers in brackets)."""
import sys
import polars as pl
sys.path.insert(0, "src")
import common as C

lab = C.load("labels").select("pair_id", "label", "behavior_family", pl.col("p1").alias("pA"), pl.col("p2").alias("pB"))
ev = C.load("evidence").select("pair_id", "hand_idx", pl.lit(1).alias("is_ev"))
ph = (pl.scan_parquet("data/derived/ph/phase0/*.parquet").join(lab.lazy(), on=["pA", "pB"]).collect()
      .join(ev, on=["pair_id", "hand_idx"], how="left").with_columns(pl.col("is_ev").fill_null(0)))
print("labelled pair-hand rows", ph.height, "evidence matched", ph["is_ev"].sum(), "of", C.load("evidence").height)
flowAB = (pl.col("net_bb_ab") < 0) & (pl.col("net_bb_ba") > 0)
flowBA = (pl.col("net_bb_ba") < 0) & (pl.col("net_bb_ab") > 0)
def dirsig(ab, ba):
    return (flowAB & ab) | (flowBA & ba)
sig = {
    "CI_S1_trash [.694/.0039]": (pl.col("trash_ab") + pl.col("trash_ba")) > 0,
    "CI_S14_utg_weak [.626/.011]": (pl.col("fpa_ord") == 0) & (pl.col("fpa_p") < 0.3),
    "CI_S5_reraise_partner [.363/.0202]": (pl.col("raise_pre_ab") + pl.col("raise_post_ab") + pl.col("raise_pre_ba") + pl.col("raise_post_ba")) > 0,
    "SP_G0 [1.0/.219]": ~((pl.col("first_fold_nonfacing_ab") == 1) | (pl.col("first_fold_nonfacing_ba") == 1)),
    "SP_G1 [.976/.100]": (pl.col("enter_ab") == 1) & (pl.col("enter_ba") == 1),
    "SP_strongcheck [.459/.0079]": (pl.col("strongcheck_ab") + pl.col("strongcheck_ba")) > 0,
    "SP_hu_checked_street [.331/.0089]": pl.col("hu_checked_streets") > 0,
    "SP_fold_to_partner_post [.505/.0338]": (pl.col("fold_post_ab") + pl.col("fold_post_ba")) > 0,
    "DT_flow [1.0/.157]": flowAB | flowBA,
    "DT_strong [.623/.0114]": dirsig((pl.col("max_hueq_fold_post_ab") >= 0.7) | (pl.col("min_hueq_call_tr_ab") <= 0.15),
                                     (pl.col("max_hueq_fold_post_ba") >= 0.7) | (pl.col("min_hueq_call_tr_ba") <= 0.15)),
    "DT_gift5 [.790/.0185]": dirsig(pl.col("gift_ab") > 5, pl.col("gift_ba") > 5),
    "DT_pcall_pop<=.15 (policy proxy)": dirsig(pl.col("min_pcall_pre_ab") <= 0.15, pl.col("min_pcall_pre_ba") <= 0.15),
}
grp = (pl.when(pl.col("is_ev") == 1).then(pl.lit("EV_") + pl.col("behavior_family"))
       .when(pl.col("label") == 1).then(pl.lit("NONEV_") + pl.col("behavior_family")).otherwise(pl.lit("NEG")))
out = ph.with_columns(grp.alias("grp")).group_by("grp").agg([pl.len()] + [e.mean().alias(k) for k, e in sig.items()]).sort("grp")
pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_float("mixed")
print(out.transpose(include_header=True, column_names="grp"))
