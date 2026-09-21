# Usage: python src/domain_detectors_check.py [out_pair_hand.parquet]   (needs data/derived/action_equity_labeled_dev.parquet from src/hand_replay.py)
"""Grounding check of hand-level domain detectors on labelled dev pairs (uses action_equity_labeled_dev)."""
import os, sys
os.environ.setdefault("POLARS_MAX_THREADS", "3")
import numpy as np
import polars as pl
import duckdb

D = r"F:\kaggle competitions\suspicious poker\data\derived"
Dp = D.replace("\\", "/")
OUT = sys.argv[1] if len(sys.argv) > 1 else None
con = duckdb.connect(); con.sql("SET threads=3; SET memory_limit='4GB'")
sh = con.sql(f"""select l.pair_id, l.label, l.behavior_family fam, s1.hand_idx, s1.seat_no sa, s2.seat_no sb,
      s1.net_chips net_a, s2.net_chips net_b, h.bb, h.hand_seq, h.table_idx
    from '{Dp}/labels.parquet' l join '{Dp}/seats.parquet' s1 on s1.player_idx=l.p1
    join '{Dp}/seats.parquet' s2 on s2.player_idx=l.p2 and s2.hand_idx=s1.hand_idx
    join '{Dp}/hands.parquet' h on h.hand_idx=s1.hand_idx where h.phase=0""").pl()
ev = pl.read_parquet(f"{D}/evidence.parquet").select("pair_id", "hand_idx").with_columns(pl.lit(1).alias("is_ev"))
sh = sh.join(ev, on=["pair_id", "hand_idx"], how="left").with_columns(pl.col("is_ev").fill_null(0))
hands = sh["hand_idx"].unique()
E = pl.read_parquet(f"{D}/action_equity_labeled_dev.parquet")
A = (pl.scan_parquet(f"{D}/actions.parquet").filter(pl.col("hand_idx").is_in(hands.implode()))
     .select("hand_idx", "action_no", "street", "player_idx", "action", "amount", "pot_before", "to_call", "players_active").collect())
S = pl.scan_parquet(f"{D}/seats.parquet").filter(pl.col("hand_idx").is_in(hands.implode())).select("hand_idx", "player_idx", "c1", "c2").collect()
cmap = pl.read_parquet(f"{D}/preflop_class_map.parquet").join(pl.read_parquet(f"{D}/preflop_equity_169.parquet").select("class_id", "top_pct_vs1"), on="class_id")
S = S.join(cmap.select("c1", "c2", "top_pct_vs1"), on=["c1", "c2"], how="left").select("hand_idx", "player_idx", "top_pct_vs1")
A = A.join(S, on=["hand_idx", "player_idx"], how="left").join(E, on=["hand_idx", "action_no"], how="inner").sort("hand_idx", "action_no")
pre = [f"eq_pre_s{s}" for s in range(6)]; post = [f"eq_post_s{s}" for s in range(6)]
A = A.with_columns(
    is_aggr=(pl.col("action").is_in([3, 4]) | ((pl.col("action") == 5) & (pl.col("amount") > pl.col("to_call")))),
    is_call=((pl.col("action") == 2) | ((pl.col("action") == 5) & (pl.col("amount") <= pl.col("to_call")) & (pl.col("to_call") > 0))),
    is_fold=(pl.col("action") == 0), is_check=(pl.col("action") == 1),
    n_live=pl.sum_horizontal([pl.col(c).is_not_nan().cast(pl.Int8) for c in pre]),
    eqpre=pl.concat_list(pre), eqpost=pl.concat_list(post),
)
A = A.with_columns(aggr_seat=pl.when(pl.col("is_aggr")).then(pl.col("seat_no")).otherwise(None))
A = A.with_columns(last_aggr=pl.col("aggr_seat").shift(1).forward_fill().over(["hand_idx", "street"]))
A = A.with_columns(eq_actor=pl.col("eqpre").list.get(pl.col("seat_no").cast(pl.Int64)))
keep = ["hand_idx", "action_no", "street", "seat_no", "action", "amount", "pot_before", "to_call", "is_aggr", "is_call", "is_fold", "is_check",
        "n_live", "eqpre", "eqpost", "last_aggr", "eq_actor", "hs_actor", "top_pct_vs1"]
A = A.select(keep)
print("actions", A.shape, flush=True)

P = sh.select("pair_id", "hand_idx", "sa", "sb", "bb").join(A, on="hand_idx", how="inner")
P = P.with_columns(role=pl.when(pl.col("seat_no") == pl.col("sa")).then(pl.lit("A")).when(pl.col("seat_no") == pl.col("sb")).then(pl.lit("B")).otherwise(pl.lit("O")))
P = P.with_columns(partner=pl.when(pl.col("role") == "A").then(pl.col("sb")).when(pl.col("role") == "B").then(pl.col("sa")).otherwise(None))
P = P.with_columns(
    eq_partner=pl.col("eqpre").list.get(pl.col("partner").fill_null(0).cast(pl.Int64)),
    eqp_post=pl.col("eqpost").list.get(pl.col("partner").fill_null(0).cast(pl.Int64)),
    eqa_post=pl.col("eqpost").list.get(pl.col("seat_no").cast(pl.Int64)),
    eqA=pl.col("eqpre").list.get(pl.col("sa").cast(pl.Int64)), eqB=pl.col("eqpre").list.get(pl.col("sb").cast(pl.Int64)),
    eqA_post=pl.col("eqpost").list.get(pl.col("sa").cast(pl.Int64)), eqB_post=pl.col("eqpost").list.get(pl.col("sb").cast(pl.Int64)),
)
P = P.with_columns(
    partner_live=pl.col("eq_partner").is_not_nan() & (pl.col("role") != "O"),
    facing_partner=(pl.col("last_aggr") == pl.col("partner")) & (pl.col("to_call") > 0),
    bbf=pl.col("bb").cast(pl.Float64),
)
# Mazrooei impact of this action on each pair member (chips, luck-free), others' mean
def dv(eq_pre_c, eq_post_c, self_flag):
    return (pl.col(eq_post_c).fill_nan(0) - pl.col(eq_pre_c).fill_nan(0)) * pl.col("pot_before") + pl.col(eq_post_c).fill_nan(0) * pl.col("amount") - pl.when(self_flag).then(pl.col("amount")).otherwise(0)
P = P.with_columns(dvA=dv("eqA", "eqA_post", pl.col("role") == "A"), dvB=dv("eqB", "eqB_post", pl.col("role") == "B"))
X = pl.col("role") != "O"
pst = pl.col("street") >= 1
det = {
    # soft play / transfer: surrender the best hand to partner aggression
    "fold_best_vs_partner": X & pl.col("is_fold") & pl.col("facing_partner") & (pl.col("eq_actor") >= 0.5),
    "fold_strong_perceived_vs_partner": X & pl.col("is_fold") & pl.col("facing_partner") & (
        ((pl.col("street") == 0) & (pl.col("top_pct_vs1") <= 0.15)) | (pst & (pl.col("hs_actor") >= 0.80))),
    # transfer: pay off partner with (near) dead hand
    "call_dead_vs_partner": X & pst & pl.col("is_call") & pl.col("facing_partner") & (pl.col("eq_actor") <= 0.15) & (pl.col("eq_partner") >= 0.7),
    "aggr_dead_into_partner": X & pst & pl.col("is_aggr") & pl.col("partner_live") & (pl.col("eq_actor") <= 0.15) & (pl.col("eq_partner") >= 0.7),
    # soft play: no value extraction heads-up vs partner
    "passive_strong_hu_vs_partner": X & pst & pl.col("partner_live") & (pl.col("n_live") == 2) & (pl.col("eq_actor") >= 0.8) & (pl.col("is_check") | pl.col("is_call")),
    "river_check_nuts_hu_vs_partner": X & (pl.col("street") == 3) & pl.col("partner_live") & (pl.col("n_live") == 2) & (pl.col("eq_actor") >= 0.99) & pl.col("is_check"),
    # isolation: partner steps aside for partner's raise; partner re-raises over 3rd party
    "fold_to_partner_raise_pf_playable": X & (pl.col("street") == 0) & pl.col("is_fold") & pl.col("facing_partner") & (pl.col("top_pct_vs1") <= 0.35),
    "fold_to_partner_raise_any": X & pl.col("is_fold") & pl.col("facing_partner"),
    "raise_over_3rd_after_partner_aggr_weak": X & pl.col("is_aggr") & (pl.col("to_call") > 0) & pl.col("partner_live") & (pl.col("last_aggr") != pl.col("partner")) & (pl.col("n_live") >= 3)
        & (((pl.col("street") == 0) & (pl.col("top_pct_vs1") >= 0.4)) | (pst & (pl.col("hs_actor") <= 0.5))),
    "third_party_fold_to_pair_aggr": (pl.col("role") == "O") & pl.col("is_fold") & ((pl.col("last_aggr") == pl.col("sa")) | (pl.col("last_aggr") == pl.col("sb")))
        & pl.col("eqA").is_not_nan() & pl.col("eqB").is_not_nan(),
    # conflict (negative for soft play)
    "raise_vs_partner": X & pl.col("is_aggr") & pl.col("facing_partner"),
}
P = P.with_columns([e.fill_null(False).alias(k) for k, e in det.items()])
P = P.with_columns(
    s_fold_best=pl.when(pl.col("fold_best_vs_partner")).then(pl.col("eq_actor") * pl.col("pot_before") / pl.col("bbf")).otherwise(0.0),
    s_call_dead=pl.when(pl.col("call_dead_vs_partner")).then((1 - pl.col("eq_actor")) * pl.col("amount") / pl.col("bbf")).otherwise(0.0),
    s_aggr_dead=pl.when(pl.col("aggr_dead_into_partner")).then(pl.col("eq_partner") * pl.col("amount") / pl.col("bbf")).otherwise(0.0),
    s_passive=pl.when(pl.col("passive_strong_hu_vs_partner")).then(pl.col("eq_actor") * pl.col("pot_before") / pl.col("bbf")).otherwise(0.0),
    imp_A_to_B=pl.when(pl.col("role") == "A").then(pl.col("dvB")).otherwise(0.0) / pl.col("bbf"),
    imp_B_to_A=pl.when(pl.col("role") == "B").then(pl.col("dvA")).otherwise(0.0) / pl.col("bbf"),
    both_aggr_street=pl.lit(0),
)
agg = [pl.col(k).sum().alias(k) for k in det] + [pl.col(c).sum() for c in ["s_fold_best", "s_call_dead", "s_aggr_dead", "s_passive", "imp_A_to_B", "imp_B_to_A"]]
# sandwich: both A and B aggressive on the same street while a 3rd party was live
st = P.group_by("pair_id", "hand_idx", "street").agg(
    nA=((pl.col("role") == "A") & pl.col("is_aggr")).sum(), nB=((pl.col("role") == "B") & pl.col("is_aggr")).sum(),
    nlive_max=pl.col("n_live").max()).with_columns(sandwich=((pl.col("nA") > 0) & (pl.col("nB") > 0) & (pl.col("nlive_max") >= 3)).cast(pl.Int32))
HH = P.group_by("pair_id", "hand_idx").agg(agg).join(st.group_by("pair_id", "hand_idx").agg(pl.col("sandwich").sum()), on=["pair_id", "hand_idx"])
HH = HH.join(sh.select("pair_id", "hand_idx", "label", "fam", "is_ev", "net_a", "net_b", "bb", "hand_seq"), on=["pair_id", "hand_idx"])
HH = HH.with_columns(
    imp_max=pl.max_horizontal("imp_A_to_B", "imp_B_to_A"), imp_sum=pl.col("imp_A_to_B") + pl.col("imp_B_to_A"),
    net_transfer=pl.max_horizontal(pl.min_horizontal(-pl.col("net_a"), pl.col("net_b")), pl.min_horizontal(-pl.col("net_b"), pl.col("net_a"))).clip(0) / pl.col("bb"),
)
if OUT:
    HH.write_parquet(OUT)
print("pair-hands", HH.shape, flush=True)

feats = list(det) + ["sandwich", "s_fold_best", "s_call_dead", "s_aggr_dead", "s_passive", "imp_max", "imp_sum", "net_transfer"]
rng = np.random.default_rng(0)

def map5(df, col):
    res = {}
    for fam, g in df.filter(pl.col("label") == 1).group_by("fam"):
        aps = []
        for pid, gg in g.group_by("pair_id"):
            s = gg[col].to_numpy().astype(float) + rng.random(gg.height) * 1e-9
            y = gg["is_ev"].to_numpy()
            order = np.argsort(-s)[:5]
            hits = 0; ps = 0.0
            for r, i in enumerate(order, 1):
                if y[i]:
                    hits += 1; ps += hits / r
            aps.append(ps / max(1, min(y.sum(), 5)))
        res[fam[0]] = float(np.mean(aps))
    return res

def auc(pos, neg):
    from scipy.stats import mannwhitneyu
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = mannwhitneyu(pos, neg).statistic
    return float(u / (len(pos) * len(neg)))

pos = HH.filter(pl.col("label") == 1)
rows = []
for f in feats:
    fire = (pl.col(f) > 0)
    r = {"detector": f}
    for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
        g = pos.filter(pl.col("fam") == fam)
        e = g.filter(pl.col("is_ev") == 1).select(fire.mean()).item()
        ne = g.filter(pl.col("is_ev") == 0).select(fire.mean()).item()
        r[f"{fam[:4]}_ev%"] = round(100 * e, 1); r[f"{fam[:4]}_nonev%"] = round(100 * ne, 2)
    r["hardneg%"] = round(100 * HH.filter(pl.col("label") == 0).select(fire.mean()).item(), 2)
    m = map5(HH, f); r.update({f"map5_{k[:4]}": round(v, 3) for k, v in m.items()})
    rows.append(r)
res = pl.DataFrame(rows)
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(300)
print(res)
# pair-level rates
PR = HH.group_by("pair_id", "label", "fam").agg([(pl.col(f) > 0).mean().alias(f) for f in feats] + [pl.len().alias("n")])
rows = []
for f in feats:
    neg = PR.filter(pl.col("label") == 0)[f].to_numpy()
    r = {"detector": f}
    for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
        r[f"auc_{fam[:4]}_vs_hardneg"] = round(auc(PR.filter(pl.col("fam") == fam)[f].to_numpy(), neg), 3)
    rows.append(r)
print(pl.DataFrame(rows))
# temporal clustering of evidence
g = pos.filter(pl.col("is_ev") == 1).sort("pair_id", "hand_seq").with_columns(gap=pl.col("hand_seq").diff().over("pair_id"))
print("evidence hand_seq gap quantiles", g["gap"].drop_nulls().quantile(0.1), g["gap"].drop_nulls().median(), g["gap"].drop_nulls().quantile(0.9))
span = pos.filter(pl.col("is_ev") == 1).group_by("pair_id", "fam").agg((pl.col("hand_seq").max() - pl.col("hand_seq").min()).alias("span"), pl.len().alias("k"))
allspan = pos.group_by("pair_id").agg((pl.col("hand_seq").max() - pl.col("hand_seq").min()).alias("pair_span"), pl.len().alias("shared"))
sp = span.join(allspan, on="pair_id").with_columns(frac=pl.col("span") / pl.col("pair_span"))
print(sp.group_by("fam").agg(pl.col("span").median(), pl.col("frac").median(), pl.col("shared").median(), pl.col("k").mean()))
