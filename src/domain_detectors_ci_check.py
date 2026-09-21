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
from scipy.stats import mannwhitneyu
P = P.sort("pair_id","hand_idx","action_no")
grp = ["pair_id","hand_idx","street"]
P = P.with_columns(
    aggA=((pl.col("role")=="A") & pl.col("is_aggr")).cast(pl.Int32), aggB=((pl.col("role")=="B") & pl.col("is_aggr")).cast(pl.Int32),
    aggO=((pl.col("role")=="O") & pl.col("is_aggr")).cast(pl.Int32), callB=((pl.col("role")=="B") & pl.col("is_call")).cast(pl.Int32),
    callA=((pl.col("role")=="A") & pl.col("is_call")).cast(pl.Int32))
P = P.with_columns(
    cA=pl.col("aggA").cum_sum().shift(1).fill_null(0).over(grp), cB=pl.col("aggB").cum_sum().shift(1).fill_null(0).over(grp),
    cO=pl.col("aggO").cum_sum().shift(1).fill_null(0).over(grp),
    kA=pl.col("callA").cum_sum().shift(1).fill_null(0).over(grp), kB=pl.col("callB").cum_sum().shift(1).fill_null(0).over(grp))
mine_before = pl.when(pl.col("role")=="A").then(pl.col("cA")).otherwise(pl.col("cB"))
part_before = pl.when(pl.col("role")=="A").then(pl.col("cB")).otherwise(pl.col("cA"))
part_called = pl.when(pl.col("role")=="A").then(pl.col("kB")).otherwise(pl.col("kA"))
new = {
  "rr_fold_vs_partner": X & pl.col("is_fold") & pl.col("facing_partner") & (mine_before>0),         # I raised, partner re-raised, I fold
  "reraise_partner": X & pl.col("is_aggr") & pl.col("facing_partner"),
  "fold_to_partner_raise": X & pl.col("is_fold") & pl.col("facing_partner"),
  "squeeze_after_partner_call": X & pl.col("is_aggr") & (pl.col("to_call")>0) & (pl.col("last_aggr")!=pl.col("partner")) & (part_called>0) & (pl.col("cO")>0),
  "reraise_3rd_after_partner_raise": X & pl.col("is_aggr") & (pl.col("to_call")>0) & (pl.col("last_aggr")!=pl.col("partner")) & (part_before>0),
  "o_fold_facing_pair_after_both_aggr": (pl.col("role")=="O") & pl.col("is_fold") & (pl.col("cA")>0) & (pl.col("cB")>0),
  "o_fold_facing_pair_aggr": (pl.col("role")=="O") & pl.col("is_fold") & ((pl.col("last_aggr")==pl.col("sa")) | (pl.col("last_aggr")==pl.col("sb"))),
}
P = P.with_columns([e.fill_null(False).cast(pl.Int32).alias(k) for k,e in new.items()])
HH = P.group_by("pair_id","hand_idx").agg([pl.col(k).sum() for k in new] + [ (pl.col("role")!="O").and_(pl.col("is_aggr")).sum().alias("pair_aggr"),
      pl.col("pot_before").max().alias("maxpot"), pl.col("bb").first()])
HH = HH.join(sh.select("pair_id","hand_idx","label","fam","is_ev","net_a","net_b"), on=["pair_id","hand_idx"])
HH = HH.with_columns(
   pair_no_conflict_war = ((pl.col("reraise_partner")>0) & ((pl.col("fold_to_partner_raise")>0))).cast(pl.Int32),
   ci_v2 = 4*pl.col("rr_fold_vs_partner") + 2*pl.col("reraise_partner") + 2*pl.col("squeeze_after_partner_call") + 2*pl.col("reraise_3rd_after_partner_raise")
           + 1.5*pl.col("o_fold_facing_pair_after_both_aggr") + pl.col("fold_to_partner_raise") + 0.5*pl.col("o_fold_facing_pair_aggr") + (pl.col("net_a")+pl.col("net_b")>0).cast(pl.Int32)*0.5,
)
rng=np.random.default_rng(0)
def map5(df,col):
    out={}
    for fam,g in df.filter(pl.col("label")==1).group_by("fam"):
        aps=[]
        for _,gg in g.group_by("pair_id"):
            s=gg[col].to_numpy().astype(float)+rng.random(gg.height)*1e-9; y=gg["is_ev"].to_numpy()
            o=np.argsort(-s)[:5]; h=0; ps=0
            for r,i in enumerate(o,1):
                if y[i]: h+=1; ps+=h/r
            aps.append(ps/max(1,min(5,y.sum())))
        out[fam[0][:4]]=round(float(np.mean(aps)),3)
    return out
cols=list(new)+["pair_no_conflict_war","ci_v2"]
neg=HH.filter(pl.col("label")==0)
PR=HH.group_by("pair_id","label","fam").agg([(pl.col(c)>0).mean().alias(c) for c in cols])
for c in cols:
    g=HH.filter(pl.col("fam")=="coordinated_isolation")
    ev_rate=g.filter(pl.col("is_ev")==1).select((pl.col(c)>0).mean()).item(); ne=g.filter(pl.col("is_ev")==0).select((pl.col(c)>0).mean()).item()
    hn=neg.select((pl.col(c)>0).mean()).item()
    a={fam[:4]: round(float(mannwhitneyu(PR.filter(pl.col("fam")==fam)[c].to_numpy(), PR.filter(pl.col("label")==0)[c].to_numpy()).statistic/(PR.filter(pl.col("fam")==fam).height*PR.filter(pl.col("label")==0).height)),3) for fam in ["directed_transfer","soft_play","coordinated_isolation"]}
    print(f"{c:36s} CI ev {100*ev_rate:5.1f}% nonev {100*ne:5.2f}% hardneg {100*hn:5.2f}%  map5 {map5(HH,c)}  pairAUC {a}")
