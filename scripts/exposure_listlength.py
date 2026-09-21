"""Does the host plant a FIXED number of events per pair, or a number that scales with exposure?

This gates the only genuinely new evidence lever the D-1 audit surfaced:
  26% of eval pairs share fewer hands than the MINIMUM dev positive (57), and in that out-of-support
  region the listing DP's implied list length falls to ~3 instead of ~4.9, so the cap-at-5 never binds
  and the DP degenerates toward the chronology-free event score (worth ~+0.10 MAP on DT per the ablation).

  * If the planted count is exposure-INDEPENDENT, that is a calibration bug and inflating q there is right.
  * If the planted count scales with exposure, the short lists are correct and inflating q would hurt.
"""
import sys
import numpy as np
import polars as pl

sys.path.insert(0, r"F:\kaggle competitions\suspicious poker\src")
import common as C  # noqa: E402

lab = C.load("labels").filter(pl.col("label") == 1)
ev = C.load("evidence")
nlist = ev.group_by("pair_id").agg(pl.len().alias("n_listed"))

# shared dev-phase hand count per labelled positive pair
hands = C.load("hands").select("hand_idx", "phase")
seats = C.scan("seats").select("hand_idx", "player_idx").collect()
dev_h = hands.filter(pl.col("phase") == 0).select("hand_idx")
s = seats.join(dev_h, on="hand_idx")

pos = lab.select("pair_id", "p1", "p2")
a = s.rename({"player_idx": "p1"})
b = s.rename({"player_idx": "p2"})
shared = (pos.join(a, on="p1").join(b, on=["hand_idx", "p2"])
          .group_by("pair_id").agg(pl.len().alias("n_shared")))

d = pos.join(shared, on="pair_id", how="left").join(nlist, on="pair_id", how="left") \
       .join(lab.select("pair_id", "behavior_family"), on="pair_id")
d = d.with_columns(pl.col("n_listed").fill_null(0))

print("=== dev positives: listed-hand count vs exposure ===")
print(f"n = {d.height}, shared hands: min {d['n_shared'].min()} / med {d['n_shared'].median():.0f} / max {d['n_shared'].max()}")
q = d["n_shared"].quantile
edges = [d["n_shared"].min() - 1, q(0.25), q(0.5), q(0.75), d["n_shared"].max() + 1]
print(f"\n{'exposure bucket':>22} {'n pairs':>8} {'mean n_listed':>14} {'share with 5':>13} {'share with 3-4':>15}")
for lo, hi in zip(edges[:-1], edges[1:]):
    b_ = d.filter((pl.col("n_shared") > lo) & (pl.col("n_shared") <= hi))
    if not b_.height:
        continue
    print(f"{int(lo)+1:>10}-{int(hi):<11} {b_.height:>8} {b_['n_listed'].mean():>14.3f} "
          f"{(b_['n_listed'] == 5).mean():>13.3f} {(b_['n_listed'] < 5).mean():>15.3f}")

# the decisive test: are the 32 short-list pairs the LOW-exposure ones?
short = d.filter(pl.col("n_listed") < 5)
full = d.filter(pl.col("n_listed") == 5)
print(f"\nshort lists (3-4 listed), n={short.height}: shared hands med {short['n_shared'].median():.0f}, "
      f"mean {short['n_shared'].mean():.1f}, min {short['n_shared'].min()}")
print(f"full lists (5 listed),   n={full.height}: shared hands med {full['n_shared'].median():.0f}, "
      f"mean {full['n_shared'].mean():.1f}, min {full['n_shared'].min()}")

from scipy import stats
r = stats.spearmanr(d["n_shared"].to_numpy(), d["n_listed"].to_numpy())
print(f"\nSpearman(n_shared, n_listed) = {r.statistic:+.4f}  p = {r.pvalue:.4f}")
u = stats.mannwhitneyu(short["n_shared"].to_numpy(), full["n_shared"].to_numpy(), alternative="less")
print(f"Mann-Whitney 'short lists have FEWER shared hands': p = {u.pvalue:.4f}")

# same within each family (exposure and family are correlated)
print("\nper family:")
for f in C.FAMILIES:
    g = d.filter(pl.col("behavior_family") == f)
    rr = stats.spearmanr(g["n_shared"].to_numpy(), g["n_listed"].to_numpy())
    print(f"  {f:<24} n={g.height:>3}  mean n_listed {g['n_listed'].mean():.3f}  "
          f"spearman {rr.statistic:+.4f} (p {rr.pvalue:.3f})")

# how many *planted-looking* hands does a pair have vs exposure?  proxy: the listed span
sp = ev.group_by("pair_id").agg(pl.col("hand_id").len().alias("k"))
print("\n=== eval exposure for reference ===")
epx = C.load("eval_pairs")
print(f"eval pairs shared hands: min {epx['shared_hands'].min()} / q05 {epx['shared_hands'].quantile(0.05):.0f} "
      f"/ q25 {epx['shared_hands'].quantile(0.25):.0f} / med {epx['shared_hands'].median():.0f}")
print(f"share of eval pairs below the dev-positive minimum (57): {(epx['shared_hands'] < 57).mean():.3f}")
