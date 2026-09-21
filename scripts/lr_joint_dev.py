"""Lane M4 stage 2: measure joint-law E[AP@5] ordering against marginal ranking on the 372 dev positives.

Two tiers:
  dp     pure DP -- rank_score = P(listed) from the DP, joint law = the DP's own law.  Self-consistent:
         answers "is marginal ranking E[AP@5]-optimal under the model that produced it?"
  blend  production -- rank_score = the shipped exp042_ramp blended score; the DP's dependence structure
         is rescaled to match the shipped marginals (copula-style), so lam = 0 reproduces the shipped list
         exactly and lam = 1 is the full joint correction.

Reported per family and overall (true-family routing, the protocol behind the shipped 0.7748), plus a
paired bootstrap over pairs and the rate at which the optimal ordering differs from the marginal one.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "scripts"))
import numpy as np
import polars as pl
import common as C
import lr_joint as JT

c = pl.col
FAMS = JT.FAMS
SH = JT.SH

ap = argparse.ArgumentParser()
ap.add_argument("--scratch", default=os.environ.get("SCRATCH", "") + "/m4")
ap.add_argument("--cand", type=int, default=10)
ap.add_argument("--lmin", type=int, default=1)
ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--mode", default="both")
ap.add_argument("--out", default="")
A = ap.parse_args()

SC = Path(A.scratch)
K = pl.read_parquet(SC / "dev_keys.parquet")
Z = np.load(SC / "dev_qpi.npz")
B = pl.read_parquet(C.DER / "evidence_scores_dev_exp042_ramp_exp027ev.parquet").drop("hand_id")
K = K.join(B, on=["pair_id", "hand_idx"], how="left", maintain_order="left")
assert K["s_soft_play"].null_count() == 0
# rows are already sorted by (pair_id, hand_seq) as load_dev_features left them
assert K.select((c("hand_seq").diff().over("pair_id") > 0).fill_null(True).all()).item(), "not time-sorted"

off = JT.run_pairs.__wrapped__ if False else None
pair_ids = K["pair_id"].to_numpy()
brk = np.flatnonzero(pair_ids[1:] != pair_ids[:-1]) + 1
offsets = np.concatenate([[0], brk, [len(pair_ids)]]).astype(np.int64)
G = len(offsets) - 1
fam_of_pair = K["family"].to_numpy()[offsets[:-1]]
n_rel = K["n_rel"].to_numpy()[offsets[:-1]]
is_ev = K["is_ev"].to_numpy().astype(np.int64)
pot = K["pot_bb"].to_numpy()
JT.log(f"{G} dev pairs, {len(K):,} rows, cand={A.cand} lmin={A.lmin} lam={A.lam}")

res = {"cand": A.cand, "lmin": A.lmin, "lam": A.lam, "pairs": G}
store = {}
for mode in (["dp", "blend"] if A.mode == "both" else [A.mode]):
    aps_m, aps_j, objs, diffs = {}, {}, {}, {}
    for f in FAMS:
        sh = SH[f]
        q = np.ascontiguousarray(Z[f"q_{sh}"])
        pi = np.ascontiguousarray(Z[f"pi_{sh}"])
        dp = np.ascontiguousarray(Z[f"dp_{sh}"])
        blend = np.ascontiguousarray(K[f"s_{f}"].to_numpy())
        # 1e-9 pot tie-break so the marginal ranking matches evidence.write_scores / map5 conventions
        tb = (pl.DataFrame({"p": pair_ids, "v": pot}).with_columns(
            r=(c("v").rank("average").over("p") / pl.len().over("p")) * 1e-9)["r"].to_numpy())
        if mode == "dp":
            rank_score, target, use_t = dp + tb, dp, 0
        else:
            rank_score, target, use_t = blend + tb, blend, 1
        t0 = time.time()
        lm, lj, ob = JT.run_pairs(offsets, q, pi, rank_score, np.ascontiguousarray(target),
                                  use_t, A.cand, A.lmin, A.lam, 5)
        sel = fam_of_pair == f
        aps_m[f] = JT.ap_from_lists(lm, is_ev, n_rel)
        aps_j[f] = JT.ap_from_lists(lj, is_ev, n_rel)
        objs[f] = ob
        same_set = np.array([set(lm[g][lm[g] >= 0]) == set(lj[g][lj[g] >= 0]) for g in range(G)])
        same_ord = (lm == lj).all(axis=1)
        diffs[f] = (same_set, same_ord)
        JT.log(f"{mode}/{sh}: {time.time() - t0:.0f}s  set-differs {1 - same_set[sel].mean():.3f}  "
               f"order-differs {1 - same_ord[sel].mean():.3f}  "
               f"E[AP] gain {np.mean(ob[sel, 1] - ob[sel, 0]):+.5f}")
    # route each pair to its true family (the protocol behind the shipped 0.7748)
    am = np.zeros(G)
    aj = np.zeros(G)
    og = np.zeros(G)
    sset = np.zeros(G, dtype=bool)
    sord = np.zeros(G, dtype=bool)
    for f in FAMS:
        s = fam_of_pair == f
        am[s], aj[s] = aps_m[f][s], aps_j[f][s]
        og[s] = objs[f][s, 1] - objs[f][s, 0]
        sset[s], sord[s] = diffs[f][0][s], diffs[f][1][s]
    row = {"overall_marginal": round(float(am.mean()), 4), "overall_joint": round(float(aj.mean()), 4)}
    p, d = JT.boot(am, aj)
    row["delta"] = round(d, 5)
    row["P_better"] = round(p, 3)
    row["model_E_AP_gain"] = round(float(og.mean()), 5)
    row["set_differs"] = round(float(1 - sset.mean()), 3)
    row["order_differs"] = round(float(1 - sord.mean()), 3)
    for f in FAMS:
        s = fam_of_pair == f
        row[SH[f]] = [round(float(am[s].mean()), 4), round(float(aj[s].mean()), 4), int(s.sum())]
    res[mode] = row
    store[mode] = (am, aj, og, sset, sord)
    JT.log(mode, json.dumps(row))

print(json.dumps(res, indent=1))
if A.out:
    Path(A.out).write_text(json.dumps(res, indent=1))
np.savez(SC / f"dev_m4_{A.mode}_c{A.cand}_l{A.lmin}_lam{A.lam}.npz",
         **{f"{m}_{k}": v for m, t in store.items() for k, v in
            zip(("am", "aj", "og", "sset", "sord"), t)}, fam=fam_of_pair.astype(str))
