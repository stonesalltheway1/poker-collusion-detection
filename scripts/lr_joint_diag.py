"""Lane M4 diagnostics: WHY the joint law does not move the ordering, and what would reopen it.

1. lambda-curve for the blend tier (lam = 0 must reproduce the shipped 0.7748 exactly).
2. The DP's law of the list length L per pair vs the truth (n_rel) -- is the 1/L channel calibrated at all?
3. The implied dependence among the top-5 candidates: corr(rel_i, rel_j) = (v_ij_unweighted - p_i p_j)/sd sd,
   i.e. how far the joint is from independence.
4. Distribution of the model's own E[AP@5] gain from reordering, and the ORACLE reordering ceiling
   (best possible AP of the shipped top-5 permuted, and of the best 5 of the shipped top-10).
"""
import os
import sys
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
FAMS, SH = JT.FAMS, JT.SH
SC = Path(os.environ["SCRATCH"]) / "m4"
K = pl.read_parquet(SC / "dev_keys.parquet")
Z = np.load(SC / "dev_qpi.npz")
B = pl.read_parquet(C.DER / "evidence_scores_dev_exp042_ramp_exp027ev.parquet").drop("hand_id")
K = K.join(B, on=["pair_id", "hand_idx"], how="left", maintain_order="left")
pair_ids = K["pair_id"].to_numpy()
brk = np.flatnonzero(pair_ids[1:] != pair_ids[:-1]) + 1
offsets = np.concatenate([[0], brk, [len(pair_ids)]]).astype(np.int64)
G = len(offsets) - 1
fam_of_pair = K["family"].to_numpy()[offsets[:-1]]
n_rel = K["n_rel"].to_numpy()[offsets[:-1]]
is_ev = K["is_ev"].to_numpy().astype(np.int64)
pot = K["pot_bb"].to_numpy()
nsh = np.diff(offsets)
tb = (pl.DataFrame({"p": pair_ids, "v": pot}).with_columns(
    r=(c("v").rank("average").over("p") / pl.len().over("p")) * 1e-9)["r"].to_numpy())

print("=" * 100)
print("1. lambda curve (blend tier, cand=10, lmin=1): lam scales the pairwise term; lam=0 == shipped ranking")
for lam in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
    am = np.zeros(G)
    aj = np.zeros(G)
    for f in FAMS:
        s = fam_of_pair == f
        bl = np.ascontiguousarray(K[f"s_{f}"].to_numpy())
        lm, lj, ob = JT.run_pairs(offsets, np.ascontiguousarray(Z[f"q_{SH[f]}"]),
                                  np.ascontiguousarray(Z[f"pi_{SH[f]}"]), bl + tb,
                                  bl, 1, 10, 1, lam, 5)
        am[s] = JT.ap_from_lists(lm, is_ev, n_rel)[s]
        aj[s] = JT.ap_from_lists(lj, is_ev, n_rel)[s]
    p, d = JT.boot(am, aj)
    print(f"  lam={lam:<5} shipped {am.mean():.4f} -> joint {aj.mean():.4f}  delta {d:+.5f}  P(better) {p:.3f}")

print("=" * 100)
print("2. the DP's law of the list length L vs the truth")
EL = np.zeros(G)
P5 = np.zeros(G)
Pshort = np.zeros(G)
for g in range(G):
    a, b = offsets[g], offsets[g + 1]
    f = fam_of_pair[g]
    q = Z[f"q_{SH[f]}"][a:b]
    F = JT._fwd(np.ascontiguousarray(q), -1, 8)[b - a]        # law of N (0..6, 7 = >=7)
    lawL = np.array([F[i] for i in range(5)] + [F[5:].sum()])  # L = min(N, 5)
    EL[g] = sum(i * lawL[i] for i in range(6))
    P5[g] = lawL[5]
    Pshort[g] = lawL[3] + lawL[4]
print(f"  model E[L]: mean {EL.mean():.3f}  p10 {np.percentile(EL, 10):.3f}  p90 {np.percentile(EL, 90):.3f}")
print(f"  model P(L = 5): mean {P5.mean():.3f}  P(L in 3,4): mean {Pshort.mean():.3f}")
print(f"  TRUTH  mean n_rel {n_rel.mean():.3f}   pairs with n_rel < 5: {(n_rel < 5).sum()} / {G}")
print(f"  corr(model E[L], true n_rel) = {np.corrcoef(EL, n_rel)[0, 1]:+.3f}   "
      f"corr(shared hands, true n_rel) = {np.corrcoef(nsh, n_rel)[0, 1]:+.3f}")
q4 = np.quantile(nsh, [0.25, 0.5, 0.75])
for lo, hi, nm in [(0, q4[0], "Q1 exposure"), (q4[2], 1e9, "Q4 exposure")]:
    s = (nsh >= lo) & (nsh < hi)
    print(f"    {nm}: true mean n_rel {n_rel[s].mean():.3f}   model E[L] {EL[s].mean():.3f}")

print("=" * 100)
print("3. dependence among the shipped top-5 (pure-DP law): corr(rel_i, rel_j) and the E[AP] gain spread")
cors = []
gains = []
for g in range(G):
    a, b = offsets[g], offsets[g + 1]
    f = fam_of_pair[g]
    q = np.ascontiguousarray(Z[f"q_{SH[f]}"][a:b])
    pi = np.ascontiguousarray(Z[f"pi_{SH[f]}"][a:b])
    dp = Z[f"dp_{SH[f]}"][a:b]
    cand = np.sort(np.argsort(-dp)[:10]).astype(np.int64)
    u, v, marg = JT.pair_uv(q, pi, cand, 1)
    # unweighted joint: recover P(rel_i, rel_j) by dividing out the (near-constant) 1/L factor
    # exact alternative: corr computed on the L-weighted quantities is the same sign/scale story
    top = np.argsort(-marg)[:5]
    for x in range(5):
        for y in range(x + 1, 5):
            i, j = top[x], top[y]
            pi_, pj = marg[i], marg[j]
            if min(pi_, pj) < 1e-6 or max(pi_, pj) > 1 - 1e-6:
                continue
            jt = v[i, j] * 5.0          # ~P(rel_i, rel_j) when L == 5 dominates
            cors.append((jt - pi_ * pj) / np.sqrt(pi_ * (1 - pi_) * pj * (1 - pj)))
    sel, bv = JT.best_order(u, v, 5)
    mo = np.argsort(-marg)[:5].astype(np.int64)
    gains.append(bv - JT.order_value(u, v, mo))
cors = np.array(cors)
gains = np.array(gains)
print(f"  implied corr(rel_i, rel_j) over top-5 pairs: mean {cors.mean():+.3f}  "
      f"p5 {np.percentile(cors, 5):+.3f}  p95 {np.percentile(cors, 95):+.3f}  |corr| > 0.2: {(np.abs(cors) > 0.2).mean():.3f}")
print(f"  per-pair model E[AP] gain from joint reordering: mean {gains.mean():.6f}  "
      f"p95 {np.percentile(gains, 95):.6f}  max {gains.max():.6f}  > 0.01 in {(gains > 0.01).mean():.4f} of pairs")

print("=" * 100)
print("4. oracle ceilings on the shipped lists (what an ordering fix could ever be worth)")
for f_ in [0]:
    base = np.zeros(G)
    orc5 = np.zeros(G)
    orc10 = np.zeros(G)
    for g in range(G):
        a, b = offsets[g], offsets[g + 1]
        f = fam_of_pair[g]
        s = K[f"s_{f}"].to_numpy()[a:b] + tb[a:b]
        ordr = np.argsort(-s)
        ev = is_ev[a:b]
        nr = min(n_rel[g], 5)
        top5 = ordr[:5]
        top10 = ordr[:10]
        hits, acc = 0, 0.0
        for k, r in enumerate(top5, 1):
            if ev[r]:
                hits += 1
                acc += hits / k
        base[g] = acc / nr
        h5 = int(ev[top5].sum())
        orc5[g] = sum((i + 1) / (i + 1) for i in range(h5)) / nr      # all hits first -> precision 1 each
        h10 = int(ev[top10].sum())
        orc10[g] = min(h10, 5) / nr
    print(f"  shipped {base.mean():.4f} | oracle re-order of the shipped top-5 {orc5.mean():.4f} "
          f"(+{orc5.mean() - base.mean():.4f}) | oracle best-5 of the shipped top-10 {orc10.mean():.4f} "
          f"(+{orc10.mean() - base.mean():.4f})")
