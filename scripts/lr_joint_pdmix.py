"""Lane M4, partner/DP tier: the shipped blend s = (1-w) s_partner + w P_DP(listed) is the MARGINAL of a
two-component mixture of laws -- with probability (1-w) the partner's (maximum-entropy, independent)
relevance law holds, with probability w the listing DP's law holds.  Ranking by s is therefore ranking by
the mixture's marginal, which is E[AP@5]-optimal only if the mixture is unimodal.  It is not: the two
experts disagree on which hands make the list (exp036 vs exp027ev share 4.28/5 on the top eval pairs).

This applies the exact E[AP@5] objective to that mixture.  Unlike the family mixture it covers 100% of
pairs, and the baseline is EXACTLY the shipped ranking, so the comparison is perfectly matched.
"""
import json
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
from numba import njit
import common as C
import lr_joint as JT

c = pl.col
FAMS, SH, CAP, NC = JT.FAMS, JT.SH, JT.CAP, JT.NC


@njit(cache=True)
def pair_uv_indep(p, cand, lmin):
    """Maximum-entropy law given the partner's per-hand P(listed): rel_i ~ Bern(p_i) independent,
    L = min(sum rel, 5).  u_i = E[rel_i/L], v_ij = E[rel_i rel_j/L]."""
    n = p.shape[0]
    Cn = cand.shape[0]
    u = np.zeros(Cn)
    v = np.zeros((Cn, Cn))
    Sq = JT._bwd(p, -1, NC)
    Fs = np.zeros((Cn, n + 1, NC))
    for a in range(Cn):
        Fa = JT._fwd(p, cand[a], NC)
        for j in range(n + 1):
            for t in range(NC):
                Fs[a, j, t] = Fa[j, t]
    for a in range(Cn):
        i = cand[a]
        s = 0.0
        for m in range(NC):
            L = m + 1 if m + 1 < CAP else CAP
            if L >= lmin:
                s += Fs[a, n, m] / L
        u[a] = p[i] * s
    Nij = np.zeros(NC)
    for a in range(Cn):
        i = cand[a]
        for b in range(a + 1, Cn):
            j = cand[b]
            tot = 0.0
            for t in range(NC - 1):
                Nij[t] = JT._conv_eq(Fs[a, j], Sq[j + 1], t)
                tot += Nij[t]
            Nij[NC - 1] = 1.0 - tot
            s = 0.0
            for m in range(NC):
                L = m + 2 if m + 2 < CAP else CAP
                if L >= lmin:
                    s += Nij[m] / L
            val = p[i] * p[j] * s
            v[a, b] = val
            v[b, a] = val
    return u, v


@njit(cache=True)
def run_pd(offsets, q, pi, sp, wdp, ncand, lmin, lam, k5):
    """sp = partner P(listed) per row; wdp = per-pair DP mixture weight (the exp042 ramp)."""
    G = offsets.shape[0] - 1
    out_m = -np.ones((G, k5), dtype=np.int64)
    out_j = -np.ones((G, k5), dtype=np.int64)
    obj = np.zeros((G, 2))
    for g in range(G):
        a, b = offsets[g], offsets[g + 1]
        n = b - a
        if n == 0:
            continue
        w = wdp[g]
        mix = np.zeros(n)
        for i in range(n):
            mix[i] = (1.0 - w) * sp[a + i] + w * q[a + i] * 0.0      # placeholder, filled below
        # the mixture marginal is exactly the shipped blend: (1-w) s_partner + w P_DP
        ordr = np.argsort(-sp[a:b])                                   # provisional, refined after the DP
        Cn = ncand if ncand < n else n
        # candidates: top-ncand of the true blended marginal, so compute the DP marginal first
        u_d, v_d, marg_d = JT.pair_uv(np.ascontiguousarray(q[a:b]), np.ascontiguousarray(pi[a:b]),
                                      np.arange(n).astype(np.int64), lmin)
        for i in range(n):
            mix[i] = (1.0 - w) * sp[a + i] + w * marg_d[i]
        ordr = np.argsort(-mix)
        cand = np.sort(ordr[:Cn])
        u_p, v_p = pair_uv_indep(np.ascontiguousarray(sp[a:b]), cand, lmin)
        U = np.zeros(Cn)
        V = np.zeros((Cn, Cn))
        for x in range(Cn):
            U[x] = (1.0 - w) * u_p[x] + w * u_d[cand[x]]
            for y in range(Cn):
                V[x, y] = (1.0 - w) * v_p[x, y] + w * lam * v_d[cand[x], cand[y]]
        pos = np.argsort(-mix[cand])
        kk = k5 if k5 < Cn else Cn
        mo = pos[:kk].astype(np.int64)
        sel, bv = JT.best_order(U, V, kk)
        obj[g, 0] = JT.order_value(U, V, mo)
        obj[g, 1] = bv
        for t in range(kk):
            out_m[g, t] = a + cand[mo[t]]
            out_j[g, t] = a + cand[sel[t]]
    return out_m, out_j, obj


SC = Path(os.environ["SCRATCH"]) / "m4"
NCAND = int(sys.argv[1]) if len(sys.argv) > 1 else 10
K = pl.read_parquet(SC / "dev_keys.parquet")
Z = np.load(SC / "dev_qpi.npz")
CP = pl.read_parquet(C.DER / "evidence_cache" / "exp042_ramp_exp027ev_dev_components.parquet")
PT = pl.read_parquet(C.DER / "evidence_scores_dev_exp027ev.parquet").drop("hand_id")
K = (K.join(CP.select("pair_id", "hand_idx", "w"), on=["pair_id", "hand_idx"], how="left", maintain_order="left")
     .join(PT, on=["pair_id", "hand_idx"], how="left", maintain_order="left"))
pair_ids = K["pair_id"].to_numpy()
brk = np.flatnonzero(pair_ids[1:] != pair_ids[:-1]) + 1
offsets = np.concatenate([[0], brk, [len(pair_ids)]]).astype(np.int64)
G = len(offsets) - 1
fam_of_pair = K["family"].to_numpy()[offsets[:-1]]
n_rel = K["n_rel"].to_numpy()[offsets[:-1]]
is_ev = K["is_ev"].to_numpy().astype(np.int64)
pot = K["pot_bb"].to_numpy()
tb = (pl.DataFrame({"p": pair_ids, "v": pot}).with_columns(
    r=(c("v").rank("average").over("p") / pl.len().over("p")) * 1e-9)["r"].to_numpy())
wdp = K["w"].to_numpy()[offsets[:-1]]
print(f"ramp weight w: mean {wdp.mean():.3f} min {wdp.min():.3f} max {wdp.max():.3f}; "
      f"pairs with w < 0.7: {(wdp < 0.699).mean():.3f}")

res = {}
for lam in (1.0,):
    am = np.zeros(G)
    aj = np.zeros(G)
    og = np.zeros(G)
    sd = np.zeros(G, dtype=bool)
    for f in FAMS:
        s = fam_of_pair == f
        sp = np.ascontiguousarray(np.clip(K[f"s_{f}"].to_numpy(), 0.0, 1.0) + tb)
        lm, lj, ob = run_pd(offsets, np.ascontiguousarray(Z[f"q_{SH[f]}"]),
                            np.ascontiguousarray(Z[f"pi_{SH[f]}"]), sp, wdp, NCAND, 1, lam, 5)
        am[s] = JT.ap_from_lists(lm, is_ev, n_rel)[s]
        aj[s] = JT.ap_from_lists(lj, is_ev, n_rel)[s]
        og[s] = (ob[:, 1] - ob[:, 0])[s]
        sd[s] = (~(lm == lj).all(axis=1))[s]
    p, d = JT.boot(am, aj)
    row = dict(shipped_equiv=round(float(am.mean()), 4), joint=round(float(aj.mean()), 4), delta=round(d, 5),
               P_better=round(p, 3), model_E_AP_gain=round(float(og.mean()), 5),
               order_differs=round(float(sd.mean()), 3))
    for f in FAMS:
        s = fam_of_pair == f
        row[SH[f]] = [round(float(am[s].mean()), 4), round(float(aj[s].mean()), 4)]
    res[f"lam{lam}"] = row
    print(f"  lam={lam}: {json.dumps(row)}")
(SC / "pdmix.json").write_text(json.dumps(res, indent=1))
