"""Lane M4: rank a pair's evidence hands by EXACT expected AP@5 under the listing DP's JOINT law.

The shipped scorer ranks by the marginal P(listed_i) from the Poisson-binomial DP of
scripts/listing_rule.py and takes the top 5.  Marginal ranking is E[AP@5]-optimal only under
INDEPENDENCE; the listing events are not independent (the cap-at-5 competition and the (type, time)
ordering couple them), and the host's AP@5 denominator min(|relevant|, 5) makes the payoff depend on
the realised list LENGTH L, which the DP also models.

Exact objective for an ordered list (h_1..h_5), with rel_i = 1{i listed} and L = min(#events, 5):

    E[AP@5] = sum_k (1/k) E[rel_{h_k}/L] + sum_{j<k} (1/k) E[rel_{h_j} rel_{h_k}/L]
            = sum_k (1/k) ( u_{h_k} + sum_{j<k} v_{h_j h_k} )

so everything reduces to the L-weighted marginal u_i and the L-weighted pairwise joint v_ij, both
computed EXACTLY from (q, pi) with the same prefix/suffix Poisson-binomial machinery:

  * L < 5  =>  every event is listed, so {rel_i = 1, L = c} == {E_i = 1, N_{-i} = c-1}
  * L = 5  =>  the residual P(listed_i) - q_i P(N_{-i} <= 3)
  * P(listed_i, listed_j): given types (k, l) one of the two precedes the other in (type, time)
    order; the later one w is listed iff at most 3 OTHER events precede it, and then the earlier one
    is listed too.  Leave-two-out Poisson-binomial with the same pre/suf split as listed_prob.

Gameplay only: the inputs are the (q, pi) of the exp042_ramp heads.  No IDs, no file/row order, no
eval-file construction.  Stage 1 (scripts/lr_joint_fit.py) caches (q, pi) per dev row.

Usage:  python scripts/lr_joint.py [--cand 10] [--lmin 1] [--lam 1.0] [--mode dp|blend|both]
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
from numba import njit

c = pl.col
FAMS = ("directed_transfer", "soft_play", "coordinated_isolation")
SH = {"directed_transfer": "dt", "soft_play": "sp", "coordinated_isolation": "ci"}
CAP = 5
NC = CAP + 1          # exact counts 0..CAP-1, last index saturates at ">= CAP"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ============================================================ Poisson-binomial helpers
@njit(cache=True)
def _fwd(p, skip, nc):
    """F[j] = law of #successes over m < j, m != skip (last index saturates)."""
    n = p.shape[0]
    F = np.zeros((n + 1, nc))
    F[0, 0] = 1.0
    for m in range(n):
        if m == skip:
            for t in range(nc):
                F[m + 1, t] = F[m, t]
            continue
        pm = p[m]
        for t in range(nc):
            v = F[m, t]
            if v == 0.0:
                continue
            F[m + 1, t] += v * (1.0 - pm)
            tt = t + 1 if t < nc - 1 else nc - 1
            F[m + 1, tt] += v * pm
    return F


@njit(cache=True)
def _bwd(p, skip, nc):
    """B[j] = law of #successes over m >= j, m != skip."""
    n = p.shape[0]
    B = np.zeros((n + 2, nc))
    B[n, 0] = 1.0
    for m in range(n - 1, -1, -1):
        if m == skip:
            for t in range(nc):
                B[m, t] = B[m + 1, t]
            continue
        pm = p[m]
        for t in range(nc):
            v = B[m + 1, t]
            if v == 0.0:
                continue
            B[m, t] += v * (1.0 - pm)
            tt = t + 1 if t < nc - 1 else nc - 1
            B[m, tt] += v * pm
    return B


@njit(cache=True)
def _conv_le(A, B, thr):
    """P(a + b <= thr) for two independent saturating count laws (thr < nc-1 required)."""
    s = 0.0
    nc = A.shape[0]
    for i in range(nc):
        ai = A[i]
        if ai == 0.0 or i > thr:
            continue
        for j in range(thr - i + 1):
            s += ai * B[j]
    return s


@njit(cache=True)
def _conv_eq(A, B, val):
    s = 0.0
    for i in range(val + 1):
        s += A[i] * B[val - i]
    return s


# ============================================================ u, v for one pair / family
@njit(cache=True)
def pair_uv(q, pi, cand, lmin):
    """q (n,), pi (n,K) in TIME order; cand (C,) ascending time indices.
    Returns u (C,), v (C,C), marg (C,) with u_i = E[rel_i/L], v_ij = E[rel_i rel_j/L],
    marg_i = P(listed_i).  lmin drops the L < lmin terms (true positives always have L >= 3)."""
    n = q.shape[0]
    K = pi.shape[1]
    C = cand.shape[0]
    u = np.zeros(C)
    v = np.zeros((C, C))
    marg = np.zeros(C)

    Sq = _bwd(q, -1, NC)
    Fskip = np.zeros((C, n + 1, NC))
    for a in range(C):
        Fa = _fwd(q, cand[a], NC)
        for j in range(n + 1):
            for t in range(NC):
                Fskip[a, j, t] = Fa[j, t]

    cw = np.zeros((n, K + 1))          # cw[m, l] = sum_{t < l} pi[m, t]
    for m in range(n):
        s = 0.0
        for t in range(K):
            cw[m, t] = s
            s += pi[m, t]
        cw[m, K] = s

    J = np.zeros((C, C, K))            # J[w, e, l] = P(#events other than w,e before w <= CAP-2)
    M = np.zeros((C, K))               # M[w, l]    = P(#events other than w before w <= CAP-1)
    pv = np.zeros(n)
    for a in range(C):
        w = cand[a]
        for l in range(K):
            for m in range(n):
                if m < w:
                    pv[m] = q[m] * (cw[m, l] + pi[m, l])
                else:
                    pv[m] = q[m] * cw[m, l]
            Pf = _fwd(pv, w, NC)
            Bf = _bwd(pv, w, NC)
            M[a, l] = _conv_le(Pf[w], Bf[w + 1], CAP - 1)
            for b in range(C):
                e = cand[b]
                if e == w:
                    continue
                J[a, b, l] = _conv_le(Pf[e], Bf[e + 1], CAP - 2)

    for a in range(C):
        i = cand[a]
        s = 0.0
        for l in range(K):
            s += pi[i, l] * M[a, l]
        marg[a] = q[i] * s

    for a in range(C):                                   # u_i
        i = cand[a]
        tail = 0.0
        acc = 0.0
        for cc in range(1, CAP):                         # L = cc, all events listed
            pr = Fskip[a, n, cc - 1]
            tail += pr
            if cc >= lmin:
                acc += pr / cc
        acc *= q[i]
        resid = marg[a] - q[i] * tail                    # P(rel_i, L = CAP)
        if resid < 0.0:
            resid = 0.0
        u[a] = acc + resid / CAP

    Nij = np.zeros(NC)
    for a in range(C):                                   # v_ij
        i = cand[a]
        for b in range(a + 1, C):
            j = cand[b]
            jp = 0.0
            for k in range(K):
                pik = pi[i, k]
                if pik == 0.0:
                    continue
                for l in range(K):
                    pjl = pi[j, l]
                    if pjl == 0.0:
                        continue
                    if k <= l:                           # cand is time-ascending -> j is later
                        jp += pik * pjl * J[b, a, l]
                    else:                                # i is later
                        jp += pik * pjl * J[a, b, k]
            jp *= q[i] * q[j]
            tot = 0.0
            for t in range(NC - 1):
                Nij[t] = _conv_eq(Fskip[a, j], Sq[j + 1], t)
                tot += Nij[t]
            Nij[NC - 1] = 1.0 - tot
            tail = 0.0
            acc = 0.0
            for cc in range(2, CAP):                     # L = cc, both listed
                pr = Nij[cc - 2]
                tail += pr
                if cc >= lmin:
                    acc += pr / cc
            acc *= q[i] * q[j]
            resid = jp - q[i] * q[j] * tail
            if resid < 0.0:
                resid = 0.0
            val = acc + resid / CAP
            v[a, b] = val
            v[b, a] = val
    return u, v, marg


# ============================================================ best ordered 5-list
@njit(cache=True)
def best_order(u, v, k5):
    """Exact search over ordered k5-subsets maximising sum_k (1/k)(u_hk + sum_{j<k} v_hj,hk)."""
    C = u.shape[0]
    best = -1e18
    sel = np.zeros(k5, dtype=np.int64)
    cur = np.zeros(k5, dtype=np.int64)
    idx = np.zeros(k5, dtype=np.int64)
    gain = np.zeros((k5 + 1, C))
    score = np.zeros(k5 + 1)
    used = np.zeros(C, dtype=np.bool_)
    for x in range(C):
        gain[0, x] = u[x]
    d = 0
    idx[0] = -1
    while d >= 0:
        idx[d] += 1
        if idx[d] >= C:
            d -= 1
            if d >= 0:
                used[cur[d]] = False
            continue
        x = idx[d]
        if used[x]:
            continue
        cur[d] = x
        used[x] = True
        score[d + 1] = score[d] + gain[d, x] / (d + 1.0)
        if d == k5 - 1:
            if score[d + 1] > best:
                best = score[d + 1]
                for t in range(k5):
                    sel[t] = cur[t]
            used[x] = False
        else:
            for y in range(C):
                gain[d + 1, y] = gain[d, y] + v[x, y]
            d += 1
            idx[d] = -1
    return sel, best


@njit(cache=True)
def order_value(u, v, order):
    s = 0.0
    for k in range(order.shape[0]):
        g = u[order[k]]
        for j in range(k):
            g += v[order[j], order[k]]
        s += g / (k + 1.0)
    return s


@njit(cache=True)
def run_pairs(offsets, q, pi, rank_score, target, use_target, ncand, lmin, lam, k5):
    """Per pair: candidates = top-ncand by rank_score.  Returns the ordered global row ids of the
    marginal ranking and of the joint-optimal list, plus both objective values."""
    G = offsets.shape[0] - 1
    out_m = -np.ones((G, k5), dtype=np.int64)
    out_j = -np.ones((G, k5), dtype=np.int64)
    obj = np.zeros((G, 2))
    for g in range(G):
        a, b = offsets[g], offsets[g + 1]
        n = b - a
        if n == 0:
            continue
        rs = rank_score[a:b]
        ordr = np.argsort(-rs)
        C = ncand if ncand < n else n
        cand = np.sort(ordr[:C])
        u, v, marg = pair_uv(q[a:b], pi[a:b], cand, lmin)
        if use_target == 1:
            # keep the DP's dependence structure but match the shipped (blended) marginals
            fdef = 0.0
            nf = 0
            for x in range(C):
                if marg[x] > 1e-9:
                    fdef += u[x] / marg[x]
                    nf += 1
            fdef = fdef / nf if nf > 0 else 1.0 / CAP
            rat = np.ones(C)
            for x in range(C):
                tx = target[a + cand[x]]
                if marg[x] > 1e-9:
                    r = tx / marg[x]
                    if r > 20.0:
                        r = 20.0
                    rat[x] = r
                    u[x] = u[x] * r
                else:
                    rat[x] = 0.0
                    u[x] = tx * fdef
            for x in range(C):
                for y in range(C):
                    v[x, y] *= rat[x] * rat[y]
        if lam != 1.0:
            for x in range(C):
                for y in range(C):
                    v[x, y] *= lam
        pos = np.argsort(-rs[cand])
        kk = k5 if k5 < C else C
        mo = pos[:kk].astype(np.int64)
        sel, bv = best_order(u, v, kk)
        obj[g, 0] = order_value(u, v, mo)
        obj[g, 1] = bv
        for t in range(kk):
            out_m[g, t] = a + cand[mo[t]]
            out_j[g, t] = a + cand[sel[t]]
    return out_m, out_j, obj


# ============================================================ evaluation helpers
def ap_from_lists(lists, is_ev, n_rel):
    aps = np.zeros(len(lists))
    for g in range(len(lists)):
        hits, s = 0, 0.0
        for k in range(lists.shape[1]):
            r = lists[g, k]
            if r < 0:
                continue
            if is_ev[r]:
                hits += 1
                s += hits / (k + 1.0)
        aps[g] = s / min(n_rel[g], 5)
    return aps


def boot(a, b, nboot=4000, seed=7):
    rng = np.random.default_rng(seed)
    d = np.asarray(b) - np.asarray(a)
    idx = rng.integers(0, len(d), size=(nboot, len(d)))
    return float((d[idx].mean(axis=1) > 0).mean()), float(d.mean())


def load_dev(scratch):
    K = pl.read_parquet(Path(scratch) / "dev_keys.parquet")
    Z = np.load(Path(scratch) / "dev_qpi.npz")
    return K, Z


@njit(cache=True)
def run_pairs_mix(offsets, q3, pi3, s3, post, ncand, lmin, lam, k5):
    """Family-MIXTURE tier.  compose_evidence ranks by mix_i = sum_f P_f s_f(i); the corresponding
    joint law is the P_f-weighted MIXTURE of the three family DPs, which is genuinely multimodal when
    the posterior is flat.  E[AP] is linear in the mixture, so u and v just mix by P_f."""
    G = offsets.shape[0] - 1
    nf = q3.shape[0]
    out_m = -np.ones((G, k5), dtype=np.int64)
    out_j = -np.ones((G, k5), dtype=np.int64)
    obj = np.zeros((G, 2))
    for g in range(G):
        a, b = offsets[g], offsets[g + 1]
        n = b - a
        if n == 0:
            continue
        mix = np.zeros(n)
        for f in range(nf):
            pf = post[g, f]
            if pf == 0.0:
                continue
            for i in range(n):
                mix[i] += pf * s3[f, a + i]
        ordr = np.argsort(-mix)
        C = ncand if ncand < n else n
        cand = np.sort(ordr[:C])
        U = np.zeros(C)
        V = np.zeros((C, C))
        for f in range(nf):
            pf = post[g, f]
            if pf <= 0.0:
                continue
            u, v, marg = pair_uv(np.ascontiguousarray(q3[f, a:b]), np.ascontiguousarray(pi3[f, a:b]), cand, lmin)
            fdef = 0.0
            nfz = 0
            for x in range(C):
                if marg[x] > 1e-9:
                    fdef += u[x] / marg[x]
                    nfz += 1
            fdef = fdef / nfz if nfz > 0 else 1.0 / CAP
            rat = np.zeros(C)
            for x in range(C):
                tx = s3[f, a + cand[x]]
                if marg[x] > 1e-9:
                    r = tx / marg[x]
                    if r > 20.0:
                        r = 20.0
                    rat[x] = r
                    U[x] += pf * u[x] * r
                else:
                    U[x] += pf * tx * fdef
            for x in range(C):
                for y in range(C):
                    V[x, y] += pf * v[x, y] * rat[x] * rat[y]
        if lam != 1.0:
            for x in range(C):
                for y in range(C):
                    V[x, y] *= lam
        pos = np.argsort(-mix[cand])
        kk = k5 if k5 < C else C
        mo = pos[:kk].astype(np.int64)
        sel, bv = best_order(U, V, kk)
        obj[g, 0] = order_value(U, V, mo)
        obj[g, 1] = bv
        for t in range(kk):
            out_m[g, t] = a + cand[mo[t]]
            out_j[g, t] = a + cand[sel[t]]
    return out_m, out_j, obj
