"""Lane M4 validation: brute-force the joint law on small synthetic pairs and check
(a) pair_uv's marginals == listing_rule.listed_prob, (b) u_i == E[rel_i/L], v_ij == E[rel_i rel_j/L],
(c) best_order returns the true E[AP@5] maximiser under exhaustive enumeration."""
import itertools
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "scripts"))
import numpy as np
import listing_rule as LR
import lr_joint as JT

CAP = 5


def states_of(q, pi):
    """Aggregate probability mass by the realised listed SET (as a bitmask)."""
    n, K = pi.shape
    st = defaultdict(float)
    for ev in itertools.product([0, 1], repeat=n):
        pe = 1.0
        for i in range(n):
            pe *= q[i] if ev[i] else (1 - q[i])
        if pe < 1e-15:
            continue
        idx = [i for i in range(n) if ev[i]]
        for ty in itertools.product(range(K), repeat=len(idx)):
            pt = pe
            for a, i in enumerate(idx):
                pt *= pi[i, ty[a]]
            if pt < 1e-15:
                continue
            order = sorted(range(len(idx)), key=lambda a: (ty[a], idx[a]))
            mask = 0
            for a in order[:CAP]:
                mask |= 1 << idx[a]
            st[mask] += pt
    return [(m, p, bin(m).count("1")) for m, p in st.items()]


def eap(states, order):
    tot = 0.0
    for mask, p, L in states:
        if L == 0:
            continue
        hits, s = 0, 0.0
        for k, h in enumerate(order, 1):
            if mask >> h & 1:
                hits += 1
                s += hits / k
        tot += p * s / L
    return tot


rng = np.random.default_rng(0)
ok = True
for trial in range(6):
    n = int(rng.integers(6, 9))
    K = 2
    q = rng.uniform(0.02, 0.55, n)
    t = rng.uniform(0.05, 0.95, n)
    pi = np.ascontiguousarray(np.stack([t, 1 - t], axis=1))
    states = states_of(q, pi)
    P2 = np.zeros(n)
    U = np.zeros(n)
    V = np.zeros((n, n))
    for mask, p, L in states:
        if L == 0:
            continue
        mem = [i for i in range(n) if mask >> i & 1]
        for i in mem:
            P2[i] += p
            U[i] += p / L
            for j in mem:
                if i != j:
                    V[i, j] += p / L
    marg_lr = LR.listed_prob(q, pi, CAP)
    cand = np.arange(n, dtype=np.int64)
    u, v, marg = JT.pair_uv(q, pi, cand, 1)
    e = [np.abs(marg - marg_lr).max(), np.abs(marg - P2).max(), np.abs(u - U).max(), np.abs(v - V).max()]
    sel, bv = JT.best_order(u, v, 5)
    e.append(abs(bv - eap(states, list(sel))))
    for _ in range(5):
        o = rng.permutation(n)[:5].astype(np.int64)
        e.append(abs(JT.order_value(u, v, o) - eap(states, list(o))))
    bestv, besto = -1.0, None
    for sub in itertools.permutations(range(n), 5):
        val = eap(states, list(sub))
        if val > bestv:
            bestv, besto = val, sub
    mo = np.argsort(-marg)[:5]
    margv = eap(states, list(mo))
    e.append(abs(bestv - eap(states, list(sel))))
    print(f"n={n} err(marg/brute/u/v/obj)={max(e[:4]):.2e} obj_err={max(e[4:10]):.2e} opt_gap={e[-1]:.2e} "
          f"| E[AP] marginal-rank {margv:.5f} -> joint-opt {bestv:.5f}  delta {bestv - margv:+.5f} "
          f"| same list: {tuple(mo) == tuple(sel)}")
    ok &= max(e) < 1e-9
print("ALL EXACT" if ok else "MISMATCH")
