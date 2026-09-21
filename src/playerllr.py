"""exp062 -- PER-PLAYER RESIDUAL POLICY (lane M2): a player-specific normal-play baseline at ACTION
granularity, so the resulting per-hand surprise can reorder hands WITHIN a pair.

Why this is not the exp050 rerun.  exp050 recalibrated the population policy per (player, phase, 8 equity
buckets) and fed pair-level aggregates into the event head (-0.0096 dev MAP, P .004).  Its own diagnosis was
that 8 buckets cannot reorder hands inside a pair.  Here the baseline is a *logistic offset over the same
state features the population policy uses*, fitted on the player's own decisions, so p_player(a | s) differs
from p_pop(a | s) hand by hand and action by action.

Contamination.  The task's suggested guard (drop hands where the partner is seated) is per-(player, partner)
and needs ~225k refits on eval.  Phase-locality gives a strictly cleaner and cheaper guard: colluder sets are
DISJOINT across the two phases, so a player's OTHER-phase decisions are collusion-free by construction.
  * dev-phase hands (the 372 labelled positive pairs, CV) are scored with a baseline fitted on eval-phase hands
  * eval-phase hands (the 112,540 eval pairs)          are scored with a baseline fitted on dev-phase hands
CLAUDE.md sanctions exactly this ("use the other phase only as a baseline").

Model.  Three binary sub-decisions, each with the population log-odds as an OFFSET so gamma = 0 reproduces the
population policy EXACTLY (hence the excess surprise is identically 0 at gamma = 0):
    k=0  facing a bet  (legal & 1):                  fold          vs  continue
    k=1  facing a bet, not folding (legal & 8):      aggressive    vs  call
    k=2  no bet to call (legal == 10):               aggressive    vs  check
    logit p_player = logit p_pop + x . gamma_{player,k},   ridge (lambda/2)||gamma||^2 toward the population.
x = [1, s, s^2, postflop, postflop*s, pot_odds] (standardised), s = own perceived strength
    (pf_eq_vs1 preflop, hs_actor postflop) -- the same information the population policy already sees.

Per (hand, player) outputs (data/derived/player_llr_hand.parquet):
    nact     number of the player's decisions in the hand
    psurp    -sum log p_player(actual)                         player-specific surprise of the whole hand
    dsurp    -sum [log p_player(actual) - log p_pop(actual)]   EXCESS surprise vs the population policy
    pmaxs    max over actions of -log p_player(actual)
    dmaxs    max over actions of -(log p_player - log p_pop)
    fsurp/fdsurp  the same two sums restricted to facing-a-bet decisions

    python src/playerllr.py fit      # fit both phases, report held-out cross-phase log-likelihood, pick lambda
    python src/playerllr.py build    # write data/derived/player_llr_hand.parquet
    python src/playerllr.py all
"""
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")

import numpy as np
import polars as pl
from numba import njit, prange

BASE = Path(__file__).resolve().parent.parent
DER = BASE / "data" / "derived"
STATES = sorted((DER / "policy_states").glob("*.parquet"))
OUT = DER / "player_llr_hand.parquet"
GAM = DER / "player_llr_gamma.npz"
D = 6
NCTX = 3
LAM = float(os.environ.get("PLLR_LAM", 25.0))
T0 = time.time()
c = pl.col
# standardisation of the non-intercept design columns (global, unsupervised, frozen)
MU = np.array([0.0, 0.5000, 0.1000, 0.4700, 0.2400, 0.2600], dtype=np.float64)
SD = np.array([1.0, 0.2200, 0.1100, 0.5000, 0.3000, 0.1450], dtype=np.float64)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ data
def load_phase(phase):
    """All decisions of one phase with the design matrix inputs, the population probabilities and the target."""
    cols = ["hand_idx", "action_no", "player_idx", "y", "legal", "street", "pf_eq_vs1", "hs_actor", "pot_odds"]
    out = []
    for f in STATES:
        S = pl.scan_parquet(f).filter(c("phase") == phase).select(cols).collect()
        if not S.height:
            continue
        lo, hi = S["hand_idx"].min(), S["hand_idx"].max()
        A = (pl.scan_parquet(DER / "action_policy.parquet")
             .filter(c("hand_idx").is_between(lo, hi))
             .select("hand_idx", "action_no", "p_fold", "p_check", "p_call", "p_agg").collect())
        out.append(S.join(A, on=["hand_idx", "action_no"], how="inner"))
        del S, A
    S = pl.concat(out)
    del out
    s = (pl.when(c("street") == 0).then(c("pf_eq_vs1")).otherwise(c("hs_actor"))).fill_nan(0.5).fill_null(0.5)
    post = (c("street") > 0).cast(pl.Float64)
    S = S.with_columns(x1=s, x2=s * s, x3=post, x4=post * s, x5=c("pot_odds").cast(pl.Float64))
    S = S.sort("player_idx", "hand_idx", "action_no")
    log(f"phase {phase}: {S.height:,} decisions")
    return S


def design(S):
    X = np.ones((S.height, D), dtype=np.float32)
    for j, k in enumerate(["x1", "x2", "x3", "x4", "x5"], start=1):
        X[:, j] = ((S[k].to_numpy() - MU[j]) / SD[j]).astype(np.float32)
    return X


def contexts(S):
    """ctx per row (-1 = none), binary target, population offset, population log P(actual)."""
    y = S["y"].to_numpy().astype(np.int64)
    lg = S["legal"].to_numpy().astype(np.int64)
    pf = np.clip(S["p_fold"].to_numpy().astype(np.float64), 1e-9, 1 - 1e-9)
    pc = np.clip(S["p_call"].to_numpy().astype(np.float64), 1e-12, 1.0)
    pa = np.clip(S["p_agg"].to_numpy().astype(np.float64), 1e-12, 1.0)
    pk = np.clip(S["p_check"].to_numpy().astype(np.float64), 1e-12, 1.0)
    facing = (lg & 1) > 0
    aggleg = (lg & 8) > 0
    n = len(y)
    ctx = np.full((n, 2), -1, dtype=np.int8)
    tgt = np.zeros((n, 2), dtype=np.float32)
    off = np.zeros((n, 2), dtype=np.float32)
    lo = np.log
    ctx[facing, 0] = 0
    tgt[facing, 0] = (y[facing] == 0).astype(float)
    off[facing, 0] = lo(pf[facing] / (1.0 - pf[facing]))
    m1 = facing & aggleg & (y != 0)
    ctx[m1, 1] = 1
    tgt[m1, 1] = (y[m1] == 3).astype(float)
    off[m1, 1] = lo(pa[m1] / pc[m1])
    m2 = ~facing
    ctx[m2, 0] = 2
    tgt[m2, 0] = (y[m2] == 3).astype(float)
    off[m2, 0] = lo(pa[m2] / pk[m2])
    lp = np.zeros(n, dtype=np.float32)
    a0 = facing & (y == 0)
    lp[a0] = lo(pf[a0])
    mm = facing & (y != 0)
    lp[mm] = lo(1.0 - pf[mm])
    mm3 = facing & aggleg & (y == 3)
    lp[mm3] += lo(pa[mm3] / (pa[mm3] + pc[mm3]))
    mm2 = facing & aggleg & (y == 2)
    lp[mm2] += lo(pc[mm2] / (pa[mm2] + pc[mm2]))
    b3 = m2 & (y == 3)
    lp[b3] = lo(pa[b3] / (pa[b3] + pk[b3]))
    b1 = m2 & (y == 1)
    lp[b1] = lo(pk[b1] / (pa[b1] + pk[b1]))
    return ctx, tgt, off, lp, facing


# ------------------------------------------------------------------ fit
@njit(cache=True, parallel=True)
def fit_groups(gstart, gend, X, ctx, tgt, off, lam, niter):
    """One ridge logistic offset model per (player, context)."""
    G = gstart.shape[0]
    nd = X.shape[1]
    out = np.zeros((G, NCTX, nd))
    for g in prange(G):
        a, b = gstart[g], gend[g]
        for k in range(NCTX):
            gam = np.zeros(nd)
            m = 0
            for i in range(a, b):
                for s in range(2):
                    if ctx[i, s] == k:
                        m += 1
            if m < 25:
                out[g, k] = gam
                continue
            for _it in range(niter):
                grad = np.zeros(nd)
                H = np.zeros((nd, nd))
                for j in range(nd):
                    grad[j] = lam * gam[j]
                    H[j, j] = lam
                for i in range(a, b):
                    for s in range(2):
                        if ctx[i, s] != k:
                            continue
                        e = off[i, s]
                        for j in range(nd):
                            e += X[i, j] * gam[j]
                        if e > 35.0:
                            e = 35.0
                        elif e < -35.0:
                            e = -35.0
                        p = 1.0 / (1.0 + np.exp(-e))
                        r = p - tgt[i, s]
                        w = p * (1.0 - p)
                        if w < 1e-7:
                            w = 1e-7
                        for j in range(nd):
                            grad[j] += X[i, j] * r
                            for j2 in range(nd):
                                H[j, j2] += w * X[i, j] * X[i, j2]
                Aug = np.zeros((nd, nd + 1))
                for j in range(nd):
                    for j2 in range(nd):
                        Aug[j, j2] = H[j, j2]
                    Aug[j, nd] = grad[j]
                ok = True
                for col in range(nd):
                    piv = col
                    best = abs(Aug[col, col])
                    for r2 in range(col + 1, nd):
                        if abs(Aug[r2, col]) > best:
                            best = abs(Aug[r2, col])
                            piv = r2
                    if best < 1e-10:
                        ok = False
                        break
                    if piv != col:
                        for j2 in range(nd + 1):
                            tmp = Aug[col, j2]
                            Aug[col, j2] = Aug[piv, j2]
                            Aug[piv, j2] = tmp
                    for r2 in range(col + 1, nd):
                        fac = Aug[r2, col] / Aug[col, col]
                        for j2 in range(col, nd + 1):
                            Aug[r2, j2] -= fac * Aug[col, j2]
                if not ok:
                    break
                step = np.zeros(nd)
                for j in range(nd - 1, -1, -1):
                    ssum = Aug[j, nd]
                    for j2 in range(j + 1, nd):
                        ssum -= Aug[j, j2] * step[j2]
                    step[j] = ssum / Aug[j, j]
                mx = 0.0
                for j in range(nd):
                    gam[j] -= step[j]
                    if abs(step[j]) > mx:
                        mx = abs(step[j])
                if mx < 1e-6:
                    break
            out[g, k] = gam
    return out


@njit(cache=True, parallel=True)
def score_rows(gidx, X, ctx, tgt, off, lp, gamma, plog):
    """Per-row player log P(actual), written into `plog` (float32, preallocated by the caller)."""
    n = X.shape[0]
    nd = X.shape[1]
    for i in prange(n):
        g = gidx[i]
        tot = 0.0
        for s in range(2):
            k = ctx[i, s]
            if k < 0:
                continue
            e = off[i, s]
            for j in range(nd):
                e += X[i, j] * gamma[g, k, j]
            if e > 35.0:
                e = 35.0
            elif e < -35.0:
                e = -35.0
            p = 1.0 / (1.0 + np.exp(-e))
            if p < 1e-12:
                p = 1e-12
            if p > 1.0 - 1e-12:
                p = 1.0 - 1e-12
            tot += np.log(p) if tgt[i, s] > 0.5 else np.log(1.0 - p)
        plog[i] = np.float32(tot)


def score_all(B, gamma, chunk=1_500_000):
    """Chunked wrapper: returns (player log P(actual), excess over population) as float32 arrays."""
    n = B["X"].shape[0]
    plog = np.zeros(n, dtype=np.float32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        score_rows(B["gidx"][a:b], B["X"][a:b], B["ctx"][a:b], B["tgt"][a:b], B["off"][a:b],
                   B["lp"][a:b], gamma, plog[a:b])
    return plog, plog - B["lp"]


def group_index(S):
    p = S["player_idx"].to_numpy()
    brk = np.flatnonzero(p[1:] != p[:-1]) + 1
    st = np.concatenate([[0], brk]).astype(np.int64)
    en = np.concatenate([brk, [len(p)]]).astype(np.int64)
    gidx = np.repeat(np.arange(len(st)), en - st).astype(np.int32)
    return st, en, gidx, p[st]


def prep(phase, keep_S=True):
    S = load_phase(phase)
    X = design(S)
    ctx, tgt, off, lp, facing = contexts(S)
    st, en, gidx, pids = group_index(S)
    K = S.select("hand_idx", "player_idx")
    del S
    gc.collect()
    return dict(S=K, X=X, ctx=ctx, tgt=tgt, off=off, lp=lp, fac=facing, st=st, en=en, gidx=gidx, pids=pids)


def align(gam, pids_src, pids_dst):
    lut = {int(q): i for i, q in enumerate(pids_src)}
    idx = np.array([lut.get(int(q), -1) for q in pids_dst])
    g2 = np.zeros((len(pids_dst), NCTX, gam.shape[2]))
    ok = idx >= 0
    g2[ok] = gam[idx[ok]]
    return g2, float(ok.mean())


# ------------------------------------------------------------------ stages
NB_TABLES = int(os.environ.get("PLLR_BATCH", 25))
COLS = ["hand_idx", "action_no", "player_idx", "y", "legal", "street", "pf_eq_vs1", "hs_actor", "pot_odds"]


def load_batch(t0, t1):
    """All decisions of tables [t0, t1] (both phases) with the population probabilities attached."""
    S = (pl.scan_parquet(STATES).filter(c("table_idx").is_between(t0, t1))
         .select(*COLS, "phase").collect())
    lo, hi = S["hand_idx"].min(), S["hand_idx"].max()
    A = (pl.scan_parquet(DER / "action_policy.parquet").filter(c("hand_idx").is_between(lo, hi))
         .select("hand_idx", "action_no", "p_fold", "p_check", "p_call", "p_agg").collect())
    S = S.join(A, on=["hand_idx", "action_no"], how="inner")
    del A
    s = (pl.when(c("street") == 0).then(c("pf_eq_vs1")).otherwise(c("hs_actor"))).fill_nan(0.5).fill_null(0.5)
    post = (c("street") > 0).cast(pl.Float64)
    return S.with_columns(x1=s, x2=s * s, x3=post, x4=post * s, x5=c("pot_odds").cast(pl.Float64))


def prep_slice(S):
    """S = one phase of one batch, already sorted by player_idx."""
    X = design(S)
    ctx, tgt, off, lp, facing = contexts(S)
    st, en, gidx, pids = group_index(S)
    return dict(S=S.select("hand_idx", "player_idx"), X=X, ctx=ctx, tgt=tgt, off=off, lp=lp, fac=facing,
                st=st, en=en, gidx=gidx, pids=pids)


def _agg(B, plog, dl):
    A = B["S"].with_columns(psurp=pl.Series(-plog), dsurp=pl.Series(-dl),
                            fac=pl.Series(B["fac"].astype(np.float32)))
    return A.group_by("hand_idx", "player_idx").agg(
        nact=pl.len(), psurp=c("psurp").sum(), dsurp=c("dsurp").sum(), pmaxs=c("psurp").max(),
        dmaxs=c("dsurp").max(), fsurp=(c("psurp") * c("fac")).sum(), fdsurp=(c("dsurp") * c("fac")).sum(),
        nfac=c("fac").sum())


def run_build(lam=LAM):
    """Table-batched: fit each player on BOTH phases, then score each phase with the OTHER phase's model."""
    dev_hands = set(pl.read_parquet(DER / "evidence_cache" / "dev_pos.parquet",
                                    columns=["hand_idx"])["hand_idx"].unique().to_list())
    log(f"dev-phase hands to keep: {len(dev_hands):,}")
    PARTS = DER / "player_llr_parts"
    PARTS.mkdir(exist_ok=True)
    gstore = []
    held = {0: [0.0, 0], 1: [0.0, 0]}
    for t0 in range(0, 400, NB_TABLES):
        t1 = min(t0 + NB_TABLES - 1, 399)
        pf_out = PARTS / f"part_{t0:03d}.parquet"
        gf_out = PARTS / f"gam_{t0:03d}.npz"
        if pf_out.exists() and gf_out.exists():
            z = np.load(gf_out)
            gstore.append((z["g0"], z["p0"], z["g1"], z["p1"]))
            log(f"tables {t0}-{t1}: cached")
            continue
        S = load_batch(t0, t1)
        P, G = {}, {}
        for ph in (0, 1):
            Sp = S.filter(c("phase") == ph).sort("player_idx", "hand_idx", "action_no")
            P[ph] = prep_slice(Sp)
            del Sp
            A = P[ph]
            G[ph] = fit_groups(A["st"], A["en"], A["X"], A["ctx"], A["tgt"], A["off"], lam, 12)
        del S
        gstore.append((np.asarray(G[0], dtype=np.float32), P[0]["pids"],
                       np.asarray(G[1], dtype=np.float32), P[1]["pids"]))
        pidmap = {ph: P[ph]["pids"] for ph in (0, 1)}
        bparts = []
        for ph in (0, 1):
            B = P[ph]
            g2, cov = align(G[1 - ph], pidmap[1 - ph], B["pids"])
            plog, dl = score_all(B, g2)
            held[ph][0] += float(dl.sum())
            held[ph][1] += len(dl)
            H = _agg(B, plog, dl)
            if ph == 0:
                H = H.filter(c("hand_idx").is_in(list(dev_hands)))
            bparts.append(H)
            del plog, dl, B
            P[ph] = None
        pl.concat(bparts).write_parquet(pf_out)
        np.savez(gf_out, g0=gstore[-1][0], p0=gstore[-1][1], g1=gstore[-1][2], p1=gstore[-1][3])
        del P, G, bparts
        gc.collect()
        log(f"tables {t0}-{t1}: cross-phase held-out gain/decision "
            f"dev {held[0][0] / max(held[0][1], 1):+.5f} eval {held[1][0] / max(held[1][1], 1):+.5f}")
    n = (pl.scan_parquet(sorted(PARTS.glob("part_*.parquet")))
         .with_columns([c(k).cast(pl.Float32) for k in ("psurp", "dsurp", "pmaxs", "dmaxs", "fsurp", "fdsurp")])
         .with_columns(nact=c("nact").cast(pl.Int16), nfac=c("nfac").cast(pl.Int16))
         .sink_parquet(OUT))
    np.savez(GAM, lam=lam,
             g0=np.concatenate([g[0] for g in gstore]), p0=np.concatenate([g[1] for g in gstore]),
             g1=np.concatenate([g[2] for g in gstore]), p1=np.concatenate([g[3] for g in gstore]))
    tot = pl.scan_parquet(OUT).select(pl.len()).collect().item()
    d = held[0][1] and f"dev {held[0][0] / held[0][1]:+.5f} eval {held[1][0] / held[1][1]:+.5f}" or "cached"
    log(f"wrote {OUT} ({tot:,} rows); held-out log-lik gain per decision {d}")


if __name__ == "__main__":
    run_build(float(os.environ.get("PLLR_LAM", LAM)))
