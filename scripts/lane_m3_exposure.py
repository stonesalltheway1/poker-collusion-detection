"""Lane M3 -- exposure-law correction of the listing-rule DP.

The DP (scripts/listing_rule.py) turns per-hand event probabilities q_i into P(hand i is on the host's
evidence list) under  list = sorted(EVENTS, key=(type, hand_seq))[:5].  Its behaviour is driven by how hard
the cap binds, i.e. by the pair's expected event count  S = sum_i q_i.  q is a per-hand model fitted on dev
positives whose exposure runs 57..303 shared hands; 26% of eval pairs sit BELOW 57, where the trees cannot
extrapolate and S collapses, the cap stops binding and P(listed_i) -> q_i (the (type,time) ordering, worth
+0.107 MAP on DT and +0.038 on SP, is discarded).

This module (a) fits a censored count law  n_listed = min(N,5),  N ~ Poisson(lam),  log lam = a + b log n_shared
(+ family effects) on dev, (b) rescales q per pair by a single multiplicative logit shift (bisection) so that
S matches the law's lam, and (c) measures dev MAP@5 with a paired bootstrap.

Stages: python scripts/lane_m3_exposure.py sweep|heldout|eval
Env: M3_SCRATCH (cache dir written by scripts/lane_m3_cache.py).
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
sys.path.insert(0, str(BASE / "src"))
import numpy as np
import polars as pl
from numba import njit
from scipy.optimize import minimize
from scipy.stats import poisson

import listing_rule as LR
import common as C

c = pl.col
FAMS = LR.FAMS
SH = LR.SH
SC = Path(os.environ.get("M3_SCRATCH", BASE / "scratch_m3"))
CAP = 5
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- the censored exposure law
def fit_law(n_shared, n_listed, family, b_fixed=None):
    """Censored Poisson: obs = min(N,5), log lam = a + b*(log n - mean log n) + d_family. Returns dict."""
    ns = np.asarray(n_shared, float)
    nl = np.asarray(n_listed, int)
    m = np.log(ns).mean()
    x = np.log(ns) - m
    D = np.stack([(np.asarray(family) == f).astype(float) for f in FAMS], 1)

    def nll(par, b=None):
        bb = par[1] if b is None else b
        eta = par[0] + bb * x + D[:, 1] * par[2] + D[:, 2] * par[3]
        lam = np.exp(np.clip(eta, -10, 12))
        ll = np.zeros_like(lam)
        u = nl < CAP
        ll[u] = poisson.logpmf(nl[u], lam[u])
        ll[~u] = np.log(np.clip(poisson.sf(CAP - 1, lam[~u]), 1e-300, 1))
        return -ll.sum()

    r = minimize(nll, [2.18, 0.5, 0.0, 0.0], method="Nelder-Mead",
                 options=dict(maxiter=40000, maxfev=40000, xatol=1e-9, fatol=1e-11))
    if b_fixed is not None:
        r2 = minimize(lambda p: nll(np.array([p[0], 0.0, p[1], p[2]]), b_fixed), [r.x[0], r.x[2], r.x[3]],
                      method="Nelder-Mead", options=dict(maxiter=40000, maxfev=40000, xatol=1e-9, fatol=1e-11))
        par = np.array([r2.x[0], b_fixed, r2.x[1], r2.x[2]])
    else:
        par = r.x
    return dict(a=float(par[0]), b=float(par[1]), d={FAMS[0]: 0.0, FAMS[1]: float(par[2]), FAMS[2]: float(par[3])},
                mean_log_ns=float(m), nll=float(r.fun))


def lam_of(law, n_shared, fam):
    return np.exp(law["a"] + law["b"] * (np.log(np.asarray(n_shared, float)) - law["mean_log_ns"]) + law["d"][fam])


# ---------------------------------------------------------------- per-pair logit shift
@njit(cache=True)
def solve_shift(offsets, qlogit, target, iters=48):
    """One multiplicative logit shift per pair so that sum_i sigmoid(logit_i + delta) == target[g]."""
    out = np.zeros(qlogit.shape[0])
    sh = np.zeros(offsets.shape[0] - 1)
    for g in range(offsets.shape[0] - 1):
        a, b = offsets[g], offsets[g + 1]
        if b <= a:
            continue
        t = target[g]
        s0 = 0.0
        n_act = 0
        for i in range(a, b):
            if qlogit[i] > -100.0:
                s0 += 1.0 / (1.0 + np.exp(-qlogit[i]))
                n_act += 1
        if n_act == 0 or t <= 0.0:
            for i in range(a, b):
                out[i] = 0.0 if qlogit[i] <= -100.0 else 1.0 / (1.0 + np.exp(-qlogit[i]))
            continue
        tt = min(t, 0.98 * n_act)                      # feasibility
        lo, hi = -14.0, 14.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            s = 0.0
            for i in range(a, b):
                if qlogit[i] > -100.0:
                    s += 1.0 / (1.0 + np.exp(-(qlogit[i] + mid)))
            if s < tt:
                lo = mid
            else:
                hi = mid
        mid = 0.5 * (lo + hi)
        sh[g] = mid
        for i in range(a, b):
            out[i] = 0.0 if qlogit[i] <= -100.0 else 1.0 / (1.0 + np.exp(-(qlogit[i] + mid)))
    return out, sh


def pair_sum(offsets, v):
    return np.add.reduceat(v, offsets[:-1])


# ---------------------------------------------------------------- AP@5
def per_pair_ap(K, score_col):
    """K: pair_id, is_ev, n_rel, pot_bb, <score_col>. Returns (pair_ids, ap) using the host AP@5."""
    t = (K.select("pair_id", "is_ev", "n_rel", c(score_col).alias("s"), "pot_bb")
         .sort(["pair_id", "s", "pot_bb"], descending=[False, True, True])
         .with_columns(k=pl.int_range(pl.len()).over("pair_id") + 1).filter(c("k") <= 5)
         .with_columns(h=c("is_ev").cast(pl.Int32).cum_sum().over("pair_id")))
    a = (t.group_by("pair_id", maintain_order=True)
         .agg(((c("h") / c("k")) * c("is_ev")).sum().alias("num"), c("n_rel").first())
         .with_columns(ap=c("num") / c("n_rel").clip(upper_bound=5)).sort("pair_id"))
    return a["pair_id"].to_numpy(), a["ap"].to_numpy()


def boot_p(a, b, n=5000, seed=0):
    """paired bootstrap over pairs: P(mean(b) > mean(a))."""
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return float((m > 0).mean()), float(d.mean()), float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))


# ---------------------------------------------------------------- dev driver
GATE_COL = {"directed_transfer": "flow", "soft_play": "sp_g0", "coordinated_isolation": None}
RAMP = (0.45, 0.65)
W_DP = 0.7


class Dev:
    def __init__(self, partner="exp027ev", npz="dev_oof.npz"):
        K = pl.read_parquet(SC / "dev_keys.parquet")
        z = np.load(SC / npz)
        assert K["pair_id"].is_sorted(), "dev keys must be pair-sorted"
        self.K = K
        self.pid = K["pair_id"].to_numpy()
        self.off = LR.group_offsets(self.pid)
        self.n = len(K)
        gate = {"directed_transfer": K["flow"].fill_null(False).cast(pl.Boolean).to_numpy(),
                "soft_play": K["sp_g0"].fill_null(False).cast(pl.Boolean).to_numpy(),
                "coordinated_isolation": ((K["ci_fb"] >= 0) & (K["ci_fb"] <= 1)).fill_null(False).to_numpy()}
        self.qlog, self.pi = {}, {}
        for f in FAMS:
            ql = np.clip(z[f"q_{SH[f]}"].copy(), -60, 60)
            ql[~gate[f]] = -1e6                     # gated-out hands: q == 0 and immune to the shift
            self.qlog[f] = ql
            if f == "coordinated_isolation":
                fb = K["ci_fb"].to_numpy()
                pi = np.zeros((self.n, 2)); pi[fb == 0, 0] = 1.0; pi[fb != 0, 1] = 1.0
            else:
                ta = z[f"t_{SH[f]}"]
                pi = np.stack([ta, 1.0 - ta], 1)
            self.pi[f] = np.ascontiguousarray(pi)
        # pair table
        P = (K.group_by("pair_id", maintain_order=True)
             .agg(c("family").first().alias("family"), c("fold").first().alias("fold"),
                  pl.len().alias("n_shared"), c("is_ev").sum().alias("n_listed")))
        self.P = P
        self.pair_ids = P["pair_id"].to_numpy()
        self.n_shared = P["n_shared"].to_numpy().astype(float)
        self.fam = P["family"].to_numpy()
        self.n_listed = P["n_listed"].to_numpy().astype(int)
        # partner + ramp weight
        S = pl.read_parquet(C.DER / f"evidence_scores_dev_{partner}.parquet").drop("hand_id")
        self.G0 = K.select("pair_id", "hand_idx", "family", "is_ev", "n_rel", "pot_bb").join(
            S, on=["pair_id", "hand_idx"], how="left", maintain_order="left")
        st = (self.G0.group_by("pair_id", maintain_order=True)
              .agg(*[c(f"s_{f}").fill_null(0).top_k(5).mean().alias(f"t5_{f}") for f in FAMS])
              .with_columns(strength=pl.max_horizontal([c(f"t5_{f}") for f in FAMS])))
        lo, hi = RAMP
        self.w_pair = (W_DP * np.clip((st["strength"].to_numpy() - lo) / (hi - lo), 0, 1))
        self.strength = st["strength"].to_numpy()
        self.S_raw = {f: pair_sum(self.off, np.where(self.qlog[f] > -100, 1 / (1 + np.exp(-self.qlog[f])), 0.0))
                      for f in FAMS}
        self.n_gate = {f: pair_sum(self.off, (self.qlog[f] > -100).astype(float)) for f in FAMS}

    def dp(self, target=None):
        """target: dict fam -> per-pair target S (None = incumbent)."""
        out = {}
        for f in FAMS:
            if target is None:
                q = np.where(self.qlog[f] > -100, 1 / (1 + np.exp(-self.qlog[f])), 0.0)
            else:
                q, _ = solve_shift(self.off, self.qlog[f], np.asarray(target[f], float))
            out[f] = LR.listed_prob_groups(self.off, q, self.pi[f], CAP)
        return out

    def score(self, dp, w_mode="ramp"):
        w = self.w_pair if w_mode == "ramp" else np.full(len(self.pair_ids), W_DP)
        wrow = np.repeat(w, np.diff(self.off))
        G = self.G0.with_columns([pl.Series(f"dp_{f}", dp[f]) for f in FAMS] + [pl.Series("w", wrow)])
        G = G.with_columns([((1 - c("w")) * c(f"s_{f}").fill_null(0) + c("w") * c(f"dp_{f}")).alias(f"b_{f}")
                            for f in FAMS])
        G = G.with_columns(routed=pl.coalesce([pl.when(c("family") == f).then(c(f"b_{f}")) for f in FAMS]))
        return G

    def ap(self, dp, w_mode="ramp"):
        G = self.score(dp, w_mode)
        pids, ap = per_pair_ap(G, "routed")
        assert (pids == np.sort(self.pair_ids)).all()
        return ap, G


def report(d, ap_base, ap_new, label, mask=None, seed=0):
    m = np.ones(len(ap_base), bool) if mask is None else mask
    p, dm, q5, q95 = boot_p(ap_base[m], ap_new[m], seed=seed)
    print(f"  {label:38s} n={m.sum():4d}  base {ap_base[m].mean():.4f} -> new {ap_new[m].mean():.4f}"
          f"  delta {dm:+.4f} [{q5:+.4f},{q95:+.4f}]  P(better) {p:.3f}", flush=True)
    return dict(n=int(m.sum()), base=float(ap_base[m].mean()), new=float(ap_new[m].mean()), delta=dm, p=p)


def fam_mask(d, f):
    return d.fam == f


def main_sweep():
    d = Dev()
    log(f"dev: {d.n:,} hand rows, {len(d.pair_ids)} pairs")
    law = fit_law(d.n_shared, d.n_listed, d.fam)
    print(f"  law: a={law['a']:.4f} b={law['b']:.4f} d_sp={law['d'][FAMS[1]]:.3f} d_ci={law['d'][FAMS[2]]:.3f}")
    for f in FAMS:
        print(f"    {SH[f]}: S_raw mean {d.S_raw[f].mean():.2f}  gated hands/pair {d.n_gate[f].mean():.1f}"
              f"  lam(mean exposure) {lam_of(law, np.exp(law['mean_log_ns']), f):.2f}")
    dp0 = d.dp()
    ap0, G0 = d.ap(dp0)
    own = np.array([dp0[f] for f in FAMS])
    print(f"  INCUMBENT dev MAP@5 (true-family routing, ramp, exp027ev) = {ap0.mean():.4f}")
    for f in FAMS:
        print(f"    {SH[f]}: {ap0[fam_mask(d, f)].mean():.4f}  (n={fam_mask(d,f).sum()})")
    qs = np.quantile(d.n_shared, [.25, .5, .75])
    qt = np.digitize(d.n_shared, qs)
    res = {"incumbent": float(ap0.mean()), "law": law, "sweep": {}, "heldout": {}}

    # ---- multiplicative sensitivity of the DP to the assumed event count
    print("\n[A] multiplicative sweep on the implied event count S (all pairs, ramp blend)")
    for fac in [0.5, 0.7, 0.85, 1.0, 1.15, 1.35, 1.6, 2.0, 3.0]:
        tgt = {f: d.S_raw[f] * fac for f in FAMS}
        ap, _ = d.ap(d.dp(tgt))
        line = f"  x{fac:<5.2f} MAP {ap.mean():.4f} ({ap.mean()-ap0.mean():+.4f})"
        for f in FAMS:
            line += f"  {SH[f]} {ap[fam_mask(d,f)].mean():.4f}"
        line += "   | bottom-quartile " + f"{ap[qt==0].mean():.4f} ({ap[qt==0].mean()-ap0[qt==0].mean():+.4f})"
        print(line, flush=True)
        res["sweep"][str(fac)] = float(ap.mean())

    # ---- law target (full-dev fit; a no-op in the middle by construction)
    print("\n[B] law target  S := lam_f(n_shared) * kappa_f   (kappa = mean S_raw / mean lam on dev)")
    for mode in ["law_level", "law_trend"]:
        tgt = {}
        for f in FAMS:
            lam = lam_of(law, d.n_shared, f)
            if mode == "law_level":
                kap = d.S_raw[f].mean() / lam.mean()
                tgt[f] = lam * kap
            else:                      # keep each pair's own S, correct only the exposure trend
                bq = np.polyfit(np.log(d.n_shared), np.log(np.clip(d.S_raw[f], 1e-3, None)), 1)[0]
                tgt[f] = d.S_raw[f] * np.exp((law["b"] - bq) * (np.log(d.n_shared) - law["mean_log_ns"]))
        ap, _ = d.ap(d.dp(tgt))
        report(d, ap0, ap, mode)
        for k in range(4):
            report(d, ap0, ap, f"   {mode} q{k} (n_shared {d.n_shared[qt==k].min():.0f}-{d.n_shared[qt==k].max():.0f})", qt == k)
        res[mode] = float(ap.mean())

    # ---- held-out: fit on the top 3 quartiles, apply to the bottom one
    print("\n[C] held-out law: fit on n_shared >= q25, apply the correction to the bottom quartile only")
    tr = qt > 0
    law_h = fit_law(d.n_shared[tr], d.n_listed[tr], d.fam[tr])
    print(f"  held-out law: a={law_h['a']:.4f} b={law_h['b']:.4f} (full-dev b={law['b']:.4f})")
    for mode in ["law_level", "law_trend"]:
        tgt = {}
        for f in FAMS:
            lam = lam_of(law_h, d.n_shared, f)
            if mode == "law_level":
                kap = d.S_raw[f][tr].mean() / lam[tr].mean()
                t = lam * kap
            else:
                bq = np.polyfit(np.log(d.n_shared[tr]), np.log(np.clip(d.S_raw[f][tr], 1e-3, None)), 1)[0]
                t = d.S_raw[f] * np.exp((law_h["b"] - bq) * (np.log(d.n_shared) - law_h["mean_log_ns"]))
            tgt[f] = np.where(tr, d.S_raw[f], t)         # correction ONLY on the held-out bottom quartile
        ap, _ = d.ap(d.dp(tgt))
        res["heldout"][mode] = report(d, ap0, ap, f"heldout {mode} (bottom quartile)", qt == 0)
    json.dump(res, open(SC / "sweep_results.json", "w"), indent=1, default=float)
    log("done")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if stage == "sweep":
        main_sweep()
