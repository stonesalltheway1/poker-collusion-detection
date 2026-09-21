"""exp036 -- the host's evidence LISTING RULE: predicates, structural diagnostics, and a rule-guided scorer.

FINDING (research/forensics/listing_rule.md):
  The dev evidence list is  sorted(EVENTS, key=(type, hand_seq))[:5]  -- all type-1 events of the pair's phase in
  time order, then type-2 events in time order, truncated to 5.
    * seg-1 hands (before the single time inversion) are spread UNIFORMLY over the phase (mean timeline position
      DT .496 / SP .493 / CI .411) while seg-2 hands are the EARLIEST type-2 events (.239 / .297 / .172) -> seg 1 is the
      COMPLETE set of type-1 events whenever a seg 2 exists (unlisted type-1-looking hands in those pairs sit at the
      natural base rate, flat before/after the last seg-1 hand).
    * CI: the type is CRISP and visible -- the number of players who FOLDED before the pair's first preflop raise:
      405/405 chronological-list hands and 43/43 seg-1 hands have 0 folds before it, 12/12 seg-2 hands exactly 1,
      and no listed CI hand has >= 2 (460/460).
    * SP: type 1 = a partner FOLDS facing the other's aggression while holding the better made hand
      (seg 1 180/213, seg 2 8/232); type 2 = passive CALL of partner aggression (seg 2 223/232); the 3 lists with two
      inversions add a type 3 = strong check (3/3).
    * DT: type 1 = the donor FOLDS facing the receiver with the better made hand (seg 1 226/236, seg 2 34/236);
      type 2 = chip-dump call-downs/bets. Identical action strings occur in both segments, so the DT/SP type is only
      partly visible (seg classifier AUC ~.95).
    * The EVENT itself (latent activation x visible action) is NOT a crisp predicate of the public log: the best
      deterministic predicates reproduce the exact listed set for only 15-25% of pairs (tables in the .md).
  Scorer = event model q_i (LightGBM on censoring-derived labels: listed = 1; unlisted hands of short lists and
  unlisted hands before the last-ranked listed hand = 0; later hands excluded) + type model (seg1-vs-seg2 on the
  one-inversion pairs; CI deterministic) + an exact Poisson-binomial DP for P(listed_i) under the rule above
  (scripts/listing_rule.py:listed_prob), blended 0.3/0.7 (W_DP=0.7) with the incumbent calibrated ranker
  (exp008 -> evidence_scores_*_exp036.parquet; exp027ev -> evidence_scores_*_exp036_exp027ev.parquet).

Stages:  python scripts/listing_rule.py [diag|cv|eval|all]
  diag   structural tables (segments, positions, CI fold-count key, fold-ahead split, deterministic-rule exact rates)
  cv     OOF (frozen table folds) event/type models + DP + blends; dev OOF MAP@5 per family; writes the dev score file
  eval   full-dev fits; scores every eval-phase shared hand of the 112,540 eval pairs (eval hands only);
         writes data/derived/evidence_scores_eval_exp036.parquet (+ dev file evidence_scores_dev_exp036.parquet)
Rules: gameplay only (actions, seats, board, stacks, table time order within the phase). No IDs / file order /
eval-file membership as signal (eval pairs are only the rows to score).
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

import numpy as np
import polars as pl
import lightgbm as lgb
from numba import njit

import common as C
import evidence as EV

c = pl.col
FAMS = list(C.FAMILIES)
SH = {"directed_transfer": "dt", "soft_play": "sp", "coordinated_isolation": "ci"}
OUT = BASE / "data" / "derived"
RES = OUT / "evidence_cache" / os.environ.get("LR_RES", "listing_rule_exp036.json")
CAP = 5
W_DP = float(os.environ.get("LR_WDP", 0.7))   # blend weight of the rule-DP probability vs the incumbent ranker
#   (dev OOF, real posteriors: w .5 -> +.052/+.054, w .7 -> +.056/+.060 over exp008/exp027ev; DP alone +.050/+.048)
PARTNER = os.environ.get("LR_PARTNER", "exp008")
TAG = os.environ.get("LR_TAG", "exp036")          # exp036 = flat blend; exp037 = + family-strength gate
GATE_TH = float(os.environ.get("LR_GATE_TH", 0.0))  # exp037: DP weight -> 0 for pairs whose best family head is weak
#   Why (exp037): on an UNSEEN family (LOFO proxy for other_coordination) the DP mixture scores .258 vs .419 for the
#   partner alone, so the DP must not be used where no modelled family fires. strength = max_f top5-mean partner s_f;
#   at 0.6 it flags 4.8% of dev positives (dev MAP +.001, no cost) and ~78% of LOFO unseen-family pairs.
EXTRA = ["ci_fb", "fold_ahead_ab", "fold_ahead_ba", "fold_made_ab", "fold_made_ba", "fold_ahead_don", "fold_ahead_any",
         "fold_made_don", "fold_made_max", "fold_st_don", "ci_raise_no"]
# ---- lane M2: per-player residual-policy surprise (src/playerllr.py). LR_LLR = "" | "small" | "full"
LLR = os.environ.get("LR_LLR", "")
LLR_BASES = ["psurp", "pmaxs", "dsurp", "fsurp"]
LLR_SMALL = ["llr_psurp_sum", "llr_psurp_mx", "llr_psurp_mn", "llr_pmaxs_sum", "llr_dsurp_sum", "llr_fsurp_sum",
             "pct_llr_psurp_sum", "pct_llr_psurp_mx", "llr_nact_sum"]
LLR_FULL = ([f"llr_{k}_{o}" for k in LLR_BASES for o in ("sum", "mx", "mn", "don", "rec")]
            + [f"pct_llr_{k}_sum" for k in LLR_BASES] + ["llr_nact_sum", "llr_nfac_sum"])
_LLRT = [None]


def llr_names():
    return {"small": LLR_SMALL, "full": LLR_FULL}.get(LLR, [])


def add_llr(E):
    """Join the per-(hand, player) player-specific surprise and build orientation-free pair views.
    E must carry hand_idx, pA, pB, pdonA, n_shared and be sorted by (pair_id, hand_seq)."""
    if not LLR:
        return E
    if _LLRT[0] is None:
        _LLRT[0] = pl.read_parquet(C.DER / "player_llr_hand.parquet")
    L = _LLRT[0]
    cols = LLR_BASES + ["nact", "nfac"]
    for side in ("A", "B"):
        E = E.join(L.select("hand_idx", "player_idx", *cols)
                   .rename({"player_idx": f"p{side}", **{k: f"_{k}_{side}" for k in cols}}),
                   on=["hand_idx", f"p{side}"], how="left", maintain_order="left")
    E = E.with_columns([c(f"_{k}_{s}").fill_null(0.0).cast(pl.Float32) for k in cols for s in ("A", "B")])
    ex = []
    for k in LLR_BASES:
        a, b = c(f"_{k}_A"), c(f"_{k}_B")
        ex += [(a + b).alias(f"llr_{k}_sum"), pl.max_horizontal(a, b).alias(f"llr_{k}_mx"),
               pl.min_horizontal(a, b).alias(f"llr_{k}_mn"),
               pl.when(c("pdonA")).then(a).otherwise(b).alias(f"llr_{k}_don"),
               pl.when(c("pdonA")).then(b).otherwise(a).alias(f"llr_{k}_rec")]
    ex += [(c("_nact_A") + c("_nact_B")).alias("llr_nact_sum"),
           (c("_nfac_A") + c("_nfac_B")).alias("llr_nfac_sum")]
    E = E.with_columns(ex).drop([f"_{k}_{s}" for k in cols for s in ("A", "B")])
    pct = [(c(f"llr_{k}_{o}").rank("average").over("pair_id") / c("n_shared")).cast(pl.Float32)
           .alias(f"pct_llr_{k}_{o}") for k in LLR_BASES for o in ("sum", "mx")]
    return E.with_columns(pct)
# exp050: counterfactual-value block (src/cfvalue.py). LR_CF = "" (off, default) | "core" | "full".
CF_MODE = os.environ.get("LR_CF", "")
CF_BASE = ["cfv", "cfvx", "cfs", "cfsx", "cfdch", "cfdfd", "cfsurp"]
CF_RANK = ["cfv_mx", "cfs_mx", "cfexc_mx", "cfvx_mx", "cfv_don", "cfsexc_mx"]
CF_FULL = ([f"{f}_{o}" for f in CF_BASE for o in ("mx", "mn", "don", "rec")]
           + [f"{f}_{o}" for f in ("cfexc", "cfexcm", "cfsexc") for o in ("mx", "don")]
           + ["cfv_sum", "cfs_sum", "cfv_net", "cfa_sum", "cfe_sum", "cfo_mx"]
           + [p + k for k in CF_RANK for p in ("pct_", "top_")])
CF_CORE = ["cfexc_mx", "cfexc_don", "cfv_mx", "cfv_don", "cfs_mx", "cfsexc_mx", "cfvx_mx", "cfdfd_mn",
           "pct_cfv_mx", "top_cfv_mx", "pct_cfexc_mx", "top_cfexc_mx"]
if CF_MODE:
    EXTRA = EXTRA + (CF_CORE if CF_MODE == "core" else CF_FULL)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ================================================================================================ DP
@njit(cache=True)
def _pb_prefix(p, cap):
    n = p.shape[0]
    out = np.zeros((n + 1, cap + 1))
    out[0, 0] = 1.0
    for i in range(n):
        pi = p[i]
        for m in range(cap + 1):
            v = out[i, m]
            if v == 0.0:
                continue
            out[i + 1, m] += v * (1.0 - pi)
            mm = m + 1 if m < cap else cap
            out[i + 1, mm] += v * pi
    return out


@njit(cache=True)
def listed_prob(q, pi, cap=5):
    """Hands of one pair in time order. q: P(event); pi[i, k]: P(type k | event). Host list = events sorted by
    (type, time)[:cap].  P(listed_i) = sum_k q_i pi_ik P(#events of type<k elsewhere + #type-k events before i < cap)."""
    n, K = pi.shape
    res = np.zeros(n)
    for k in range(K):
        p_pre = np.zeros(n)
        p_suf = np.zeros(n)
        for j in range(n):
            a = 0.0
            b = 0.0
            for t in range(K):
                if t <= k:
                    a += pi[j, t]
                if t < k:
                    b += pi[j, t]
            p_pre[j] = q[j] * a
            p_suf[j] = q[j] * b
        pre = _pb_prefix(p_pre, cap)
        suf = _pb_prefix(p_suf[::-1].copy(), cap)
        for i in range(n):
            A = pre[i]
            B = suf[n - 1 - i]
            s = 0.0
            for m1 in range(cap):
                for m2 in range(cap - m1):
                    s += A[m1] * B[m2]
            res[i] += q[i] * pi[i, k] * s
    return res


@njit(cache=True)
def listed_prob_groups(offsets, q, pi, cap=5):
    out = np.zeros(q.shape[0])
    for g in range(offsets.shape[0] - 1):
        a, b = offsets[g], offsets[g + 1]
        if b > a:
            out[a:b] = listed_prob(q[a:b], pi[a:b], cap)
    return out


def group_offsets(pair_ids):
    p = np.asarray(pair_ids)
    brk = np.flatnonzero(p[1:] != p[:-1]) + 1
    return np.concatenate([[0], brk, [len(p)]]).astype(np.int64)


def _auc(y, s):
    y = np.asarray(y, dtype=int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.empty(len(s))
    o = np.argsort(s, kind="mergesort")
    sv = np.asarray(s)[o]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ap5(pred, rel):
    if not rel:
        return 0.0
    hits, s = 0, 0.0
    for k, h in enumerate(pred[:5], 1):
        if h in rel:
            hits += 1
            s += hits / k
    return s / min(len(rel), 5)


def map5(df, score_col, fam=None):
    """host AP@5 per pair (df: pair_id, hand_idx, is_ev, family, score); returns mean over pairs [of family]."""
    d = df if fam is None else df.filter(c("family") == fam)
    t = (d.select("pair_id", "is_ev", "n_rel", c(score_col).alias("s"), "pot_bb")
         .sort(["pair_id", "s", "pot_bb"], descending=[False, True, True])
         .with_columns(k=pl.int_range(pl.len()).over("pair_id") + 1).filter(c("k") <= 5)
         .with_columns(h=c("is_ev").cast(pl.Int32).cum_sum().over("pair_id")))
    a = t.group_by("pair_id").agg(((c("h") / c("k")) * c("is_ev")).sum().alias("num"), c("n_rel").first())
    return float(a.with_columns(ap=c("num") / c("n_rel").clip(upper_bound=5))["ap"].mean())


# ================================================================================================ features
def extra_features(R):
    """Vectorized action facts for rows R (hand_idx, pA, pB [, pdonA]) from the public action log + seat strength.
    ci_fb        = # players who FOLDED before the pair's first preflop raise (-1 if the pair never raises preflop)
    fold_made_xy = made-hand rank of X minus Y on the street where X folded (preflop: 1000 * class-equity diff)
    fold_ahead_xy= X folds facing Y's aggression (Y = last aggressor on that street) with the better made hand."""
    hids = R.select("hand_idx").unique()
    A = (C.scan("actions").join(hids.lazy(), on="hand_idx")
         .select("hand_idx", "action_no", "street", "player_idx", "action", "amount", "to_call").collect()
         .sort("hand_idx", "action_no"))
    A = A.with_columns(aggr=c("action").is_in([3, 4]) | ((c("action") == 5) & (c("amount") > c("to_call"))),
                       isf=(c("action") == 0).cast(pl.Int32))
    A = A.with_columns(aggp=pl.when(c("aggr")).then(c("player_idx")).otherwise(None))
    A = A.with_columns(last_aggr=c("aggp").shift(1).forward_fill().over(["hand_idx", "street"]),
                       fb=(c("isf").cum_sum() - c("isf")).over("hand_idx"))
    FR = (A.filter((c("street") == 0) & c("aggr")).group_by("hand_idx", "player_idx")
          .agg(c("action_no").min().alias("r_no"), c("fb").sort_by("action_no").first().alias("r_fb")))
    FO = (A.filter(c("isf") == 1).group_by("hand_idx", "player_idx")
          .agg(c("street").first().alias("f_st"), c("last_aggr").first().alias("f_face")))
    SS = (pl.scan_parquet(C.DER / "seat_strength.parquet").join(hids.lazy(), on="hand_idx")
          .select("hand_idx", "player_idx", "rank_flop", "rank_turn", "final_rank_7", "pf_eq_vs1").collect())
    X = R.select("hand_idx", "pA", "pB")
    for side in ("A", "B"):
        p = f"p{side}"
        X = (X.join(FR.rename({"player_idx": p, "r_no": f"r_no{side}", "r_fb": f"r_fb{side}"}), on=["hand_idx", p], how="left")
             .join(FO.rename({"player_idx": p, "f_st": f"f_st{side}", "f_face": f"f_face{side}"}), on=["hand_idx", p], how="left")
             .join(SS.rename({"player_idx": p, "rank_flop": f"rf{side}", "rank_turn": f"rt{side}",
                              "final_rank_7": f"rr{side}", "pf_eq_vs1": f"pe{side}"}), on=["hand_idx", p], how="left"))

    def made(st, x, y):
        return (pl.when(st == 0).then(((c(f"pe{x}") - c(f"pe{y}")) * 1000).cast(pl.Float32))
                .when(st == 1).then((c(f"rf{x}") - c(f"rf{y}")).cast(pl.Float32))
                .when(st == 2).then((c(f"rt{x}") - c(f"rt{y}")).cast(pl.Float32))
                .when(st == 3).then((c(f"rr{x}") - c(f"rr{y}")).cast(pl.Float32)))
    X = X.with_columns(
        ci_fb=pl.when(c("r_noA").is_null() & c("r_noB").is_null()).then(-1)
        .when(c("r_noB").is_null() | (c("r_noA") < c("r_noB"))).then(c("r_fbA")).otherwise(c("r_fbB")).cast(pl.Float32),
        ci_raise_no=pl.min_horizontal("r_noA", "r_noB").fill_null(-1).cast(pl.Float32),
        fold_made_ab=made(c("f_stA"), "A", "B").fill_null(0.0),
        fold_made_ba=made(c("f_stB"), "B", "A").fill_null(0.0),
    ).with_columns(
        fold_ahead_ab=((c("f_faceA") == c("pB")) & (c("fold_made_ab") > 0)).fill_null(False).cast(pl.Float32),
        fold_ahead_ba=((c("f_faceB") == c("pA")) & (c("fold_made_ba") > 0)).fill_null(False).cast(pl.Float32),
        f_stA=c("f_stA").fill_null(-1).cast(pl.Float32), f_stB=c("f_stB").fill_null(-1).cast(pl.Float32),
    )
    return X.select("hand_idx", "pA", "pB", "ci_fb", "ci_raise_no", "fold_made_ab", "fold_made_ba", "fold_ahead_ab",
                    "fold_ahead_ba", "f_stA", "f_stB")


def cf_extra(E):
    """exp050 counterfactual-value block: join src/cfvalue.py per-hand directional deltas, then orient them
    (max/min, donor/receiver), form the hand-level DIRECTEDNESS excess over the other seated non-partners, and
    rank the key quantities WITHIN the pair (the axis the listing decision actually lives on)."""
    import cfvalue as CF
    cp = Path(os.environ["LR_CF_CACHE"]) if os.environ.get("LR_CF_CACHE") else None
    X = None
    if cp is not None and cp.exists():
        X = pl.read_parquet(cp)
        keys = E.select("hand_idx", "pA", "pB").unique()
        if keys.join(X.select("hand_idx", "pA", "pB"), on=["hand_idx", "pA", "pB"], how="anti").height:
            log("LR_CF_CACHE does not cover these rows - recomputing")
            X = None
    if X is None:
        X = CF.pair_features(E.select("hand_idx", "pA", "pB")).unique(subset=["hand_idx", "pA", "pB"])
        if cp is not None:
            X.write_parquet(cp)
    E = E.join(X, on=["hand_idx", "pA", "pB"], how="left", maintain_order="left")
    d = c("pdonA")
    ex = []
    for f in CF_BASE:
        a, b = c(f + "_ab"), c(f + "_ba")
        ex += [pl.max_horizontal(a, b).alias(f + "_mx"), pl.min_horizontal(a, b).alias(f + "_mn"),
               pl.when(d).then(a).otherwise(b).alias(f + "_don"), pl.when(d).then(b).otherwise(a).alias(f + "_rec")]
    ex += [(c("cfv_ab") - c("cfo_ab")).alias("cfexc_ab"), (c("cfv_ba") - c("cfo_ba")).alias("cfexc_ba"),
           (c("cfv_ab") - c("cfom_ab")).alias("cfexcm_ab"), (c("cfv_ba") - c("cfom_ba")).alias("cfexcm_ba"),
           (c("cfs_ab") - c("cfso_ab")).alias("cfsexc_ab"), (c("cfs_ba") - c("cfso_ba")).alias("cfsexc_ba")]
    E = E.with_columns(ex)
    ex2 = []
    for f in ("cfexc", "cfexcm", "cfsexc"):
        a, b = c(f + "_ab"), c(f + "_ba")
        ex2 += [pl.max_horizontal(a, b).alias(f + "_mx"), pl.when(d).then(a).otherwise(b).alias(f + "_don")]
    ex2 += [(c("cfv_ab") + c("cfv_ba")).alias("cfv_sum"), (c("cfs_ab") + c("cfs_ba")).alias("cfs_sum"),
            (c("cfv_ab") - c("cfv_ba")).abs().alias("cfv_net"),
            (c("cfa_ab") + c("cfa_ba")).alias("cfa_sum"), (c("cfe_ab") + c("cfe_ba")).alias("cfe_sum"),
            pl.max_horizontal("cfo_ab", "cfo_ba").alias("cfo_mx")]
    E = E.with_columns(ex2)
    rk = []
    for k in CF_RANK:
        rk += [(c(k).rank("average").over("pair_id") / pl.len().over("pair_id")).cast(pl.Float32).alias("pct_" + k),
               (c(k).rank("min", descending=True).over("pair_id") - 1).cast(pl.Float32).alias("top_" + k)]
    return E.with_columns(rk)


def add_extra(E):
    """E = EV.hand_features output (has pA, pB, pdonA). Adds EXTRA columns aligned to E rows."""
    X = extra_features(E.select("hand_idx", "pA", "pB"))
    E = E.join(X, on=["hand_idx", "pA", "pB"], how="left", maintain_order="left")
    d = c("pdonA")
    E = E.with_columns(
        fold_ahead_don=pl.when(d).then(c("fold_ahead_ab")).otherwise(c("fold_ahead_ba")),
        fold_made_don=pl.when(d).then(c("fold_made_ab")).otherwise(c("fold_made_ba")),
        fold_st_don=pl.when(d).then(c("f_stA")).otherwise(c("f_stB")),
        fold_ahead_any=pl.max_horizontal("fold_ahead_ab", "fold_ahead_ba"),
        fold_made_max=pl.max_horizontal("fold_made_ab", "fold_made_ba"),
    )
    return cf_extra(E) if CF_MODE else E


def strength_gate(K, partner_cols, th):
    """Per-pair DP weight: W_DP where the best modelled family head fires, else 0.
    K must hold one row per (pair, hand) with the partner's s_<fam> columns."""
    ramp = os.environ.get("LR_GATE_RAMP")
    if ramp:
        low, high = [float(x) for x in ramp.split(",")]
        st = K.group_by("pair_id").agg(*[c(k).fill_null(0).top_k(5).mean().alias(f"t5_{k}") for k in partner_cols])
        st = st.with_columns(strength=pl.max_horizontal([c(f"t5_{k}") for k in partner_cols])).select("pair_id", "strength")
        return K.join(st, on="pair_id", how="left").with_columns(
            w=(W_DP * ((c("strength") - low) / (high - low)).clip(0.0, 1.0)).cast(pl.Float64))
    if th <= 0:
        return K.with_columns(w=pl.lit(W_DP))
    st = K.group_by("pair_id").agg(*[c(k).fill_null(0).top_k(5).mean().alias(f"t5_{k}") for k in partner_cols])
    st = st.with_columns(strength=pl.max_horizontal([c(f"t5_{k}") for k in partner_cols])).select("pair_id", "strength")
    return K.join(st, on="pair_id", how="left").with_columns(
        w=pl.when(c("strength") >= th).then(W_DP).otherwise(0.0))


def gate(fam):
    """family gate (100% coverage of dev listed hands); q := 0 outside."""
    if fam == "directed_transfer":
        return c("flow")
    if fam == "soft_play":
        return c("sp_g0")
    return (c("ci_fb") >= 0) & (c("ci_fb") <= 1)


def type_matrix(E, fam, tau=None):
    if fam == "coordinated_isolation":
        fb = E["ci_fb"].to_numpy()
        pi = np.zeros((len(fb), 2))
        pi[fb == 0, 0] = 1.0
        pi[fb != 0, 1] = 1.0
        return pi
    return np.stack([tau, 1.0 - tau], axis=1)


def seg_labels(D):
    """segment of each listed hand from the single time inversion (1-based); pairs with >=1 inversion flagged."""
    L = D.filter(c("is_ev") == 1).sort("pair_id", "evidence_rank")
    L = L.with_columns(inv=(c("hand_seq") < c("hand_seq").shift(1)).over("pair_id").fill_null(False).cast(pl.Int32))
    L = L.with_columns(seg=c("inv").cum_sum().over("pair_id") + 1, ninv=c("inv").sum().over("pair_id"),
                       last_seq=c("hand_seq").sort_by("evidence_rank").last().over("pair_id"))
    return L.select("pair_id", "hand_idx", "seg", "ninv"), L.group_by("pair_id").agg(c("last_seq").first(), c("ninv").first())


def load_dev_features():
    t = time.time()
    F = EV.load_dev()                       # dev positive-pair rows, 321 orientation-free engine views, sorted
    F = add_llr(add_extra(F))
    S, P = seg_labels(F)
    F = F.join(S, on=["pair_id", "hand_idx"], how="left", maintain_order="left").join(P, on="pair_id", how="left", maintain_order="left")
    F = F.with_columns(ev_label=pl.when(c("is_ev") == 1).then(1).when(c("n_rel") < 5).then(0)
                       .when(c("hand_seq") < c("last_seq")).then(0).otherwise(None))
    F = F.with_columns(q_pos=((pl.int_range(pl.len()).over("pair_id") + 0.5) / pl.len().over("pair_id")))
    log(f"dev features: {F.height:,} rows x {F.width} cols ({time.time() - t:.0f}s)")
    return F


def feature_names():
    res = json.loads((C.DER / "evidence_cache" / "cv_results.json").read_text())
    return list(res["features_007"]) + EXTRA + llr_names()


PRM = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.6,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4, seed=42)
PRM_T = dict(PRM, num_leaves=7, min_data_in_leaf=10)
R_EV, R_T = 300, 150


def fit_event(F, X, fam, mask):
    m = mask & (F["family"].to_numpy() == fam) & F["ev_label"].is_not_null().to_numpy()
    return lgb.train(PRM, lgb.Dataset(X[m], F["ev_label"].to_numpy()[m].astype(int)), R_EV)


def fit_type(F, X, fam, mask):
    m = mask & (F["family"].to_numpy() == fam) & (F["is_ev"].to_numpy() == 1) & (F["ninv"].fill_null(0).to_numpy() >= 1)
    y = (F["seg"].to_numpy()[m] == 1).astype(int)
    return lgb.train(PRM_T, lgb.Dataset(X[m], y), R_T)


def dp_scores(E, qlogit, tau, fam):
    """E sorted by pair_id, hand_seq. Returns P(listed) array for family fam."""
    q = 1.0 / (1.0 + np.exp(-np.clip(qlogit, -40, 40)))
    q = np.where(E.select(gate(fam).fill_null(False)).to_series().to_numpy(), q, 0.0)
    pi = type_matrix(E, fam, tau)
    return listed_prob_groups(group_offsets(E["pair_id"].to_numpy()), q, np.ascontiguousarray(pi), CAP)


# ================================================================================================ diag
def run_diag():
    F = load_dev_features()
    out = {}
    L = F.filter(c("is_ev") == 1).with_columns(grp=pl.when(c("ninv") == 0).then(pl.lit("chrono")).otherwise(pl.format("inv{}_s{}", c("ninv"), c("seg"))))
    pos = F.with_columns(pos=c("q_pos")).filter(c("is_ev") == 1)
    for fam in FAMS:
        Lf = L.filter(c("family") == fam)
        tab = {}
        tab["timeline_pos"] = (pos.filter(c("family") == fam).join(Lf.select("pair_id", "hand_idx", "grp"), on=["pair_id", "hand_idx"])
                               .group_by("grp").agg(pl.len(), c("pos").mean().round(3)).sort("grp").rows())
        tab["ci_fb"] = Lf.group_by("grp", "ci_fb").len().sort("grp", "ci_fb").rows()
        tab["fold_ahead_any"] = Lf.group_by("grp", "fold_ahead_any").len().sort("grp", "fold_ahead_any").rows()
        tab["fold_ahead_don"] = Lf.group_by("grp", "fold_ahead_don").len().sort("grp", "fold_ahead_don").rows()
        out[fam] = tab
        log(SH[fam], json.dumps(tab))
    # deterministic rules: first-|L| P-hands in (type, time) order
    rules = {
        "coordinated_isolation": {"P: pair raise with <=1 fold before (sort fb,time)": ((c("ci_fb") >= 0) & (c("ci_fb") <= 1), c("ci_fb")),
                                  "P & first raiser p_aggr<0.2": ((c("ci_fb") >= 0) & (c("ci_fb") <= 1) & (c("fpa_p") < 0.2), c("ci_fb")),
                                  "P & first raiser p_aggr<0.3": ((c("ci_fb") >= 0) & (c("ci_fb") <= 1) & (c("fpa_p") < 0.3), c("ci_fb")),
                                  "P & fpa_p<0.3 & partner folds or re-raises": ((c("ci_fb") >= 0) & (c("ci_fb") <= 1) & (c("fpa_p") < 0.3) & c("ci_resp").fill_null(False), c("ci_fb"))},
        "directed_transfer": {"flow & donor-dir fold-ahead first, then flow & gift>5": (c("flow") & c("don_pair") & ((c("fold_ahead_don") > 0) | (c("gift_don") > 5)), 1 - c("fold_ahead_don")),
                              "flow & dt_strong|pfweak (donor dir), fold-ahead first": (c("flow") & c("don_pair") & (c("dt_strong") | c("dt_pfweak")), 1 - c("fold_ahead_don"))},
        "soft_play": {"sp_soft, fold-ahead first": (c("sp_soft"), 1 - c("fold_ahead_any")),
                      "S_PL & (fold-ahead | strong check | call post), fold-ahead first": (c("sp_pl") & ((c("fold_ahead_any") > 0) | c("sp_sc") | (c("call_post_mx") > 0)), 1 - c("fold_ahead_any"))},
    }
    out["rules"] = {}
    for fam, rs in rules.items():
        Ff = F.filter(c("family") == fam)
        for name, (P, T) in rs.items():
            r = eval_rule(Ff.with_columns(P=P.fill_null(False), T=T.cast(pl.Float64)))
            out["rules"][f"{SH[fam]}: {name}"] = r
            log(f"{SH[fam]} rule [{name}]: {r}")
    RES.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(RES.read_text()) if RES.exists() else {}
    prev["diag"] = out
    RES.write_text(json.dumps(prev, indent=1, default=str))


def eval_rule(Ff):
    res = []
    for (pid,), g in Ff.group_by(["pair_id"]):
        rel = set(g.filter(c("is_ev") == 1)["hand_idx"].to_list())
        nl = len(rel)
        cand = g.filter(c("P")).sort("T", "hand_seq")
        rule = cand["hand_idx"].to_list()[:5]
        Lg = g.filter(c("is_ev") == 1)
        last = Lg["hand_seq"].max()
        res.append(dict(exact=int(set(rule[:nl]) == rel), ap=ap5(rule, rel), nl=nl,
                        unl_before=g.filter(c("P") & (c("is_ev") == 0) & (c("hand_seq") < last)).height,
                        unl_total=g.filter(c("P") & (c("is_ev") == 0)).height,
                        listed_notP=Lg.filter(~c("P")).height))
    r = pl.DataFrame(res)
    s5, sh = r.filter(c("nl") == 5), r.filter(c("nl") < 5)
    return dict(pairs=r.height, exact_prefix=round(r["exact"].mean(), 3), map5=round(r["ap"].mean(), 4),
                listed_not_P_per_pair=round(r["listed_notP"].mean(), 3),
                P_unlisted_before_last_listed_5lists=round(s5["unl_before"].mean(), 3),
                P_unlisted_total_shortlists=(round(sh["unl_total"].mean(), 3) if sh.height else None))


# ================================================================================================ cv
def run_cv():
    F = load_dev_features()
    feats = feature_names()
    X = F.select([c(k).cast(pl.Float32) for k in feats]).to_numpy()
    fold = F["fold"].to_numpy()
    pair_fam = F["family"].to_numpy()
    qlog = {f: np.zeros(len(F)) for f in FAMS}
    tau = {f: np.zeros(len(F)) for f in FAMS}
    for k in range(5):
        tr = fold != k
        va = fold == k
        for f in FAMS:
            m = fit_event(F, X, f, tr)
            qlog[f][va] = m.predict(X[va], raw_score=True)
            if f != "coordinated_isolation":
                mt = fit_type(F, X, f, tr)
                tau[f][va] = mt.predict(X[va])
        log(f"cv fold {k} done")
    auc = {}
    lab_ok = F["ev_label"].is_not_null().to_numpy()
    for f in FAMS:
        m = (pair_fam == f) & lab_ok
        auc[SH[f]] = round(_auc(F["ev_label"].to_numpy()[m].astype(int), qlog[f][m]), 4)
        auc[SH[f] + "_n"] = [int(m.sum()), int(F["ev_label"].to_numpy()[m].sum())]
    log("event-head OOF AUC vs censoring-derived negatives:", auc)
    S8 = pl.read_parquet(C.DER / f"evidence_scores_dev_{PARTNER}.parquet").drop("hand_id")
    G = F.select("pair_id", "hand_idx", "family", "is_ev", "n_rel", "pot_bb").join(S8, on=["pair_id", "hand_idx"], how="left", maintain_order="left")
    out = {"partner": PARTNER, "w_dp": W_DP, "cf_mode": CF_MODE, "n_features": len(feats), "event_auc": auc}
    cols = {}
    for f in FAMS:
        p = dp_scores(F, qlog[f], tau[f], f)
        cols[f"dp_{f}"] = p
        cols[f"q_{f}"] = 1 / (1 + np.exp(-qlog[f]))
    G = G.with_columns([pl.Series(k, v) for k, v in cols.items()])
    G = strength_gate(G, [f"s_{f}" for f in FAMS], GATE_TH)
    for f in FAMS:
        G = G.with_columns(((1 - c("w")) * c(f"s_{f}").fill_null(0) + c("w") * c(f"dp_{f}")).alias(f"b_{f}"))
    if GATE_TH > 0:
        sh = G.group_by("pair_id").agg(c("w").first()).select((c("w") == 0).mean()).item()
        log(f"strength gate th={GATE_TH}: DP switched off for {sh:.3f} of dev positive pairs")
    table = {}
    for f in FAMS:
        row = {"partner": map5(G, f"s_{f}", f), "event_q_only": map5(G, f"q_{f}", f), "dp_only": map5(G, f"dp_{f}", f)}
        for w in (0.3, 0.5, 0.7):
            G2 = G.with_columns(((1 - w) * c(f"s_{f}").fill_null(0) + w * c(f"dp_{f}")).alias("tmp"))
            row[f"blend_w{w}"] = map5(G2, "tmp", f)
        table[f] = {k: round(v, 4) for k, v in row.items()}
        log(SH[f], table[f])
    out["dev_map5_true_family"] = table
    npairs = {f: F.filter(c("family") == f)["pair_id"].n_unique() for f in FAMS}
    tot = sum(npairs.values())
    for key in table[FAMS[0]]:
        out.setdefault("overall", {})[key] = round(sum(table[f][key] * npairs[f] for f in FAMS) / tot, 4)
    log("overall", out["overall"])
    # uniform-posterior routing check (compose over family mixture)
    Gm = G.with_columns(mix=pl.mean_horizontal([c(f"b_{f}") for f in FAMS]), mix8=pl.mean_horizontal([c(f"s_{f}").fill_null(0) for f in FAMS]))
    out["uniform_posterior"] = {"blend": round(map5(Gm, "mix"), 4), "partner": round(map5(Gm, "mix8"), 4)}
    log("uniform posteriors", out["uniform_posterior"])
    # dev score file: blended scores, all rows (372 pairs), exp008 format
    hid = C.load("hands").select("hand_idx", "hand_id")
    dev_out = (G.join(hid, on="hand_idx", how="left")
               .select("pair_id", "hand_id", "hand_idx", *[c(f"b_{f}").cast(pl.Float64).alias(f"s_{f}") for f in FAMS])
               .sort("pair_id", "hand_idx"))
    tag = TAG if PARTNER == "exp008" else f"{TAG}_{PARTNER}"
    dev_out.write_parquet(C.DER / f"evidence_scores_dev_{tag}.parquet")
    G.select("pair_id", "hand_idx", "w", *cols.keys()).write_parquet(C.DER / "evidence_cache" / f"{tag}_dev_components.parquet")
    log(f"wrote evidence_scores_dev_{tag}.parquet ({dev_out.height:,} rows)")
    prev = json.loads(RES.read_text()) if RES.exists() else {}
    prev[f"cv_{TAG}_{PARTNER}"] = out
    RES.write_text(json.dumps(prev, indent=1, default=str))
    return out


# ================================================================================================ eval
def run_eval(tables_per_batch=20, topk=15):
    t_start = time.time()
    F = load_dev_features()
    feats = feature_names()
    X = F.select([c(k).cast(pl.Float32) for k in feats]).to_numpy()
    allm = np.ones(len(F), dtype=bool)
    mq = {f: fit_event(F, X, f, allm) for f in FAMS}
    mt = {f: fit_type(F, X, f, allm) for f in FAMS if f != "coordinated_isolation"}
    del X
    log(f"eval: full-dev models fitted ({time.time() - t_start:.0f}s)")
    partners = {p: pl.read_parquet(C.DER / f"evidence_scores_eval_{p}.parquet").drop("hand_id")
                for p in os.environ.get("LR_PARTNERS", "exp008,exp027ev").split(",")}
    ep = C.load("eval_pairs").select("pair_id", c("p1").alias("pA"), c("p2").alias("pB"), "shared_hands")
    files = sorted((EV.PH / "phase1").glob("*.parquet"))
    outs = {p: [] for p in partners}
    dps = []
    n_rows = n_pairs = 0
    for tb in range(0, 400, tables_per_batch):
        t = time.time()
        R = (pl.scan_parquet(files).filter(c("table_idx").is_between(tb, tb + tables_per_batch - 1))
             .join(ep.lazy(), on=["pA", "pB"]).collect().sort("pair_id", "hand_seq"))
        chk = R.group_by("pair_id").agg(pl.len(), c("shared_hands").first())
        assert (chk["len"] == chk["shared_hands"]).all(), "shared-hand count mismatch vs eval_pairs"
        n_pairs += chk.height
        E = EV.hand_features(R.drop("shared_hands"))
        del R
        E = add_llr(add_extra(E))
        XE = E.select([c(k).cast(pl.Float32) for k in feats]).to_numpy()
        cols = {}
        for f in FAMS:
            ql = mq[f].predict(XE, raw_score=True)
            ta = mt[f].predict(XE) if f in mt else None
            cols[f"dp_{f}"] = dp_scores(E, ql, ta, f)
        K0 = E.select("pair_id", "hand_idx", "pot_bb").with_columns([pl.Series(k, v) for k, v in cols.items()])
        # 1e-7 pot-size tie-break (same convention as evidence.write_scores)
        K0 = K0.with_columns(tbk=(c("pot_bb").rank("average").over("pair_id") / pl.len().over("pair_id")) * 1e-7)
        dps.append(K0.filter(pl.max_horizontal([c(f"dp_{f}") for f in FAMS]) > 1e-4).select("pair_id", "hand_idx", *cols.keys()))
        for pn, S8 in partners.items():
            K = K0.join(S8, on=["pair_id", "hand_idx"], how="left")
            K = strength_gate(K, [f"s_{f}" for f in FAMS], GATE_TH)
            K = K.with_columns([((1 - c("w")) * c(f"s_{f}").fill_null(0) + c("w") * c(f"dp_{f}") + c("tbk")).alias(f"b_{f}") for f in FAMS])
            rk = [c(f"b_{f}").rank("ordinal", descending=True).over("pair_id") for f in FAMS]
            K = K.filter(pl.min_horizontal(rk) <= topk)
            outs[pn].append(K.select("pair_id", "hand_idx", *[c(f"b_{f}").alias(f"s_{f}") for f in FAMS]))
        n_rows += len(XE)
        del E, XE, K0
        log(f"eval tables {tb}-{tb + tables_per_batch - 1}: {n_rows:,} rows so far, {time.time() - t:.0f}s")
    assert n_pairs == ep.height, (n_pairs, ep.height)
    hid = C.load("hands").select("hand_idx", "hand_id", "phase")
    pl.concat(dps).write_parquet(C.DER / "evidence_cache" / f"{TAG}_eval_dp.parquet")
    prev = json.loads(RES.read_text()) if RES.exists() else {}
    for pn, parts in outs.items():
        out = pl.concat(parts).join(hid, on="hand_idx", how="left")
        assert (out["phase"] == 1).all(), "non-eval hand in eval evidence"
        tag = TAG if pn == "exp008" else f"{TAG}_{pn}"
        final = out.select("pair_id", "hand_id", "hand_idx", *[c(f"s_{f}").cast(pl.Float64) for f in FAMS]).sort("pair_id", "hand_idx")
        final.write_parquet(C.DER / f"evidence_scores_eval_{tag}.parquet")
        cnt = final.group_by("pair_id").len()
        log(f"wrote evidence_scores_eval_{tag}.parquet: {final.height:,} rows, {final['pair_id'].n_unique():,} pairs, "
            f"rows/pair min {cnt['len'].min()} median {cnt['len'].median()}")
        prev[f"eval_{TAG}_{pn}"] = dict(rows_scored=n_rows, rows_kept=final.height, pairs=final["pair_id"].n_unique(),
                                  w_dp=W_DP, gate_th=GATE_TH, runtime_s=round(time.time() - t_start, 1))
    RES.write_text(json.dumps(prev, indent=1, default=str))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("diag", "all"):
        run_diag()
    if stage in ("cv", "all"):
        run_cv()
    if stage in ("eval", "all"):
        run_eval()
