"""exp050 -- COUNTERFACTUAL VALUE features for the evidence event head.

The event head asks "was this colluder's action altered by the generator in this hand?".  Every existing input
describes WHAT HAPPENED (engine views, gift/impact, fold-ahead keys).  None describes WHAT WOULD HAVE HAPPENED
under normal play, which is the definition of an altered action.  This module supplies that difference.

One-step value accounting (Mazrooei, Archibald & Bowling AAAI-13, as in src/hand_replay.py): the impact of an
action by actor k (chips a, pot P before) on seat j is
    dV_j = (eq_post_j - eq_pre_j) * P + eq_post_j * a - [j == k] * a          [chips; /bb -> big blinds]
with omniscient per-seat equities.  eq_post == eq_pre except on a fold.

For every decision we replay the SAME accounting under each legal alternative, weighted by the label-free
population policy (data/derived/action_policy.parquet, exp003; cross-fitted by table half, positive-pair hands
excluded from its training set):
    fold  : eq_post_j = eq_pre_j / (1 - eq_pre_k)   (proportional redistribution; |err| .032 / corr .971 against
            the exact recomputation on 646k real folds), a = 0
    check : nothing moves                                   (legal only when to_call == 0)
    call  : a = min(to_call, stack)                         eq unchanged
    agg   : a = min(stack, to_call + r_street * (P + to_call)), r = per-street median raise ratio
    E[dV_j] = sum_action p_action * dV_j(action)            (p renormalised over legal actions)
    delta_j = dV_j(actual) - E[dV_j]        > 0  <=>  k handed seat j more value than normal play would have.

Features exported per (hand, ordered seat pair k -> j) and per (hand, actor):
    v_sum/v_max        sum / max of delta_j over k's decisions in the hand
    s_sum/s_max        the same weighted by the policy surprise -ln p(actual action): unusual AND profitable
    a_sum/e_sum        the actual and the expected (normal-play) halves, kept separately
    dchips/dfold/surp  chips invested vs normal, fold rate vs normal, total surprise of k in the hand
and, at pair level, the DIRECTEDNESS at HAND level: the same delta computed toward every other seated
non-partner, so `excess = delta_to_partner - max/mean(delta_to_others)` separates a planted transfer from a
cooler that happened to be profitable for everyone downstream of the actor.

Usage:  import cfvalue;  cfvalue.pair_features(R)   with R = (hand_idx, pA, pB) rows; returns one row per input row.
Rules: gameplay only (actions, seats, cards, policy fitted on actions).  No IDs / file order / eval membership.
"""
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit

BASE = Path(__file__).resolve().parent.parent
DER = BASE / "data" / "derived"
c = pl.col

# per-street median (amount - to_call) / (pot_before + to_call) over aggressive actions
R_STREET = np.array([0.60, 0.73, 0.69, 0.62], dtype=np.float64)
NST = 6          # v_sum, v_max, s_sum, s_max, a_sum, e_sum
NAC = 4          # dchips, dfold, surp_sum, n_act
EQP = [f"eq_pre_s{i}" for i in range(6)]
EQQ = [f"eq_post_s{i}" for i in range(6)]


@njit(cache=True)
def _kernel(h_start, street, seat, act, amount, pot, stack, to_call, legal,
            p_fold, p_check, p_call, p_agg, surprise, eqpre, eqpost, bb, out, oac):
    nh = h_start.shape[0] - 1
    for h in range(nh):
        b = bb[h]
        if b <= 0:
            continue
        for t in range(h_start[h], h_start[h + 1]):
            k = seat[t]
            if k < 0 or k > 5:
                continue
            P = pot[t] / b
            T = to_call[t] / b
            ST = stack[t] / b
            aa = amount[t] / b
            ek = eqpre[t, k]
            if not (ek == ek):
                continue
            lg = legal[t]
            pf = p_fold[t] if (lg & 1) else 0.0
            pc = p_check[t] if (lg & 2) else 0.0
            pl_ = p_call[t] if (lg & 4) else 0.0
            pa = p_agg[t] if (lg & 8) else 0.0
            s = pf + pc + pl_ + pa
            if s <= 1e-9:
                continue
            pf /= s
            pc /= s
            pl_ /= s
            pa /= s
            a_call = T if T < ST else ST
            r = R_STREET[street[t]] if 0 <= street[t] <= 3 else 0.65
            a_agg = T + r * (P + T)
            if a_agg > ST:
                a_agg = ST
            den = 1.0 - ek
            if den < 0.02:
                den = 0.02
            e_a = pl_ * a_call + pa * a_agg          # expected chips in
            sw = surprise[t]
            oac[h, k, 0] += aa - e_a
            oac[h, k, 1] += (1.0 if act[t] == 0 else 0.0) - pf
            oac[h, k, 2] += sw
            oac[h, k, 3] += 1.0
            for j in range(6):
                if j == k:
                    continue
                ej = eqpre[t, j]
                if not (ej == ej):
                    continue
                qj = eqpost[t, j]
                if not (qj == qj):
                    qj = ej
                imp_act = (qj - ej) * P + qj * aa
                imp_fold = ej * (ek / den) * P
                imp_call = ej * a_call
                imp_agg = ej * a_agg
                e_imp = pf * imp_fold + pl_ * imp_call + pa * imp_agg
                d = imp_act - e_imp
                out[h, k, j, 0] += d
                if d > out[h, k, j, 1]:
                    out[h, k, j, 1] = d
                sd = sw * d
                out[h, k, j, 2] += sd
                if sd > out[h, k, j, 3]:
                    out[h, k, j, 3] = sd
                out[h, k, j, 4] += imp_act
                out[h, k, j, 5] += e_imp


def hand_tables(hids, phase=None):
    """hids: DataFrame with a hand_idx column. Returns (sorted hand ids, out[n,6,6,NST], oac[n,6,NAC])."""
    hl = hids.select("hand_idx").unique().sort("hand_idx")
    H = (pl.scan_parquet(DER / "hands.parquet").join(hl.lazy(), on="hand_idx")
         .select("hand_idx", "bb", "phase").sort("hand_idx").collect())
    if phase is None:
        assert H["phase"].min() == H["phase"].max(), "mixed-phase hand batch"
        phase = int(H["phase"].max())
    tag = "dev" if phase == 0 else "eval"
    A = (pl.scan_parquet(DER / "actions.parquet").join(hl.lazy(), on="hand_idx")
         .select("hand_idx", "action_no", "street", "player_idx", "action", "amount", "pot_before",
                 "stack_before", "to_call"))
    P = (pl.scan_parquet(DER / "action_policy.parquet").join(hl.lazy(), on="hand_idx")
         .select("hand_idx", "action_no", "player_idx", "legal", "p_fold", "p_check", "p_call", "p_agg", "surprise"))
    E = (pl.scan_parquet(DER / f"action_equity_{tag}.parquet").join(hl.lazy(), on="hand_idx")
         .select("hand_idx", "action_no", "seat_no", *EQP, *EQQ))
    J = (A.join(P, on=["hand_idx", "action_no", "player_idx"], how="left")
         .join(E, on=["hand_idx", "action_no"], how="left").sort("hand_idx", "action_no").collect())
    hand_ids = H["hand_idx"].to_numpy()
    nh = len(hand_ids)
    hrow = np.searchsorted(hand_ids, J["hand_idx"].to_numpy())
    h_start = np.searchsorted(hrow, np.arange(nh + 1)).astype(np.int64)
    eqpre = np.ascontiguousarray(J.select(EQP).to_numpy().astype(np.float64))
    eqpost = np.ascontiguousarray(J.select(EQQ).to_numpy().astype(np.float64))
    out = np.zeros((nh, 6, 6, NST), dtype=np.float64)
    out[:, :, :, 1] = -1e18
    out[:, :, :, 3] = -1e18
    oac = np.zeros((nh, 6, NAC), dtype=np.float64)
    _kernel(h_start, J["street"].to_numpy().astype(np.int64),
            np.nan_to_num(J["seat_no"].to_numpy().astype(np.float64), nan=-1).astype(np.int64),
            J["action"].to_numpy().astype(np.int64), J["amount"].to_numpy().astype(np.float64),
            J["pot_before"].to_numpy().astype(np.float64), J["stack_before"].to_numpy().astype(np.float64),
            J["to_call"].to_numpy().astype(np.float64),
            np.nan_to_num(J["legal"].to_numpy().astype(np.float64), nan=0).astype(np.int64),
            *[np.nan_to_num(J[k].to_numpy().astype(np.float64)) for k in ("p_fold", "p_check", "p_call", "p_agg", "surprise")],
            eqpre, eqpost, H["bb"].to_numpy().astype(np.float64), out, oac)
    out[:, :, :, 1] = np.where(out[:, :, :, 1] < -1e17, 0.0, out[:, :, :, 1])
    out[:, :, :, 3] = np.where(out[:, :, :, 3] < -1e17, 0.0, out[:, :, :, 3])
    return hand_ids, out, oac


RAW = ["cfv", "cfvx", "cfs", "cfsx", "cfa", "cfe", "cfo", "cfom", "cfso", "cfxo"]
ACC = ["cfdch", "cfdfd", "cfsurp"]
COLS = [f"{k}_{s}" for k in RAW + ACC for s in ("ab", "ba")]


def pair_features(R, phase=None, seats=None):
    """R: rows with hand_idx, pA, pB. Returns hand_idx, pA, pB + the _ab/_ba counterfactual columns, same order."""
    hl = R.select("hand_idx").unique()
    hand_ids, out, oac = hand_tables(hl, phase)
    if seats is None:
        seats = (pl.scan_parquet(DER / "seats.parquet").join(hl.lazy(), on="hand_idx")
                 .select("hand_idx", "player_idx", "seat_no").collect())
    S = seats
    X = (R.select("hand_idx", "pA", "pB")
         .join(S.rename({"player_idx": "pA", "seat_no": "sA"}), on=["hand_idx", "pA"], how="left", maintain_order="left")
         .join(S.rename({"player_idx": "pB", "seat_no": "sB"}), on=["hand_idx", "pB"], how="left", maintain_order="left"))
    hr = np.searchsorted(hand_ids, X["hand_idx"].to_numpy())
    sA = np.nan_to_num(X["sA"].to_numpy().astype(np.float64), nan=-1).astype(np.int64)
    sB = np.nan_to_num(X["sB"].to_numpy().astype(np.float64), nan=-1).astype(np.int64)
    ok = (sA >= 0) & (sB >= 0)
    sAc, sBc = np.where(ok, sA, 0), np.where(ok, sB, 0)
    n = len(hr)
    ar = np.arange(n)
    live = oac[:, :, 3] > 0
    cols = {}
    for side, (sx, sy) in (("ab", (sAc, sBc)), ("ba", (sBc, sAc))):
        blk = out[hr, sx]                                                # (n, 6, NST): actor sx -> every seat
        tgt = blk[ar, sy]                                                # (n, NST)
        msk = np.ones((n, 6), dtype=bool)
        msk[ar, sx] = False
        msk[ar, sy] = False
        msk &= live[hr]
        cnt = msk.sum(axis=1)
        any_o = cnt > 0
        vs = np.where(msk, blk[:, :, 0], np.nan)
        ss = np.where(msk, blk[:, :, 2], np.nan)
        o_max = np.zeros(n)
        o_mean = np.zeros(n)
        s_max = np.zeros(n)
        if any_o.any():
            o_max[any_o] = np.nanmax(vs[any_o], axis=1)
            o_mean[any_o] = np.nanmean(vs[any_o], axis=1)
            s_max[any_o] = np.nanmax(ss[any_o], axis=1)
        for i, nm in enumerate(("cfv", "cfvx", "cfs", "cfsx", "cfa", "cfe")):
            cols[f"{nm}_{side}"] = np.where(ok, tgt[:, i], 0.0)
        cols[f"cfo_{side}"] = np.where(ok, np.nan_to_num(o_max), 0.0)
        cols[f"cfom_{side}"] = np.where(ok, np.nan_to_num(o_mean), 0.0)
        cols[f"cfso_{side}"] = np.where(ok, np.nan_to_num(s_max), 0.0)
        cols[f"cfxo_{side}"] = np.where(ok, cnt.astype(np.float64), 0.0)
        for i, nm in enumerate(ACC):
            cols[f"{nm}_{side}"] = np.where(ok, oac[hr, sx, i], 0.0)
    return R.select("hand_idx", "pA", "pB").with_columns(
        [pl.Series(k, v.astype(np.float32)) for k, v in cols.items()])


if __name__ == "__main__":
    import time
    D = pl.read_parquet(DER / "evidence_cache" / "dev_pos.parquet",
                        columns=["hand_idx", "pA", "pB", "is_ev", "pair_id", "family"])
    t = time.time()
    F = pair_features(D.select("hand_idx", "pA", "pB"), 0)
    print(f"{F.height:,} rows x {F.width} cols in {time.time() - t:.0f}s", flush=True)
    G = D.hstack(F.drop("hand_idx", "pA", "pB"))
    G = G.with_columns(cfv_mx=pl.max_horizontal("cfv_ab", "cfv_ba"),
                       cfs_mx=pl.max_horizontal("cfs_ab", "cfs_ba"),
                       cfexc=pl.max_horizontal(c("cfv_ab") - c("cfo_ab"), c("cfv_ba") - c("cfo_ba")))
    for k in ["cfv_ab", "cfv_ba", "cfv_mx", "cfs_mx", "cfexc", "cfo_ab", "cfdch_ab", "cfdfd_ab", "cfa_ab", "cfe_ab"]:
        a = G.filter(c("is_ev") == 1)[k].mean()
        b = G.filter(c("is_ev") == 0)[k].mean()
        print(f"{k:10s} listed {a: .4f}  unlisted {b: .4f}")
    for f in ("directed_transfer", "soft_play", "coordinated_isolation"):
        H = G.filter(c("family") == f)
        print(f[:2].upper(), {k: round(float(H.filter(c("is_ev") == 1)[k].mean() - H.filter(c("is_ev") == 0)[k].mean()), 4)
                              for k in ("cfv_mx", "cfs_mx", "cfexc")})
