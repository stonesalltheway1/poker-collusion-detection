"""Pair-hand engine (exp001): one numba pass per hand fills features for all 15 seat pairs.

Output: data/derived/ph/phase{0,1}/part_XX.parquet, one row per (hand, unordered pair of seated players).
Pair orientation is canonical only: A = lower player_idx, B = higher. Directional columns come as
<feat>_ab (X=A acting towards Y=B) and <feat>_ba (X=B towards Y=A).

Inputs (all gameplay, label-free): actions, seats, hands, action_equity_<phase> (omniscient per-seat equity
before/after each action, actor perceived strength hs_actor), hu_equity_<phase> (heads-up equity per street
and seat pair), population preflop tables (class x #prior aggr x position x #prior callers) built on all hands,
optional per-action normal-play policy (data/derived/action_policy.parquet, exp003).

Definitions follow research/forensics/*.md:
  facing Y  = Y is the last aggressor on the current street and to_call > 0 at X's action
  aggr      = bet / raise / all_in with amount > to_call; call = call or all_in with amount <= to_call
  gift X->Y = EV X gives up to Y: fold facing Y max(0, eq*(pot+to_call)-to_call); call facing Y
              max(0, amt - eq*(pot+amt)); X bets/raises and Y then calls/raises max(0, amt*(1-2*eqHU))  [bb]
  imp X->Y  = Mazrooei et al. (AAAI-13) luck-free value impact of X's actions on Y's equity-weighted pot share [bb]
Usage: python src/engine.py [phase ...]      (default: 0 1)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit, prange

BASE = Path(__file__).resolve().parent.parent
DER = BASE / "data" / "derived"
OUT = DER / os.environ.get("PH_DIR", "ph")

# ------------------------------------------------------------------------------------------- layout
SYM = ["pot_bb", "board_n", "players_vol", "both_streets", "hu_streets", "hu_checked_streets", "sd_both",
       "hu_final", "fpa_role", "fpa_ord", "fpa_p", "fpa_partner_behind", "sandwich", "tf_facing", "tf_pre",
       "tf_both_active", "third_vol", "pre_actions"]
DIR = ["face", "fold_pre", "call_pre", "raise_pre", "fold_post", "call_post", "raise_post",
       "min_pfold_pre", "min_pcall_pre", "max_pfeq_call_pre", "max_hueq_fold_post", "min_hueq_call_post",
       "min_hueq_call_tr", "max_hs_fold_post", "max_hs_call_post", "strongcheck", "max_hs_check",
       "bluffinto", "gift", "gift_max", "imp", "aggpre", "trash", "weak10", "minpa", "fold_after_aggr",
       "vol", "first_fold_nonfacing", "enter", "net_bb", "aggpost", "min_polcall_pre", "max_polagg_resp",
       "surp_face", "min_polfold_face", "max_eq_fold_face"]
NS, ND = len(SYM), len(DIR)
NF = NS + 2 * ND
S = {f: i for i, f in enumerate(SYM)}
Dx = {f: i for i, f in enumerate(DIR)}
COLS = SYM + [f"{f}_ab" for f in DIR] + [f"{f}_ba" for f in DIR]
INIT = np.zeros(NF, dtype=np.float32)
for f, v in {"fpa_ord": 9, "fpa_p": 1}.items():
    INIT[S[f]] = v
for f, v in {"min_pfold_pre": 1, "min_pcall_pre": 1, "max_pfeq_call_pre": -1, "max_hueq_fold_post": -1,
             "min_hueq_call_post": 2, "min_hueq_call_tr": 2, "max_hs_fold_post": -1, "max_hs_call_post": -1,
             "max_hs_check": -1, "minpa": 1, "min_polcall_pre": 1, "max_polagg_resp": -1,
             "min_polfold_face": 1, "max_eq_fold_face": -1}.items():
    for o in (0, 1):
        INIT[NS + o * ND + Dx[f]] = v

PI = np.full((6, 6), -1, dtype=np.int64)          # seat pair -> 0..14
_k = 0
for _i in range(6):
    for _j in range(_i + 1, 6):
        PI[_i, _j] = PI[_j, _i] = _k
        _k += 1

# numba constants
D_FACE, D_FOLD_PRE, D_CALL_PRE, D_RAISE_PRE, D_FOLD_POST, D_CALL_POST, D_RAISE_POST = [Dx[f] for f in DIR[:7]]
(D_MIN_PFOLD_PRE, D_MIN_PCALL_PRE, D_MAX_PFEQ_CALL_PRE, D_MAX_HUEQ_FOLD_POST, D_MIN_HUEQ_CALL_POST,
 D_MIN_HUEQ_CALL_TR, D_MAX_HS_FOLD_POST, D_MAX_HS_CALL_POST, D_STRONGCHECK, D_MAX_HS_CHECK, D_BLUFFINTO,
 D_GIFT, D_GIFT_MAX, D_IMP, D_AGGPRE, D_TRASH, D_WEAK10, D_MINPA, D_FOLD_AFTER_AGGR, D_VOL,
 D_FIRST_FOLD_NONFACING, D_ENTER, D_NET_BB, D_AGGPOST, D_MIN_POLCALL_PRE, D_MAX_POLAGG_RESP, D_SURP_FACE,
 D_MIN_POLFOLD_FACE, D_MAX_EQ_FOLD_FACE) = [Dx[f] for f in DIR[7:]]
(S_POT_BB, S_BOARD_N, S_PLAYERS_VOL, S_BOTH_STREETS, S_HU_STREETS, S_HU_CHECKED_STREETS, S_SD_BOTH, S_HU_FINAL,
 S_FPA_ROLE, S_FPA_ORD, S_FPA_P, S_FPA_PARTNER_BEHIND, S_SANDWICH, S_TF_FACING, S_TF_PRE, S_TF_BOTH_ACTIVE,
 S_THIRD_VOL, S_PRE_ACTIONS) = range(NS)


@njit(cache=True, inline="always")
def class_id(c1, c2):
    r1, r2 = c1 // 4, c2 // 4
    hi, lo = max(r1, r2), min(r1, r2)
    if r1 == r2:
        return r1 * 13 + r1
    if c1 % 4 == c2 % 4:
        return hi * 13 + lo
    return lo * 13 + hi


@njit(cache=True)
def _col(o, X, A):
    """base column offset for directional features of actor seat X in pair whose canonical A-seat is A."""
    return NS + (0 if X == A else ND)


@njit(cache=True, parallel=True)
def run_chunk(h0, h1, s_player, s_c1, s_c2, s_net, s_sd, s_folded, h_button, h_bb, h_pot, h_nboard,
              a_start, a_end, a_street, a_player, a_action, a_amount, a_to_call, a_pot, a_pactive,
              eq_pre, eq_post, hs_act, hu, pol, has_pol, pf_eq1, pa_tab, pf_tab, pc_tab, PI, INIT):
    nh = h1 - h0
    out = np.empty((nh * 15, INIT.shape[0]), dtype=np.float32)
    seatA = np.empty(nh * 15, dtype=np.int8)
    seatB = np.empty(nh * 15, dtype=np.int8)
    for r in prange(h0, h1):
        base = (r - h0) * 15
        for p in range(15):
            for f in range(INIT.shape[0]):
                out[base + p, f] = INIT[f]
        so = r * 6
        bb = h_bb[r]
        pl_ = np.empty(6, dtype=np.int64)
        pos = np.empty(6, dtype=np.int64)
        cls = np.empty(6, dtype=np.int64)
        for s in range(6):
            pl_[s] = s_player[so + s]
            pos[s] = (s - h_button[r] + 6) % 6          # 0 BTN 1 SB 2 BB 3 UTG 4 HJ 5 CO
            cls[s] = class_id(s_c1[so + s], s_c2[so + s])
        # canonical A/B per seat pair
        Aof = np.empty(15, dtype=np.int64)
        Bof = np.empty(15, dtype=np.int64)
        for i in range(6):
            for j in range(i + 1, 6):
                p = PI[i, j]
                if pl_[i] < pl_[j]:
                    Aof[p] = i; Bof[p] = j
                else:
                    Aof[p] = j; Bof[p] = i
                seatA[base + p] = Aof[p]
                seatB[base + p] = Bof[p]
        active = np.ones(6, dtype=np.int64)
        first_act = np.full(6, -1, dtype=np.int64)
        first_fold_facing = np.full(6, -1, dtype=np.int64)
        vol = np.zeros(6, dtype=np.int64)
        acted_pre = np.zeros(6, dtype=np.int64)
        first_aggr_pos = np.full(6, 1 << 30, dtype=np.int64)
        fold_pos = np.full(6, 1 << 30, dtype=np.int64)
        pend_eq = np.full((6, 6), -1.0)                 # X aggr this street: HU eq vs Y (for bluff-into)
        pend_amt = np.zeros(6)
        hu_aggr_seen = np.zeros(15, dtype=np.int64)
        hu_street_on = np.zeros(15, dtype=np.int64)
        fpa_done = np.zeros(15, dtype=np.int64)
        last_pair_aggr = np.full(15, -1, dtype=np.int64)   # seat of latest preflop aggr within pair
        third_since = np.zeros(15, dtype=np.int64)
        cur_st = -1
        last_aggr = -1
        nr = 0
        ncall = 0
        third_vol_tot = 0
        n_vol = 0
        n_pre = 0
        k_pos = 0
        for k in range(a_start[r], a_end[r]):
            st = a_street[k]
            if st != cur_st:
                # close previous street HU flags
                for p in range(15):
                    if hu_street_on[p] == 1 and hu_aggr_seen[p] == 0:
                        out[base + p, S_HU_CHECKED_STREETS] += 1
                    hu_street_on[p] = 0
                    hu_aggr_seen[p] = 0
                cur_st = st
                last_aggr = -1
                nr = 0
                for x in range(6):
                    pend_amt[x] = 0.0
                    for y in range(6):
                        pend_eq[x, y] = -1.0
                nact = 0
                for s in range(6):
                    nact += active[s]
                for i in range(6):
                    for j in range(i + 1, 6):
                        if active[i] == 1 and active[j] == 1:
                            p = PI[i, j]
                            out[base + p, S_BOTH_STREETS] += 1
                            if st > 0 and nact == 2:
                                out[base + p, S_HU_STREETS] += 1
                                hu_street_on[p] = 1
            x = -1
            for s in range(6):
                if pl_[s] == a_player[k]:
                    x = s
            act = a_action[k]
            amt = a_amount[k]
            tc = a_to_call[k]
            P = a_pot[k]
            aggr = act == 3 or act == 4 or (act == 5 and amt > tc)
            iscall = act == 2 or (act == 5 and amt <= tc)
            L = last_aggr
            eqx = eq_pre[k, x]
            if eqx != eqx:
                eqx = 0.0
            # ---- Mazrooei impact of X's action on every other seat j
            for j in range(6):
                if j == x:
                    continue
                e0 = eq_pre[k, j]
                e1 = eq_post[k, j]
                if e0 != e0:
                    e0 = 0.0
                if e1 != e1:
                    e1 = 0.0
                dv = (e1 - e0) * P + e1 * amt
                p = PI[x, j]
                out[base + p, _col(0, x, Aof[p]) + D_IMP] += dv / bb
            # ---- preflop bookkeeping
            if st == 0:
                n_pre += 1
                if first_act[x] < 0:
                    first_act[x] = act
                    first_fold_facing[x] = L if act == 0 else -1
                if iscall or aggr:
                    if vol[x] == 0:
                        vol[x] = 1
                        n_vol += 1
                    for p in range(15):
                        if Aof[p] != x and Bof[p] != x and last_pair_aggr[p] >= 0:
                            third_since[p] += 1
                    # third players voluntarily in (for any pair not containing x) counted at hand end
            hs = hs_act[k]
            # ---- facing the last aggressor Y
            if L >= 0 and L != x and tc > 0:
                y = L
                p = PI[x, y]
                c = _col(0, x, Aof[p])
                o = base + p
                out[o, c + D_FACE] += 1
                if has_pol:
                    pa_ = pol[k, act_class(act, amt, tc)]
                    out[o, c + D_SURP_FACE] += -np.log(max(pa_, 1e-4))
                if act == 0:
                    if st == 0:
                        out[o, c + D_FOLD_PRE] += 1
                        pf = pf_tab[cls[x], min(nr, 3), pos[x], min(ncall, 2)]
                        if pf < out[o, c + D_MIN_PFOLD_PRE]:
                            out[o, c + D_MIN_PFOLD_PRE] = pf
                    else:
                        out[o, c + D_FOLD_POST] += 1
                        he = hu[r, st, PI[x, y]]
                        if he == he:
                            if x > y:
                                he = 1.0 - he
                            if he > out[o, c + D_MAX_HUEQ_FOLD_POST]:
                                out[o, c + D_MAX_HUEQ_FOLD_POST] = he
                        if hs == hs and hs > out[o, c + D_MAX_HS_FOLD_POST]:
                            out[o, c + D_MAX_HS_FOLD_POST] = hs
                    if eqx > out[o, c + D_MAX_EQ_FOLD_FACE]:
                        out[o, c + D_MAX_EQ_FOLD_FACE] = eqx
                    g = eqx * (P + tc) - tc
                    if g > 0:
                        out[o, c + D_GIFT] += g / bb
                        if g / bb > out[o, c + D_GIFT_MAX]:
                            out[o, c + D_GIFT_MAX] = g / bb
                    if has_pol and pol[k, 0] < out[o, c + D_MIN_POLFOLD_FACE]:
                        out[o, c + D_MIN_POLFOLD_FACE] = pol[k, 0]
                elif iscall:
                    if st == 0:
                        out[o, c + D_CALL_PRE] += 1
                        pc = pc_tab[cls[x], min(nr, 3), pos[x], min(ncall, 2)]
                        if pc < out[o, c + D_MIN_PCALL_PRE]:
                            out[o, c + D_MIN_PCALL_PRE] = pc
                        e1v = pf_eq1[cls[x]]
                        if e1v > out[o, c + D_MAX_PFEQ_CALL_PRE]:
                            out[o, c + D_MAX_PFEQ_CALL_PRE] = e1v
                        if has_pol and pol[k, 2] < out[o, c + D_MIN_POLCALL_PRE]:
                            out[o, c + D_MIN_POLCALL_PRE] = pol[k, 2]
                    else:
                        out[o, c + D_CALL_POST] += 1
                        he = hu[r, st, PI[x, y]]
                        if he == he:
                            if x > y:
                                he = 1.0 - he
                            if he < out[o, c + D_MIN_HUEQ_CALL_POST]:
                                out[o, c + D_MIN_HUEQ_CALL_POST] = he
                            if st >= 2 and he < out[o, c + D_MIN_HUEQ_CALL_TR]:
                                out[o, c + D_MIN_HUEQ_CALL_TR] = he
                        if hs == hs and hs > out[o, c + D_MAX_HS_CALL_POST]:
                            out[o, c + D_MAX_HS_CALL_POST] = hs
                    g = amt - eqx * (P + amt)
                    if g > 0:
                        out[o, c + D_GIFT] += g / bb
                        if g / bb > out[o, c + D_GIFT_MAX]:
                            out[o, c + D_GIFT_MAX] = g / bb
                    if has_pol and pol[k, 3] > out[o, c + D_MAX_POLAGG_RESP]:
                        out[o, c + D_MAX_POLAGG_RESP] = pol[k, 3]
                elif aggr:
                    if st == 0:
                        out[o, c + D_RAISE_PRE] += 1
                    else:
                        out[o, c + D_RAISE_POST] += 1
            # ---- strong check with other players active (per pair with each active y)
            if act == 1 and st > 0 and hs == hs:
                for y in range(6):
                    if y != x and active[y] == 1:
                        p = PI[x, y]
                        c = _col(0, x, Aof[p])
                        if hs >= 0.8:
                            out[base + p, c + D_STRONGCHECK] += 1
                        if hs > out[base + p, c + D_MAX_HS_CHECK]:
                            out[base + p, c + D_MAX_HS_CHECK] = hs
            # ---- Y responds (call/raise) to X's earlier aggression this street -> bluff-into / gift
            if iscall or aggr:
                for z in range(6):
                    if z != x and pend_eq[z, x] >= 0:
                        p = PI[z, x]
                        c = _col(0, z, Aof[p])
                        e = pend_eq[z, x]
                        if e <= 0.30:
                            out[base + p, c + D_BLUFFINTO] += 1
                        g = pend_amt[z] * (1.0 - 2.0 * e)
                        if g > 0:
                            out[base + p, c + D_GIFT] += g / bb
                            if g / bb > out[base + p, c + D_GIFT_MAX]:
                                out[base + p, c + D_GIFT_MAX] = g / bb
                        pend_eq[z, x] = -1.0
            # ---- X aggression
            if aggr:
                if first_aggr_pos[x] > k_pos:
                    first_aggr_pos[x] = k_pos
                pend_amt[x] = amt
                for y in range(6):
                    if y != x and active[y] == 1:
                        he = hu[r, st, PI[x, y]]
                        if he == he:
                            if x > y:
                                he = 1.0 - he
                            pend_eq[x, y] = he
                        if st > 0:
                            p = PI[x, y]
                            out[base + p, _col(0, x, Aof[p]) + D_AGGPOST] += 1
                if st == 0:
                    pa = pa_tab[cls[x], min(nr, 3), pos[x], min(ncall, 2)]
                    for y in range(6):
                        if y == x:
                            continue
                        p = PI[x, y]
                        c = _col(0, x, Aof[p])
                        o = base + p
                        out[o, c + D_AGGPRE] += 1
                        if pa < out[o, c + D_MINPA]:
                            out[o, c + D_MINPA] = pa
                        if pa < 0.02:
                            out[o, c + D_TRASH] += 1
                        if pa < 0.10:
                            out[o, c + D_WEAK10] += 1
                        if fpa_done[p] == 0:
                            fpa_done[p] = 1
                            out[o, S_FPA_ROLE] = 1 if x == Aof[p] else 2
                            out[o, S_FPA_ORD] = (pos[x] - 3 + 6) % 6
                            out[o, S_FPA_P] = pa
                            out[o, S_FPA_PARTNER_BEHIND] = 1 - acted_pre[y]
                        if last_pair_aggr[p] == y and third_since[p] > 0:
                            out[o, S_SANDWICH] = 1
                        last_pair_aggr[p] = x
                        third_since[p] = 0
                for p in range(15):
                    if hu_street_on[p] == 1 and (Aof[p] == x or Bof[p] == x):
                        hu_aggr_seen[p] = 1
            # ---- third-party fold facing a pair member's aggression
            if act == 0 and L >= 0 and L != x:
                for y in range(6):
                    if y == x or y == L:
                        continue
                    p = PI[L, y]
                    o = base + p
                    out[o, S_TF_FACING] += 1
                    if st == 0:
                        out[o, S_TF_PRE] += 1
                    if active[y] == 1:
                        out[o, S_TF_BOTH_ACTIVE] += 1
            if act == 0:
                active[x] = 0
                if fold_pos[x] > k_pos:
                    fold_pos[x] = k_pos
            if st == 0:
                acted_pre[x] = 1
                if iscall:
                    ncall += 1
            if aggr:
                last_aggr = x
                nr += 1
            k_pos += 1
        for p in range(15):
            if hu_street_on[p] == 1 and hu_aggr_seen[p] == 0:
                out[base + p, S_HU_CHECKED_STREETS] += 1
        # ---- hand-level fills
        nb = h_nboard[r]
        for i in range(6):
            for j in range(i + 1, 6):
                p = PI[i, j]
                o = base + p
                A = Aof[p]
                B = Bof[p]
                out[o, S_POT_BB] = h_pot[r] / bb
                out[o, S_BOARD_N] = nb
                out[o, S_PLAYERS_VOL] = n_vol
                out[o, S_PRE_ACTIONS] = n_pre
                out[o, S_THIRD_VOL] = n_vol - vol[i] - vol[j]
                out[o, S_SD_BOTH] = 1 if (s_sd[so + i] and s_sd[so + j]) else 0
                nleft = 0
                for s in range(6):
                    nleft += active[s]
                out[o, S_HU_FINAL] = 1 if (nleft == 2 and active[i] == 1 and active[j] == 1) else 0
                for X, Y in ((A, B), (B, A)):
                    c = NS if X == A else NS + ND
                    out[o, c + D_NET_BB] = s_net[so + X] / bb
                    out[o, c + D_VOL] = vol[X]
                    out[o, c + D_ENTER] = 1 if (first_act[X] >= 0 and first_act[X] != 0) else 0
                    out[o, c + D_FIRST_FOLD_NONFACING] = 1 if (first_act[X] == 0 and first_fold_facing[X] != Y) else 0
                    out[o, c + D_FOLD_AFTER_AGGR] = 1 if fold_pos[X] > first_aggr_pos[Y] and fold_pos[X] < (1 << 30) else 0
    return out, seatA, seatB


@njit(cache=True, inline="always")
def act_class(act, amt, tc):
    """4-class policy index: 0 fold, 1 check, 2 call, 3 aggressive."""
    if act == 0:
        return 0
    if act == 1:
        return 1
    if act == 2 or (act == 5 and amt <= tc):
        return 2
    return 3


# ------------------------------------------------------------------------------------------- driver
def pop_tables(min_n=150):
    b = pl.read_parquet(BASE / "research" / "forensics" / "ci_baseline_pre.parquet")
    b2 = pl.read_parquet(BASE / "research" / "forensics" / "ci_baseline_pre_pos.parquet")
    tabs = {}
    for col in ("p_aggr", "p_fold", "p_call"):
        t = np.full((169, 4, 6, 3), 0.5, dtype=np.float64)
        for r in b.iter_rows(named=True):
            t[r["class_id"], r["nr"], :, :] = r[col]
        for r in b2.filter(pl.col("n") >= min_n).iter_rows(named=True):
            t[r["class_id"], r["nr"], r["pos"], r["nc"]] = r[col]
        tabs[col] = t
    return tabs["p_aggr"], tabs["p_fold"], tabs["p_call"]


def run_phase(phase, chunk=100_000):
    t0 = time.time()
    tag = "dev" if phase == 0 else "eval"
    hands = pl.read_parquet(DER / "hands.parquet", columns=["hand_idx", "table_idx", "hand_seq", "phase", "button_seat",
                                                           "bb", "final_pot", "board"]).filter(pl.col("phase") == phase)
    limit = int(os.environ.get("ENGINE_LIMIT", "0"))
    if limit:
        hands = hands.head(limit)
    hids = hands["hand_idx"].to_numpy()
    assert np.all(np.diff(hids) > 0)
    lo, hi = int(hids[0]), int(hids[-1]) + 1
    # phase hands are contiguous per table but interleaved in hand_idx; work on the phase subset with row index r
    hrow = np.full(hi, -1, dtype=np.int64)
    hrow[hids] = np.arange(len(hids))
    seats = (pl.scan_parquet(DER / "seats.parquet").filter(pl.col("hand_idx").is_in(hids))
             .select("hand_idx", "seat_no", "player_idx", "c1", "c2", "net_chips", "went_to_showdown", "folded")
             .collect().sort("hand_idx", "seat_no"))
    assert seats.height == 6 * len(hids)
    acts = (pl.scan_parquet(DER / "actions.parquet").filter(pl.col("hand_idx").is_in(hids)).collect()
            .sort("hand_idx", "action_no"))
    eq = pl.scan_parquet(DER / f"action_equity_{tag}.parquet").filter(pl.col("hand_idx").is_in(hids)).collect()
    assert eq.height == acts.height and (eq["hand_idx"].to_numpy() == acts["hand_idx"].to_numpy()).all() \
        and (eq["action_no"].to_numpy() == acts["action_no"].to_numpy()).all()
    hu_df = pl.scan_parquet(DER / f"hu_equity_{tag}.parquet").filter(pl.col("hand_idx").is_in(hids)).collect()
    assert (hu_df["hand_idx"].to_numpy() == hids).all()
    print(f"[{tag}] loaded {len(hids):,} hands, {acts.height:,} actions in {time.time() - t0:.1f}s", flush=True)

    ah = acts["hand_idx"].to_numpy()
    a_start = np.searchsorted(ah, hids, side="left").astype(np.int64)
    a_end = np.searchsorted(ah, hids, side="right").astype(np.int64)
    eq_pre = eq.select([f"eq_pre_s{s}" for s in range(6)]).to_numpy().astype(np.float32)
    eq_post = eq.select([f"eq_post_s{s}" for s in range(6)]).to_numpy().astype(np.float32)
    hs_act = eq["hs_actor"].to_numpy().astype(np.float32)
    del eq
    hu = np.full((len(hids), 4, 15), np.nan, dtype=np.float32)
    for st in range(4):
        for i in range(6):
            for j in range(i + 1, 6):
                hu[:, st, PI[i, j]] = hu_df[f"hu_s{st}_{i}{j}"].to_numpy()
    del hu_df
    pol_path = DER / "action_policy.parquet"
    if pol_path.exists():
        pol_df = (pl.scan_parquet(pol_path).filter(pl.col("hand_idx").is_in(hids))
                  .select("hand_idx", "action_no", "p_fold", "p_check", "p_call", "p_agg").collect()
                  .sort("hand_idx", "action_no"))
        assert pol_df.height == acts.height
        pol = pol_df.select("p_fold", "p_check", "p_call", "p_agg").to_numpy().astype(np.float32)
        has_pol = True
        del pol_df
    else:
        pol = np.zeros((1, 4), dtype=np.float32)
        has_pol = False
    pfe = pl.read_parquet(DER / "preflop_equity_169.parquet").sort("class_id")
    pf_eq1 = pfe["equity_vs1"].to_numpy().astype(np.float64)
    assert len(pf_eq1) == 169
    pa_tab, pf_tab, pc_tab = pop_tables()
    board_n = hands["board"].list.len().to_numpy().astype(np.int64)

    arrs = dict(
        s_player=seats["player_idx"].to_numpy().astype(np.int64), s_c1=seats["c1"].to_numpy().astype(np.int64),
        s_c2=seats["c2"].to_numpy().astype(np.int64), s_net=seats["net_chips"].to_numpy().astype(np.float64),
        s_sd=seats["went_to_showdown"].to_numpy(), s_folded=seats["folded"].to_numpy(),
        h_button=hands["button_seat"].to_numpy().astype(np.int64), h_bb=hands["bb"].to_numpy().astype(np.float64),
        h_pot=hands["final_pot"].to_numpy().astype(np.float64), h_nboard=board_n,
        a_start=a_start, a_end=a_end, a_street=acts["street"].to_numpy().astype(np.int64),
        a_player=acts["player_idx"].to_numpy().astype(np.int64), a_action=acts["action"].to_numpy().astype(np.int64),
        a_amount=acts["amount"].to_numpy().astype(np.float64), a_to_call=acts["to_call"].to_numpy().astype(np.float64),
        a_pot=acts["pot_before"].to_numpy().astype(np.float64), a_pactive=acts["players_active"].to_numpy().astype(np.int64))
    del acts, seats
    outdir = OUT / (f"phase{phase}" if not limit else f"test{phase}")
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.glob("part_*.parquet"):
        f.unlink()
    table = hands["table_idx"].to_numpy()
    seq = hands["hand_seq"].to_numpy()
    for ci, h0 in enumerate(range(0, len(hids), chunk)):
        h1 = min(h0 + chunk, len(hids))
        t1 = time.time()
        X, sA, sB = run_chunk(h0, h1, arrs["s_player"], arrs["s_c1"], arrs["s_c2"], arrs["s_net"], arrs["s_sd"],
                              arrs["s_folded"], arrs["h_button"], arrs["h_bb"], arrs["h_pot"], arrs["h_nboard"],
                              arrs["a_start"], arrs["a_end"], arrs["a_street"], arrs["a_player"], arrs["a_action"],
                              arrs["a_amount"], arrs["a_to_call"], arrs["a_pot"], arrs["a_pactive"],
                              eq_pre, eq_post, hs_act, hu, pol, has_pol, pf_eq1, pa_tab, pf_tab, pc_tab, PI, INIT)
        rows = np.repeat(np.arange(h0, h1), 15)
        splayer = arrs["s_player"]
        pA = splayer[rows * 6 + sA.astype(np.int64)]
        pB = splayer[rows * 6 + sB.astype(np.int64)]
        df = pl.DataFrame({"hand_idx": hids[rows].astype(np.int32), "table_idx": table[rows].astype(np.int16),
                           "hand_seq": seq[rows].astype(np.int16), "pA": pA.astype(np.int32), "pB": pB.astype(np.int32),
                           "seatA": sA, "seatB": sB})
        df = df.hstack(pl.DataFrame({c: X[:, i] for i, c in enumerate(COLS)}))
        df.write_parquet(outdir / f"part_{ci:02d}.parquet", compression="zstd")
        print(f"[{tag}] chunk {ci} hands {h0}-{h1} rows {df.height:,} in {time.time() - t1:.1f}s", flush=True)
    print(f"[{tag}] done in {time.time() - t0:.1f}s (policy={'yes' if has_pol else 'no'})", flush=True)


if __name__ == "__main__":
    phases = [int(a) for a in sys.argv[1:]] or [0, 1]
    for ph in phases:
        run_phase(ph)
