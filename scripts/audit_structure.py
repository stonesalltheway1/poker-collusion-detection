"""Structure / labels / hard-negative / validation-design audit.

Stages (re-runnable, memory-light: numpy + one numba pass over all hands, < ~2.5 GB peak):
  python scripts/audit_structure.py build     # -> data/derived/player_stats_{dev,eval}.parquet
                                              #    data/derived/pair_generic_{dev,eval}.parquet
                                              #    + *_dev_w2000 (dev hand_seq 0..1999, eval-length window)
  python scripts/audit_structure.py analyze   # -> research/forensics/structure_results.json (+ stdout)
  python scripts/audit_structure.py extra     # random-placement nulls, eval-phase persistence, PU mixture
                                              #    estimate, matched-window CV, cross-phase overlap (appends JSON)
Notes: research/forensics/structure.md

Everything is derived from poker activity (hands/seats/actions) + host-approved player metadata.
IDs are used only as join keys; player "local index" inside a table is the rank of player_idx
(dense int from sorted player_id) and is only an array coordinate, never a feature.

Conventions in pair tables: p1 < p2 (player_idx), same orientation as labels/eval_pairs.
Suffix _12 = p1 acting toward / losing to p2; _21 the reverse. All chip values in big blinds.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "3")
import numpy as np
import polars as pl
from numba import njit

BASE = Path(__file__).resolve().parent.parent
DER = BASE / "data" / "derived"
OUT = BASE / "research" / "forensics"
FOLDS_CSV = BASE / "data" / "folds_tables_5.csv"
OUT.mkdir(parents=True, exist_ok=True)

N_TABLES, POOL = 400, 30
PHASE_HANDS = (3000, 2000)
HALF_SPLIT = (1500, 4000)  # hand_seq split for first/second half of each phase
TILT_LOSS_BB = 30.0        # a hand losing >= this many bb opens a tilt window
TILT_WINDOW = 10           # next own hands observed after a big loss
JOIN_WIN = 2               # co-join / co-leave tolerance in hands

# ---------------------------------------------------------------- directed pair feature ids
DF = ["shared", "both_vpip", "both_flop", "sd_tog", "hu_final", "hu_flop", "checkdown_hu",
      "both_aggr", "flow", "flow_hu", "flow_sd", "off1", "off2", "off3", "acts_v", "aggr_v",
      "acts_v_post", "aggr_v_post", "acts_hu", "aggr_hu", "faced", "fold_faced", "reraise_faced",
      "faced_post", "fold_faced_post", "sd_win", "cojoin", "coleave", "faced_pre_strong",
      "fold_faced_pre_strong"]
D = {k: i for i, k in enumerate(DF)}
PF = ["hands", "vpip", "pfr", "tb_opp", "tb", "limp", "pre_acts", "post_aggr", "post_calls",
      "post_checks", "post_folds", "post_faced", "post_fold_faced", "saw_flop", "wsd", "wsd_won",
      "net_bb", "stack_bb", "allin", "weak_dealt", "weak_vpip", "strong_dealt", "strong_foldpre",
      "vpip_pct_sum", "h1_hands", "h1_vpip", "h1_pfr", "h1_net", "h2_hands", "h2_vpip", "h2_pfr",
      "h2_net", "tilt_hands", "tilt_vpip", "tilt_pfr", "tilt_net", "bigloss_events", "maxdd_bb",
      "maxup_bb", "joins", "tot_acts", "tot_aggr", "h1_aggr", "h1_acts", "h2_aggr", "h2_acts",
      "won_hands", "fold_pre", "pre_aggr"]
P = {k: i for i, k in enumerate(PF)}


def chen_pct_table():
    """52x52 preflop strength percentile (Chen formula, ties averaged over the 1326 combos)."""
    hi = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10]
    sc = np.zeros((52, 52))
    for a in range(52):
        for b in range(52):
            if a == b:
                continue
            ra, rb = a // 4, b // 4
            h, l = max(ra, rb), min(ra, rb)
            if ra == rb:
                s = max(hi[h] * 2, 5)
            else:
                s = hi[h]
                if a % 4 == b % 4:
                    s += 2
                gap = h - l - 1
                s -= [0, 1, 2, 4][gap] if gap < 4 else 5
                if gap <= 1 and h < 10:
                    s += 1
            sc[a, b] = np.ceil(s)
    iu = np.triu_indices(52, 1)
    v = sc[iu]
    from scipy.stats import rankdata
    pct = (rankdata(v) - 0.5) / len(v)
    out = np.zeros((52, 52))
    out[iu] = pct
    out = out + out.T
    return out


@njit(cache=False)
def kernel(h_table, h_seq, h_phase, h_bb, h_nboard, h_pot,
           s_player, s_seat, s_stack, s_c1, s_c2, s_net, s_wsd, s_won,
           a_off, a_street, a_player, a_action, a_amount, a_tocall, a_pactive,
           loc, pf_pct, Dm, Pm, half_split, tilt_loss, tilt_window, join_win):
    N = h_table.shape[0]
    n_players = loc.shape[0]
    BIG = 1 << 30
    cum = np.zeros((2, n_players))
    peak = np.zeros((2, n_players))
    trough = np.zeros((2, n_players))
    tilt_left = np.zeros((2, n_players), np.int64)
    last_join = np.full(n_players, -1000, np.int64)
    sess_start = np.zeros(n_players, np.int64)
    rl_player = np.full(64, -1, np.int64)
    rl_seq = np.full(64, -1000, np.int64)
    rl_ptr = 0
    pl = np.zeros(6, np.int64)
    lc = np.zeros(6, np.int64)
    fold_no = np.zeros(6, np.int64)
    fold_st = np.zeros(6, np.int64)
    vp = np.zeros(6, np.int64)
    pr = np.zeros(6, np.int64)
    agg_any = np.zeros(6, np.int64)
    saw = np.zeros(6, np.int64)
    joined = np.zeros(6, np.int64)
    for h in range(N):
        t = h_table[h]
        ph = h_phase[h]
        bb = float(h_bb[h])
        nb = h_nboard[h]
        seq = h_seq[h]
        lo = a_off[h]
        hi = a_off[h + 1]
        for k in range(6):
            pl[k] = s_player[h, k]
            lc[k] = loc[pl[k]]
            fold_no[k] = BIG
            fold_st[k] = 9
            vp[k] = 0
            pr[k] = 0
            agg_any[k] = 0
        # ---- pass 1: folds
        for r in range(lo, hi):
            i = -1
            for k in range(6):
                if pl[k] == a_player[r]:
                    i = k
                    break
            if a_action[r] == 0:
                fold_no[i] = r - lo
                fold_st[i] = a_street[r]
        # ---- pass 2: actions
        street_cur = -1
        last_agg = -1
        nraise_pre = 0
        post_agg_in_hand = 0
        for r in range(lo, hi):
            kk = r - lo
            st = a_street[r]
            if st != street_cur:
                street_cur = st
                last_agg = -1
            i = -1
            for k in range(6):
                if pl[k] == a_player[r]:
                    i = k
                    break
            p = pl[i]
            act = a_action[r]
            amt = a_amount[r]
            tc = a_tocall[r]
            pa = a_pactive[r]
            isagg = (act == 3) or (act == 4) or (act == 5 and amt > tc)
            iscall = (act == 2) or (act == 5 and amt <= tc)
            Pm[P_TOT_ACTS, ph, p] += 1
            if isagg:
                Pm[P_TOT_AGGR, ph, p] += 1
                agg_any[i] = 1
            if act == 5:
                Pm[P_ALLIN, ph, p] += 1
            if seq < half_split[ph]:
                Pm[P_H1_ACTS, ph, p] += 1
                if isagg:
                    Pm[P_H1_AGGR, ph, p] += 1
            else:
                Pm[P_H2_ACTS, ph, p] += 1
                if isagg:
                    Pm[P_H2_AGGR, ph, p] += 1
            if st == 0:
                Pm[P_PRE_ACTS, ph, p] += 1
                if act >= 2:
                    vp[i] = 1
                if isagg:
                    pr[i] = 1
                    Pm[P_PRE_AGGR, ph, p] += 1
                if nraise_pre == 1:
                    Pm[P_TB_OPP, ph, p] += 1
                    if isagg:
                        Pm[P_TB, ph, p] += 1
                if nraise_pre == 0 and iscall:
                    Pm[P_LIMP, ph, p] += 1
                if isagg:
                    nraise_pre += 1
            else:
                if isagg:
                    Pm[P_POST_AGGR, ph, p] += 1
                    post_agg_in_hand += 1
                elif iscall:
                    Pm[P_POST_CALLS, ph, p] += 1
                elif act == 1:
                    Pm[P_POST_CHECKS, ph, p] += 1
                elif act == 0:
                    Pm[P_POST_FOLDS, ph, p] += 1
                if tc > 0:
                    Pm[P_POST_FACED, ph, p] += 1
                    if act == 0:
                        Pm[P_POST_FOLD_FACED, ph, p] += 1
            for j in range(6):
                if j == i:
                    continue
                if fold_no[j] > kk:
                    a = lc[i]
                    b = lc[j]
                    Dm[D_ACTS_V, ph, t, a, b] += 1
                    if isagg:
                        Dm[D_AGGR_V, ph, t, a, b] += 1
                    if st > 0:
                        Dm[D_ACTS_V_POST, ph, t, a, b] += 1
                        if isagg:
                            Dm[D_AGGR_V_POST, ph, t, a, b] += 1
                    if pa == 2:
                        Dm[D_ACTS_HU, ph, t, a, b] += 1
                        if isagg:
                            Dm[D_AGGR_HU, ph, t, a, b] += 1
            if tc > 0 and last_agg >= 0 and last_agg != i:
                a = lc[i]
                b = lc[last_agg]
                Dm[D_FACED, ph, t, a, b] += 1
                if act == 0:
                    Dm[D_FOLD_FACED, ph, t, a, b] += 1
                if isagg:
                    Dm[D_RERAISE_FACED, ph, t, a, b] += 1
                if st > 0:
                    Dm[D_FACED_POST, ph, t, a, b] += 1
                    if act == 0:
                        Dm[D_FOLD_FACED_POST, ph, t, a, b] += 1
                elif pf_pct[s_c1[h, i], s_c2[h, i]] >= 0.85:
                    Dm[D_FACED_PRE_STRONG, ph, t, a, b] += 1
                    if act == 0:
                        Dm[D_FOLD_FACED_PRE_STRONG, ph, t, a, b] += 1
            if isagg:
                last_agg = i
        # ---- end-of-hand structure
        nf = 0
        for k in range(6):
            if fold_no[k] == BIG:
                nf += 1
            saw[k] = 1 if (nb >= 3 and fold_st[k] >= 1) else 0
        hu_a = -1
        hu_b = -1
        if nf == 2:
            for k in range(6):
                if fold_no[k] == BIG:
                    if hu_a < 0:
                        hu_a = k
                    else:
                        hu_b = k
        elif nf == 1:
            lastf = -1
            lastk = -1
            for k in range(6):
                if fold_no[k] == BIG:
                    hu_a = k
                elif fold_no[k] > lastf:
                    lastf = fold_no[k]
                    lastk = k
            hu_b = lastk
        totwin = 0.0
        for k in range(6):
            if s_net[h, k] > 0:
                totwin += s_net[h, k]
        for i in range(6):
            a = lc[i]
            for j in range(6):
                if i == j:
                    continue
                b = lc[j]
                Dm[D_SHARED, ph, t, a, b] += 1
                if vp[i] == 1 and vp[j] == 1:
                    Dm[D_BOTH_VPIP, ph, t, a, b] += 1
                if saw[i] == 1 and saw[j] == 1:
                    Dm[D_BOTH_FLOP, ph, t, a, b] += 1
                bothsd = s_wsd[h, i] and s_wsd[h, j]
                if bothsd:
                    Dm[D_SD_TOG, ph, t, a, b] += 1
                    if s_won[h, i] > s_won[h, j]:
                        Dm[D_SD_WIN, ph, t, a, b] += 1
                ishu = (i == hu_a and j == hu_b) or (i == hu_b and j == hu_a)
                if ishu:
                    Dm[D_HU_FINAL, ph, t, a, b] += 1
                    if saw[i] == 1 and saw[j] == 1:
                        Dm[D_HU_FLOP, ph, t, a, b] += 1
                        if post_agg_in_hand == 0:
                            Dm[D_CHECKDOWN_HU, ph, t, a, b] += 1
                if agg_any[i] == 1 and agg_any[j] == 1:
                    Dm[D_BOTH_AGGR, ph, t, a, b] += 1
                if s_net[h, i] < 0 and s_net[h, j] > 0 and totwin > 0:
                    fl = (-s_net[h, i]) * s_net[h, j] / totwin / bb
                    Dm[D_FLOW, ph, t, a, b] += fl
                    if ishu:
                        Dm[D_FLOW_HU, ph, t, a, b] += fl
                    if bothsd:
                        Dm[D_FLOW_SD, ph, t, a, b] += fl
                off = (s_seat[h, j] - s_seat[h, i]) % 6
                if off == 1:
                    Dm[D_OFF1, ph, t, a, b] += 1
                elif off == 2:
                    Dm[D_OFF2, ph, t, a, b] += 1
                elif off == 3:
                    Dm[D_OFF3, ph, t, a, b] += 1
        # ---- player per-hand stats
        for k in range(6):
            p = pl[k]
            nbb = s_net[h, k] / bb
            Pm[P_HANDS, ph, p] += 1
            Pm[P_VPIP, ph, p] += vp[k]
            Pm[P_PFR, ph, p] += pr[k]
            Pm[P_SAW_FLOP, ph, p] += saw[k]
            if fold_st[k] == 0:
                Pm[P_FOLD_PRE, ph, p] += 1
            if s_wsd[h, k]:
                Pm[P_WSD, ph, p] += 1
                if s_won[h, k] > 0:
                    Pm[P_WSD_WON, ph, p] += 1
            Pm[P_NET_BB, ph, p] += nbb
            if nbb > 0:
                Pm[P_WON_HANDS, ph, p] += 1
            Pm[P_STACK_BB, ph, p] += s_stack[h, k] / bb
            pct = pf_pct[s_c1[h, k], s_c2[h, k]]
            if vp[k] == 1:
                Pm[P_VPIP_PCT_SUM, ph, p] += pct
            if pct < 0.4:
                Pm[P_WEAK_DEALT, ph, p] += 1
                Pm[P_WEAK_VPIP, ph, p] += vp[k]
            if pct >= 0.92:
                Pm[P_STRONG_DEALT, ph, p] += 1
                if fold_st[k] == 0:
                    Pm[P_STRONG_FOLDPRE, ph, p] += 1
            if seq < half_split[ph]:
                Pm[P_H1_HANDS, ph, p] += 1
                Pm[P_H1_VPIP, ph, p] += vp[k]
                Pm[P_H1_PFR, ph, p] += pr[k]
                Pm[P_H1_NET, ph, p] += nbb
            else:
                Pm[P_H2_HANDS, ph, p] += 1
                Pm[P_H2_VPIP, ph, p] += vp[k]
                Pm[P_H2_PFR, ph, p] += pr[k]
                Pm[P_H2_NET, ph, p] += nbb
            if tilt_left[ph, p] > 0:
                Pm[P_TILT_HANDS, ph, p] += 1
                Pm[P_TILT_VPIP, ph, p] += vp[k]
                Pm[P_TILT_PFR, ph, p] += pr[k]
                Pm[P_TILT_NET, ph, p] += nbb
                tilt_left[ph, p] -= 1
            if nbb <= -tilt_loss:
                tilt_left[ph, p] = tilt_window
                Pm[P_BIGLOSS_EVENTS, ph, p] += 1
            cum[ph, p] += nbb
            if cum[ph, p] > peak[ph, p]:
                peak[ph, p] = cum[ph, p]
            if cum[ph, p] < trough[ph, p]:
                trough[ph, p] = cum[ph, p]
            if peak[ph, p] - cum[ph, p] > Pm[P_MAXDD_BB, ph, p]:
                Pm[P_MAXDD_BB, ph, p] = peak[ph, p] - cum[ph, p]
            if cum[ph, p] - trough[ph, p] > Pm[P_MAXUP_BB, ph, p]:
                Pm[P_MAXUP_BB, ph, p] = cum[ph, p] - trough[ph, p]
        # ---- sessions: joins / leaves (lineup changes between consecutive table hands)
        has_prev = h > 0 and h_table[h - 1] == t
        if not has_prev:
            rl_player[:] = -1
            rl_seq[:] = -1000
            for k in range(6):
                sess_start[pl[k]] = seq
                last_join[pl[k]] = -1000
            continue
        for k in range(6):
            joined[k] = 1
            for q in range(6):
                if s_player[h - 1, q] == pl[k]:
                    joined[k] = 0
                    break
            if joined[k] == 1:
                Pm[P_JOINS, ph, pl[k]] += 1
                sess_start[pl[k]] = seq
                last_join[pl[k]] = seq
        for i in range(6):
            for j in range(i + 1, 6):
                if (joined[i] == 1 or joined[j] == 1) and last_join[pl[i]] >= seq - join_win \
                        and last_join[pl[j]] >= seq - join_win:
                    Dm[D_COJOIN, ph, t, lc[i], lc[j]] += 1
                    Dm[D_COJOIN, ph, t, lc[j], lc[i]] += 1
        for q in range(6):
            pq = s_player[h - 1, q]
            left = True
            for k in range(6):
                if pl[k] == pq:
                    left = False
                    break
            if not left:
                continue
            for e in range(64):
                if rl_player[e] >= 0 and rl_player[e] != pq and rl_seq[e] >= seq - join_win \
                        and sess_start[pq] <= rl_seq[e] - 1 and sess_start[rl_player[e]] <= seq - 1:
                    Dm[D_COLEAVE, ph, t, loc[pq], loc[rl_player[e]]] += 1
                    Dm[D_COLEAVE, ph, t, loc[rl_player[e]], loc[pq]] += 1
            rl_player[rl_ptr] = pq
            rl_seq[rl_ptr] = seq
            rl_ptr = (rl_ptr + 1) % 64


def _inject_constants():
    g = globals()
    for k, i in D.items():
        g["D_" + k.upper()] = i
    for k, i in P.items():
        g["P_" + k.upper()] = i


_inject_constants()


def _ratio(num, den):
    return pl.when(den > 0).then(num / den).otherwise(None)


# ============================================================================ build
def stage_build(window=False):
    """window=True: phase 0 := dev hands with hand_seq < 2000 (eval-length matched window) -> *_dev_w2000.parquet"""
    t0 = time.time()
    hands = pl.read_parquet(DER / "hands.parquet",
                            columns=["hand_idx", "table_idx", "hand_seq", "phase", "bb", "board", "final_pot"])
    assert (hands["hand_idx"].to_numpy() == np.arange(hands.height)).all()
    h_table = hands["table_idx"].to_numpy().astype(np.int64)
    h_seq = hands["hand_seq"].to_numpy().astype(np.int64)
    h_phase = hands["phase"].to_numpy().astype(np.int64)
    if window:
        h_phase = np.where((h_phase == 0) & (h_seq < 2000), 0, 1).astype(np.int64)
    phases = ((0, "dev_w2000"),) if window else ((0, "dev"), (1, "eval"))
    ph_hands = (2000, 3000) if window else PHASE_HANDS
    half_split = (1000, 4000) if window else HALF_SPLIT
    h_bb = hands["bb"].to_numpy().astype(np.int64)
    h_nboard = hands["board"].list.len().to_numpy().astype(np.int64)
    h_pot = hands["final_pot"].to_numpy().astype(np.int64)
    N = hands.height
    del hands
    seats = pl.read_parquet(DER / "seats.parquet",
                            columns=["hand_idx", "player_idx", "seat_no", "starting_stack", "c1", "c2",
                                     "net_chips", "went_to_showdown", "won_share"])
    assert seats.height == 6 * N
    sh = lambda c, dt: seats[c].to_numpy().astype(dt).reshape(N, 6)
    s_player, s_seat, s_stack = sh("player_idx", np.int64), sh("seat_no", np.int64), sh("starting_stack", np.float64)
    s_c1, s_c2, s_net = sh("c1", np.int64), sh("c2", np.int64), sh("net_chips", np.float64)
    s_wsd, s_won = sh("went_to_showdown", np.bool_), sh("won_share", np.float64)
    del seats
    # player -> table, table-local index (rank of player_idx within table)
    n_players = 12000
    ptable = np.full(n_players, -1, np.int64)
    ptable[s_player.ravel()] = np.repeat(h_table, 6)
    assert (ptable >= 0).all()
    loc = np.zeros(n_players, np.int64)
    tab_players = np.zeros((N_TABLES, POOL), np.int64)
    for t in range(N_TABLES):
        ps = np.sort(np.flatnonzero(ptable == t))
        assert len(ps) == POOL, (t, len(ps))
        tab_players[t] = ps
        loc[ps] = np.arange(POOL)
    acts = pl.read_parquet(DER / "actions.parquet",
                           columns=["hand_idx", "street", "player_idx", "action", "amount", "to_call",
                                    "players_active"])
    a_hand = acts["hand_idx"].to_numpy()
    a_off = np.searchsorted(a_hand, np.arange(N + 1)).astype(np.int64)
    a_street = acts["street"].to_numpy().astype(np.int64)
    a_player = acts["player_idx"].to_numpy().astype(np.int64)
    a_action = acts["action"].to_numpy().astype(np.int64)
    a_amount = acts["amount"].to_numpy().astype(np.int64)
    a_tocall = acts["to_call"].to_numpy().astype(np.int64)
    a_pactive = acts["players_active"].to_numpy().astype(np.int64)
    del acts, a_hand
    pf_pct = chen_pct_table()
    Dm = np.zeros((len(DF), 2, N_TABLES, POOL, POOL), np.float64)
    Pm = np.zeros((len(PF), 2, n_players), np.float64)
    print(f"loaded in {time.time() - t0:.1f}s; running kernel", flush=True)
    kernel(h_table, h_seq, h_phase, h_bb, h_nboard, h_pot, s_player, s_seat, s_stack, s_c1, s_c2, s_net, s_wsd,
           s_won, a_off, a_street, a_player, a_action, a_amount, a_tocall, a_pactive, loc, pf_pct, Dm, Pm,
           np.array(half_split, np.int64), TILT_LOSS_BB, TILT_WINDOW, JOIN_WIN)
    print(f"kernel done {time.time() - t0:.1f}s", flush=True)
    np.save(OUT / "_audit_tab_players.npy", tab_players)

    # ---- player stats
    for ph, name in phases:
        cols = {k: Pm[i, ph] for k, i in P.items()}
        df = pl.DataFrame({"player_idx": np.arange(n_players, dtype=np.int32),
                           "table_idx": ptable.astype(np.int16), "phase": np.full(n_players, ph, np.int8),
                           **{k: v for k, v in cols.items()}})
        c = pl.col
        df = df.with_columns(
            (c("hands") / ph_hands[ph]).alias("seat_share"),
            _ratio(c("vpip"), c("hands")).alias("vpip_rate"),
            _ratio(c("pfr"), c("hands")).alias("pfr_rate"),
            _ratio(c("tb"), c("tb_opp")).alias("threebet_rate"),
            _ratio(c("limp"), c("hands")).alias("limp_rate"),
            _ratio(c("post_aggr"), c("post_calls")).alias("af_post"),
            _ratio(c("post_aggr"), c("post_aggr") + c("post_calls") + c("post_checks") + c("post_folds")).alias("afq_post"),
            _ratio(c("tot_aggr"), c("tot_acts")).alias("aggr_rate"),
            _ratio(c("wsd"), c("saw_flop")).alias("wtsd"),
            _ratio(c("wsd_won"), c("wsd")).alias("wsd_won_rate"),
            _ratio(c("saw_flop"), c("hands")).alias("saw_flop_rate"),
            _ratio(c("wsd"), c("hands")).alias("sd_freq"),
            _ratio(c("net_bb") * 100, c("hands")).alias("bb100"),
            _ratio(c("post_fold_faced"), c("post_faced")).alias("fold_to_bet_post"),
            _ratio(c("stack_bb"), c("hands")).alias("avg_stack_bb"),
            _ratio(c("allin"), c("hands")).alias("allin_rate"),
            _ratio(c("weak_vpip"), c("weak_dealt")).alias("weak_vpip_rate"),
            _ratio(c("strong_foldpre"), c("strong_dealt")).alias("strong_foldpre_rate"),
            _ratio(c("vpip_pct_sum"), c("vpip")).alias("vpip_avg_strength"),
            _ratio(c("won_hands"), c("hands")).alias("won_hand_rate"),
            (_ratio(c("h2_vpip"), c("h2_hands")) - _ratio(c("h1_vpip"), c("h1_hands"))).alias("drift_vpip"),
            (_ratio(c("h2_pfr"), c("h2_hands")) - _ratio(c("h1_pfr"), c("h1_hands"))).alias("drift_pfr"),
            (_ratio(c("h2_aggr"), c("h2_acts")) - _ratio(c("h1_aggr"), c("h1_acts"))).alias("drift_aggr"),
            (_ratio(c("h2_net") * 100, c("h2_hands")) - _ratio(c("h1_net") * 100, c("h1_hands"))).alias("drift_bb100"),
            (_ratio(c("tilt_vpip"), c("tilt_hands")) - _ratio(c("vpip"), c("hands"))).alias("tilt_vpip_delta"),
            (_ratio(c("tilt_pfr"), c("tilt_hands")) - _ratio(c("pfr"), c("hands"))).alias("tilt_pfr_delta"),
            (_ratio(c("tilt_net") * 100, c("tilt_hands")) - _ratio(c("net_bb") * 100, c("hands"))).alias("tilt_bb100_delta"),
            _ratio(c("hands"), c("joins") + 1).alias("mean_session_len"),
        )
        df.write_parquet(DER / f"player_stats_{name}.parquet", compression="zstd")
        print(f"player_stats_{name}: {df.shape}", flush=True)

    # ---- pair generic
    iu, ju = np.triu_indices(POOL, 1)
    T = np.repeat(np.arange(N_TABLES), len(iu))
    A = np.tile(iu, N_TABLES)
    B = np.tile(ju, N_TABLES)
    p1 = tab_players[T, A]
    p2 = tab_players[T, B]
    sym = ["shared", "both_vpip", "both_flop", "sd_tog", "hu_final", "hu_flop", "checkdown_hu", "both_aggr",
           "off3", "cojoin", "coleave"]
    symname = {"shared": "shared_hands", "both_flop": "both_saw_flop", "sd_tog": "sd_together",
               "off3": "seat_opposite", "both_vpip": "both_vpip"}
    dirn = ["flow", "flow_hu", "flow_sd", "off1", "off2", "acts_v", "aggr_v", "acts_v_post", "aggr_v_post",
            "acts_hu", "aggr_hu", "faced", "fold_faced", "reraise_faced", "faced_post", "fold_faced_post",
            "sd_win", "faced_pre_strong", "fold_faced_pre_strong"]
    for ph, name in phases:
        cols = {"table_idx": T.astype(np.int16), "p1": p1.astype(np.int32), "p2": p2.astype(np.int32),
                "phase": np.full(len(T), ph, np.int8)}
        for f in sym:
            v = Dm[D[f], ph, T, A, B]
            cols[symname.get(f, f)] = v.astype(np.int32)
        for f in dirn:
            v12 = Dm[D[f], ph, T, A, B]
            v21 = Dm[D[f], ph, T, B, A]
            dt = np.float32 if f.startswith("flow") else np.int32
            cols[f + "_12"] = v12.astype(dt)
            cols[f + "_21"] = v21.astype(dt)
        df = pl.DataFrame(cols)
        ps = pl.read_parquet(DER / f"player_stats_{name}.parquet",
                             columns=["player_idx", "hands", "tot_acts", "tot_aggr", "joins"])
        c = pl.col
        df = (df.join(ps.rename({k: k + "_1" for k in ps.columns}).rename({"player_idx_1": "p1"}), on="p1")
                .join(ps.rename({k: k + "_2" for k in ps.columns}).rename({"player_idx_2": "p2"}), on="p2")
                .rename({"hands_1": "n1_hands", "hands_2": "n2_hands"})
                .with_columns(
                    _ratio(c("shared_hands") * ph_hands[ph], c("n1_hands") * c("n2_hands")).alias("copresence_lift"),
                    _ratio(c("shared_hands"), pl.min_horizontal("n1_hands", "n2_hands")).alias("copresence_min"),
                    (c("flow_12") - c("flow_21")).alias("net_flow_12_bb"),
                    (c("flow_hu_12") - c("flow_hu_21")).alias("net_flow_hu_12_bb"),
                    (c("flow_sd_12") - c("flow_sd_21")).alias("net_flow_sd_12_bb"),
                    _ratio(c("aggr_v_12"), c("acts_v_12")).alias("aggr_rate_1v2"),
                    _ratio(c("aggr_v_21"), c("acts_v_21")).alias("aggr_rate_2v1"),
                    _ratio(c("tot_aggr_1") - c("aggr_v_12"), c("tot_acts_1") - c("acts_v_12")).alias("aggr_rate_1_not2"),
                    _ratio(c("tot_aggr_2") - c("aggr_v_21"), c("tot_acts_2") - c("acts_v_21")).alias("aggr_rate_2_not1"),
                    _ratio(c("aggr_hu_12"), c("acts_hu_12")).alias("aggr_rate_hu_1v2"),
                    _ratio(c("aggr_hu_21"), c("acts_hu_21")).alias("aggr_rate_hu_2v1"),
                    _ratio(c("fold_faced_12"), c("faced_12")).alias("fold_rate_1_to_2"),
                    _ratio(c("fold_faced_21"), c("faced_21")).alias("fold_rate_2_to_1"),
                    _ratio(c("off1_12") + c("off1_21"), c("shared_hands")).alias("seat_adjacent_rate"),
                )
                .drop("tot_acts_1", "tot_aggr_1", "tot_acts_2", "tot_aggr_2")
                .rename({"flow_12": "flow_12_bb", "flow_21": "flow_21_bb", "flow_hu_12": "flow_hu_12_bb",
                         "flow_hu_21": "flow_hu_21_bb", "flow_sd_12": "flow_sd_12_bb", "flow_sd_21": "flow_sd_21_bb"}))
        df.write_parquet(DER / f"pair_generic_{name}.parquet", compression="zstd")
        print(f"pair_generic_{name}: {df.shape}  ({time.time() - t0:.1f}s)", flush=True)
    if window:
        return
    # sanity: eval shared hands vs host
    ep = pl.read_parquet(DER / "eval_pairs.parquet", columns=["p1", "p2", "shared_hands"])
    chk = ep.join(pl.read_parquet(DER / "pair_generic_eval.parquet", columns=["p1", "p2", "shared_hands"]),
                  on=["p1", "p2"], how="left", suffix="_calc")
    print("eval shared_hands mismatches:", int((chk["shared_hands"] != chk["shared_hands_calc"]).sum()),
          "missing:", chk["shared_hands_calc"].null_count())


# ============================================================================ analysis helpers
def auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ap(y, s):
    y = np.asarray(y, int)
    order = np.argsort(-np.asarray(s, float), kind="mergesort")
    r = y[order]
    return float(np.sum(np.cumsum(r) / np.arange(1, len(r) + 1) * r) / max(r.sum(), 1))


def q(x, qs=(0, .1, .25, .5, .75, .9, 1)):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return [round(float(v), 3) for v in np.quantile(x, qs)] if len(x) else None


R = {}


def rec(key, val, show=True):
    R[key] = val
    if show:
        print(f"{key}: {val}", flush=True)


# ============================================================================ analyze
def stage_analyze():
    c = pl.col
    lab = pl.read_parquet(DER / "labels.parquet")
    ev = pl.read_parquet(DER / "evidence.parquet")
    ep = pl.read_parquet(DER / "eval_pairs.parquet")
    players = pl.read_parquet(DER / "players.parquet")
    hands = pl.read_parquet(DER / "hands.parquet", columns=["hand_idx", "table_idx", "hand_seq", "ts", "phase"])
    gd = pl.read_parquet(DER / "pair_generic_dev.parquet")
    ge = pl.read_parquet(DER / "pair_generic_eval.parquet")
    psd = pl.read_parquet(DER / "player_stats_dev.parquet")
    pse = pl.read_parquet(DER / "player_stats_eval.parquet")
    folds = pl.read_csv(FOLDS_CSV).with_columns(c("table_idx").cast(pl.Int16), c("fold").cast(pl.Int8))
    ptab = psd.select("player_idx", "table_idx")

    # ------------------------------------------------------------------ 1. pair universe
    print("\n==== 1. PAIR UNIVERSE ====")
    rec("n_within_table_pairs", gd.height)
    rec("dev_pairs_shared_ge1", int((gd["shared_hands"] >= 1).sum()))
    rec("eval_pairs_shared_ge1_all_within_table", int((ge["shared_hands"] >= 1).sum()))
    rec("dev_shared_quantiles_all", q(gd["shared_hands"]))
    rec("eval_shared_quantiles_all", q(ge["shared_hands"]))
    lab = lab.join(ptab.rename({"player_idx": "p1"}), on="p1").join(folds, on="table_idx")
    pos = lab.filter(c("label") == 1)
    neg = lab.filter(c("label") == 0)
    rec("labels_same_table_check", int(lab.join(ptab.rename({"player_idx": "p2", "table_idx": "t2"}), on="p2")
                                       .filter(c("table_idx") != c("t2")).height))
    pos_players = set(pos["p1"].to_list()) | set(pos["p2"].to_list())
    neg_players = set(neg["p1"].to_list()) | set(neg["p2"].to_list())
    rec("n_pos_players", len(pos_players))
    rec("n_neg_players", len(neg_players))
    rec("n_players_pos_and_neg", len(pos_players & neg_players))
    lab_keys = lab.select("p1", "p2", "label", "behavior_family")
    U = (gd.select("table_idx", "p1", "p2", "shared_hands").rename({"shared_hands": "dev_shared"})
         .join(ge.select("p1", "p2", "shared_hands").rename({"shared_hands": "eval_shared"}), on=["p1", "p2"])
         .join(lab_keys, on=["p1", "p2"], how="left")
         .join(ep.select("p1", "p2", "shared_hands").rename({"shared_hands": "ep_shared"}).with_columns(pl.lit(True).alias("in_eval")),
               on=["p1", "p2"], how="left")
         .with_columns(c("in_eval").fill_null(False),
                       (c("p1").is_in(list(pos_players)) | c("p2").is_in(list(pos_players))).alias("has_pos_player"),
                       (c("p1").is_in(list(neg_players)) | c("p2").is_in(list(neg_players))).alias("has_neg_player"),
                       c("label").is_not_null().alias("is_labelled")))
    rec("eval_pairs_matched_in_universe", int(U["in_eval"].sum()))
    rec("eval_shared_mismatch", int(U.filter(c("in_eval") & (c("ep_shared") != c("eval_shared"))).height))
    rec("eval_pairs_that_are_labelled", int(U.filter(c("in_eval") & c("is_labelled")).height))
    rec("eval_pairs_with_pos_player", int(U.filter(c("in_eval") & c("has_pos_player")).height))
    rec("eval_pairs_with_neg_player", int(U.filter(c("in_eval") & c("has_neg_player")).height))
    rec("eval_pairs_with_neg_player_only_nonpos", int(U.filter(c("in_eval") & c("has_neg_player") & ~c("has_pos_player")).height))
    elig = U.filter(~c("is_labelled") & ~c("has_pos_player"))
    rec("eligible_pairs(not labelled, no pos player)", elig.height)
    rec("eligible_by_eval_shared_ge_k", {k: int((elig["eval_shared"] >= k).sum()) for k in (1, 20, 30, 37, 38, 39, 40, 50)})
    rec("eligible_not_in_eval_eval_shared_quantiles", q(elig.filter(~c("in_eval"))["eval_shared"]))
    rec("eligible_in_eval_min_eval_shared", int(elig.filter(c("in_eval"))["eval_shared"].min()))
    rec("in_eval_but_ineligible", int(U.filter(c("in_eval") & (c("is_labelled") | c("has_pos_player"))).height))
    # dev pair count consistent with 155,812 unknown?
    for k in (1, 20, 38, 57):
        rec(f"dev_unlabelled_pairs_with_dev_shared_ge{k}", int(U.filter(~c("is_labelled") & (c("dev_shared") >= k)).height))
    rec("labelled_dev_shared_min_by_label", U.filter(c("is_labelled")).group_by("label").agg(
        c("dev_shared").min().alias("min"), c("dev_shared").median().alias("med"), c("dev_shared").max().alias("max")).sort("label").to_dicts())
    rec("dev_shared_quantiles_unknown_ge57", q(U.filter(~c("is_labelled") & (c("dev_shared") >= 57))["dev_shared"]))
    rec("dev_shared_quantiles_pos", q(U.filter(c("label") == 1)["dev_shared"]))
    rec("dev_shared_quantiles_neg", q(U.filter(c("label") == 0)["dev_shared"]))
    rec("dev_shared_quantiles_unknown_all", q(U.filter(~c("is_labelled"))["dev_shared"]))
    rec("eval_shared_quantiles_evalpairs", q(ep["shared_hands"]))
    rec("eval_shared_quantiles_labelled_pos_pairs(not in eval)", q(U.filter(c("label") == 1)["eval_shared"]))
    rec("eval_shared_quantiles_labelled_neg_pairs", q(U.filter(c("label") == 0)["eval_shared"]))
    rec("ratio_eval_to_dev_shared_median_all", round(float(U["eval_shared"].median() / U["dev_shared"].median()), 3))

    # per-table counts
    tabs = (pl.DataFrame({"table_idx": np.arange(N_TABLES, dtype=np.int16)})
            .join(pos.group_by("table_idx").len().rename({"len": "n_pos"}), on="table_idx", how="left")
            .join(neg.group_by("table_idx").len().rename({"len": "n_neg"}), on="table_idx", how="left")
            .join(ep.join(ptab.rename({"player_idx": "p1"}), on="p1").group_by("table_idx").len().rename({"len": "n_eval"}),
                  on="table_idx", how="left")
            .fill_null(0))
    ppt = (pl.concat([pos.select("table_idx", c("p1").alias("p")), pos.select("table_idx", c("p2").alias("p"))])
           .unique().group_by("table_idx").len().rename({"len": "n_pos_players"}))
    tabs = tabs.join(ppt, on="table_idx", how="left").fill_null(0).join(folds, on="table_idx")
    rec("tables_with_pos", int((tabs["n_pos"] > 0).sum()))
    rec("n_pos_per_table_dist", dict(sorted(tabs.group_by("n_pos").len().rows())))
    rec("n_neg_per_table_dist", dict(sorted(tabs.group_by("n_neg").len().rows())))
    rec("neg_eq_4x_pos_tables", int((tabs["n_neg"] == 4 * tabs["n_pos"]).sum()))
    rec("tables_zero_pos_but_neg", int(((tabs["n_pos"] == 0) & (tabs["n_neg"] > 0)).sum()))
    rec("corr_npos_nneg", round(float(np.corrcoef(tabs["n_pos"], tabs["n_neg"])[0, 1]), 3))
    rec("n_eval_pairs_per_table_by_npos", tabs.group_by("n_pos").agg(c("n_eval").mean().round(1), c("n_pos_players").mean().round(2)).sort("n_pos").to_dicts())
    # zero-truncated Poisson fit on positives-per-table
    npos = tabs.filter(c("n_pos") > 0)["n_pos"].to_numpy()
    m = npos.mean()
    lam = m
    for _ in range(200):
        lam = m * (1 - np.exp(-lam))
    rec("zt_poisson_lambda_pos_per_table", round(float(lam), 3))
    rec("zt_poisson_expected_zero_tables_of_400", round(float(400 * np.exp(-lam)), 1))
    rec("observed_zero_tables", int((tabs["n_pos"] == 0).sum()))
    # per-pos-pair negatives
    rec("pos_family_counts", dict(pos.group_by("behavior_family").len().rows()))
    rec("pos_family_by_fold", pos.group_by("fold", "behavior_family").len().sort("fold", "behavior_family").pivot(
        on="behavior_family", index="fold", values="len").to_dicts())
    rec("neg_by_fold", dict(sorted(neg.group_by("fold").len().rows())))
    rec("tables_by_fold", dict(sorted(folds.group_by("fold").len().rows())))

    # are zero-positive tables different? (table-level aggregates of pair stats)
    tstat = (gd.group_by("table_idx").agg(
        c("copresence_lift").filter(c("shared_hands") >= 57).max().alias("max_lift"),
        c("net_flow_12_bb").abs().filter(c("shared_hands") >= 57).max().alias("max_absflow"),
        (c("net_flow_12_bb").abs() / c("shared_hands")).filter(c("shared_hands") >= 57).quantile(0.99).alias("p99_flow_rate"),
        c("seat_adjacent_rate").filter(c("shared_hands") >= 57).max().alias("max_adj"),
    ).join(psd.group_by("table_idx").agg(c("vpip_rate").mean().alias("mean_vpip"), c("bb100").std().alias("sd_bb100"),
                                          c("hands").std().alias("sd_hands"), c("joins").mean().alias("mean_joins")), on="table_idx")
     .join(pl.read_parquet(DER / "hands.parquet", columns=["table_idx", "bb", "final_pot"]).group_by("table_idx").agg(
         c("bb").first().alias("bb"), (c("final_pot") / c("bb")).mean().alias("mean_pot_bb")), on="table_idx")
     .join(tabs, on="table_idx"))
    zt = {}
    for f in ["max_lift", "max_absflow", "p99_flow_rate", "max_adj", "mean_vpip", "sd_bb100", "sd_hands", "mean_joins", "mean_pot_bb", "bb"]:
        y = (tstat["n_pos"] > 0).to_numpy().astype(int)
        zt[f] = {"auc_haspos_vs_zero": round(auc(y, tstat[f].to_numpy()), 3),
                 "mean_haspos": round(float(tstat.filter(c("n_pos") > 0)[f].mean()), 3),
                 "mean_zero": round(float(tstat.filter(c("n_pos") == 0)[f].mean()), 3)}
    rec("zero_pos_tables_vs_pos_tables", zt)
    rec("stake_by_haspos", tstat.group_by("bb").agg(c("n_pos").mean().round(3).alias("mean_npos"), (c("n_pos") > 0).mean().round(3).alias("share_haspos"), pl.len()).sort("bb").to_dicts())

    # ------------------------------------------------------------------ 2. graph structure
    print("\n==== 2. GRAPH / SEATING / SESSIONS ====")
    deg = (pl.concat([pos.select(c("p1").alias("p"), "behavior_family", "table_idx"),
                      pos.select(c("p2").alias("p"), "behavior_family", "table_idx")])
           .group_by("p").agg(pl.len().alias("deg"), c("behavior_family").n_unique().alias("nfam")))
    rec("pos_player_degree_dist", dict(sorted(deg.group_by("deg").len().rows())))
    rec("multi_pos_players_mixed_family", int(deg.filter((c("deg") > 1) & (c("nfam") > 1)).height))
    # connected components per table
    import collections
    adj = collections.defaultdict(set)
    fam_edge = {}
    for a_, b_, f_ in pos.select("p1", "p2", "behavior_family").rows():
        adj[a_].add(b_)
        adj[b_].add(a_)
        fam_edge[(a_, b_)] = f_
    seen, comps = set(), []
    for v in list(adj):
        if v in seen:
            continue
        st, comp = [v], set()
        while st:
            x = st.pop()
            if x in comp:
                continue
            comp.add(x)
            st.extend(adj[x] - comp)
        seen |= comp
        ne = sum(1 for (a_, b_) in fam_edge if a_ in comp)
        fams = {f_ for (a_, b_), f_ in fam_edge.items() if a_ in comp}
        comps.append((len(comp), ne, len(fams)))
    cc = collections.Counter((s_, e_) for s_, e_, _ in comps)
    rec("pos_components_(nodes,edges)->count", {f"{k[0]}n_{k[1]}e": v for k, v in sorted(cc.items())})
    rec("pos_components_mixed_family", sum(1 for _, _, nf in comps if nf > 1))
    # components where a triangle-completing pair is unlabelled
    comp_open = 0
    for s_, e_, _ in comps:
        if s_ >= 3 and e_ < s_ * (s_ - 1) // 2:
            comp_open += 1
    rec("pos_components_ge3_not_complete", comp_open)
    # negatives containing a positive player
    negx = neg.with_columns(c("p1").is_in(list(pos_players)).alias("a"), c("p2").is_in(list(pos_players)).alias("b"))
    rec("neg_pairs_with_0_1_2_pos_players", dict(sorted(negx.group_by((c("a").cast(int) + c("b").cast(int)).alias("k")).len().rows())))
    # negatives whose pos player is at same table as pos pair -> neg partner is a non-partner of a colluder
    negpp = negx.filter(c("a") | c("b"))
    rec("neg_pairs_touching_pos_player", negpp.height)
    # per positive pair: how many negatives share a player
    pos_np = pos.with_row_index("ri")
    share_cnt = []
    negset = neg.select("p1", "p2").rows()
    negp = collections.Counter()
    for a_, b_ in negset:
        negp[a_] += 1
        negp[b_] += 1
    for a_, b_ in pos.select("p1", "p2").rows():
        share_cnt.append(negp[a_] + negp[b_])
    rec("negs_sharing_player_per_pos_pair_dist", dict(sorted(collections.Counter(share_cnt).items())))
    # tables: negatives in tables with positives share players?
    rec("neg_players_degree_dist", dict(sorted(collections.Counter(collections.Counter(
        [x for ab in negset for x in ab]).values()).items())))

    # labelled pairs with generic features (dev)
    G = (gd.join(lab_keys, on=["p1", "p2"], how="left")
         .with_columns(pl.when(c("label") == 1).then(pl.lit("pos")).when(c("label") == 0).then(pl.lit("neg"))
                       .otherwise(pl.lit("unk")).alias("grp")))
    G = G.with_columns((c("has_pp") if False else (c("p1").is_in(list(pos_players)) | c("p2").is_in(list(pos_players)))).alias("touch_pos"))
    Gs = G.filter(c("shared_hands") >= 57)
    # seat adjacency / offsets
    rec("seat_offsets_share_by_group", Gs.group_by("grp").agg(
        ((c("off1_12") + c("off1_21")).sum() / c("shared_hands").sum()).round(4).alias("adjacent(off1|off5)"),
        ((c("off2_12") + c("off2_21")).sum() / c("shared_hands").sum()).round(4).alias("off2|off4"),
        (c("seat_opposite").sum() / c("shared_hands").sum()).round(4).alias("opposite"),
        (c("off1_12").sum() / (c("off1_12") + c("off1_21")).sum()).round(4).alias("p2_left_of_p1_share"),
        c("seat_adjacent_rate").median().round(4).alias("median_pair_adj_rate"),
    ).sort("grp").to_dicts())
    for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
        sub = Gs.filter(c("behavior_family") == fam)
        R[f"adjacent_rate_{fam}"] = round(float((sub["off1_12"] + sub["off1_21"]).sum() / sub["shared_hands"].sum()), 4)
    print({k: v for k, v in R.items() if k.startswith("adjacent_rate_")})
    rec("copresence_lift_by_group", Gs.group_by("grp").agg(
        c("copresence_lift").quantile(.1).round(3).alias("q10"), c("copresence_lift").median().round(3).alias("q50"),
        c("copresence_lift").quantile(.9).round(3).alias("q90"), c("copresence_lift").mean().round(3).alias("mean"), pl.len()).sort("grp").to_dicts())
    rec("copresence_lift_by_group_all_shared", G.group_by("grp").agg(c("copresence_lift").mean().round(3).alias("mean")).sort("grp").to_dicts())
    rec("cojoin_coleave_by_group", Gs.group_by("grp").agg(
        (c("cojoin") / (c("joins_1") + c("joins_2") + 1)).mean().round(4).alias("cojoin_per_join"),
        (c("coleave") / (c("joins_1") + c("joins_2") + 1)).mean().round(4).alias("coleave_per_join"),
        c("joins_1").mean().round(1).alias("joins_1_mean")).sort("grp").to_dicts())
    rec("n_hands_players_by_group", Gs.group_by("grp").agg(
        pl.min_horizontal("n1_hands", "n2_hands").median().alias("min_hands_med"),
        pl.max_horizontal("n1_hands", "n2_hands").median().alias("max_hands_med")).sort("grp").to_dicts())

    # session timing: evidence hands' spread within the pair's shared hands
    evh = ev.join(hands.select("hand_idx", "hand_seq"), on="hand_idx").group_by("pair_id").agg(
        c("hand_seq").min().alias("ev_min"), c("hand_seq").max().alias("ev_max"), pl.len().alias("n_ev"))
    rec("evidence_count_dist", dict(sorted(evh.group_by("n_ev").len().rows())))
    rec("evidence_span_hands_quantiles", q((evh["ev_max"] - evh["ev_min"]).to_numpy()))
    rec("evidence_first_seq_quantiles", q(evh["ev_min"].to_numpy()))
    rec("evidence_last_seq_quantiles", q(evh["ev_max"].to_numpy()))
    rec("pos_without_evidence", int(pos.join(evh, on="pair_id", how="anti").height))

    # ------------------------------------------------------------------ 3. hard negatives vs pos vs unknown
    print("\n==== 3. FEATURE SEPARATION (pos / neg / unknown) ====")
    ps_cols = ["hands", "vpip_rate", "pfr_rate", "threebet_rate", "af_post", "afq_post", "aggr_rate", "wtsd",
               "wsd_won_rate", "bb100", "fold_to_bet_post", "sd_freq", "limp_rate", "allin_rate", "weak_vpip_rate",
               "strong_foldpre_rate", "vpip_avg_strength", "drift_vpip", "drift_pfr", "drift_aggr", "drift_bb100",
               "tilt_vpip_delta", "tilt_pfr_delta", "tilt_bb100_delta", "bigloss_events", "maxdd_bb", "maxup_bb",
               "joins", "mean_session_len", "avg_stack_bb"]
    pss = psd.select("player_idx", *[x for x in ps_cols if x not in ("joins",)])
    F = (Gs.drop("joins_1", "joins_2")
         .join(pss.rename({k: k + "_a" for k in pss.columns}).rename({"player_idx_a": "p1"}), on="p1")
         .join(pss.rename({k: k + "_b" for k in pss.columns}).rename({"player_idx_b": "p2"}), on="p2")
         .join(psd.select(c("player_idx").alias("p1"), c("joins").alias("joins_a")), on="p1")
         .join(psd.select(c("player_idx").alias("p2"), c("joins").alias("joins_b")), on="p2"))
    feats = {}
    S = c("shared_hands")
    feats.update({
        "shared_hands": S, "copresence_lift": c("copresence_lift"), "copresence_min": c("copresence_min"),
        "both_vpip_rate": c("both_vpip") / S, "both_flop_rate": c("both_saw_flop") / S, "sd_tog_rate": c("sd_together") / S,
        "hu_final_rate": c("hu_final") / S, "hu_flop_rate": c("hu_flop") / S,
        "checkdown_share_of_huflop": _ratio(c("checkdown_hu"), c("hu_flop")),
        "both_aggr_rate": c("both_aggr") / S,
        "abs_net_flow_per100": (c("net_flow_12_bb").abs() * 100 / S),
        "abs_net_flow_hu_per100": (c("net_flow_hu_12_bb").abs() * 100 / S),
        "gross_flow_per100": ((c("flow_12_bb") + c("flow_21_bb")) * 100 / S),
        "flow_asym": _ratio(c("net_flow_12_bb").abs(), c("flow_12_bb") + c("flow_21_bb")),
        "seat_adjacent_rate": c("seat_adjacent_rate"),
        "aggr_v_min_rate": pl.min_horizontal(c("aggr_rate_1v2"), c("aggr_rate_2v1")),
        "aggr_toward_minus_others_min": pl.min_horizontal(c("aggr_rate_1v2") - c("aggr_rate_1_not2"), c("aggr_rate_2v1") - c("aggr_rate_2_not1")),
        "aggr_hu_min_rate": pl.min_horizontal(c("aggr_rate_hu_1v2"), c("aggr_rate_hu_2v1")),
        "aggr_hu_sum_rate": _ratio(c("aggr_hu_12") + c("aggr_hu_21"), c("acts_hu_12") + c("acts_hu_21")),
        "fold_to_partner_max": pl.max_horizontal(c("fold_rate_1_to_2"), c("fold_rate_2_to_1")),
        "fold_post_to_partner_max": pl.max_horizontal(_ratio(c("fold_faced_post_12"), c("faced_post_12")), _ratio(c("fold_faced_post_21"), c("faced_post_21"))),
        "reraise_partner_rate": _ratio(c("reraise_faced_12") + c("reraise_faced_21"), c("faced_12") + c("faced_21")),
        "fold_strongpre_to_partner": _ratio(c("fold_faced_pre_strong_12") + c("fold_faced_pre_strong_21"), c("faced_pre_strong_12") + c("faced_pre_strong_21")),
        "cojoin_rate": c("cojoin") / (c("joins_a") + c("joins_b") + 1),
        "coleave_rate": c("coleave") / (c("joins_a") + c("joins_b") + 1),
    })
    for x in ps_cols:
        if x == "joins":
            continue
        a_, b_ = c(x + "_a"), c(x + "_b")
        feats[f"{x}__min"] = pl.min_horizontal(a_, b_)
        feats[f"{x}__max"] = pl.max_horizontal(a_, b_)
        feats[f"{x}__absdiff"] = (a_ - b_).abs()
    # metadata
    md = players.select("player_idx", "account_age_days", "experience_hands_bucket", "preferred_stake", "region_bucket", "client_family")
    F = (F.join(md.rename({k: k + "_ma" for k in md.columns}).rename({"player_idx_ma": "p1"}), on="p1")
          .join(md.rename({k: k + "_mb" for k in md.columns}).rename({"player_idx_mb": "p2"}), on="p2"))
    for m_ in ["experience_hands_bucket", "preferred_stake", "region_bucket", "client_family"]:
        feats[f"same_{m_}"] = (c(m_ + "_ma") == c(m_ + "_mb")).cast(pl.Float64)
    feats["age_gap"] = (c("account_age_days_ma") - c("account_age_days_mb")).abs().cast(pl.Float64)
    feats["age_min"] = pl.min_horizontal("account_age_days_ma", "account_age_days_mb").cast(pl.Float64)
    FX = F.select("table_idx", "p1", "p2", "grp", "behavior_family", "touch_pos", *[e.alias(k) for k, e in feats.items()])
    FX.write_parquet(OUT / "_audit_pair_feats_dev.parquet")
    grp = FX["grp"].to_numpy()
    rows = []
    for k in feats:
        v = FX[k].cast(pl.Float64).to_numpy()
        mp, mn, mu = grp == "pos", grp == "neg", grp == "unk"
        sel = lambda m1, m2: (np.r_[np.ones(m1.sum()), np.zeros(m2.sum())], np.r_[v[m1], v[m2]])
        rows.append({"feat": k,
                     "auc_pos_vs_unk": round(auc(*sel(mp, mu)), 3),
                     "auc_pos_vs_neg": round(auc(*sel(mp, mn)), 3),
                     "auc_neg_vs_unk": round(auc(*sel(mn, mu)), 3),
                     "med_pos": round(float(np.nanmedian(v[mp])), 4), "med_neg": round(float(np.nanmedian(v[mn])), 4),
                     "med_unk": round(float(np.nanmedian(v[mu])), 4)})
    sep = pl.DataFrame(rows)
    sep = sep.with_columns((c("auc_pos_vs_unk") - 0.5).abs().alias("d_pu"), (c("auc_pos_vs_neg") - 0.5).abs().alias("d_pn"),
                           (c("auc_neg_vs_unk") - 0.5).abs().alias("d_nu"))
    sep.write_csv(OUT / "structure_feature_separation.csv")
    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=250):
        print(sep.sort("d_pu", descending=True).head(60))
    R["feature_separation_top_pos_vs_unk"] = sep.sort("d_pu", descending=True).head(25).drop("d_pu", "d_pn", "d_nu").to_dicts()
    R["feature_separation_top_pos_vs_neg"] = sep.sort("d_pn", descending=True).head(25).drop("d_pu", "d_pn", "d_nu").to_dicts()
    R["feature_separation_top_neg_vs_unk"] = sep.sort("d_nu", descending=True).head(25).drop("d_pu", "d_pn", "d_nu").to_dicts()
    R["danger_pos_vs_unk_but_not_vs_neg"] = sep.filter((c("d_pu") > 0.1) & (c("d_pn") < 0.05)).drop("d_pu", "d_pn", "d_nu").to_dicts()
    R["neg_vs_unk_strong(|auc-.5|>.1)"] = sep.filter(c("d_nu") > 0.1).drop("d_pu", "d_pn", "d_nu").to_dicts()

    # per-family separation (pos family vs neg) for a few key features
    fam_rows = []
    for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
        mf = (FX["behavior_family"] == fam).fill_null(False).to_numpy()
        mn = grp == "neg"
        mu = grp == "unk"
        for k in feats:
            v = FX[k].cast(pl.Float64).to_numpy()
            fam_rows.append({"family": fam, "feat": k,
                             "auc_vs_neg": round(auc(np.r_[np.ones(mf.sum()), np.zeros(mn.sum())], np.r_[v[mf], v[mn]]), 3),
                             "auc_vs_unk": round(auc(np.r_[np.ones(mf.sum()), np.zeros(mu.sum())], np.r_[v[mf], v[mu]]), 3)})
    famdf = pl.DataFrame(fam_rows).with_columns((c("auc_vs_neg") - .5).abs().alias("d"))
    famdf.write_csv(OUT / "structure_feature_separation_by_family.csv")
    R["family_top_vs_neg"] = {fam: famdf.filter(c("family") == fam).sort("d", descending=True).head(10).drop("d", "family").to_dicts()
                              for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]}
    for fam in R["family_top_vs_neg"]:
        print(fam, R["family_top_vs_neg"][fam][:6])

    # hard-negative sub-types: which "suspicious-looking" axes are extreme in negatives vs unknown baseline
    axes = {
        "tilt": ["tilt_vpip_delta__max", "tilt_pfr_delta__max", "bigloss_events__max", "maxdd_bb__max"],
        "weak_play": ["weak_vpip_rate__max", "vpip_rate__max", "bb100__min", "vpip_avg_strength__min"],
        "similar_strategy": ["vpip_rate__absdiff", "pfr_rate__absdiff", "aggr_rate__absdiff", "af_post__absdiff"],
        "opponent_selection": ["copresence_lift", "copresence_min", "cojoin_rate", "coleave_rate"],
        "streaks": ["maxup_bb__max", "bb100__max", "abs_net_flow_per100", "gross_flow_per100"],
        "strategy_change": ["drift_vpip__max", "drift_pfr__max", "drift_aggr__max", "drift_bb100__max"],
    }
    unk_tbl = FX.filter(c("grp") == "unk")
    sub = {}
    for ax, fs in axes.items():
        zsc = []
        for f_ in fs:
            vals = unk_tbl[f_].cast(pl.Float64).drop_nulls().to_numpy()
            lo_, hi_ = np.quantile(vals, [0.05, 0.95])
            zsc.append(f_)
        out_ax = {}
        for f_ in fs:
            vals = unk_tbl[f_].cast(pl.Float64).drop_nulls().to_numpy()
            lo_, hi_ = np.quantile(vals, [0.05, 0.95])
            d_ = {}
            for g_ in ["pos", "neg", "unk"]:
                vv = FX.filter(c("grp") == g_)[f_].cast(pl.Float64).drop_nulls().to_numpy()
                d_[g_] = {"gt_unk_p95": round(float((vv > hi_).mean()), 3), "lt_unk_p05": round(float((vv < lo_).mean()), 3)}
            out_ax[f_] = d_
        sub[ax] = out_ax
    rec("tail_rates_vs_unknown_p05_p95", sub, show=False)
    for ax in sub:
        print(ax, {f_: (sub[ax][f_]["neg"], sub[ax][f_]["pos"]) for f_ in sub[ax]})

    # clustering hard negatives on standardized axis features (robust z vs unknown)
    from sklearn.cluster import KMeans
    cl_feats = [f_ for fs in axes.values() for f_ in fs]
    Z = []
    for f_ in cl_feats:
        vals = unk_tbl[f_].cast(pl.Float64).to_numpy()
        med = np.nanmedian(vals)
        iqr = np.nanquantile(vals, .75) - np.nanquantile(vals, .25) + 1e-9
        Z.append(((FX[f_].cast(pl.Float64).to_numpy() - med) / iqr))
    Z = np.nan_to_num(np.vstack(Z).T.clip(-6, 6))
    mneg = grp == "neg"
    mpos = grp == "pos"
    km_res = {}
    for kk in (4, 6):
        km = KMeans(kk, n_init=10, random_state=42).fit(Z[mneg])
        lab_ = km.labels_
        cents = km.cluster_centers_
        desc = []
        for ci in range(kk):
            top = np.argsort(-np.abs(cents[ci]))[:4]
            desc.append({"n": int((lab_ == ci).sum()),
                         "top_axes": {cl_feats[j]: round(float(cents[ci, j]), 2) for j in top}})
        # assign positives & unknown sample to the neg clusters
        pos_assign = np.bincount(km.predict(Z[mpos]), minlength=kk).tolist()
        rng = np.random.default_rng(42)
        uidx = rng.choice(np.flatnonzero(grp == "unk"), 20000, replace=False)
        unk_assign = np.bincount(km.predict(Z[uidx]), minlength=kk).tolist()
        km_res[kk] = {"clusters": desc, "pos_assign": pos_assign, "unk20k_assign": unk_assign}
    rec("neg_kmeans", km_res)
    # neg pairs that contain a positive player vs not: any difference?
    rec("neg_touching_pos_vs_not_key_medians", FX.filter(c("grp") == "neg").group_by("touch_pos").agg(
        pl.len(), c("copresence_lift").median().round(3), c("abs_net_flow_per100").median().round(3),
        c("aggr_v_min_rate").median().round(3), c("shared_hands").median()).to_dicts())

    # ------------------------------------------------------------------ 4. metadata lift
    print("\n==== 4. METADATA ====")
    MD = (U.select("table_idx", "p1", "p2", "label", "in_eval", "dev_shared", "has_pos_player")
          .join(md.rename({k: k + "_ma" for k in md.columns}).rename({"player_idx_ma": "p1"}), on="p1")
          .join(md.rename({k: k + "_mb" for k in md.columns}).rename({"player_idx_mb": "p2"}), on="p2")
          .with_columns(pl.when(c("label") == 1).then(pl.lit("pos")).when(c("label") == 0).then(pl.lit("neg"))
                        .when(c("in_eval")).then(pl.lit("eval")).when(c("has_pos_player")).then(pl.lit("unk_touch_pos"))
                        .otherwise(pl.lit("unk_other")).alias("grp")))
    mdres = {}
    for m_ in ["experience_hands_bucket", "preferred_stake", "region_bucket", "client_family"]:
        mdres[f"same_{m_}"] = MD.group_by("grp").agg((c(m_ + "_ma") == c(m_ + "_mb")).mean().round(4).alias("rate"), pl.len()).sort("grp").to_dicts()
    mdres["age_gap_median"] = MD.group_by("grp").agg((c("account_age_days_ma") - c("account_age_days_mb")).abs().median().alias("med"),
                                                     (c("account_age_days_ma") - c("account_age_days_mb")).abs().mean().round(1).alias("mean")).sort("grp").to_dicts()
    mdres["n_same_4_fields_mean"] = MD.with_columns(sum((c(m_ + "_ma") == c(m_ + "_mb")).cast(pl.Int8) for m_ in
                                                        ["experience_hands_bucket", "preferred_stake", "region_bucket", "client_family"]).alias("ns")) \
        .group_by("grp").agg(c("ns").mean().round(3), (c("ns") == 4).mean().round(4).alias("all4")).sort("grp").to_dicts()
    for m_ in ["experience_hands_bucket", "preferred_stake", "region_bucket", "client_family"]:
        mdres[f"{m_}_levels"] = dict(players.group_by(m_).len().rows())
    # player-level metadata: pos players vs neg players vs others
    pl_grp = players.with_columns(pl.when(c("player_idx").is_in(list(pos_players))).then(pl.lit("pos_player"))
                                  .when(c("player_idx").is_in(list(neg_players))).then(pl.lit("neg_only_player"))
                                  .otherwise(pl.lit("other")).alias("g"))
    mdres["player_level_age_median"] = pl_grp.group_by("g").agg(c("account_age_days").median(), pl.len()).sort("g").to_dicts()
    for m_ in ["experience_hands_bucket", "preferred_stake", "region_bucket", "client_family"]:
        mdres[f"player_level_{m_}"] = pl_grp.group_by("g", m_).len().with_columns(
            (c("len") / c("len").sum().over("g")).round(3).alias("share")).sort("g", m_).to_dicts()
    # preferred_stake vs table stake
    tstake = pl.read_parquet(DER / "hands.parquet", columns=["table_idx", "bb"]).unique("table_idx")
    mdres["preferred_stake_vs_table_bb"] = (pl_grp.join(ptab, on="player_idx").join(tstake, on="table_idx")
                                            .group_by("bb", "preferred_stake").len().sort("bb", "preferred_stake").to_dicts())
    rec("metadata", mdres, show=False)
    for k_, v_ in mdres.items():
        if not k_.startswith("player_level_") and not k_.endswith("_levels"):
            print(k_, v_)

    # ------------------------------------------------------------------ 7. validation design
    print("\n==== 7. VALIDATION DESIGN ====")
    rec("folds_md5", __import__("hashlib").md5(FOLDS_CSV.read_bytes()).hexdigest())
    fold_tbl = tabs.group_by("fold").agg(pl.len().alias("tables"), c("n_pos").sum(), c("n_neg").sum(),
                                         (c("n_pos") > 0).sum().alias("tables_with_pos"), c("n_eval").sum()).sort("fold")
    rec("fold_balance", fold_tbl.to_dicts())
    # simulated exclusion: within-table random split of positive pairs into "labelled" / "held-out"
    rng = np.random.default_rng(42)
    posr = pos.select("table_idx", "p1", "p2", "behavior_family").rows()
    by_t = collections.defaultdict(list)
    for r_ in posr:
        by_t[r_[0]].append(r_)
    surv = []
    for rep in range(200):
        kept = tot = 0
        for t_, prs in by_t.items():
            msk = rng.random(len(prs)) < 0.5
            labp = {x for r_, m_ in zip(prs, msk) if m_ for x in r_[1:3]}
            for r_, m_ in zip(prs, msk):
                if not m_:
                    tot += 1
                    kept += (r_[1] not in labp and r_[2] not in labp)
        surv.append(kept / max(tot, 1))
    rec("sim_within_table_50pct_split_heldout_pos_surviving_exclusion", round(float(np.mean(surv)), 4))
    # fraction of positive pairs sharing a player with another positive pair
    rec("pos_pairs_sharing_player_with_other_pos", int(sum(1 for r_ in posr if deg.filter(c("p").is_in([r_[1], r_[2]]))["deg"].max() > 1)))
    # dev pairs eligible under eval-like rules inside a CV fold: exclude pairs touching labelled positive
    # players *of the training folds* is automatic (tables disjoint); inside the held-out fold, the
    # scoring universe mirroring eval = all pairs with dev_shared >= 57 (38*1.5) not touching ...
    elig_dev = U.filter(c("dev_shared") >= 57)
    rec("dev_universe_shared_ge57", elig_dev.height)
    rec("dev_universe_shared_ge57_by_label", dict(elig_dev.group_by(c("label").fill_null(-1)).len().rows()))
    rec("dev_universe_shared_ge57_unknown_touching_pos_player",
        int(elig_dev.filter(c("label").is_null() & c("has_pos_player")).height))
    rec("dev_unknown_ge57_not_touching_pos", int(elig_dev.filter(c("label").is_null() & ~c("has_pos_player")).height))
    rec("prevalence_labelled_only", round(pos.height / lab.height, 4))
    rec("prevalence_pos_over_dev_universe_ge57", round(pos.height / elig_dev.height, 5))

    # quick PU illustration: how the three local AP variants react to a single strong generic feature
    FXa = FX.with_columns(c("copresence_lift").alias("s1"), c("abs_net_flow_per100").alias("s2"))
    for sname in ["s1", "s2"]:
        d_ = FXa.select("grp", "touch_pos", sname).drop_nulls()
        y_lab = d_.filter(c("grp") != "unk")
        y_all = d_
        y_clean = d_.filter((c("grp") != "unk") | ~c("touch_pos"))
        rec(f"ap_variants_{sname}", {
            "labelled_only(pos vs neg)": round(ap((y_lab["grp"] == "pos").to_numpy(), y_lab[sname].to_numpy()), 4),
            "pos_vs_neg+unknown(all ge57)": round(ap((y_all["grp"] == "pos").to_numpy(), y_all[sname].to_numpy()), 4),
            "pos_vs_neg+unknown(no unk touching pos player)": round(ap((y_clean["grp"] == "pos").to_numpy(), y_clean[sname].to_numpy()), 4),
            "random_baseline_all": round(pos.height / y_all.height, 5)})

    # dev -> eval shift of generic features for unknown pairs (rate features should be stable)
    shift = []
    ge_s = ge.filter(c("shared_hands") >= 38)
    gd_s = gd.filter(c("shared_hands") >= 57)
    for nm, e_ in [("copresence_lift", c("copresence_lift")), ("both_vpip_rate", c("both_vpip") / S), ("hu_final_rate", c("hu_final") / S),
                   ("abs_net_flow_per100", c("net_flow_12_bb").abs() * 100 / S), ("seat_adjacent_rate", c("seat_adjacent_rate")),
                   ("shared_hands", S), ("aggr_rate_1v2", c("aggr_rate_1v2"))]:
        a1 = gd_s.select(e_.alias("v"))["v"].to_numpy().astype(float)
        a2 = ge_s.select(e_.alias("v"))["v"].to_numpy().astype(float)
        shift.append({"feat": nm, "dev_q": q(a1, (.1, .5, .9)), "eval_q": q(a2, (.1, .5, .9))})
    rec("dev_eval_shift_generic", shift)
    # player stats stability dev vs eval (rank corr) - how much player identity persists
    from scipy.stats import spearmanr
    j_ = psd.join(pse, on="player_idx", suffix="_e")
    stab = {}
    for x in ["vpip_rate", "pfr_rate", "aggr_rate", "af_post", "wtsd", "bb100", "fold_to_bet_post", "threebet_rate", "weak_vpip_rate", "hands"]:
        m_ = j_.select(x, x + "_e").drop_nulls()
        stab[x] = round(float(spearmanr(m_[x], m_[x + "_e"]).correlation), 3)
    rec("player_stat_dev_eval_spearman", stab)
    # do labelled-positive players behave differently in eval than in dev? (drift of pos players vs others)
    j2 = j_.with_columns(c("player_idx").is_in(list(pos_players)).alias("posp"), c("player_idx").is_in(list(neg_players)).alias("negp"))
    rec("dev_eval_delta_by_player_group", j2.group_by("posp", "negp").agg(
        pl.len(), (c("vpip_rate_e") - c("vpip_rate")).mean().round(4).alias("d_vpip"),
        (c("bb100_e") - c("bb100")).mean().round(2).alias("d_bb100"),
        (c("aggr_rate_e") - c("aggr_rate")).mean().round(4).alias("d_aggr"),
        c("bb100").mean().round(2).alias("bb100_dev"), c("bb100_e").mean().round(2).alias("bb100_eval"),
        c("vpip_rate").mean().round(4).alias("vpip_dev")).sort("posp", "negp").to_dicts())

    (OUT / "structure_results.json").write_text(json.dumps(R, indent=1, default=str))
    print("\nwrote", OUT / "structure_results.json")


def rate_features(g, ps):
    """Phase-invariant pair rate features from a pair_generic frame + player_stats (same phase)."""
    c = pl.col
    S = c("shared_hands")
    pss = ps.select("player_idx", "vpip_rate", "pfr_rate", "threebet_rate", "aggr_rate", "af_post", "wtsd",
                    "bb100", "fold_to_bet_post", "limp_rate", "weak_vpip_rate", "allin_rate", "sd_freq")
    g = (g.join(pss.rename({k: k + "_a" for k in pss.columns}).rename({"player_idx_a": "p1"}), on="p1")
          .join(pss.rename({k: k + "_b" for k in pss.columns}).rename({"player_idx_b": "p2"}), on="p2"))
    feats = {
        "both_vpip_rate": c("both_vpip") / S, "both_flop_rate": c("both_saw_flop") / S, "sd_tog_rate": c("sd_together") / S,
        "hu_final_rate": c("hu_final") / S, "hu_flop_rate": c("hu_flop") / S, "checkdown_rate": c("checkdown_hu") / S,
        "both_aggr_rate": c("both_aggr") / S, "gross_flow_per100": (c("flow_12_bb") + c("flow_21_bb")) * 100 / S,
        "abs_net_flow_per100": c("net_flow_12_bb").abs() * 100 / S, "abs_net_flow_hu_per100": c("net_flow_hu_12_bb").abs() * 100 / S,
        "abs_net_flow_sd_per100": c("net_flow_sd_12_bb").abs() * 100 / S,
        "aggr_v_min": pl.min_horizontal("aggr_rate_1v2", "aggr_rate_2v1"), "aggr_v_max": pl.max_horizontal("aggr_rate_1v2", "aggr_rate_2v1"),
        "aggr_lift_min": pl.min_horizontal(c("aggr_rate_1v2") - c("aggr_rate_1_not2"), c("aggr_rate_2v1") - c("aggr_rate_2_not1")),
        "aggr_lift_max": pl.max_horizontal(c("aggr_rate_1v2") - c("aggr_rate_1_not2"), c("aggr_rate_2v1") - c("aggr_rate_2_not1")),
        "aggr_hu_rate": _ratio(c("aggr_hu_12") + c("aggr_hu_21"), c("acts_hu_12") + c("acts_hu_21")),
        "fold_to_partner_min": pl.min_horizontal("fold_rate_1_to_2", "fold_rate_2_to_1"),
        "fold_to_partner_max": pl.max_horizontal("fold_rate_1_to_2", "fold_rate_2_to_1"),
        "reraise_partner_rate": _ratio(c("reraise_faced_12") + c("reraise_faced_21"), c("faced_12") + c("faced_21")),
        "fold_post_partner": _ratio(c("fold_faced_post_12") + c("fold_faced_post_21"), c("faced_post_12") + c("faced_post_21")),
        "faced_partner_rate": (c("faced_12") + c("faced_21")) / S,
        "sd_win_asym": _ratio((c("sd_win_12") - c("sd_win_21")).abs(), c("sd_together")),
    }
    for x in ["vpip_rate", "pfr_rate", "threebet_rate", "aggr_rate", "af_post", "wtsd", "bb100", "fold_to_bet_post",
              "limp_rate", "weak_vpip_rate", "allin_rate", "sd_freq"]:
        feats[x + "__min"] = pl.min_horizontal(x + "_a", x + "_b")
        feats[x + "__max"] = pl.max_horizontal(x + "_a", x + "_b")
    return g.select("table_idx", "p1", "p2", "shared_hands", *[e.cast(pl.Float64).alias(k) for k, e in feats.items()])


def stage_extra():
    """Random-placement nulls for label structure, eval-phase persistence of labelled pairs, PU mixture estimate."""
    import collections
    import lightgbm as lgb
    c = pl.col
    res_path = OUT / "structure_results.json"
    global R
    R = json.loads(res_path.read_text()) if res_path.exists() else {}
    rng = np.random.default_rng(42)
    lab = pl.read_parquet(DER / "labels.parquet")
    ep = pl.read_parquet(DER / "eval_pairs.parquet")
    gd = pl.read_parquet(DER / "pair_generic_dev.parquet")
    ge = pl.read_parquet(DER / "pair_generic_eval.parquet")
    psd = pl.read_parquet(DER / "player_stats_dev.parquet")
    pse = pl.read_parquet(DER / "player_stats_eval.parquet")
    folds = pl.read_csv(FOLDS_CSV).with_columns(c("table_idx").cast(pl.Int16), c("fold").cast(pl.Int8))
    ptab = psd.select("player_idx", "table_idx")
    lab = lab.join(ptab.rename({"player_idx": "p1"}), on="p1")
    pos = lab.filter(c("label") == 1)
    neg = lab.filter(c("label") == 0)
    pos_players = set(pos["p1"].to_list()) | set(pos["p2"].to_list())
    tab_players = np.load(OUT / "_audit_tab_players.npy")

    print("\n==== E1. random-placement nulls ====")
    # (a) positives: players in >1 positive pair vs random placement of k pairs among pairs with dev_shared>=57
    cand = gd.filter(c("shared_hands") >= 57).select("table_idx", "p1", "p2")
    cand_by_t = {t: np.array(v) for t, v in
                 cand.group_by("table_idx").agg(pl.concat_list("p1", "p2").alias("pp")).select("table_idx", "pp").rows()}
    k_pos = dict(pos.group_by("table_idx").len().rows())
    k_neg = dict(neg.group_by("table_idx").len().rows())
    obs_multi = sum(1 for v in collections.Counter([x for ab in pos.select("p1", "p2").rows() for x in ab]).values() if v > 1)
    sims = []
    for _ in range(500):
        tot = 0
        for t, k in k_pos.items():
            arr = cand_by_t[t]
            idx = rng.choice(len(arr), k, replace=False)
            cnt = collections.Counter(arr[idx].ravel().tolist())
            tot += sum(1 for v in cnt.values() if v > 1)
        sims.append(tot)
    rec("E1_pos_players_in_multiple_pos_pairs_observed", obs_multi)
    rec("E1_pos_players_in_multiple_pos_pairs_random_placement_mean_p05_p95",
        [round(float(np.mean(sims)), 1), float(np.quantile(sims, .05)), float(np.quantile(sims, .95))])
    # (b) negatives touching positive players vs random sampling among non-positive candidate pairs
    pos_set = set(map(tuple, pos.select("p1", "p2").rows()))
    obs_touch = neg.filter(c("p1").is_in(list(pos_players)) | c("p2").is_in(list(pos_players))).height
    pp_t = collections.defaultdict(set)
    for t, a_, b_ in pos.select("table_idx", "p1", "p2").rows():
        pp_t[t] |= {a_, b_}
    cand_np = {t: arr[np.array([(int(a_), int(b_)) not in pos_set for a_, b_ in arr])] for t, arr in cand_by_t.items()}
    sims = []
    for _ in range(300):
        tot = 0
        for t, k in k_neg.items():
            arr2 = cand_np[t]
            idx = rng.choice(len(arr2), k, replace=False)
            ps_ = pp_t.get(t, set())
            tot += sum(1 for a_, b_ in arr2[idx] if a_ in ps_ or b_ in ps_)
        sims.append(tot)
    rec("E1_neg_touching_pos_player_observed", obs_touch)
    rec("E1_neg_touching_pos_player_random_mean_p05_p95",
        [round(float(np.mean(sims)), 1), float(np.quantile(sims, .05)), float(np.quantile(sims, .95))])
    # (c) per-table count of negatives vs Poisson/multinomial (1488 over 400 tables)
    nn = np.array([k_neg.get(t, 0) for t in range(N_TABLES)])
    rec("E1_neg_per_table_var_over_mean", round(float(nn.var() / nn.mean()), 3))
    npp = np.array([k_pos.get(t, 0) for t in range(N_TABLES)])
    rec("E1_pos_per_table_var_over_mean", round(float(npp.var() / npp.mean()), 3))

    print("\n==== E2. eval-phase persistence of labelled pairs ====")
    Fd = rate_features(gd, psd).join(lab.select("p1", "p2", "label", "behavior_family"), on=["p1", "p2"], how="left")
    Fe = rate_features(ge, pse).join(lab.select("p1", "p2", "label", "behavior_family"), on=["p1", "p2"], how="left")
    key = ["both_vpip_rate", "both_flop_rate", "sd_tog_rate", "gross_flow_per100", "abs_net_flow_per100", "aggr_lift_min",
           "both_aggr_rate", "reraise_partner_rate", "fold_to_partner_max"]
    pers = {}
    for f_ in key:
        d_ = Fd.filter(c("label").is_not_null())
        e_ = Fe.filter(c("label").is_not_null() & (c("shared_hands") >= 38))
        pers[f_] = {"dev_auc_pos_vs_neg": round(auc(d_["label"].to_numpy(), d_[f_].to_numpy()), 3),
                    "eval_auc_pos_vs_neg(eval_shared>=38)": round(auc(e_["label"].to_numpy(), e_[f_].to_numpy()), 3)}
        for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
            ee = e_.filter((c("behavior_family") == fam) | (c("label") == 0))
            dd = d_.filter((c("behavior_family") == fam) | (c("label") == 0))
            pers[f_][fam] = [round(auc(dd["label"].to_numpy(), dd[f_].to_numpy()), 3),
                             round(auc(ee["label"].to_numpy(), ee[f_].to_numpy()), 3)]
    rec("E2_persistence_dev_vs_eval_auc", pers)
    rec("E2_labelled_pairs_eval_shared_ge38", dict(Fe.filter(c("label").is_not_null() & (c("shared_hands") >= 38))
                                                   .group_by("label").len().rows()))

    print("\n==== E3. PU mixture estimate of hidden positives ====")
    feat_cols = [x for x in Fd.columns if x not in ("table_idx", "p1", "p2", "shared_hands", "label", "behavior_family")]
    D0 = (Fd.filter(c("shared_hands") >= 57).join(folds, on="table_idx")
          .with_columns(pl.when(c("label") == 1).then(pl.lit("pos")).when(c("label") == 0).then(pl.lit("neg"))
                        .otherwise(pl.lit("unk")).alias("grp"),
                        (c("p1").is_in(list(pos_players)) | c("p2").is_in(list(pos_players))).alias("touch_pos")))
    X = D0.select(feat_cols).to_numpy()
    grp = D0["grp"].to_numpy()
    fold = D0["fold"].to_numpy()
    E0 = Fe.join(ep.select("p1", "p2"), on=["p1", "p2"], how="semi")
    XE = E0.select(feat_cols).to_numpy()
    NE = (Fe.filter((c("label") == 0) & (c("shared_hands") >= 38)))
    PE = (Fe.filter((c("label") == 1) & (c("shared_hands") >= 38)))
    gw = pl.read_parquet(DER / "pair_generic_dev_w2000.parquet")
    psw = pl.read_parquet(DER / "player_stats_dev_w2000.parquet")
    Fw = rate_features(gw, psw)
    W0 = D0.select("p1", "p2").join(Fw, on=["p1", "p2"], how="left")
    XW = W0.select(feat_cols).to_numpy()
    w_shared = W0["shared_hands"].to_numpy()
    oof_w = np.zeros(len(grp))
    oof = np.zeros(len(grp))
    pe = np.zeros(E0.height)
    pne = np.zeros(NE.height)
    ppe = np.zeros(PE.height)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=3, verbose=-1, seed=42)
    for k in range(5):
        tr = (fold != k) & (grp != "neg")
        y = (grp[tr] == "pos").astype(int)
        w = np.where(y == 1, 20.0, 1.0)
        m = lgb.train(params, lgb.Dataset(X[tr], y, weight=w), num_boost_round=300)
        te = fold == k
        oof[te] = m.predict(X[te])
        oof_w[te] = m.predict(XW[te])
        pe += m.predict(XE) / 5
        pne += m.predict(NE.select(feat_cols).to_numpy()) / 5
        ppe += m.predict(PE.select(feat_cols).to_numpy()) / 5
    mp, mn, mu = grp == "pos", grp == "neg", grp == "unk"
    rec("E3_oof_auc_pos_vs_neg", round(auc(np.r_[np.ones(mp.sum()), np.zeros(mn.sum())], np.r_[oof[mp], oof[mn]]), 4))
    rec("E3_oof_auc_pos_vs_unk", round(auc(np.r_[np.ones(mp.sum()), np.zeros(mu.sum())], np.r_[oof[mp], oof[mu]]), 4))
    rec("E3_oof_auc_neg_vs_unk", round(auc(np.r_[np.ones(mn.sum()), np.zeros(mu.sum())], np.r_[oof[mn], oof[mu]]), 4))
    yl = np.r_[np.ones(mp.sum()), np.zeros(mn.sum())]
    rec("E3_oof_ap_labelled_only", round(ap(yl, np.r_[oof[mp], oof[mn]]), 4))
    rec("E3_oof_ap_pos_vs_neg_plus_unknown", round(ap(mp.astype(int), oof), 4))
    mt = D0["touch_pos"].to_numpy()
    keep = ~(mu & mt)
    rec("E3_oof_ap_pos_vs_neg_plus_unknown_not_touching_pos", round(ap(mp[keep].astype(int), oof[keep]), 4))
    # stress AP with unknown reweighted to eval pool size is equivalent in ranking; report per-fold spread
    rec("E3_oof_ap_labelled_only_by_fold", [round(ap((grp[(fold == k) & ~mu] == "pos").astype(int), oof[(fold == k) & ~mu]), 3) for k in range(5)])
    rec("E3_oof_ap_all_by_fold", [round(ap((grp[fold == k] == "pos").astype(int), oof[fold == k]), 3) for k in range(5)])
    np.save(OUT / "_audit_pu_oof.npy", oof)

    def wilson(x, n, z=1.96):
        p = x / n
        d = 1 + z * z / n
        cen = (p + z * z / (2 * n)) / d
        hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return cen - hw, cen + hw

    est = []
    for rq in (0.3, 0.5, 0.7, 0.8, 0.9):
        tau = np.quantile(oof[mp], 1 - rq)
        r_pos = float((oof[mp] >= tau).mean())
        xn = int((oof[mn] >= tau).sum())
        r_neg = xn / mn.sum()
        r_unk = float((oof[mu] >= tau).mean())
        r_unk_nt = float((oof[mu & ~mt] >= tau).mean())
        lo, hi = wilson(xn, mn.sum())
        pi = (r_unk - r_neg) / (r_pos - r_neg)
        pi_lo = (r_unk - hi) / (r_pos - hi)
        pi_hi = (r_unk - lo) / (r_pos - lo)
        # eval pairs
        r_ev = float((pe >= tau).mean())
        xne = int((pne >= tau).sum())
        r_ne = xne / len(pne)
        r_pe = float((ppe >= tau).mean())
        lo_e, hi_e = wilson(xne, len(pne))
        pi_e = (r_ev - r_ne) / (r_pos - r_ne)
        est.append({"pos_recall": rq, "tau": round(float(tau), 4), "neg_rate": round(float(r_neg), 4), "neg_hits": xn,
                    "unk_rate": round(r_unk, 4), "unk_rate_not_touching_pos": round(r_unk_nt, 4),
                    "pi_unknown": round(float(pi), 5), "hidden_pos_in_unknown_ge57": round(pi * mu.sum(), 0),
                    "hidden_pos_CI95": [round(pi_lo * mu.sum(), 0), round(pi_hi * mu.sum(), 0)],
                    "evalpairs_rate": round(r_ev, 4), "neg_evalphase_rate": round(r_ne, 4),
                    "labelled_pos_evalphase_recall": round(r_pe, 4),
                    "pi_eval(dev recall)": round(pi_e, 5), "eval_pos_est(dev recall)": round(pi_e * len(pe), 0),
                    "eval_pos_CI95": [round((r_ev - hi_e) / (r_pos - hi_e) * len(pe), 0), round((r_ev - lo_e) / (r_pos - lo_e) * len(pe), 0)],
                    "eval_pos_est(evalphase recall)": (None if r_pe - r_ne < 0.05 else round((r_ev - r_ne) / (r_pe - r_ne) * len(pe), 0))})
    rec("E3_mixture_estimates", est)
    # ---- E4: eval-length matched window (dev hands 0..1999) as clean-FPR reference and eval-like CV
    mw = w_shared >= 38
    rec("E4_w2000_oof_ap_labelled_only", round(ap((grp[mw & ~mu] == "pos").astype(int), oof_w[mw & ~mu]), 4))
    rec("E4_w2000_oof_ap_pos_vs_neg_plus_unknown", round(ap((grp[mw] == "pos").astype(int), oof_w[mw]), 4))
    rec("E4_w2000_oof_auc_pos_vs_neg", round(auc((grp[mw & ~mu] == "pos").astype(int), oof_w[mw & ~mu]), 4))
    rec("E4_w2000_n_pos_neg_unk", [int((mw & mp).sum()), int((mw & mn).sum()), int((mw & mu).sum())])
    est_w = []
    for rq in (0.3, 0.5, 0.7, 0.8, 0.9):
        tau = np.quantile(oof_w[mw & mp], 1 - rq)
        r_pos = float((oof_w[mw & mp] >= tau).mean())
        xn = int((oof_w[mw & mn] >= tau).sum())
        r_neg = xn / (mw & mn).sum()
        r_unk = float((oof_w[mw & mu] >= tau).mean())
        r_ev = float((pe >= tau).mean())
        lo, hi = wilson(xn, (mw & mn).sum())
        est_w.append({"pos_recall_w2000": rq, "tau": round(float(tau), 4), "neg_w_rate": round(r_neg, 5), "neg_w_hits": xn,
                      "unk_w_rate": round(r_unk, 5), "eval_rate": round(r_ev, 5), "eval_hits": int((pe >= tau).sum()),
                      "eval_pos_est(fp=unk_w)": round((r_ev - r_unk) / (r_pos - r_unk) * len(pe), 0),
                      "eval_pos_est(fp=neg_w)": round((r_ev - r_neg) / (r_pos - r_neg) * len(pe), 0),
                      "eval_pos_est(fp=neg_w CI95)": [round((r_ev - hi) / (r_pos - hi) * len(pe), 0), round((r_ev - lo) / (r_pos - lo) * len(pe), 0)],
                      "dev_unk_hidden_est(w2000, fp=neg_w)": round((r_unk - r_neg) / (r_pos - r_neg) * (mw & mu).sum(), 0)})
    rec("E4_mixture_estimates_matched_window", est_w)
    # ---- E5: are eval-phase suspicious pairs also suspicious in dev? (same relationship across phases?)
    E0k = E0.select("p1", "p2").with_columns(pl.Series("pe", pe))
    Dk = D0.select("p1", "p2", "grp").with_columns(pl.Series("oof", oof))
    J = E0k.join(Dk, on=["p1", "p2"], how="inner")
    from scipy.stats import spearmanr
    rec("E5_eval_pairs_with_dev_ge57", J.height)
    rec("E5_spearman_dev_oof_vs_eval_score", round(float(spearmanr(J["pe"], J["oof"]).correlation), 4))
    dev_pct = J["oof"].rank() / J.height
    J = J.with_columns(pl.Series("dev_pct", dev_pct), (pl.col("pe").rank() / J.height).alias("eval_pct"))
    cross = {}
    for K in (50, 100, 250, 500):
        topE = J.sort("pe", descending=True).head(K)
        topD = J.sort("oof", descending=True).head(K)
        cross[K] = {"top_eval_mean_dev_pct": round(float(topE["dev_pct"].mean()), 3),
                    "top_eval_share_in_dev_top1pct": round(float((topE["dev_pct"] >= 0.99).mean()), 3),
                    "top_dev_mean_eval_pct": round(float(topD["eval_pct"].mean()), 3),
                    "top_dev_share_in_eval_top1pct": round(float((topD["eval_pct"] >= 0.99).mean()), 3)}
    rec("E5_cross_phase_top_overlap", cross)
    rec("E3_n_unknown_ge57", int(mu.sum()))
    rec("E3_n_eval_pairs", int(len(pe)))
    # top unknown-score concentration: tables & touching pos players
    top = np.argsort(-oof[mu])[:500]
    rec("E3_top500_unknown_touching_pos_share", round(float(mt[mu][top].mean()), 3))
    rec("E3_unknown_touching_pos_share_overall", round(float(mt[mu].mean()), 3))
    res_path.write_text(json.dumps(R, indent=1, default=str))
    print("updated", res_path)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("build", "all"):
        stage_build()
        stage_build(window=True)
    if stage in ("analyze", "all"):
        stage_analyze()
    if stage in ("extra", "all"):
        stage_extra()
