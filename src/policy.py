"""exp003 -- LABEL-FREE normal-play action policy for every action (dev + eval phases).

    python src/policy.py states     # numba decision-state features -> data/derived/policy_states/part_*.parquet
    python src/policy.py train      # 2 cross-fitted LightGBM multiclass models (train on one table half)
    python src/policy.py predict    # held-out half predictions -> data/derived/action_policy.parquet
    python src/policy.py report     # per-street logloss/accuracy + p_call calibration (held-out)
    python src/policy.py validate   # pfcall_weak_max (DT) and SP passed-up raise check on labelled dev pairs
    python src/policy.py all        # everything in order

The policy sees only what a real player sees: public state (street, position, pot, stacks, bets, action history
of the hand) + own hole cards (preflop class equity, perceived postflop hand strength `hs_actor` vs one random hand,
made-hand category, draws, board texture). It never sees other players' cards (no omniscient eq_* columns),
player identities, or labels. Labels are used only to CLEAN training data: actions of hands where both players of a
labelled-positive pair are seated are excluded from training (they are still predicted).

Classes: 0 fold, 1 check, 2 call (call, or all_in with amount <= to_call), 3 aggressive (bet, raise, all_in amount > to_call).
Legal-action renormalisation: to_call == 0 -> {check, agg}; to_call > 0 -> {fold, call, agg}; agg illegal when
stack_before <= to_call. (A fold with to_call == 0 never occurs in the data: 0 of 18.6M actions.)
surprise = -ln max(p_renorm(actual class), 1e-6).

Cross-fitting: half = frozen table fold (data/folds_tables_5.csv) % 2. The model trained on half 1-h predicts half h,
so every probability is out-of-table. Street-stratified subsample of the training half (sampling on a covariate
leaves p(y | x) unbiased). Early stopping on a 5% hand-level holdout drawn from the training tables only.

Output data/derived/action_policy.parquet (one row per action, sorted by hand_idx, action_no):
  hand_idx i32, action_no i16, player_idx i32, street i8, y i8 (actual class), n_agg_street i8 (aggressive actions
  before this one on the street), last_aggr_player i32 (-1 = none; last bet/raise/aggressive all-in on this street),
  to_call_bb f32, legal i8 (bitmask 1 fold, 2 check, 4 call, 8 agg), p_fold, p_check, p_call, p_agg f32 (renormalised
  over legal actions), p_act f32, surprise f32, p_agg_nf f32 (= p_agg / (1 - p_fold), aggression given continuing),
  half i8, clean bool (False = hand with a labelled-positive pair seated together).
Feature states (all FEATS + keys) stay in data/derived/policy_states/ for downstream reuse.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "6")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
import polars as pl
from numba import njit, prange

BASE = Path(__file__).resolve().parent.parent
DER = BASE / "data" / "derived"
STATES = DER / "policy_states"
MODELS = DER / "policy_models"
PREDS = DER / "policy_preds"
OUT = DER / "action_policy.parquet"
FOLDS_CSV = BASE / "data" / "folds_tables_5.csv"
SEED = 42
NTHREADS = 6
N_HANDS = 2_000_000
CHUNK_HANDS = 250_000
# street-stratified training sample rates per training half (half ~ 6.6M/1.43M/0.81M/0.46M actions by street)
STREET_RATE = {0: 0.25, 1: 0.70, 2: 0.75, 3: 1.00}
HOLDOUT_FRAC = 0.05

FEATS = [
    # 0-7 position / players
    "street", "pos_pf", "pos_post", "n_active", "n_can_act_oth", "n_allin_oth", "n_left_to_act", "n_after",
    # 8-16 money
    "to_call_bb", "pot_bb", "stack_bb", "pot_odds", "spr", "eff_stack_bb", "eff_spr", "tocall_frac_stack", "call_is_allin",
    # 17-27 street action history
    "n_agg_street", "n_agg_hand", "n_calls_street", "n_calls_since_agg", "n_checks_street", "bet_level_bb",
    "last_inc_bb", "last_agg_frac_pot", "last_aggr_pos_pf", "last_aggr_allin", "n_active_street_start",
    # 28-37 actor history
    "act_street_inv_bb", "act_hand_inv_bb", "act_n_act_street", "act_n_agg_street", "act_n_agg_hand",
    "act_init_prev", "act_is_pf_aggr", "n_agg_prev_street", "committed_frac", "bb",
    # 38-47 preflop strength (own cards)
    "pf_eq_vsN", "pf_eq_vs1", "pf_eq_vs5", "pf_top_vsN", "sklansky", "chen", "hi_rank", "lo_rank", "is_pair", "is_suited",
    # 48-64 postflop strength / board texture (own cards + board)
    "hs_actor", "hs_powN", "rank_norm", "own_cat", "board_cat", "hole_improves", "nboard", "board_max_suit",
    "board_mult", "board_high", "board_str_win", "top_pair", "overpair", "n_hole_match", "n_overcards",
    "flush_draw", "straight_draw",
    # 65
    "last_aggr_rel",
]
NF = len(FEATS)
assert NF == 66
CLASS_NAMES = ["fold", "check", "call", "agg"]
STREET_NAMES = ["preflop", "flop", "turn", "river"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================================ numba kernel
CAT_THR = np.array([0, 1277, 4137, 4995, 5853, 5863, 7140, 7296, 7452], dtype=np.int64)


@njit(cache=True, parallel=True)
def kernel(h0, h1, h_start, a_street, a_player, a_action, a_amount, a_pot, a_stack, a_tocall, a_pactive, a_hs,
           button, bbv, sbv, board, nboard, s_player, s_stack, s_c1, s_c2, s_pfc, s_rank,
           eqtab, toptab, skl, chen, F, Y, LAP, NAGG, LEGAL, chk):
    base = h_start[h0]
    nan = np.float32(np.nan)
    for h in prange(h0, h1):
        btn = np.int64(button[h])
        bb = np.int64(bbv[h])
        fbb = float(bb)
        folded = np.zeros(6, np.bool_)
        allin = np.zeros(6, np.bool_)
        acted = np.zeros(6, np.bool_)
        inv_st = np.zeros(6, np.int64)
        inv_hand = np.zeros(6, np.int64)
        stack_rem = np.zeros(6, np.int64)
        n_act_st_p = np.zeros(6, np.int64)
        n_agg_st_p = np.zeros(6, np.int64)
        n_agg_hand_p = np.zeros(6, np.int64)
        for s in range(6):
            stack_rem[s] = s_stack[h, s]
        sbs = (btn + 1) % 6
        bbs = (btn + 2) % 6
        x = min(np.int64(sbv[h]), stack_rem[sbs])
        inv_st[sbs] += x
        inv_hand[sbs] += x
        stack_rem[sbs] -= x
        x = min(bb, stack_rem[bbs])
        inv_st[bbs] += x
        inv_hand[bbs] += x
        stack_rem[bbs] -= x
        cur_bet = max(inv_st[sbs], inv_st[bbs])
        street = -1
        n_agg_st = 0
        n_agg_hand = 0
        n_calls_st = 0
        n_calls_since = 0
        n_checks_st = 0
        last_aggr = -1
        last_inc = fbb
        last_frac = 0.0
        last_allin = False
        prev_nagg = -1
        prev_aggr = -1
        pf_aggr = -1
        n_start = 6
        hc = h - h0
        for t in range(h_start[h], h_start[h + 1]):
            r = t - base
            st = np.int64(a_street[t])
            if st != street:
                if street >= 0:
                    prev_nagg = n_agg_st
                    prev_aggr = last_aggr
                if st > 0:
                    for s in range(6):
                        inv_st[s] = 0
                        acted[s] = False
                        n_act_st_p[s] = 0
                        n_agg_st_p[s] = 0
                    cur_bet = 0
                    n_agg_st = 0
                    n_calls_st = 0
                    n_calls_since = 0
                    n_checks_st = 0
                    last_aggr = -1
                    last_inc = 0.0
                    last_frac = 0.0
                    last_allin = False
                n_start = 0
                for s in range(6):
                    if not folded[s]:
                        n_start += 1
                street = st
            k = -1
            for s in range(6):
                if s_player[h, s] == a_player[t]:
                    k = s
                    break
            if k < 0:
                chk[hc, 0] += 1
                Y[r] = -1
                continue
            if st == 0:
                order = (k - btn + 9) % 6
            else:
                order = (k - btn + 11) % 6
            n_active = 0
            n_can = 0
            n_ai = 0
            n_left = 0
            n_after = 0
            others_max = 0
            for s in range(6):
                if folded[s]:
                    continue
                n_active += 1
                if s == k:
                    continue
                tot = stack_rem[s] + inv_st[s]
                if tot > others_max:
                    others_max = tot
                if allin[s]:
                    n_ai += 1
                    continue
                n_can += 1
                if (not acted[s]) or inv_st[s] < cur_bet:
                    n_left += 1
                if st == 0:
                    o2 = (s - btn + 9) % 6
                else:
                    o2 = (s - btn + 11) % 6
                if o2 > order:
                    n_after += 1
            tc = np.int64(a_tocall[t])
            pot = np.int64(a_pot[t])
            stk = np.int64(a_stack[t])
            an = a_action[t]
            amt = np.int64(a_amount[t])
            # consistency checks of the state machine against the logged fields
            my_tc = min(cur_bet - inv_st[k], stack_rem[k])
            if my_tc < 0:
                my_tc = 0
            if my_tc != tc:
                chk[hc, 1] += 1
            if stack_rem[k] != stk:
                chk[hc, 2] += 1
            if n_active != a_pactive[t]:
                chk[hc, 3] += 1
            fpot = max(float(pot), 1.0)
            F[r, 0] = st
            F[r, 1] = (k - btn + 9) % 6
            F[r, 2] = (k - btn + 11) % 6
            F[r, 3] = n_active
            F[r, 4] = n_can
            F[r, 5] = n_ai
            F[r, 6] = n_left
            F[r, 7] = n_after
            F[r, 8] = tc / fbb
            F[r, 9] = pot / fbb
            F[r, 10] = stk / fbb
            F[r, 11] = tc / max(float(pot + tc), 1.0)
            F[r, 12] = stk / fpot
            eff = min(stk + inv_st[k], others_max) - inv_st[k]
            if eff < 0:
                eff = 0
            F[r, 13] = eff / fbb
            F[r, 14] = eff / fpot
            F[r, 15] = tc / max(float(stk), 1.0)
            F[r, 16] = 1.0 if (tc > 0 and tc >= stk) else 0.0
            F[r, 17] = n_agg_st
            F[r, 18] = n_agg_hand
            F[r, 19] = n_calls_st
            F[r, 20] = n_calls_since
            F[r, 21] = n_checks_st
            F[r, 22] = cur_bet / fbb
            F[r, 23] = last_inc / fbb
            F[r, 24] = last_frac
            if last_aggr >= 0:
                F[r, 25] = (last_aggr - btn + 9) % 6
                F[r, 26] = 1.0 if last_allin else 0.0
                if st == 0:
                    oa = (last_aggr - btn + 9) % 6
                else:
                    oa = (last_aggr - btn + 11) % 6
                F[r, 65] = 1.0 if oa > order else -1.0
            else:
                F[r, 25] = -1.0
                F[r, 26] = 0.0
                F[r, 65] = 0.0
            F[r, 27] = n_start
            F[r, 28] = inv_st[k] / fbb
            F[r, 29] = inv_hand[k] / fbb
            F[r, 30] = n_act_st_p[k]
            F[r, 31] = n_agg_st_p[k]
            F[r, 32] = n_agg_hand_p[k]
            F[r, 33] = 1.0 if (st > 0 and prev_aggr == k) else 0.0
            F[r, 34] = 1.0 if (st > 0 and pf_aggr == k) else 0.0
            F[r, 35] = prev_nagg if st > 0 else -1
            F[r, 36] = inv_hand[k] / max(float(inv_hand[k] + stk), 1.0)
            F[r, 37] = fbb
            # ---- own preflop strength
            g = s_pfc[h, k]
            nopp = min(max(n_active - 1, 1), 5)
            F[r, 38] = eqtab[nopp, g]
            F[r, 39] = eqtab[1, g]
            F[r, 40] = eqtab[5, g]
            F[r, 41] = toptab[nopp, g]
            F[r, 42] = skl[g]
            F[r, 43] = chen[g]
            c1 = np.int64(s_c1[h, k])
            c2 = np.int64(s_c2[h, k])
            r1 = c1 // 4
            r2 = c2 // 4
            F[r, 44] = max(r1, r2)
            F[r, 45] = min(r1, r2)
            F[r, 46] = 1.0 if r1 == r2 else 0.0
            F[r, 47] = 1.0 if (c1 % 4) == (c2 % 4) else 0.0
            # ---- postflop strength + board texture
            nb = 0
            if st == 1:
                nb = 3
            elif st == 2:
                nb = 4
            elif st == 3:
                nb = 5
            if st > 0 and nboard[h] >= nb:
                hs = a_hs[t]
                F[r, 48] = hs
                F[r, 49] = hs ** max(n_active - 1, 1)
                rk = np.int64(s_rank[h, k, st - 1])
                own_cat = -1
                if rk >= 0:
                    F[r, 50] = rk / 7461.0
                    own_cat = 0
                    for q in range(9):
                        if rk >= CAT_THR[q]:
                            own_cat = q
                else:
                    F[r, 50] = nan
                F[r, 51] = own_cat
                rc = np.zeros(13, np.int64)
                sc = np.zeros(4, np.int64)
                bhigh = -1
                for i in range(nb):
                    c = np.int64(board[h, i])
                    rc[c // 4] += 1
                    sc[c % 4] += 1
                    if c // 4 > bhigh:
                        bhigh = c // 4
                mult = 0
                npair = 0
                for q in range(13):
                    if rc[q] > mult:
                        mult = rc[q]
                    if rc[q] >= 2:
                        npair += 1
                msuit = 0
                for q in range(4):
                    if sc[q] > msuit:
                        msuit = sc[q]
                bcat = 0
                if mult == 4:
                    bcat = 7
                elif mult == 3 and npair >= 2:
                    bcat = 6
                elif mult == 3:
                    bcat = 3
                elif npair >= 2:
                    bcat = 2
                elif npair == 1:
                    bcat = 1
                # straight windows over 14 slots (slot 0 = ace low, slot r+1 = rank r)
                bw = 0
                aw = 0
                for w in range(10):
                    cb = 0
                    ca = 0
                    for q in range(w, w + 5):
                        rr = 12 if q == 0 else q - 1
                        if rc[rr] > 0:
                            cb += 1
                            ca += 1
                        elif rr == r1 or rr == r2:
                            ca += 1
                    if cb > bw:
                        bw = cb
                    if ca > aw:
                        aw = ca
                if nb == 5 and bw == 5 and bcat < 4:
                    bcat = 4
                if nb == 5 and msuit == 5 and bcat < 5:
                    bcat = 5
                F[r, 52] = bcat
                F[r, 53] = 1.0 if own_cat > bcat else 0.0
                F[r, 54] = nb
                F[r, 55] = msuit
                F[r, 56] = mult
                F[r, 57] = bhigh
                F[r, 58] = bw
                F[r, 59] = 1.0 if (r1 != r2 and (r1 == bhigh or r2 == bhigh)) else 0.0
                F[r, 60] = 1.0 if (r1 == r2 and r1 > bhigh) else 0.0
                hm = 0
                if rc[r1] > 0:
                    hm += 1
                if r2 != r1 and rc[r2] > 0:
                    hm += 1
                F[r, 61] = hm
                ov = 0
                if r1 != r2:
                    if r1 > bhigh:
                        ov += 1
                    if r2 > bhigh:
                        ov += 1
                F[r, 62] = ov
                fd = 0.0
                if nb < 5:
                    for q in range(4):
                        hsu = (1 if c1 % 4 == q else 0) + (1 if c2 % 4 == q else 0)
                        if hsu >= 1 and sc[q] + hsu == 4:
                            fd = 1.0
                F[r, 63] = fd
                F[r, 64] = 1.0 if (nb < 5 and own_cat < 4 and aw >= 4 and bw < 4) else 0.0
            else:
                for q in range(48, 54):
                    F[r, q] = nan
                F[r, 54] = 0.0
                for q in range(55, 65):
                    F[r, q] = nan
            # ---- label + legal mask + bookkeeping outputs
            is_agg = an == 3 or an == 4 or (an == 5 and amt > tc)
            if an == 0:
                Y[r] = 0
            elif an == 1:
                Y[r] = 1
            elif is_agg:
                Y[r] = 3
            else:
                Y[r] = 2
            lg = 0
            if tc > 0:
                lg |= 1 | 4
            else:
                lg |= 2
            if stk > tc:
                lg |= 8
            LEGAL[r] = lg
            NAGG[r] = min(n_agg_st, 100)
            LAP[r] = s_player[h, last_aggr] if last_aggr >= 0 else -1
            # ---- state update (trust logged stack)
            stack_rem[k] = stk - amt
            inv_st[k] += amt
            inv_hand[k] += amt
            acted[k] = True
            n_act_st_p[k] += 1
            if an == 0:
                folded[k] = True
            if amt > 0 and stack_rem[k] <= 0:
                allin[k] = True
            if is_agg:
                nb2 = inv_st[k]
                last_inc = float(nb2 - cur_bet)
                last_frac = amt / fpot
                if nb2 > cur_bet:
                    cur_bet = nb2
                n_agg_st += 1
                n_agg_hand += 1
                n_agg_st_p[k] += 1
                n_agg_hand_p[k] += 1
                last_aggr = k
                last_allin = allin[k]
                n_calls_since = 0
                if st == 0:
                    pf_aggr = k
            elif an == 2 or an == 5:
                n_calls_st += 1
                n_calls_since += 1
            elif an == 1:
                n_checks_st += 1
    return 0


# ============================================================================================ stage: states
def _pf_tables():
    t = pl.read_parquet(DER / "preflop_equity_169.parquet").sort("grid_idx")
    assert t["grid_idx"].to_list() == list(range(169))
    eq = np.zeros((6, 169), np.float32)
    top = np.zeros((6, 169), np.float32)
    for k in range(1, 6):
        eq[k] = t[f"equity_vs{k}"].to_numpy()
        top[k] = t[f"top_pct_vs{k}"].to_numpy()
    return eq, top, t["sklansky_group"].to_numpy().astype(np.float32), t["chen"].to_numpy().astype(np.float32)


def _positive_pair_hands():
    """bool[N_HANDS]: hand has both players of some labelled-positive pair seated (any phase)."""
    lab = pl.read_parquet(DER / "labels.parquet").filter(pl.col("label") == 1).select("p1", "p2")
    players = pl.concat([lab["p1"], lab["p2"]]).unique()
    se = pl.scan_parquet(DER / "seats.parquet").select("hand_idx", "player_idx").filter(pl.col("player_idx").is_in(players.implode())).collect()
    sh = (lab.join(se.rename({"player_idx": "p1"}), on="p1")
          .join(se.rename({"player_idx": "p2"}), on=["p2", "hand_idx"]))
    m = np.zeros(N_HANDS, np.bool_)
    m[sh["hand_idx"].unique().to_numpy()] = True
    return m


def stage_states(max_parts="99"):
    t0 = time.time()
    STATES.mkdir(parents=True, exist_ok=True)
    A = pl.read_parquet(DER / "actions.parquet")
    N = A.height
    hand = A["hand_idx"].to_numpy()
    cnt = np.bincount(hand, minlength=N_HANDS)
    h_start = np.zeros(N_HANDS + 1, np.int64)
    np.cumsum(cnt, out=h_start[1:])
    arr = {c: A[c].to_numpy() for c in ["street", "player_idx", "action", "amount", "pot_before", "stack_before", "to_call", "players_active"]}
    action_no = A["action_no"].to_numpy()
    del A
    # perceived strength hs_actor aligned to actions
    hs_parts = [pl.read_parquet(DER / f"action_equity_{t}.parquet", columns=["hand_idx", "action_no", "hs_actor"]) for t in ("dev", "eval")]
    HS = pl.concat(hs_parts).sort("hand_idx", "action_no")
    del hs_parts
    assert HS.height == N
    assert np.array_equal(HS["hand_idx"].to_numpy(), hand) and np.array_equal(HS["action_no"].to_numpy(), action_no)
    a_hs = HS["hs_actor"].to_numpy().astype(np.float32)
    del HS
    log(f"loaded {N:,} actions + hs_actor in {time.time() - t0:.0f}s")
    H = pl.read_parquet(DER / "hands.parquet", columns=["hand_idx", "table_idx", "phase", "button_seat", "sb", "bb", "board"])
    button = H["button_seat"].to_numpy().astype(np.int8)
    bbv = H["bb"].to_numpy().astype(np.int32)
    sbv = H["sb"].to_numpy().astype(np.int32)
    phase = H["phase"].to_numpy().astype(np.int8)
    nboard = H["board"].list.len().to_numpy().astype(np.int8)
    board = np.full((N_HANDS, 5), -1, np.int8)
    ex = H.select("hand_idx", "board").explode("board").drop_nulls("board")
    ex = ex.with_columns((pl.int_range(pl.len()).over("hand_idx")).alias("pos"))
    board[ex["hand_idx"].to_numpy(), ex["pos"].to_numpy()] = ex["board"].to_numpy()
    del ex
    folds = pl.read_csv(FOLDS_CSV).with_columns(pl.col("table_idx").cast(pl.Int16))
    half_h = H.select("table_idx").join(folds, on="table_idx", how="left", maintain_order="left")["fold"].to_numpy() % 2
    half_h = half_h.astype(np.int8)
    table_h = H["table_idx"].to_numpy()
    del H
    S = pl.read_parquet(DER / "seats.parquet", columns=["player_idx", "starting_stack", "c1", "c2"])
    s_player = S["player_idx"].to_numpy().reshape(N_HANDS, 6)
    s_stack = S["starting_stack"].to_numpy().reshape(N_HANDS, 6)
    s_c1 = S["c1"].to_numpy().reshape(N_HANDS, 6)
    s_c2 = S["c2"].to_numpy().reshape(N_HANDS, 6)
    del S
    SS = pl.read_parquet(DER / "seat_strength.parquet", columns=["preflop_class", "rank_flop", "rank_turn", "final_rank_7"])
    s_pfc = SS["preflop_class"].to_numpy().reshape(N_HANDS, 6).astype(np.int16)
    s_rank = np.stack([SS[c].fill_null(-1).to_numpy().reshape(N_HANDS, 6) for c in ["rank_flop", "rank_turn", "final_rank_7"]], axis=2).astype(np.int16)
    del SS
    eqtab, toptab, skl, chen = _pf_tables()
    pos_hand = _positive_pair_hands()
    log(f"inputs ready {time.time() - t0:.0f}s; positive-pair hands {pos_hand.sum():,}")
    rng_h = np.random.default_rng(SEED)
    u_hand = rng_h.random(N_HANDS, dtype=np.float32)
    chk_tot = np.zeros(4, np.int64)
    part = 0
    for h0 in range(0, N_HANDS, CHUNK_HANDS):
        if part >= int(max_parts):
            break
        h1 = min(h0 + CHUNK_HANDS, N_HANDS)
        r0, r1 = h_start[h0], h_start[h1]
        n = r1 - r0
        F = np.empty((n, NF), np.float32)
        Y = np.full(n, -1, np.int8)
        LAP = np.full(n, -1, np.int32)
        NAGG = np.zeros(n, np.int8)
        LEGAL = np.zeros(n, np.int8)
        chk = np.zeros((h1 - h0, 4), np.int64)
        kernel(h0, h1, h_start, arr["street"], arr["player_idx"], arr["action"], arr["amount"], arr["pot_before"],
               arr["stack_before"], arr["to_call"], arr["players_active"], a_hs, button, bbv, sbv, board, nboard,
               s_player, s_stack, s_c1, s_c2, s_pfc, s_rank, eqtab, toptab, skl, chen, F, Y, LAP, NAGG, LEGAL, chk)
        chk_tot += chk.sum(0)
        hh = hand[r0:r1]
        rng_a = np.random.default_rng(SEED + 1000 + part)
        cols = {
            "hand_idx": hand[r0:r1], "action_no": action_no[r0:r1], "player_idx": arr["player_idx"][r0:r1],
            "y": Y, "last_aggr_player": LAP, "n_agg_st_i8": NAGG, "legal": LEGAL,
            "phase": phase[hh], "half": half_h[hh], "table_idx": table_h[hh], "clean": ~pos_hand[hh],
            "u_hand": u_hand[hh], "u_act": rng_a.random(n, dtype=np.float32),
        }
        df = pl.DataFrame(cols)
        df = df.hstack([pl.Series(FEATS[j], F[:, j]) for j in range(NF)])
        del F
        df.write_parquet(STATES / f"part_{part:02d}.parquet", compression="zstd", row_group_size=500_000)
        log(f"part {part} hands {h0}-{h1} rows {n:,} ({time.time() - t0:.0f}s)")
        del df
        part += 1
    log(f"state-machine checks over {N:,} actions: unknown actor {chk_tot[0]}, to_call mismatch {chk_tot[1]}, "
        f"stack mismatch {chk_tot[2]}, players_active mismatch {chk_tot[3]}")
    (STATES / "checks.json").write_text(json.dumps({"n_actions": int(N), "unknown_actor": int(chk_tot[0]),
                                                    "to_call_mismatch": int(chk_tot[1]), "stack_mismatch": int(chk_tot[2]),
                                                    "players_active_mismatch": int(chk_tot[3])}))
    log(f"states done {time.time() - t0:.0f}s")


# ============================================================================================ stage: train
def _lgb_params():
    return dict(objective="multiclass", num_class=4, learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
                feature_fraction=0.9, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, max_bin=255,
                num_threads=NTHREADS, seed=SEED, verbose=-1)


def _sample_expr():
    rate = pl.lit(1.0)
    for s, v in STREET_RATE.items():
        rate = pl.when(pl.col("street") == s).then(pl.lit(v)).otherwise(rate)
    return pl.col("u_act") < rate


def stage_train(max_rounds="600"):
    import lightgbm as lgb
    MODELS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta = {}
    for h in (0, 1):
        # model "h" is trained on tables of half 1-h and predicts half h
        lf = pl.scan_parquet(STATES / "part_*.parquet")
        tr = (lf.filter((pl.col("half") != h) & pl.col("clean") & (pl.col("y") >= 0) & _sample_expr())
              .select(["y", "u_hand"] + FEATS).collect())
        va_mask = (tr["u_hand"] < HOLDOUT_FRAC).to_numpy()
        y = tr["y"].to_numpy().astype(np.int32)
        X = tr.select(FEATS).to_numpy()
        assert X.dtype == np.float32
        street_counts = tr.group_by("street").len().sort("street").to_dicts()
        del tr
        log(f"[model {h}] train rows {(~va_mask).sum():,} holdout rows {va_mask.sum():,} by street {street_counts} "
            f"class shares {np.bincount(y, minlength=4) / len(y)} ({time.time() - t0:.0f}s)")
        dtr = lgb.Dataset(X[~va_mask], y[~va_mask], feature_name=FEATS, free_raw_data=True)
        dva = lgb.Dataset(X[va_mask], y[va_mask], reference=dtr, free_raw_data=True)
        del X
        evals = {}
        bst = lgb.train(_lgb_params(), dtr, int(max_rounds), valid_sets=[dva], valid_names=["holdout"],
                        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50), lgb.record_evaluation(evals)])
        bst.save_model(str(MODELS / f"policy_half{h}.txt"))
        imp = sorted(zip(FEATS, bst.feature_importance("gain")), key=lambda z: -z[1])
        tot = sum(v for _, v in imp)
        meta[h] = dict(best_iter=bst.best_iteration, holdout_logloss=float(evals["holdout"]["multi_logloss"][bst.best_iteration - 1]),
                       n_train=int((~va_mask).sum()), n_holdout=int(va_mask.sum()), street_counts=street_counts,
                       top_gain=[(f, round(v / tot, 4)) for f, v in imp[:25]])
        log(f"[model {h}] best_iter {bst.best_iteration} holdout mlogloss {meta[h]['holdout_logloss']:.4f} ({time.time() - t0:.0f}s)")
        log(f"[model {h}] top gain: {meta[h]['top_gain'][:15]}")
        del dtr, dva, bst
    meta["train_seconds"] = time.time() - t0
    (MODELS / "train_meta.json").write_text(json.dumps(meta, indent=1))


# ============================================================================================ stage: predict
def renorm(P, legal):
    P = P.astype(np.float64)
    for j, bit in enumerate((1, 2, 4, 8)):
        P[(legal & bit) == 0, j] = 0.0
    s = P.sum(1, keepdims=True)
    bad = s[:, 0] <= 0
    if bad.any():  # should not happen; fall back to uniform over legal
        L = np.stack([(legal[bad] & b) > 0 for b in (1, 2, 4, 8)], 1).astype(np.float64)
        P[bad] = L / L.sum(1, keepdims=True)
        s = P.sum(1, keepdims=True)
    return P / s


def stage_predict():
    import lightgbm as lgb
    PREDS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    models = {h: lgb.Booster(model_file=str(MODELS / f"policy_half{h}.txt")) for h in (0, 1)}
    parts = sorted(STATES.glob("part_*.parquet"))
    for p in parts:
        D = pl.read_parquet(p, columns=["hand_idx", "action_no", "player_idx", "y", "n_agg_st_i8", "last_aggr_player",
                                        "legal", "half", "clean"] + FEATS)
        n = D.height
        P = np.zeros((n, 4), np.float64)
        half = D["half"].to_numpy()
        for h in (0, 1):
            idx = np.flatnonzero(half == h)
            if len(idx) == 0:
                continue
            X = D.select(FEATS)[idx].to_numpy()
            bst = models[h]
            P[idx] = bst.predict(X, num_iteration=bst.best_iteration, num_threads=NTHREADS)
            del X
        legal = D["legal"].to_numpy().astype(np.int64)
        P = renorm(P, legal)
        y = D["y"].to_numpy().astype(np.int64)
        yy = np.clip(y, 0, 3)
        p_act = P[np.arange(n), yy]
        out = D.select("hand_idx", "action_no", "player_idx", pl.col("street").cast(pl.Int8), pl.col("y"),
                       pl.col("n_agg_st_i8").alias("n_agg_street"), "last_aggr_player", "to_call_bb", "legal").with_columns(
            pl.Series("p_fold", P[:, 0].astype(np.float32)), pl.Series("p_check", P[:, 1].astype(np.float32)),
            pl.Series("p_call", P[:, 2].astype(np.float32)), pl.Series("p_agg", P[:, 3].astype(np.float32)),
            pl.Series("p_act", p_act.astype(np.float32)),
            pl.Series("surprise", (-np.log(np.clip(p_act, 1e-6, 1.0))).astype(np.float32)),
            pl.Series("p_agg_nf", (P[:, 3] / np.clip(1.0 - P[:, 0], 1e-9, None)).astype(np.float32)),
            D["half"], D["clean"])
        out.write_parquet(PREDS / p.name, compression="zstd")
        log(f"predicted {p.name} rows {n:,} ({time.time() - t0:.0f}s)")
        del D, P, out
    pl.scan_parquet(PREDS / "part_*.parquet").sink_parquet(OUT, compression="zstd", row_group_size=1_000_000)
    n = pl.scan_parquet(OUT).select(pl.len()).collect().item()
    log(f"wrote {OUT} rows {n:,} ({time.time() - t0:.0f}s)")


# ============================================================================================ stage: report
def stage_report():
    t0 = time.time()
    lf = pl.scan_parquet(OUT)
    ph = pl.scan_parquet(STATES / "part_*.parquet").select("hand_idx", "action_no", "phase")
    base = lf.with_columns(
        pl.max_horizontal("p_fold", "p_check", "p_call", "p_agg").alias("pmax"),
        pl.concat_list("p_fold", "p_check", "p_call", "p_agg").list.arg_max().cast(pl.Int8).alias("pred"))
    res = {}
    by_street = (base.group_by("street").agg(pl.len().alias("n"), pl.col("surprise").mean().alias("logloss"),
                                             (pl.col("pred") == pl.col("y")).mean().alias("accuracy"))
                 .sort("street").collect())
    # marginal baseline: class frequencies per (street, facing, n_agg_street clipped 3)
    ctx = [pl.col("street"), (pl.col("to_call_bb") > 0).alias("facing"), pl.col("n_agg_street").clip(0, 3).alias("nagg")]
    marg = (lf.with_columns(ctx).group_by("street", "facing", "nagg", "y").agg(pl.len().alias("c"))
            .with_columns((pl.col("c") / pl.col("c").sum().over("street", "facing", "nagg")).alias("pm"))
            .collect())
    bl = (lf.with_columns(ctx).join(marg.lazy().select("street", "facing", "nagg", "y", "pm"), on=["street", "facing", "nagg", "y"])
          .group_by("street").agg((-pl.col("pm").log()).mean().alias("baseline_logloss")).sort("street").collect())
    by_street = by_street.join(bl, on="street")
    allrow = base.select(pl.len().alias("n"), pl.col("surprise").mean().alias("logloss"),
                         (pl.col("pred") == pl.col("y")).mean().alias("accuracy")).collect()
    print("held-out (cross-fitted) per street:\n", by_street)
    print("all actions:", allrow.to_dicts())
    res["by_street"] = by_street.to_dicts()
    res["all"] = allrow.to_dicts()[0]
    by_phase = (base.join(ph, on=["hand_idx", "action_no"]).group_by("phase", "street")
                .agg(pl.len().alias("n"), pl.col("surprise").mean().alias("logloss"), (pl.col("pred") == pl.col("y")).mean().alias("accuracy"))
                .sort("phase", "street").collect())
    print("by phase x street:\n", by_phase)
    res["by_phase_street"] = by_phase.to_dicts()
    by_clean = (base.group_by("clean", "street").agg(pl.len().alias("n"), pl.col("surprise").mean().alias("logloss"))
                .sort("clean", "street").collect())
    print("clean vs positive-pair hands:\n", by_clean)
    res["by_clean_street"] = by_clean.to_dicts()
    # calibration: p_call when facing a preflop raise
    edges = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    cal = (lf.filter((pl.col("street") == 0) & (pl.col("n_agg_street") >= 1) & (pl.col("to_call_bb") > 0))
           .with_columns(pl.col("p_call").cut(edges[1:-1], left_closed=True).alias("bin"))
           .group_by("bin").agg(pl.len().alias("n"), pl.col("p_call").mean().alias("mean_p_call"),
                                (pl.col("y") == 2).mean().alias("obs_call_rate"))
           .sort("mean_p_call").collect())
    print("calibration p_call facing preflop raise:\n", cal)
    res["calibration_pfcall"] = [{k: (str(v) if k == "bin" else v) for k, v in d.items()} for d in cal.to_dicts()]
    cal2 = (lf.filter(pl.col("street") > 0)
            .with_columns(pl.col("p_agg").cut(edges[1:-1], left_closed=True).alias("bin"))
            .group_by("bin").agg(pl.len().alias("n"), pl.col("p_agg").mean().alias("mean_p_agg"), (pl.col("y") == 3).mean().alias("obs_agg_rate"))
            .sort("mean_p_agg").collect())
    print("calibration p_agg postflop:\n", cal2)
    res["calibration_postflop_agg"] = [{k: (str(v) if k == "bin" else v) for k, v in d.items()} for d in cal2.to_dicts()]
    res["checks"] = json.loads((STATES / "checks.json").read_text())
    res["train_meta"] = json.loads((MODELS / "train_meta.json").read_text())
    (MODELS / "report.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"report done {time.time() - t0:.0f}s")


# ============================================================================================ stage: validate
def _auc_ap(pos, neg):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg].astype(float)
    return roc_auc_score(y, s), average_precision_score(y, s)


def stage_validate(thr="0.15"):
    thr = float(thr)
    t0 = time.time()
    lab = pl.read_parquet(DER / "labels.parquet").select("pair_id", "p1", "p2", "label", "behavior_family")
    players = pl.concat([lab["p1"], lab["p2"]]).unique()
    dev = pl.scan_parquet(DER / "hands.parquet").filter(pl.col("phase") == 0).select("hand_idx")
    se = (pl.scan_parquet(DER / "seats.parquet").select("hand_idx", "player_idx")
          .filter(pl.col("player_idx").is_in(players.implode())).join(dev, on="hand_idx").collect())
    shared = (lab.join(se.rename({"player_idx": "p1"}), on="p1").join(se.rename({"player_idx": "p2"}), on=["p2", "hand_idx"]))
    n_shared = shared.group_by("pair_id").len("shared")
    hands = shared["hand_idx"].unique()
    pol = (pl.scan_parquet(OUT).filter(pl.col("hand_idx").is_in(hands.implode()))
           .select("hand_idx", "action_no", "player_idx", "street", "y", "n_agg_street", "last_aggr_player", "to_call_bb",
                   "p_fold", "p_check", "p_call", "p_agg", "surprise").collect())
    res = {}
    lines = []
    # ---------------- DT: pfcall_weak_max
    for t in sorted({0.10, thr, 0.20}):
        wc = pol.filter((pl.col("street") == 0) & (pl.col("y") == 2) & (pl.col("n_agg_street") >= 1)
                        & (pl.col("last_aggr_player") >= 0) & (pl.col("p_call") <= t)).select("hand_idx", "player_idx", "last_aggr_player")
        ab = shared.join(wc, left_on=["hand_idx", "p1", "p2"], right_on=["hand_idx", "player_idx", "last_aggr_player"]).group_by("pair_id").agg(pl.col("hand_idx").n_unique().alias("wAB"))
        ba = shared.join(wc, left_on=["hand_idx", "p2", "p1"], right_on=["hand_idx", "player_idx", "last_aggr_player"]).group_by("pair_id").agg(pl.col("hand_idx").n_unique().alias("wBA"))
        PP = (lab.join(n_shared, on="pair_id", how="left").join(ab, on="pair_id", how="left").join(ba, on="pair_id", how="left")
              .with_columns(pl.col("shared", "wAB", "wBA").fill_null(0))
              .with_columns(pl.max_horizontal("wAB", "wBA").alias("pfcall_weak_max"))
              .with_columns((pl.col("pfcall_weak_max") / pl.col("shared").clip(1)).alias("pfcall_weak_max_rate")))
        if t == thr:
            PP.write_parquet(MODELS / "validate_pfcall_weak_pairs.parquet")
        for c in ["pfcall_weak_max", "pfcall_weak_max_rate"]:
            dt = PP.filter(pl.col("behavior_family") == "directed_transfer")[c].to_numpy()
            ng = PP.filter(pl.col("label") == 0)[c].to_numpy()
            spci = PP.filter(pl.col("behavior_family").is_in(["soft_play", "coordinated_isolation"]))[c].to_numpy()
            a1, p1 = _auc_ap(dt, ng)
            a2, p2 = _auc_ap(dt, spci)
            a3, p3 = _auc_ap(spci, ng)
            key = f"{c}@{t:.2f}"
            res[key] = dict(auc_dt_vs_neg=a1, ap_dt_vs_neg=p1, auc_dt_vs_spci=a2, ap_dt_vs_spci=p2, auc_spci_vs_neg=a3)
            lines.append(f"{key:28s} DT vs NEG AUC {a1:.4f} AP {p1:.4f} | DT vs SP+CI AUC {a2:.4f} AP {p2:.4f} | SP+CI vs NEG AUC {a3:.4f}")
        if t == thr:
            g = (PP.with_columns(pl.when(pl.col("label") == 0).then(pl.lit("NEG")).otherwise(pl.col("behavior_family")).alias("grp"))
                 .group_by("grp").agg(pl.len(), pl.col("shared").mean().alias("shared_mean"),
                                      pl.col("pfcall_weak_max").mean().alias("mean"), (pl.col("pfcall_weak_max") >= 3).mean().alias("ge3"),
                                      (pl.col("pfcall_weak_max") >= 5).mean().alias("ge5")).sort("grp"))
            print(g)
            res["pfcall_weak_groups"] = g.to_dicts()
    # role-level: call / weak-call rates facing partner's preflop raise
    ev = pl.read_parquet(DER / "evidence.parquet").select("pair_id", "hand_idx").with_columns(pl.lit(True).alias("isev"))
    resp = pl.concat([
        shared.join(pol, left_on=["hand_idx", X], right_on=["hand_idx", "player_idx"]).filter(pl.col("last_aggr_player") == pl.col(Yp))
        .with_columns(pl.lit(X).alias("actor")) for X, Yp in (("p1", "p2"), ("p2", "p1"))], how="diagonal_relaxed")
    resp = resp.join(ev, on=["pair_id", "hand_idx"], how="left").with_columns(pl.col("isev").fill_null(False))
    resp = resp.with_columns(pl.when(pl.col("label") == 0).then(pl.lit("NEG"))
                             .otherwise(pl.col("behavior_family") + pl.when(pl.col("isev")).then(pl.lit(":ev")).otherwise(pl.lit(":non"))).alias("grp"))
    pf_tab = (resp.filter(pl.col("street") == 0).group_by("grp")
              .agg(pl.len().alias("spots"), (pl.col("y") == 2).mean().alias("call"), (pl.col("y") == 3).mean().alias("reraise"),
                   ((pl.col("y") == 2) & (pl.col("p_call") <= thr)).mean().alias("weak_call"),
                   pl.col("p_call").mean().alias("mean_p_call"), pl.col("surprise").mean().alias("mean_surprise"))
              .sort("grp"))
    print("responses to partner's preflop raise:\n", pf_tab)
    res["pf_partner_response"] = pf_tab.to_dicts()
    # ---------------- SP: passed-up raises at responses to partner aggression
    passive = pl.col("y").is_in([0, 2])
    sp_tab = (resp.group_by("grp")
              .agg(pl.len().alias("responses"), pl.col("p_agg").mean().alias("mean_p_agg_all"),
                   pl.col("p_agg").filter(passive).mean().alias("mean_p_agg_passive"),
                   pl.col("p_agg").filter(passive & (pl.col("street") > 0)).mean().alias("mean_p_agg_passive_post"),
                   (pl.col("y") == 3).mean().alias("obs_agg_rate"),
                   pl.col("surprise").filter(passive).mean().alias("mean_surprise_passive"))
              .sort("grp"))
    print("responses to partner aggression (SP check):\n", sp_tab)
    res["sp_partner_response"] = sp_tab.to_dicts()
    # hand-level: max passed-up p_agg among passive responses (policy analogue of forensics max_miss)
    hand_miss = (resp.filter(passive).group_by("pair_id", "hand_idx", "grp").agg(pl.col("p_agg").max().alias("max_miss"),
                                                                                  pl.col("p_agg").sum().alias("sum_miss")))
    allh = (shared.join(ev, on=["pair_id", "hand_idx"], how="left").with_columns(pl.col("isev").fill_null(False))
            .with_columns(pl.when(pl.col("label") == 0).then(pl.lit("NEG"))
                          .otherwise(pl.col("behavior_family") + pl.when(pl.col("isev")).then(pl.lit(":ev")).otherwise(pl.lit(":non"))).alias("grp"))
            .join(hand_miss.drop("grp"), on=["pair_id", "hand_idx"], how="left").with_columns(pl.col("max_miss", "sum_miss").fill_null(0.0)))
    for c in ["max_miss", "sum_miss"]:
        spev = allh.filter(pl.col("grp") == "soft_play:ev")[c].to_numpy()
        neg = allh.filter(pl.col("grp") == "NEG")[c].to_numpy()
        spnon = allh.filter(pl.col("grp") == "soft_play:non")[c].to_numpy()
        a1, p1 = _auc_ap(spev, neg)
        a2, _ = _auc_ap(spev, spnon)
        res[f"sp_hand_{c}"] = dict(auc_spev_vs_neg_hands=a1, ap_spev_vs_neg_hands=p1, auc_spev_vs_spnon=a2,
                                   mean_spev=float(spev.mean()), mean_neg=float(neg.mean()))
        lines.append(f"SP hand {c:9s}: SP_ev vs NEG hands AUC {a1:.4f} (AP {p1:.4f}); SP_ev vs SP_non AUC {a2:.4f}; mean {spev.mean():.3f} vs NEG {neg.mean():.3f}")
    print("\n".join(lines))
    res["lines"] = lines
    (MODELS / "validate.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"validate done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    args = sys.argv[2:]
    if stage == "all":
        for f in (stage_states, stage_train, stage_predict, stage_report, stage_validate):
            f()
    else:
        {"states": stage_states, "train": stage_train, "predict": stage_predict, "report": stage_report,
         "validate": stage_validate}[stage](*args)
