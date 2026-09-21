"""exp013: hunt for the hidden 4th mechanism (other_coordination, OC).

Stages (python scripts/oc_hunt.py <stage>):
  events   numba pass over all hands of both phases -> directed (X acts, Y seated) event counts + opportunity counts
           per ordered within-table pair, windows: 0 dev_full, 1 dev_w2000 (hand_seq<2000), 2 eval.
           -> data/derived/oc/events_{ev,op,sm}.npy  [W, 400, 30, 30, E]
  stats    own-baseline (IPF row x column log-linear expectation) lift, Poisson signed LLR G, row/col tops, mutual tops
           per unordered pair and event -> data/derived/oc/pairstats_w{W}.parquet (+ Poisson null replicate)
  excess   tail counts per event/statistic in groups (dev candidates / negatives / positives by family / eval
           candidates / Poisson null) -> research/forensics/oc_excess_*.csv
  print    print full hands for pairs (python scripts/oc_hunt.py print <pA> <pB> <phase> [k])
Label-free everywhere except group definitions (labels only used to define the null/reference groups).
Rules: only gameplay (actions, cards, seats, hand_seq/lineups); no ID formats / row order.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "3")
os.environ.setdefault("NUMBA_NUM_THREADS", "3")
import numpy as np
import polars as pl
from numba import njit, prange

BASE = Path(__file__).resolve().parent.parent
DER = BASE / "data" / "derived"
OC = DER / "oc"
OC.mkdir(exist_ok=True)
FOR = BASE / "research" / "forensics"
col = pl.col

# ------------------------------------------------------------------------------------------- event catalogue
# directed: X = actor, Y = co-seated player. (opportunity definition | event)
EVENTS = [
    ("shared", "both dealt | 1"),
    ("pf_vol", "shared | X voluntarily enters preflop"),
    ("pf_raise", "shared | X raises preflop"),
    ("pf_allin", "shared | X all-in preflop"),
    ("inf_fold_Ystrong", "X first pf decision (to_call>0) before Y acts, Y top15% | X folds"),
    ("inf_fold_Yweak", "same, Y bottom50% | X folds"),
    ("inf_raise_Ystrong", "same, Y top15% | X raises"),
    ("inf_raise_Yweak", "same, Y bottom50% | X raises"),
    ("bhp_fold_Ybetter", "X first pf decision before Y, X top40%, HU pf eq X<.45 vs Y | X folds"),
    ("bhp_fold_Yworse", "same, HU pf eq X>.55 | X folds"),
    ("pf_enter_afterY", "X first pf decision with Y already voluntarily in | X enters"),
    ("pf_limpbehind", "X first pf decision, unraised, Y limped | X calls"),
    ("pf_iso_overY", "X first pf decision, unraised, Y limped | X raises"),
    ("pf_overcall_3rd_Yin", "X faces 3rd raise pf, Y vol in and active | X calls"),
    ("pf_fold_3rd_Yin", "same | X folds"),
    ("pf_squeeze_3rd_Yin", "same | X raises"),
    ("pf_fold_3rd_Ybehind", "X faces 3rd raise pf, Y still to act | X folds"),
    ("pf_call_Yraise", "X faces Y raise pf | X calls"),
    ("pf_3bet_Yraise", "X faces Y raise pf | X raises"),
    ("pf_fold_Yraise", "X faces Y raise pf | X folds"),
    ("bvb_walk", "X SB folded to, Y BB | X folds (walk to Y)"),
    ("bvb_raise", "X SB folded to, Y BB | X raises"),
    ("blind_fold_steal", "X in blind faces Y unopened late open (no callers) | X folds"),
    ("steal_open_Yblind", "X CO/BTN/SB folded to, Y in blind to act | X raises"),
    ("mw_bet_Yactive", "X postflop to_call=0, Y active, >=3 active | X bets"),
    ("mw_fold_3rd", "X postflop faces 3rd bet, Y active | X folds"),
    ("mw_call_3rd", "same | X calls"),
    ("mw_raise_3rd", "same | X raises"),
    ("mw_fold_3rd_eq60", "same | X folds with HU eq vs bettor >= .6"),
    ("fold_Yahead", "X postflop faces 3rd bet, Y active, X HU eq vs Y <= .35 | X folds"),
    ("fold_Ybehind", "same, X HU eq vs Y >= .65 | X folds"),
    ("pump_Yahead", "X postflop decision, Y + 3rd active, X eq vs Y <= .25 | X aggr"),
    ("aggr_Ybehind", "X postflop decision, Y + 3rd active, X eq vs Y >= .75 | X aggr"),
    ("mw_checkthrough", "postflop street w/ actions, X,Y active at start, >=3 active | no aggression (symmetric)"),
    ("mw_strongcheck", "X postflop to_call=0 hs>=.8, Y + 3rd active | X checks"),
    ("hu_strongcheck", "X postflop to_call=0 hs>=.8, HU vs Y | X checks"),
    ("hu_bet", "X postflop to_call=0 HU vs Y | X bets"),
    ("hu_fold", "X postflop faces Y bet HU | X folds"),
    ("hu_raise", "X postflop faces Y bet HU | X raises"),
    ("flow", "shared | X net<0 and Y net>0"),
    ("flow_big", "shared | X net<=-10bb and Y net>=10bb"),
    ("imp5", "shared | Mazrooei impact of X actions on Y >= 5bb"),
    ("impneg5", "shared | impact of X actions on Y <= -5bb"),
    ("both_sd", "both see flop | both at showdown (symmetric)"),
    ("foldpost_Ysd", "both see flop | X folds postflop and Y goes to showdown"),
    ("betfold_Y", "X aggr then faces Y raise same street | X folds"),
    ("pot3_Ywins", "X contrib>=3bb, 3rd contrib>=3bb, Y vol | X net<0 and Y net>0"),
    ("seat_next", "shared | Y sits directly after X"),
    ("cojoin", "X newly seated this hand | Y newly seated too"),
    ("coleave", "X leaves after this hand | Y leaves too"),
    ("sd_win_vsY", "both at showdown | X net>0 and Y net<0"),
    ("call_Yallin", "X faces Y all-in | X calls"),
    ("foldeq50_Yactive", "X folds postflop with Y active | X multiway eq >= .5"),
    ("hu_smallbet", "X bets HU vs Y postflop | size <= .35 pot"),
    ("mw_bigbet", "X aggr postflop with Y + 3rd active | amount >= pot"),
    ("pf_fold_first_Yin", "X first pf decision, Y vol in | X folds"),
    ("pf_raise_first_Yin", "X first pf decision, Y vol in | X raises"),
    ("hu_checkdown", "HU postflop street X,Y only active | street checked through (symmetric)"),
    ("pre_both_vol", "shared | both voluntarily in preflop (symmetric)"),
    ("third_folds_pair", "X aggr facing 3rd, Y active | the 3rd folds to X"),
    ("gift_to_Y", "X faces Y aggr | X fold with eq*(pot+tc)-tc>=2bb or call with amt-eq*(pot+amt)>=2bb"),
    ("allin_Yactive", "X aggr all-in any street with Y active | 1 (count over shared)"),
    ("limp_first", "X first pf decision unraised, nobody in | X limps"),
]
NE = len(EVENTS)
EI = {n: i for i, (n, _) in enumerate(EVENTS)}
NK = 3  # sums: imp X->Y (bb), transfer X->Y (bb, flow hands), gift X->Y (bb)
W_NAMES = ["dev_full", "dev_w2000a", "eval", "dev_w2000b"]   # window index -> pf_<name>.parquet window
NW = len(W_NAMES)

PI = np.full((6, 6), -1, dtype=np.int64)
_k = 0
for _i in range(6):
    for _j in range(_i + 1, 6):
        PI[_i, _j] = PI[_j, _i] = _k
        _k += 1

(E_SHARED, E_PF_VOL, E_PF_RAISE, E_PF_ALLIN, E_INF_FOLD_YS, E_INF_FOLD_YW, E_INF_RAISE_YS, E_INF_RAISE_YW,
 E_BHP_FOLD_YB, E_BHP_FOLD_YW, E_PF_ENTER_AFTERY, E_PF_LIMPBEHIND, E_PF_ISO, E_PF_OVERCALL3, E_PF_FOLD3, E_PF_SQUEEZE3,
 E_PF_FOLD3_YBEHIND, E_PF_CALL_YR, E_PF_3BET_YR, E_PF_FOLD_YR, E_BVB_WALK, E_BVB_RAISE, E_BLIND_FOLD_STEAL,
 E_STEAL_OPEN, E_MW_BET, E_MW_FOLD3, E_MW_CALL3, E_MW_RAISE3, E_MW_FOLD3_EQ60, E_FOLD_YAHEAD, E_FOLD_YBEHIND,
 E_PUMP_YAHEAD, E_AGGR_YBEHIND, E_MW_CHECKTHROUGH, E_MW_STRONGCHECK, E_HU_STRONGCHECK, E_HU_BET, E_HU_FOLD,
 E_HU_RAISE, E_FLOW, E_FLOW_BIG, E_IMP5, E_IMPNEG5, E_BOTH_SD, E_FOLDPOST_YSD, E_BETFOLD_Y, E_POT3_YWINS,
 E_SEAT_NEXT, E_COJOIN, E_COLEAVE, E_SD_WIN, E_CALL_YALLIN, E_FOLDEQ50, E_HU_SMALLBET, E_MW_BIGBET,
 E_PF_FOLD_FIRST_YIN, E_PF_RAISE_FIRST_YIN, E_HU_CHECKDOWN, E_PRE_BOTH_VOL, E_THIRD_FOLDS_PAIR, E_GIFT_TO_Y,
 E_ALLIN_YACTIVE, E_LIMP_FIRST) = range(NE)


@njit(cache=True, inline="always")
def class_id(c1, c2):
    r1, r2 = c1 // 4, c2 // 4
    hi, lo = max(r1, r2), min(r1, r2)
    if r1 == r2:
        return r1 * 13 + r1
    if c1 % 4 == c2 % 4:
        return hi * 13 + lo
    return lo * 13 + hi


@njit(cache=True, inline="always")
def hueq(hu, r, st, x, y, PI):
    """HU equity of seat x vs seat y on street st (NaN if not both active at street start)."""
    he = hu[r, st, PI[x, y]]
    if x > y:
        he = 1.0 - he
    return he


@njit(cache=True)
def fill_hand(r, hp, hn, s_player, s_c1, s_c2, s_net, s_sd, s_contrib, h_button, h_bb, h_nboard, a_start, a_end,
              a_street, a_player, a_action, a_amount, a_to_call, a_pot, eq_pre, eq_post, hs_act, hu, tp169, PI,
              hev, hop, hsm):
        hev[:] = 0
        hop[:] = 0
        hsm[:] = 0.0
        so = r * 6
        bb = h_bb[r]
        pl_ = np.empty(6, dtype=np.int64)
        pos = np.empty(6, dtype=np.int64)
        tp = np.empty(6)
        for s in range(6):
            pl_[s] = s_player[so + s]
            pos[s] = (s - h_button[r] + 6) % 6          # 0 BTN 1 SB 2 BB 3 UTG 4 HJ 5 CO
            tp[s] = tp169[class_id(s_c1[so + s], s_c2[so + s])]
        active = np.ones(6, dtype=np.int64)
        acted_pre = np.zeros(6, dtype=np.int64)
        vol = np.zeros(6, dtype=np.int64)
        limped = np.zeros(6, dtype=np.int64)
        pfr = np.zeros(6, dtype=np.int64)
        allin_pre = np.zeros(6, dtype=np.int64)
        fold_st = np.full(6, 9, dtype=np.int64)
        aggr_st = np.zeros(6, dtype=np.int64)
        st_start = np.zeros(6, dtype=np.int64)
        imp = np.zeros((6, 6))
        nvol = 0
        open_raiser = -1
        open_unopened = 0
        calls_after_open = 0
        cur_st = -1
        L = -1
        nr = 0
        st_aggr = 0
        st_nact = 0
        last_allin = 0
        for k in range(a_start[r], a_end[r] + 1):
            # ---- street close / open (k == a_end[r] is a sentinel close)
            st = a_street[k] if k < a_end[r] else 99
            if st != cur_st:
                if cur_st >= 1:
                    n0 = 0
                    for s in range(6):
                        n0 += st_start[s]
                    for i in range(6):
                        for j in range(6):
                            if i != j and st_start[i] == 1 and st_start[j] == 1:
                                if n0 >= 3:
                                    hop[i, j, E_MW_CHECKTHROUGH] = 1
                                    if st_aggr == 0:
                                        hev[i, j, E_MW_CHECKTHROUGH] = 1
                                elif n0 == 2:
                                    hop[i, j, E_HU_CHECKDOWN] = 1
                                    if st_aggr == 0:
                                        hev[i, j, E_HU_CHECKDOWN] = 1
                if k == a_end[r]:
                    break
                cur_st = st
                L = -1
                nr = 0
                st_aggr = 0
                last_allin = 0
                for s in range(6):
                    aggr_st[s] = 0
                    st_start[s] = active[s]
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
            isfold = act == 0
            nact = 0
            for s in range(6):
                nact += active[s]
            # Mazrooei impact of X's action on each other seat
            for j in range(6):
                if j == x:
                    continue
                e0 = eq_pre[k, j]
                e1 = eq_post[k, j]
                if e0 != e0:
                    e0 = 0.0
                if e1 != e1:
                    e1 = 0.0
                imp[x, j] += ((e1 - e0) * P + e1 * amt) / bb
            eqx = eq_pre[k, x]
            if eqx != eqx:
                eqx = 0.0
            # gift X -> L when facing L's aggression
            if L >= 0 and L != x and tc > 0:
                g = 0.0
                if isfold:
                    g = eqx * (P + tc) - tc
                elif iscall:
                    g = amt - eqx * (P + amt)
                hop[x, L, E_GIFT_TO_Y] = 1
                if g > 0:
                    hsm[x, L, 2] += g / bb
                    if g / bb >= 2.0:
                        hev[x, L, E_GIFT_TO_Y] = 1
                if last_allin == 1:
                    hop[x, L, E_CALL_YALLIN] = 1
                    if iscall:
                        hev[x, L, E_CALL_YALLIN] = 1
            if st == 0:
                first = acted_pre[x] == 0
                for y in range(6):
                    if y == x:
                        continue
                    if first:
                        if tc > 0 and acted_pre[y] == 0:
                            if tp[y] <= 0.15:
                                hop[x, y, E_INF_FOLD_YS] = 1
                                hop[x, y, E_INF_RAISE_YS] = 1
                                if isfold:
                                    hev[x, y, E_INF_FOLD_YS] = 1
                                if aggr:
                                    hev[x, y, E_INF_RAISE_YS] = 1
                            if tp[y] >= 0.5:
                                hop[x, y, E_INF_FOLD_YW] = 1
                                hop[x, y, E_INF_RAISE_YW] = 1
                                if isfold:
                                    hev[x, y, E_INF_FOLD_YW] = 1
                                if aggr:
                                    hev[x, y, E_INF_RAISE_YW] = 1
                            if tp[x] <= 0.4:
                                he = hueq(hu, r, 0, x, y, PI)
                                if he == he:
                                    if he < 0.45:
                                        hop[x, y, E_BHP_FOLD_YB] = 1
                                        if isfold:
                                            hev[x, y, E_BHP_FOLD_YB] = 1
                                    elif he > 0.55:
                                        hop[x, y, E_BHP_FOLD_YW] = 1
                                        if isfold:
                                            hev[x, y, E_BHP_FOLD_YW] = 1
                        if vol[y] == 1 and active[y] == 1:
                            hop[x, y, E_PF_ENTER_AFTERY] = 1
                            hop[x, y, E_PF_FOLD_FIRST_YIN] = 1
                            hop[x, y, E_PF_RAISE_FIRST_YIN] = 1
                            if iscall or aggr:
                                hev[x, y, E_PF_ENTER_AFTERY] = 1
                            if isfold:
                                hev[x, y, E_PF_FOLD_FIRST_YIN] = 1
                            if aggr:
                                hev[x, y, E_PF_RAISE_FIRST_YIN] = 1
                        if nr == 0 and limped[y] == 1 and active[y] == 1:
                            hop[x, y, E_PF_LIMPBEHIND] = 1
                            hop[x, y, E_PF_ISO] = 1
                            if iscall:
                                hev[x, y, E_PF_LIMPBEHIND] = 1
                            if aggr:
                                hev[x, y, E_PF_ISO] = 1
                        if nvol == 0 and nr == 0 and pos[x] == 1 and pos[y] == 2:
                            hop[x, y, E_BVB_WALK] = 1
                            hop[x, y, E_BVB_RAISE] = 1
                            if isfold:
                                hev[x, y, E_BVB_WALK] = 1
                            if aggr:
                                hev[x, y, E_BVB_RAISE] = 1
                        if (nvol == 0 and nr == 0 and (pos[x] == 5 or pos[x] == 0 or pos[x] == 1)
                                and (pos[y] == 1 or pos[y] == 2) and acted_pre[y] == 0):
                            hop[x, y, E_STEAL_OPEN] = 1
                            if aggr:
                                hev[x, y, E_STEAL_OPEN] = 1
                        if ((pos[x] == 1 or pos[x] == 2) and L == y and open_raiser == y and open_unopened == 1
                                and calls_after_open == 0 and nr == 1 and (pos[y] == 5 or pos[y] == 0 or pos[y] == 1)):
                            hop[x, y, E_BLIND_FOLD_STEAL] = 1
                            if isfold:
                                hev[x, y, E_BLIND_FOLD_STEAL] = 1
                    if L >= 0 and L != x and tc > 0:
                        if y == L:
                            hop[x, y, E_PF_CALL_YR] = 1
                            hop[x, y, E_PF_3BET_YR] = 1
                            hop[x, y, E_PF_FOLD_YR] = 1
                            if iscall:
                                hev[x, y, E_PF_CALL_YR] = 1
                            if aggr:
                                hev[x, y, E_PF_3BET_YR] = 1
                            if isfold:
                                hev[x, y, E_PF_FOLD_YR] = 1
                        elif active[y] == 1:
                            if vol[y] == 1:
                                hop[x, y, E_PF_OVERCALL3] = 1
                                hop[x, y, E_PF_FOLD3] = 1
                                hop[x, y, E_PF_SQUEEZE3] = 1
                                if iscall:
                                    hev[x, y, E_PF_OVERCALL3] = 1
                                if isfold:
                                    hev[x, y, E_PF_FOLD3] = 1
                                if aggr:
                                    hev[x, y, E_PF_SQUEEZE3] = 1
                            elif acted_pre[y] == 0:
                                hop[x, y, E_PF_FOLD3_YBEHIND] = 1
                                if isfold:
                                    hev[x, y, E_PF_FOLD3_YBEHIND] = 1
                if first and tc > 0 and nr == 0 and nvol == 0:
                    for y in range(6):
                        if y != x:
                            hop[x, y, E_LIMP_FIRST] = 1
                            if iscall:
                                hev[x, y, E_LIMP_FIRST] = 1
                if act == 5:
                    allin_pre[x] = 1
                nvol_before = nvol
                acted_pre[x] = 1
                if iscall or aggr:
                    if vol[x] == 0:
                        vol[x] = 1
                        nvol += 1
                if first and iscall and nr == 0:
                    limped[x] = 1
                if aggr:
                    pfr[x] = 1
                    if open_raiser < 0:
                        open_raiser = x
                        open_unopened = 1 if nvol_before == 0 else 0
                if iscall and open_raiser >= 0:
                    calls_after_open += 1
            else:
                hs = hs_act[k]
                third = nact >= 3
                for y in range(6):
                    if y == x or active[y] == 0:
                        continue
                    he = hueq(hu, r, st, x, y, PI)
                    if tc == 0:
                        if third:
                            hop[x, y, E_MW_BET] = 1
                            if aggr:
                                hev[x, y, E_MW_BET] = 1
                        if hs == hs and hs >= 0.8:
                            if third:
                                hop[x, y, E_MW_STRONGCHECK] = 1
                                if act == 1:
                                    hev[x, y, E_MW_STRONGCHECK] = 1
                            elif nact == 2:
                                hop[x, y, E_HU_STRONGCHECK] = 1
                                if act == 1:
                                    hev[x, y, E_HU_STRONGCHECK] = 1
                        if nact == 2:
                            hop[x, y, E_HU_BET] = 1
                            if aggr:
                                hev[x, y, E_HU_BET] = 1
                                hop[x, y, E_HU_SMALLBET] = 1
                                if amt <= 0.35 * P:
                                    hev[x, y, E_HU_SMALLBET] = 1
                    if tc > 0 and L >= 0 and L != x:
                        if L == y:
                            if nact == 2:
                                hop[x, y, E_HU_FOLD] = 1
                                hop[x, y, E_HU_RAISE] = 1
                                if isfold:
                                    hev[x, y, E_HU_FOLD] = 1
                                if aggr:
                                    hev[x, y, E_HU_RAISE] = 1
                            if aggr_st[x] == 1:
                                hop[x, y, E_BETFOLD_Y] = 1
                                if isfold:
                                    hev[x, y, E_BETFOLD_Y] = 1
                        else:
                            hop[x, y, E_MW_FOLD3] = 1
                            hop[x, y, E_MW_CALL3] = 1
                            hop[x, y, E_MW_RAISE3] = 1
                            hop[x, y, E_MW_FOLD3_EQ60] = 1
                            if isfold:
                                hev[x, y, E_MW_FOLD3] = 1
                                heL = hueq(hu, r, st, x, L, PI)
                                if heL == heL and heL >= 0.6:
                                    hev[x, y, E_MW_FOLD3_EQ60] = 1
                            if iscall:
                                hev[x, y, E_MW_CALL3] = 1
                            if aggr:
                                hev[x, y, E_MW_RAISE3] = 1
                            if he == he:
                                if he <= 0.35:
                                    hop[x, y, E_FOLD_YAHEAD] = 1
                                    if isfold:
                                        hev[x, y, E_FOLD_YAHEAD] = 1
                                if he >= 0.65:
                                    hop[x, y, E_FOLD_YBEHIND] = 1
                                    if isfold:
                                        hev[x, y, E_FOLD_YBEHIND] = 1
                    if third and he == he:
                        if he <= 0.25:
                            hop[x, y, E_PUMP_YAHEAD] = 1
                            if aggr:
                                hev[x, y, E_PUMP_YAHEAD] = 1
                        if he >= 0.75:
                            hop[x, y, E_AGGR_YBEHIND] = 1
                            if aggr:
                                hev[x, y, E_AGGR_YBEHIND] = 1
                    if aggr and third:
                        hop[x, y, E_MW_BIGBET] = 1
                        if amt >= P:
                            hev[x, y, E_MW_BIGBET] = 1
                    if isfold:
                        hop[x, y, E_FOLDEQ50] = 1
                        if eqx >= 0.5:
                            hev[x, y, E_FOLDEQ50] = 1
            # third party folds to X's aggression while Y active
            if isfold and L >= 0 and L != x:
                for y in range(6):
                    if y != x and y != L and active[y] == 1:
                        hev[L, y, E_THIRD_FOLDS_PAIR] = 1
            if aggr and act == 5:
                for y in range(6):
                    if y != x and active[y] == 1:
                        hev[x, y, E_ALLIN_YACTIVE] = 1
            if isfold:
                active[x] = 0
                fold_st[x] = st
            if aggr:
                L = x
                nr += 1
                st_aggr = 1
                aggr_st[x] = 1
                last_allin = 1 if act == 5 else 0
        # ---- hand-level
        nb = h_nboard[r]
        has_prev = hp
        has_next = hn
        for x in range(6):
            newx = 0
            leavex = 0
            if has_prev:
                newx = 1
                for s in range(6):
                    if s_player[(r - 1) * 6 + s] == pl_[x]:
                        newx = 0
            if has_next:
                leavex = 1
                for s in range(6):
                    if s_player[(r + 1) * 6 + s] == pl_[x]:
                        leavex = 0
            for y in range(6):
                if y == x:
                    continue
                newy = 0
                leavey = 0
                if has_prev:
                    newy = 1
                    for s in range(6):
                        if s_player[(r - 1) * 6 + s] == pl_[y]:
                            newy = 0
                if has_next:
                    leavey = 1
                    for s in range(6):
                        if s_player[(r + 1) * 6 + s] == pl_[y]:
                            leavey = 0
                for e in (E_SHARED, E_PF_VOL, E_PF_RAISE, E_PF_ALLIN, E_FLOW, E_FLOW_BIG, E_IMP5, E_IMPNEG5,
                          E_SEAT_NEXT, E_PRE_BOTH_VOL, E_ALLIN_YACTIVE):
                    hop[x, y, e] = 1
                hev[x, y, E_SHARED] = 1
                hev[x, y, E_PF_VOL] = vol[x]
                hev[x, y, E_PF_RAISE] = pfr[x]
                hev[x, y, E_PF_ALLIN] = allin_pre[x]
                nx = s_net[so + x]
                ny = s_net[so + y]
                if nx < 0 and ny > 0:
                    hev[x, y, E_FLOW] = 1
                    hsm[x, y, 1] += min(-nx, ny) / bb
                if nx <= -10 * bb and ny >= 10 * bb:
                    hev[x, y, E_FLOW_BIG] = 1
                hsm[x, y, 0] = imp[x, y]
                if imp[x, y] >= 5.0:
                    hev[x, y, E_IMP5] = 1
                if imp[x, y] <= -5.0:
                    hev[x, y, E_IMPNEG5] = 1
                if (y - x + 6) % 6 == 1:
                    hev[x, y, E_SEAT_NEXT] = 1
                if vol[x] == 1 and vol[y] == 1:
                    hev[x, y, E_PRE_BOTH_VOL] = 1
                if nb >= 3 and fold_st[x] >= 1 and fold_st[y] >= 1:
                    hop[x, y, E_BOTH_SD] = 1
                    hop[x, y, E_FOLDPOST_YSD] = 1
                    if s_sd[so + x] and s_sd[so + y]:
                        hev[x, y, E_BOTH_SD] = 1
                    if fold_st[x] >= 1 and fold_st[x] <= 3 and s_sd[so + y]:
                        hev[x, y, E_FOLDPOST_YSD] = 1
                if s_sd[so + x] and s_sd[so + y]:
                    hop[x, y, E_SD_WIN] = 1
                    if nx > 0 and ny < 0:
                        hev[x, y, E_SD_WIN] = 1
                if s_contrib[so + x] >= 3 * bb and vol[y] == 1:
                    ok3 = False
                    for z in range(6):
                        if z != x and z != y and s_contrib[so + z] >= 3 * bb:
                            ok3 = True
                    if ok3:
                        hop[x, y, E_POT3_YWINS] = 1
                        if nx < 0 and ny > 0:
                            hev[x, y, E_POT3_YWINS] = 1
                if newx == 1:
                    hop[x, y, E_COJOIN] = 1
                    if newy == 1:
                        hev[x, y, E_COJOIN] = 1
                if leavex == 1:
                    hop[x, y, E_COLEAVE] = 1
                    if leavey == 1:
                        hev[x, y, E_COLEAVE] = 1
        # third-folds opp: X aggr while Y active facing a 3rd is approximated by "X aggressive postflop/pf with Y seated"
        for x in range(6):
            for y in range(6):
                if x != y:
                    hop[x, y, E_THIRD_FOLDS_PAIR] = 1
        return 0


@njit(cache=True, parallel=True)
def run_tables(tab_lo, tab_hi, widx0, widx1, widx2, h_seq, s_player, s_c1, s_c2, s_net, s_sd, s_contrib, h_button, h_bb,
               h_nboard, a_start, a_end, a_street, a_player, a_action, a_amount, a_to_call, a_pot,
               eq_pre, eq_post, hs_act, hu, tp169, lidx, PI, EV, OP, SM, NE):
    ntab = tab_lo.shape[0]
    for t in prange(ntab):
        hev = np.zeros((6, 6, NE), dtype=np.uint8)
        hop = np.zeros((6, 6, NE), dtype=np.uint8)
        hsm = np.zeros((6, 6, 3), dtype=np.float64)
        for r in range(tab_lo[t], tab_hi[t]):
            hp = r > tab_lo[t]
            hn = r + 1 < tab_hi[t]
            pl_ = np.empty(6, dtype=np.int64)
            for s in range(6):
                pl_[s] = s_player[r * 6 + s]
            fill_hand(r, hp, hn, s_player, s_c1, s_c2, s_net, s_sd, s_contrib, h_button, h_bb, h_nboard, a_start, a_end, a_street, a_player, a_action, a_amount, a_to_call, a_pot, eq_pre, eq_post, hs_act, hu, tp169, PI, hev, hop, hsm)
            # ---- accumulate
            for w in (widx0[r], widx1[r], widx2[r]):
                if w < 0:
                    continue
                for x in range(6):
                    lx = lidx[pl_[x]]
                    for y in range(6):
                        if x == y:
                            continue
                        ly = lidx[pl_[y]]
                        for e in range(NE):
                            EV[w, t, lx, ly, e] += hev[x, y, e]
                            OP[w, t, lx, ly, e] += hop[x, y, e]
                        for q in range(3):
                            SM[w, t, lx, ly, q] += hsm[x, y, q]
    return 0


@njit(cache=True, parallel=True)
def hand_events(r_lo, r_hi, s_player, s_c1, s_c2, s_net, s_sd, s_contrib, h_button, h_bb, h_nboard, a_start, a_end, a_street, a_player, a_action, a_amount, a_to_call, a_pot, eq_pre, eq_post, hs_act, hu, tp169, PI, NE):
    """per-hand directed event/opportunity flags for hand rows [r_lo, r_hi) of ONE table (lineup from neighbours)."""
    nh = r_hi - r_lo
    HEV = np.zeros((nh, 6, 6, NE), dtype=np.uint8)
    HOP = np.zeros((nh, 6, 6, NE), dtype=np.uint8)
    HSM = np.zeros((nh, 6, 6, 3), dtype=np.float64)
    for q in prange(nh):
        r = r_lo + q
        fill_hand(r, r > r_lo, r + 1 < r_hi, s_player, s_c1, s_c2, s_net, s_sd, s_contrib, h_button, h_bb, h_nboard, a_start, a_end, a_street, a_player, a_action, a_amount, a_to_call, a_pot, eq_pre, eq_post, hs_act, hu, tp169, PI, HEV[q], HOP[q], HSM[q])
    return HEV, HOP, HSM


def stage_events():
    t0 = time.time()
    seats_all = pl.scan_parquet(DER / "seats.parquet").select("player_idx").unique().collect()
    ptab = (pl.scan_parquet(DER / "seats.parquet").select("hand_idx", "player_idx").unique("player_idx")
            .collect().join(pl.read_parquet(DER / "hands.parquet", columns=["hand_idx", "table_idx"]), on="hand_idx"))
    ptab = ptab.sort("table_idx", "player_idx").with_columns(pl.int_range(pl.len()).over("table_idx").alias("lidx"))
    assert ptab.group_by("table_idx").len()["len"].max() == 30 and ptab.height == 12000
    lidx = np.full(12000, -1, dtype=np.int64)
    lidx[ptab["player_idx"].to_numpy()] = ptab["lidx"].to_numpy()
    ptab.select("player_idx", "table_idx", "lidx").write_parquet(OC / "player_local.parquet")
    pfe = pl.read_parquet(DER / "preflop_equity_169.parquet").sort("class_id")
    tp169 = pfe["top_pct_vs1"].to_numpy().astype(np.float64)
    EV = np.zeros((NW, 400, 30, 30, NE), dtype=np.int32)
    OP = np.zeros((NW, 400, 30, 30, NE), dtype=np.int32)
    SM = np.zeros((NW, 400, 30, 30, NK), dtype=np.float32)
    for phase in (0, 1):
        tag = "dev" if phase == 0 else "eval"
        hands = (pl.read_parquet(DER / "hands.parquet", columns=["hand_idx", "table_idx", "hand_seq", "phase", "button_seat", "bb", "board"])
                 .filter(col("phase") == phase).sort("hand_idx"))
        hids = hands["hand_idx"].to_numpy()
        tab = hands["table_idx"].to_numpy()
        assert np.all(np.diff(tab) >= 0)
        tab_lo = np.searchsorted(tab, np.arange(400), side="left").astype(np.int64)
        tab_hi = np.searchsorted(tab, np.arange(400), side="right").astype(np.int64)
        lim = int(os.environ.get("OC_LIMIT_TABLES", "0"))
        if lim:
            tab_hi[lim:] = tab_lo[lim:]
        seq = hands["hand_seq"].to_numpy()
        if phase == 0:
            widx0 = np.zeros(len(hids), dtype=np.int64)
            widx1 = np.where(seq < 2000, 1, -1).astype(np.int64)
            widx2 = np.where(seq >= 1000, 3, -1).astype(np.int64)
        else:
            widx0 = np.full(len(hids), 2, dtype=np.int64)
            widx1 = np.full(len(hids), -1, dtype=np.int64)
            widx2 = np.full(len(hids), -1, dtype=np.int64)
        seats = (pl.scan_parquet(DER / "seats.parquet").filter(col("hand_idx").is_in(hids))
                 .select("hand_idx", "seat_no", "player_idx", "c1", "c2", "net_chips", "went_to_showdown", "total_contribution")
                 .collect().sort("hand_idx", "seat_no"))
        assert seats.height == 6 * len(hids)
        acts = (pl.scan_parquet(DER / "actions.parquet").filter(col("hand_idx").is_in(hids))
                .select("hand_idx", "action_no", "street", "player_idx", "action", "amount", "to_call", "pot_before")
                .collect().sort("hand_idx", "action_no"))
        eq = pl.scan_parquet(DER / f"action_equity_{tag}.parquet").filter(col("hand_idx").is_in(hids)).collect()
        assert eq.height == acts.height and (eq["action_no"].to_numpy() == acts["action_no"].to_numpy()).all()
        eq_pre = eq.select([f"eq_pre_s{s}" for s in range(6)]).to_numpy().astype(np.float32)
        eq_post = eq.select([f"eq_post_s{s}" for s in range(6)]).to_numpy().astype(np.float32)
        hs_act = eq["hs_actor"].to_numpy().astype(np.float32)
        del eq
        hu_df = pl.scan_parquet(DER / f"hu_equity_{tag}.parquet").filter(col("hand_idx").is_in(hids)).collect()
        assert (hu_df["hand_idx"].to_numpy() == hids).all()
        hu = np.full((len(hids), 4, 15), np.nan, dtype=np.float32)
        for st in range(4):
            for i in range(6):
                for j in range(i + 1, 6):
                    hu[:, st, PI[i, j]] = hu_df[f"hu_s{st}_{i}{j}"].to_numpy()
        del hu_df
        ah = acts["hand_idx"].to_numpy()
        a_start = np.searchsorted(ah, hids, side="left").astype(np.int64)
        a_end = np.searchsorted(ah, hids, side="right").astype(np.int64)
        print(f"[{tag}] loaded in {time.time() - t0:.0f}s", flush=True)
        run_tables(tab_lo, tab_hi, widx0, widx1, widx2, seq.astype(np.int64),
                   seats["player_idx"].to_numpy().astype(np.int64), seats["c1"].to_numpy().astype(np.int64),
                   seats["c2"].to_numpy().astype(np.int64), seats["net_chips"].to_numpy().astype(np.float64),
                   seats["went_to_showdown"].to_numpy(), seats["total_contribution"].to_numpy().astype(np.float64),
                   hands["button_seat"].to_numpy().astype(np.int64), hands["bb"].to_numpy().astype(np.float64),
                   hands["board"].list.len().to_numpy().astype(np.int64), a_start, a_end,
                   np.append(acts["street"].to_numpy().astype(np.int64), 0),
                   acts["player_idx"].to_numpy().astype(np.int64), acts["action"].to_numpy().astype(np.int64),
                   acts["amount"].to_numpy().astype(np.float64), acts["to_call"].to_numpy().astype(np.float64),
                   acts["pot_before"].to_numpy().astype(np.float64), eq_pre, eq_post, hs_act, hu, tp169, lidx, PI,
                   EV, OP, SM, NE)
        print(f"[{tag}] events done in {time.time() - t0:.0f}s", flush=True)
        del acts, seats, eq_pre, eq_post, hs_act, hu
    sfx = "_test" if os.environ.get("OC_LIMIT_TABLES") else ""
    np.save(OC / f"events_ev{sfx}.npy", EV)
    np.save(OC / f"events_op{sfx}.npy", OP)
    np.save(OC / f"events_sm{sfx}.npy", SM)
    print(f"saved in {time.time() - t0:.0f}s")




# ------------------------------------------------------------------------------------------- stats
SUM_EVENTS = ["sum_imp", "sum_transfer", "sum_gift"]
ALL_EVENTS = [n for n, _ in EVENTS] + SUM_EVENTS
SYMMETRIC = {"mw_checkthrough", "both_sd", "hu_checkdown", "pre_both_vol", "shared"}


def ipf_expect(c, o, m=20.0):
    """c, o: [T, 30, 30] (diag 0). Returns (E_fit, E_loo).
    E_fit: row x column independence fit (null generator). E_loo: leave-pair-out, shrunk double-centred expectation
    rate_xy = p_x^(-y) * q_y^(-x) / p0 with p_x^(-y) = (C_x. - c_xy + m p0) / (O_x. - o_xy + m) (same for column y)."""
    p0 = c.sum(axis=(1, 2), keepdims=True) / np.maximum(o.sum(axis=(1, 2), keepdims=True), 1.0)   # per-table base rate
    p0 = np.maximum(p0, 1e-6)
    rs, ro = c.sum(2, keepdims=True), o.sum(2, keepdims=True)
    cs, co = c.sum(1, keepdims=True), o.sum(1, keepdims=True)
    px = (rs + m * p0) / (ro + m)
    qy = (cs + m * p0) / (co + m)
    E_fit = o * np.minimum(px * qy / p0, 1.0)
    px_l = (rs - c + m * p0) / (ro - o + m)
    qy_l = (cs - c + m * p0) / (co - o + m)
    E_loo = o * np.minimum(px_l * qy_l / p0, 1.0)
    return E_fit, E_loo


def pois_G(c, E):
    E = np.maximum(E, 1e-3)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(c > 0, c * np.log(np.maximum(c, 1e-12) / E), 0.0) - (c - E)
    return np.sign(c - E) * 2.0 * np.maximum(t, 0.0)


def event_arrays(w, rng=None):
    """Return dict of per-ordered-pair arrays [T,30,30,NE'] : c, E, G, llift, rowtop, coltop (optionally Poisson null)."""
    EV = np.load(OC / "events_ev.npy", mmap_mode="r")
    OP = np.load(OC / "events_op.npy", mmap_mode="r")
    SM = np.load(OC / "events_sm.npy", mmap_mode="r")
    nall = len(ALL_EVENTS)
    out = {k: np.zeros((400, 30, 30, nall), dtype=np.float32) for k in ("c", "o", "E", "G", "L")}
    out["rt"] = np.zeros((400, 30, 30, nall), dtype=np.int8)
    out["ct"] = np.zeros((400, 30, 30, nall), dtype=np.int8)
    diag = np.eye(30, dtype=bool)[None]
    for e in range(nall):
        if e < NE:
            c = np.asarray(EV[w, :, :, :, e], dtype=np.float64)
            o = np.asarray(OP[w, :, :, :, e], dtype=np.float64)
        else:
            c = np.maximum(np.asarray(SM[w, :, :, :, e - NE], dtype=np.float64), 0) / 5.0   # units of 5bb
            o = np.asarray(OP[w, :, :, :, E_SHARED], dtype=np.float64)
        c[:, diag[0]] = 0
        o[:, diag[0]] = 0
        if rng is not None:                       # binomial null (Poisson for the continuous sums)
            E0, _ = ipf_expect(c, o)
            if e < NE:
                pr = np.clip(np.where(o > 0, E0 / np.maximum(o, 1e-12), 0.0), 0.0, 1.0)
                c = rng.binomial(o.astype(np.int64), pr).astype(np.float64)
            else:
                c = rng.poisson(E0).astype(np.float64)
        E, E_loo = ipf_expect(c, o)
        G = pois_G(c, E_loo)
        Ll = np.log((c + 1.0) / (E_loo + 1.0))
        Lm = np.where(o > 0, Ll, -np.inf)
        Lm[:, diag[0]] = -np.inf
        rt = (Lm == Lm.max(axis=2, keepdims=True)) & (o > 0)
        ct = (Lm == Lm.max(axis=1, keepdims=True)) & (o > 0)
        out["c"][..., e] = c
        out["o"][..., e] = o
        out["E"][..., e] = E_loo
        out["G"][..., e] = G
        out["L"][..., e] = Ll
        out["rt"][..., e] = rt
        out["ct"][..., e] = ct
    return out


def pair_index():
    pt = pl.read_parquet(OC / "player_local.parquet").sort("table_idx", "lidx")
    P = np.full((400, 30), -1, dtype=np.int64)
    P[pt["table_idx"].to_numpy(), pt["lidx"].to_numpy()] = pt["player_idx"].to_numpy()
    iu, ju = np.triu_indices(30, 1)
    t = np.repeat(np.arange(400), len(iu))
    i = np.tile(iu, 400)
    j = np.tile(ju, 400)
    return t, i, j, P[t, i], P[t, j]


def stage_stats():
    t0 = time.time()
    t, i, j, pA, pB = pair_index()
    rng = np.random.default_rng(13)
    nulls = (True,) if os.environ.get("OC_NULL_ONLY") else ((False,) if os.environ.get("OC_NO_NULL") else (False, True))
    wins = [int(x) for x in os.environ.get("OC_WINDOWS", "0,1,2").split(",")]
    for w in wins:
        for null in nulls:
            A = event_arrays(w, rng if null else None)
            res = {"table_idx": t.astype(np.int16), "pA": pA.astype(np.int32), "pB": pB.astype(np.int32)}
            arrs = {}
            for k in ("c", "o", "E", "G", "L", "rt", "ct"):
                arrs[k + "_ab"] = A[k][t, i, j]     # A acts toward B
                arrs[k + "_ba"] = A[k][t, j, i]
            np.savez(OC / f"pairstats_w{w}{'_null' if null else ''}.npz", **res, **arrs)
            print(f"w{w} null={null} done {time.time() - t0:.0f}s", flush=True)
            del A, arrs


# ------------------------------------------------------------------------------------------- groups + excess
def load_ps(w, null=False):
    z = np.load(OC / f"pairstats_w{w}{'_null' if null else ''}.npz")
    return {k: z[k] for k in z.files}


def groups():
    """Unordered pair frame (same order as pair_index) with group flags. Labels only define reference groups."""
    t, i, j, pA, pB = pair_index()
    df = pl.DataFrame({"row": np.arange(len(t)), "table_idx": t.astype(np.int16), "pA": pA.astype(np.int32), "pB": pB.astype(np.int32)})
    o = pl.read_parquet(BASE / "oof" / "exp005_oof_full.parquet").select("pA", "pB", "n", "y", "is_lab", "behavior_family", "oof")
    q05 = o.filter(col("y") == 1)["oof"].quantile(0.05)
    lab = pl.read_parquet(DER / "labels.parquet")
    labp = set(lab["p1"].to_list()) | set(lab["p2"].to_list())
    posp = set(lab.filter(col("label") == 1)["p1"].to_list()) | set(lab.filter(col("label") == 1)["p2"].to_list())
    ep = (pl.read_parquet(DER / "eval_pairs.parquet")
          .select(pl.min_horizontal("p1", "p2").alias("pA"), pl.max_horizontal("p1", "p2").alias("pB"), "pair_id", col("shared_hands").alias("n_eval")))
    es = pl.read_parquet(BASE / "oof" / "exp005_eval_scores.parquet").select("pair_id", "risk_raw")
    ep = ep.join(es, on="pair_id").with_columns(col("risk_raw").rank("ordinal", descending=True).alias("eval_rank"))
    df = df.join(o, on=["pA", "pB"], how="left").join(ep, on=["pA", "pB"], how="left")
    nolab = ~col("pA").is_in(list(labp)) & ~col("pB").is_in(list(labp))
    df = df.with_columns(
        (col("is_lab").fill_null(False) & (col("y") == 0)).alias("NEG"),
        (col("is_lab").fill_null(False) & (col("behavior_family") == "directed_transfer")).alias("DT"),
        (col("is_lab").fill_null(False) & (col("behavior_family") == "soft_play")).alias("SP"),
        (col("is_lab").fill_null(False) & (col("behavior_family") == "coordinated_isolation")).alias("CI"),
        (~col("is_lab").fill_null(True) & (col("n") >= 57) & nolab & (col("oof") < q05)).alias("CAND"),
        (~col("is_lab").fill_null(True) & (col("n") >= 57) & nolab & (col("oof") >= q05)).alias("HID"),
        (col("pair_id").is_not_null() & (col("eval_rank") > 800) & (col("n_eval") >= 57)).alias("EVC"),
        (col("pair_id").is_not_null() & (col("eval_rank") <= 800)).alias("EVHI"),
        (~col("pA").is_in(list(posp)) & ~col("pB").is_in(list(posp))).alias("noposp"),
    ).sort("row")
    return df


def flag_defs(S, e):
    Gab, Gba = S["G_ab"][:, e], S["G_ba"][:, e]
    dtop_ab = (S["rt_ab"][:, e] == 1) & (S["ct_ab"][:, e] == 1)
    dtop_ba = (S["rt_ba"][:, e] == 1) & (S["ct_ba"][:, e] == 1)
    stop = (S["rt_ab"][:, e] == 1) & (S["rt_ba"][:, e] == 1)
    f = {}
    for g in (6, 10, 16, 25):
        f[f"dirtop_G{g}"] = (dtop_ab & (Gab >= g)) | (dtop_ba & (Gba >= g))
    for g in (3, 6, 10):
        f[f"symtop_G{g}"] = stop & (np.minimum(Gab, Gba) >= g)
    for g in (16, 25, 40):
        f[f"gmax_{g}"] = np.maximum(Gab, Gba) >= g
    for g in (10, 16, 25):
        f[f"deficit_{g}"] = (Gab + Gba) <= -g
    return f


def stage_excess():
    """Tail counts per event x flag. CANDX = dev candidates not eval-high; EVCX = eval candidates not dev-high.
    Nulls: binomial IPF null (same pairs), labelled negatives scaled, other phase of the same pairs."""
    t0 = time.time()
    df = groups()
    q05 = df.filter(col("DT") | col("SP") | col("CI"))["oof"].quantile(0.05)
    df = df.with_columns((col("CAND") & ~col("EVHI")).alias("CANDX"),
                         (col("EVC") & (col("oof").fill_null(0.0) < q05)).alias("EVCX"))
    gm = {g: df[g].fill_null(False).to_numpy() for g in ("NEG", "DT", "SP", "CI", "HID", "CANDX", "EVCX")}
    print({g: int(v.sum()) for g, v in gm.items()}, flush=True)
    S = {w: load_ps(w) for w in (0, 1, 2)}
    N = {w: load_ps(w, True) for w in (0, 1, 2)}
    rows = []
    nneg, ncx, nex = gm["NEG"].sum(), gm["CANDX"].sum(), gm["EVCX"].sum()
    for e, name in enumerate(ALL_EVENTS):
        if name in ("shared", "impneg5", "seat_next"):
            continue
        F = {w: flag_defs(S[w], e) for w in (0, 1, 2)}
        FN = {w: flag_defs(N[w], e) for w in (0, 1, 2)}
        for fk in F[0]:
            r = {"event": name, "flag": fk}
            for g in ("NEG", "DT", "SP", "CI", "HID"):
                r[g] = int(F[0][fk][gm[g]].sum())
            r["NEG_null"] = int(FN[0][fk][gm["NEG"]].sum())
            r["CX_dev"] = int(F[0][fk][gm["CANDX"]].sum())
            r["CX_dev_null"] = int(FN[0][fk][gm["CANDX"]].sum())
            r["CX_dev_negexp"] = round(r["NEG"] / nneg * ncx, 1)
            r["CX_w1"] = int(F[1][fk][gm["CANDX"]].sum())
            r["CX_w1_null"] = int(FN[1][fk][gm["CANDX"]].sum())
            r["CX_evph"] = int(F[2][fk][gm["CANDX"]].sum())
            r["EX_eval"] = int(F[2][fk][gm["EVCX"]].sum())
            r["EX_eval_null"] = int(FN[2][fk][gm["EVCX"]].sum())
            r["EX_devph"] = int(F[1][fk][gm["EVCX"]].sum())
            r["all_both"] = int((F[1][fk] & F[2][fk]).sum())
            r["both_exp"] = round(float(F[1][fk].sum()) * float(F[2][fk].sum()) / len(F[1][fk]), 1)
            rows.append(r)
    X = pl.DataFrame(rows).with_columns(
        (col("CX_dev") - pl.max_horizontal(col("CX_dev_null"), col("CX_dev_negexp"))).alias("xs_dev"),
        (col("EX_eval") - col("EX_eval_null")).alias("xs_eval"),
        ((col("DT") / 148 + col("SP") / 132 + col("CI") / 92) / 3).round(3).alias("pos_recall"),
    )
    X.write_csv(FOR / "oc_excess_all.csv")
    print(f"done {time.time() - t0:.0f}s")
    return X


# ------------------------------------------------------------------------------------------- hand-level reading
_TP169 = None


def load_tables(phase, tables):
    """Arrays for all hands of the given tables in one phase (for per-hand events / printing)."""
    global _TP169
    if _TP169 is None:
        _TP169 = pl.read_parquet(DER / "preflop_equity_169.parquet").sort("class_id")["top_pct_vs1"].to_numpy().astype(np.float64)
    tag = "dev" if phase == 0 else "eval"
    tables = list(tables)
    hands = (pl.read_parquet(DER / "hands.parquet", columns=["hand_idx", "hand_id", "table_idx", "hand_seq", "phase", "button_seat", "bb", "board", "final_pot"])
             .filter((col("phase") == phase) & col("table_idx").is_in(tables)).sort("hand_idx"))
    hids = hands["hand_idx"].to_numpy()
    lo, hi = int(hids.min()), int(hids.max())
    rng_f = (col("hand_idx") >= lo) & (col("hand_idx") <= hi)
    seats = (pl.scan_parquet(DER / "seats.parquet").filter(rng_f).filter(col("hand_idx").is_in(hids))
             .select("hand_idx", "seat_no", "player_idx", "c1", "c2", "net_chips", "went_to_showdown", "total_contribution", "starting_stack")
             .collect().sort("hand_idx", "seat_no"))
    acts = (pl.scan_parquet(DER / "actions.parquet").filter(rng_f).filter(col("hand_idx").is_in(hids))
            .collect().sort("hand_idx", "action_no"))
    eq = (pl.scan_parquet(DER / f"action_equity_{tag}.parquet").filter(rng_f).filter(col("hand_idx").is_in(hids))
          .collect().sort("hand_idx", "action_no"))
    hu_df = pl.scan_parquet(DER / f"hu_equity_{tag}.parquet").filter(rng_f).filter(col("hand_idx").is_in(hids)).collect().sort("hand_idx")
    hu = np.full((len(hids), 4, 15), np.nan, dtype=np.float32)
    for st in range(4):
        for i in range(6):
            for j in range(i + 1, 6):
                hu[:, st, PI[i, j]] = hu_df[f"hu_s{st}_{i}{j}"].to_numpy()
    ah = acts["hand_idx"].to_numpy()
    D = dict(hands=hands, seats=seats, acts=acts, eq=eq, hu=hu, hids=hids,
             a_start=np.searchsorted(ah, hids, side="left").astype(np.int64),
             a_end=np.searchsorted(ah, hids, side="right").astype(np.int64))
    D["args"] = (seats["player_idx"].to_numpy().astype(np.int64), seats["c1"].to_numpy().astype(np.int64),
                 seats["c2"].to_numpy().astype(np.int64), seats["net_chips"].to_numpy().astype(np.float64),
                 seats["went_to_showdown"].to_numpy(), seats["total_contribution"].to_numpy().astype(np.float64),
                 hands["button_seat"].to_numpy().astype(np.int64), hands["bb"].to_numpy().astype(np.float64),
                 hands["board"].list.len().to_numpy().astype(np.int64), D["a_start"], D["a_end"],
                 np.append(acts["street"].to_numpy().astype(np.int64), 0),
                 acts["player_idx"].to_numpy().astype(np.int64), acts["action"].to_numpy().astype(np.int64),
                 acts["amount"].to_numpy().astype(np.float64), acts["to_call"].to_numpy().astype(np.float64),
                 acts["pot_before"].to_numpy().astype(np.float64),
                 eq.select([f"eq_pre_s{s}" for s in range(6)]).to_numpy().astype(np.float32),
                 eq.select([f"eq_post_s{s}" for s in range(6)]).to_numpy().astype(np.float32),
                 eq["hs_actor"].to_numpy().astype(np.float32), hu, _TP169, PI)
    return D


def pair_hand_events(D, table, pA, pB):
    """Per shared hand of (pA,pB) in one table: directed flags A->B and B->A for every event (-1 = no opportunity)."""
    tab = D["hands"]["table_idx"].to_numpy()
    r_lo = int(np.searchsorted(tab, table, side="left"))
    r_hi = int(np.searchsorted(tab, table, side="right"))
    HEV, HOP, HSM = hand_events(r_lo, r_hi, *D["args"], NE)
    sp = D["args"][0].reshape(-1, 6)[r_lo:r_hi]
    rows = []
    for q in range(r_hi - r_lo):
        sa = np.flatnonzero(sp[q] == pA)
        sb = np.flatnonzero(sp[q] == pB)
        if len(sa) == 0 or len(sb) == 0:
            continue
        a, b = int(sa[0]), int(sb[0])
        rec = {"hand_idx": int(D["hids"][r_lo + q]), "hand_seq": int(D["hands"]["hand_seq"][r_lo + q]), "seatA": a, "seatB": b}
        for e, (n, _) in enumerate(EVENTS):
            rec[f"{n}_ab"] = int(HEV[q, a, b, e]) if HOP[q, a, b, e] else -1
            rec[f"{n}_ba"] = int(HEV[q, b, a, e]) if HOP[q, b, a, e] else -1
        rec["imp_ab"], rec["imp_ba"] = float(HSM[q, a, b, 0]), float(HSM[q, b, a, 0])
        rec["gift_ab"], rec["gift_ba"] = float(HSM[q, a, b, 2]), float(HSM[q, b, a, 2])
        rows.append(rec)
    return pl.DataFrame(rows)


ACT = {0: "fold", 1: "check", 2: "call", 3: "bet", 4: "raise", 5: "allin"}
POSN = {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "HJ", 5: "CO"}
STN = {0: "PRE", 1: "FLOP", 2: "TURN", 3: "RIVER"}


def cstr(c):
    return "23456789TJQKA"[c // 4] + "cdhs"[c % 4]


def format_hand(D, hand_idx, pA, pB):
    r = int(np.searchsorted(D["hids"], hand_idx))
    h = D["hands"].row(r, named=True)
    bb = h["bb"]
    S = D["seats"].slice(r * 6, 6)
    tag = {}
    lines = []
    board = " ".join(cstr(c) for c in h["board"])
    lines.append(f"== hand {hand_idx} seq {h['hand_seq']} table {h['table_idx']} bb {bb} pot {h['final_pot'] / bb:.1f}bb board [{board}]")
    for s in S.iter_rows(named=True):
        who = "A" if s["player_idx"] == pA else ("B" if s["player_idx"] == pB else f"o{s['seat_no']}")
        tag[s["seat_no"]] = who
        pos = (s["seat_no"] - h["button_seat"] + 6) % 6
        cls = class_id(s["c1"], s["c2"])
        lines.append(f"   {who:>3} seat{s['seat_no']} {POSN[pos]:>3} {cstr(s['c1'])}{cstr(s['c2'])} top{_TP169[cls]:.2f} "
                     f"stack {s['starting_stack'] / bb:6.1f}bb net {s['net_chips'] / bb:+7.1f}bb {'SD' if s['went_to_showdown'] else ''}")
    seat_of = {s["player_idx"]: s["seat_no"] for s in S.iter_rows(named=True)}
    a0, a1 = D["a_start"][r], D["a_end"][r]
    A = D["acts"].slice(a0, a1 - a0)
    E = D["eq"].slice(a0, a1 - a0)
    sA, sB = seat_of[pA], seat_of[pB]
    for a, e in zip(A.iter_rows(named=True), E.iter_rows(named=True)):
        x = seat_of[a["player_idx"]]
        st = a["street"]
        huab = D["hu"][r, st, PI[sA, sB]]
        if sA > sB:
            huab = 1 - huab
        eqs = " ".join(f"{tag[s]}:{e[f'eq_pre_s{s}']:.2f}" for s in range(6) if e[f"eq_pre_s{s}"] == e[f"eq_pre_s{s}"])
        hs = e["hs_actor"]
        lines.append(f"   {STN[st]:>5} {tag[x]:>3} {ACT[a['action']]:>5} {a['amount'] / bb:6.1f}bb tc {a['to_call'] / bb:5.1f} pot {a['pot_before'] / bb:6.1f}"
                     f" | hs {hs:.2f} | huA>B {huab:.2f} | eq {eqs}")
    return "\n".join(lines)


def stage_print():
    """python scripts/oc_hunt.py print pA pB phase [k] [event]"""
    pA, pB, phase = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    k = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    table = pl.read_parquet(OC / "player_local.parquet").filter(col("player_idx") == pA)["table_idx"][0]
    D = load_tables(phase, [table])
    ph = pair_hand_events(D, table, pA, pB)
    ev = sys.argv[6] if len(sys.argv) > 6 else None
    if ev:
        ph = ph.filter((col(f"{ev}_ab") == 1) | (col(f"{ev}_ba") == 1))
    for hid in ph["hand_idx"].to_list()[:k]:
        print(format_hand(D, hid, pA, pB))



# ------------------------------------------------------------------------------------------- multi-event mining
# events that carry pair-interaction information (drop pure exposure / seat autocorrelation / continuous sums)
MINE_DROP = {"shared", "impneg5", "seat_next", "pf_vol", "pf_raise", "pf_allin"}


def mine_scores(w, null=False):
    """Per unordered pair: multi-event anomaly scores from the event bank."""
    S = load_ps(w, null)
    keep = [e for e, n in enumerate(ALL_EVENTS) if n not in MINE_DROP]
    Gab, Gba = S["G_ab"][:, keep], S["G_ba"][:, keep]
    rt_ab, rt_ba = S["rt_ab"][:, keep] == 1, S["rt_ba"][:, keep] == 1
    ct_ab, ct_ba = S["ct_ab"][:, keep] == 1, S["ct_ba"][:, keep] == 1
    G = np.maximum(Gab, Gba)
    top = np.where(Gab >= Gba, rt_ab & ct_ab, rt_ba & ct_ba)
    out = {
        "K6": ((G >= 6) & top).sum(1).astype(np.int16),
        "K10": ((G >= 10) & top).sum(1).astype(np.int16),
        "K16": ((G >= 16) & top).sum(1).astype(np.int16),
        "Gpos": np.clip(G, 0, 30).sum(1).astype(np.float32),
        "Gtop": np.clip(np.where(top, G, 0.0), 0, 30).sum(1).astype(np.float32),
        "Gmax": G.max(1).astype(np.float32),
        "Gsym": np.clip(np.minimum(Gab, Gba), 0, 30).sum(1).astype(np.float32),
        "n": S["o_ab"][:, EI["shared"]].astype(np.int32),
    }
    return out, keep, {"G": G, "top": top}


def stage_mine():
    t0 = time.time()
    df = groups()
    q05 = df.filter(col("DT") | col("SP") | col("CI"))["oof"].quantile(0.05)
    df = df.with_columns((col("CAND") & ~col("EVHI")).alias("CANDX"),
                         (col("EVC") & (col("oof").fill_null(0.0) < q05)).alias("EVCX"))
    keepnames = None
    frames = {}
    for w, tag in ((0, "dev_full"), (1, "dev_w2000"), (2, "eval")):
        for null in (False, True):
            sc, keep, _ = mine_scores(w, null)
            keepnames = [ALL_EVENTS[e] for e in keep]
            frames[(tag, null)] = sc
    gm = {g: df[g].fill_null(False).to_numpy() for g in ("NEG", "DT", "SP", "CI", "HID", "CANDX", "EVCX")}
    stat = "Gtop"
    print(f"--- distribution of {stat} (and K10) by group")
    for (tag, null), sc in frames.items():
        line = [f"{tag:9s} null={int(null)}"]
        for g in ("NEG", "DT", "SP", "CI", "HID", "CANDX", "EVCX"):
            v = sc[stat][gm[g]]
            line.append(f"{g} p50 {np.median(v):6.1f} p99 {np.quantile(v, .99):7.1f} max {v.max():7.1f}")
        print(" | ".join(line), flush=True)
    for (tag, null), sc in frames.items():
        thr = np.quantile(frames[("dev_w2000", True)]["Gtop"][gm["NEG"]], 0.99)
        print(tag, "null" if null else "obs", "frac>thr(nullNEG p99):",
              {g: round(float((sc["Gtop"][gm[g]] > thr).mean()), 4) for g in ("NEG", "DT", "SP", "CI", "HID", "CANDX", "EVCX")})
    # save per-pair scores for the top lists
    out = df.select("row", "table_idx", "pA", "pB", "n", "y", "is_lab", "behavior_family", "oof", "pair_id",
                    "n_eval", "eval_rank", "NEG", "DT", "SP", "CI", "HID", "CANDX", "EVCX")
    for (tag, null), sc in frames.items():
        if null:
            continue
        for k, v in sc.items():
            out = out.with_columns(pl.Series(f"{k}_{tag}", v))
    out.write_parquet(OC / "mine_scores.parquet")
    print(f"saved mine_scores {time.time() - t0:.0f}s")
    return out



# ------------------------------------------------------------------------------------------- symmetric phase test
def stage_phase():
    """Paired dev-vs-eval (McNemar) test on a SYMMETRICALLY selected population.

    SYM = pairs with no labelled player, dev(w2000) shared >= 50 AND eval shared >= 50, exp005 dev OOF below the
    positives' 5th pct, and not in the exp005 eval top-800. Both phases are then filtered the same way and have
    comparable exposure, so under the null (no phase-local pair effect) the number of pairs flagged in dev only (b)
    equals the number flagged in eval only (c). A hidden mechanism active only in the eval phase gives c >> b.
    """
    t0 = time.time()
    df = groups()
    q05 = df.filter(col("DT") | col("SP") | col("CI"))["oof"].quantile(0.05)
    S1, S2 = load_ps(1), load_ps(2)
    n1 = S1["o_ab"][:, EI["shared"]]
    n2 = S2["o_ab"][:, EI["shared"]]
    lab = pl.read_parquet(DER / "labels.parquet")
    labp = set(lab["p1"].to_list()) | set(lab["p2"].to_list())
    nolab = ~df["pA"].is_in(list(labp)).to_numpy() & ~df["pB"].is_in(list(labp)).to_numpy()
    devlow = df["oof"].fill_null(0.0).to_numpy() < q05
    evallow = ~df["EVHI"].fill_null(False).to_numpy()
    sym = nolab & devlow & evallow & (n1 >= 50) & (n2 >= 50)
    print(f"SYM pairs {sym.sum():,} | mean n dev {n1[sym].mean():.1f} eval {n2[sym].mean():.1f}", flush=True)
    tight = sym & (np.abs(n1 - n2) <= 10)
    rows = []
    for e, name in enumerate(ALL_EVENTS):
        if name in ("shared", "impneg5", "seat_next"):
            continue
        F1, F2 = flag_defs(S1, e), flag_defs(S2, e)
        for fk in F1:
            for grp, msk in (("sym", sym), ("tight", tight)):
                b = int((F1[fk] & ~F2[fk] & msk).sum())
                c = int((F2[fk] & ~F1[fk] & msk).sum())
                both = int((F1[fk] & F2[fk] & msk).sum())
                z = (c - b) / np.sqrt(max(b + c, 1))
                rows.append({"event": name, "flag": fk, "grp": grp, "dev_only": b, "eval_only": c, "both": both,
                             "z_eval_excess": round(float(z), 2)})
    X = pl.DataFrame(rows)
    X.write_csv(FOR / "oc_phase_mcnemar.csv")
    top = X.filter((col("grp") == "sym") & ((col("dev_only") + col("eval_only")) >= 10)).sort("z_eval_excess", descending=True)
    print(top.head(25))
    print(top.tail(15))
    print(f"done {time.time() - t0:.0f}s")
    return X



# ------------------------------------------------------------------------------------------- pair profiles
def pair_profile(pA, pB, w, S=None, idx=None):
    """Top events by G for one pair in window w (S/idx cached)."""
    if S is None:
        S = load_ps(w)
    if idx is None:
        idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(S["pA"], S["pB"]))}
    i = idx[(min(pA, pB), max(pA, pB))]
    rows = []
    for e, name in enumerate(ALL_EVENTS):
        for d in ("ab", "ba"):
            g = float(S[f"G_{d}"][i, e])
            c, o, E = float(S[f"c_{d}"][i, e]), float(S[f"o_{d}"][i, e]), float(S[f"E_{d}"][i, e])
            if o == 0:
                continue
            rows.append({"event": name, "dir": d, "c": c, "o": o, "E": round(E, 2), "G": round(g, 1),
                         "rt": int(S[f"rt_{d}"][i, e]), "ct": int(S[f"ct_{d}"][i, e])})
    return pl.DataFrame(rows).sort("G", descending=True)


def stage_profile():
    """python scripts/oc_hunt.py profile <phase> <nhands> pA:pB [pA:pB ...]"""
    phase = int(sys.argv[2])
    nh = int(sys.argv[3])
    prs = [tuple(int(x) for x in a.split(":")) for a in sys.argv[4:]]
    w = 0 if phase == 0 else 2
    S = load_ps(w)
    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(S["pA"], S["pB"]))}
    ptab = pl.read_parquet(OC / "player_local.parquet")
    pl.Config.set_tbl_rows(24)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    for pA, pB in prs:
        table = int(ptab.filter(col("player_idx") == pA)["table_idx"][0])
        print(f"\n########## pair {pA}:{pB} table {table} phase {phase}")
        prof = pair_profile(pA, pB, w, S, idx)
        print(prof.filter(col("G") > 2).head(20))
        D = load_tables(phase, [table])
        ph = pair_hand_events(D, table, pA, pB)
        print(f"shared hands {ph.height}")
        both = ph.filter((col("pre_both_vol_ab") == 1))
        sc = (both["imp_ab"].abs() + both["imp_ba"].abs() + both["gift_ab"] + both["gift_ba"]).to_numpy() if both.height else np.array([])
        ordh = both["hand_idx"].to_numpy()[np.argsort(-sc)] if both.height else []
        print(f"-- both-voluntary hands: {both.height}; printing top {min(nh, len(ordh))} by |imp|+gift")
        for hid in list(ordh)[:nh]:
            print(format_hand(D, int(hid), pA, pB))



# ------------------------------------------------------------------------------------------- export
EXPORT_EVENTS = ["limp_first", "pre_both_vol", "pf_limpbehind", "pf_overcall_3rd_Yin", "pf_enter_afterY",
                 "pf_call_Yraise", "mw_checkthrough", "hu_checkdown", "hu_strongcheck", "mw_strongcheck",
                 "third_folds_pair", "betfold_Y", "gift_to_Y", "imp5", "flow_big", "pot3_Ywins", "foldpost_Ysd",
                 "fold_Yahead", "pump_Yahead", "bvb_walk", "blind_fold_steal", "call_Yallin", "mw_bigbet",
                 "sum_imp", "sum_transfer", "sum_gift"]


def famflags(win, negwin_rows=None):
    """High-precision per-family flags: any of the family's 12 best pf statistics above the labelled-negative
    maximum in the SAME window (thresholds re-derived per window, label-free at scoring time)."""
    import json
    sel = json.loads((FOR / "oc_family_feats.json").read_text())
    use = sorted({c for v in sel.values() for c in v})
    pf = pl.read_parquet(DER / f"pf_{win}.parquet", columns=["pA", "pB", "n"] + use)
    lab = pl.read_parquet(DER / "labels.parquet").filter(col("label") == 0)
    neg = pf.join(lab.select(pl.min_horizontal("p1", "p2").alias("pA"), pl.max_horizontal("p1", "p2").alias("pB")),
                  on=["pA", "pB"], how="semi")
    X = np.nan_to_num(pf.select(use).to_numpy(), nan=-1e9)
    T = np.nan_to_num(neg.select(use).to_numpy(), nan=-1e9).max(axis=0)
    ix = {c: i for i, c in enumerate(use)}
    F = {t: np.any(np.vstack([X[:, ix[c]] > T[ix[c]] for c in cs]), axis=0) for t, cs in sel.items()}
    return pf.select("pA", "pB", "n").with_columns(pl.Series("f_DT", F["DT"]), pl.Series("f_SP", F["SP"]),
                                                   pl.Series("f_CI", F["CI"]),
                                                   pl.Series("fam_unexplained", ~(F["DT"] | F["SP"] | F["CI"])))


def stage_export():
    """Write data/derived/oc_pair_scores_{dev_full,eval}.parquet + oc_hand_candidates_eval.parquet."""
    t0 = time.time()
    wins = [(int(x), W_NAMES[int(x)]) for x in os.environ.get("OC_EXPORT_WINDOWS", "0,2").split(",")]
    for w, win in wins:
        S = load_ps(w)
        keep = [e for e, n in enumerate(ALL_EVENTS) if n not in MINE_DROP]
        G = np.maximum(S["G_ab"], S["G_ba"])
        top = np.where(S["G_ab"] >= S["G_ba"], (S["rt_ab"] == 1) & (S["ct_ab"] == 1), (S["rt_ba"] == 1) & (S["ct_ba"] == 1))
        out = pl.DataFrame({"pA": S["pA"], "pB": S["pB"], "table_idx": S["table_idx"],
                            "n": S["o_ab"][:, EI["shared"]].astype(np.int32),
                            "oc_score": np.clip(np.where(top, G, 0.0)[:, keep], 0, 30).sum(1),
                            "oc_gsum": np.clip(G[:, keep], 0, 30).sum(1),
                            "oc_k10": ((G[:, keep] >= 10) & top[:, keep]).sum(1).astype(np.int16),
                            "oc_gmax": G[:, keep].max(1)})
        for name in EXPORT_EVENTS:
            e = ALL_EVENTS.index(name)
            out = out.with_columns(pl.Series(f"G_{name}", G[:, e]), pl.Series(f"mtop_{name}", top[:, e].astype(np.int8)))
        out = out.join(famflags(win), on=["pA", "pB"], how="left")
        if win == "eval":
            ep = pl.read_parquet(DER / "eval_pairs.parquet").select(
                pl.min_horizontal("p1", "p2").alias("pA"), pl.max_horizontal("p1", "p2").alias("pB"), "pair_id")
            out = out.join(ep, on=["pA", "pB"], how="left")
        out.write_parquet(DER / f"oc_pair_scores_{win}.parquet")
        print(f"{win}: {out.height:,} pairs, {out.width} cols -> oc_pair_scores_{win}.parquet", flush=True)
    if os.environ.get("OC_EXPORT_WINDOWS"):
        return
    # hand-level generic candidates for eval pairs that carry no known-family signature but rank high
    sc = pl.read_parquet(DER / "oc_pair_scores_eval.parquet")
    es = pl.read_parquet(BASE / "oof" / "exp005_eval_scores.parquet").with_columns(
        col("risk_raw").rank("ordinal", descending=True).alias("rk"))
    want = (sc.filter(col("pair_id").is_not_null()).join(es.select("pair_id", "rk"), on="pair_id")
            .filter(col("fam_unexplained") & (col("rk") <= 1200)))
    print(f"hand candidates for {want.height} unexplained top-1200 eval pairs", flush=True)
    tabs = sorted(want["table_idx"].unique().to_list())
    rows = []
    D = load_tables(1, tabs)
    tab = D["hands"]["table_idx"].to_numpy()
    for t in tabs:
        r_lo = int(np.searchsorted(tab, t, side="left"))
        r_hi = int(np.searchsorted(tab, t, side="right"))
        HEV, HOP, HSM = hand_events(r_lo, r_hi, *D["args"], NE)
        sp = D["args"][0].reshape(-1, 6)[r_lo:r_hi]
        hid = D["hids"][r_lo:r_hi]
        for pa, pb, pid in want.filter(col("table_idx") == t).select("pA", "pB", "pair_id").iter_rows():
            sa = (sp == pa).argmax(1); ha = (sp == pa).any(1)
            sb = (sp == pb).argmax(1); hb = (sp == pb).any(1)
            m = ha & hb
            if not m.any():
                continue
            q = np.flatnonzero(m)
            a, b = sa[q], sb[q]
            imp = np.abs(HSM[q, a, b, 0]) + np.abs(HSM[q, b, a, 0])
            gift = HSM[q, a, b, 2] + HSM[q, b, a, 2]
            both = HEV[q, a, b, E_PRE_BOTH_VOL].astype(float)
            score = imp + 2 * gift + 3 * both
            order = np.argsort(-score)[:5]
            for rank, o in enumerate(order, 1):
                rows.append({"pair_id": pid, "pA": pa, "pB": pb, "hand_idx": int(hid[q[o]]),
                             "oc_hand_score": float(score[o]), "rank": rank})
    H = pl.DataFrame(rows).join(pl.read_parquet(DER / "hands.parquet", columns=["hand_idx", "hand_id"]), on="hand_idx", how="left")
    H.write_parquet(DER / "oc_hand_candidates_eval.parquet")
    print(f"oc_hand_candidates_eval: {H.height} rows for {H['pair_id'].n_unique()} pairs, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "events"
    {"events": stage_events, "stats": stage_stats, "excess": stage_excess, "print": stage_print, "mine": stage_mine, "phase": stage_phase, "profile": stage_profile, "export": stage_export}[stage]()
