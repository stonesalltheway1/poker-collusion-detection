"""handeval.py -- fast numba Texas Hold'em evaluator + exact/MC equity for our card codes.

Card code = rank*4 + suit, rank '23456789TJQKA' (0..12), suit 'cdhs' (0..3), i.e. 2c=0 ... As=51.
(This is identical to phevaluator's card id, so phevaluator.evaluate_cards(*codes) works directly.)

RANK CONVENTION
  rank = dense hand-class index 0..7461, HIGHER = BETTER (0 = 7-5-4-3-2 offsuit, 7461 = royal flush).
  Exactly equals 7462 - phevaluator.evaluate_cards(...)   (verified, see __main__ self-test).
  Works for 5, 6 or 7 cards (best 5-card hand). -1 means "not available".

PUBLIC API (numpy in / numpy out; all heavy loops are numba, parallel ones use prange)
  rank_cards(cards)                         -> int     single hand, 5..7 codes
  rank_batch(cards2d, counts=None)          -> int16[n] rows of up to 7 codes (-1 padded ok when counts given)
  rank_hole_board(c1, c2, board5, nboard, upto)
                                            -> int16[n] rank of hole + first `upto` board cards (3/4/5), -1 if nboard<upto
  table_ranks(holes, board5, nboard, upto)  -> int16[n,6] same for (n,P,2) hole matrix (P<=6; c<0 => empty seat -> -1)
  hand_category(rank)                       -> int8 0..8 (-1 for -1): 0 high,1 pair,2 two_pair,3 trips,4 straight,
                                               5 flush,6 full_house,7 quads,8 straight_flush.  CATEGORY_NAMES
  preflop_class(c1, c2)                     -> int16 0..168 = 13x13 grid index: row/col 0=A..12=2;
                                               pairs on diagonal, suited above (row=hi), offsuit below (row=lo).
                                               AA=0, AKs=1, AKo=13, 22=168.  preflop_class_name(idx), PREFLOP_COMBOS
  equity_hu(h1, h2, board=())               -> (eq1, win1, tie)  exact enumeration of all runouts
  equity_multi(holes, board=(), dead=())    -> float64[k] exact equity (win + split share) for k<=6 hands
  equity_batch(holes, active, board5, nboard, max_exact=20000, n_mc=2000, seed=0)
                                            -> float32[n,P]  per-hand all-in equity of the ACTIVE seats
                                               (inactive seats -> NaN). All holes in the row (active or not,
                                               c>=0) are dead cards. Exact enumeration when #runouts<=max_exact,
                                               else Monte-Carlo with n_mc runouts (deterministic per row+seed).
  preflop_equity_table(n_samples, seed)     -> (eq_vs1[169], eq_vs5[169]) MC equity vs 1 / 5 random hands.

THREADS: numba threads default to env HANDEVAL_THREADS (default 3) -- shared machine.
"""
from __future__ import annotations

import os
from itertools import combinations

import numba as nb
import numpy as np
from numba import njit, prange

try:
    nb.set_num_threads(max(1, min(int(os.environ.get("HANDEVAL_THREADS", "3")), nb.config.NUMBA_NUM_THREADS)))
except Exception:  # pragma: no cover
    pass

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"
CATEGORY_NAMES = ("high_card", "pair", "two_pair", "trips", "straight", "flush",
                  "full_house", "quads", "straight_flush")
N_CLASSES = 7462


def card_code(s: str) -> int:
    """'As' -> 51."""
    return RANK_CHARS.index(s[0].upper()) * 4 + SUIT_CHARS.index(s[1].lower())


def card_str(c: int) -> str:
    return RANK_CHARS[c >> 2] + SUIT_CHARS[c & 3]


def parse_cards(s: str) -> list[int]:
    """'Ah Kd 2c' or 'AhKd2c' -> codes."""
    s = s.replace(" ", "")
    return [card_code(s[i:i + 2]) for i in range(0, len(s), 2)]


# ----------------------------------------------------------------------------------------------
# small 13-bit rank-mask tables
# ----------------------------------------------------------------------------------------------
def _build_mask_tables():
    straight_hi = np.full(8192, -1, np.int8)
    hibit = np.zeros(8192, np.int8)
    popcnt = np.zeros(8192, np.int8)
    top5 = np.zeros(8192, np.int32)  # packed top-5 ranks (4 bits each, highest first)
    for m in range(8192):
        popcnt[m] = bin(m).count("1")
        if m:
            hibit[m] = m.bit_length() - 1
        for hi in range(12, 3, -1):
            w = 0b11111 << (hi - 4)
            if m & w == w:
                straight_hi[m] = hi
                break
        else:
            if m & 0b1000000001111 == 0b1000000001111:  # wheel A2345 -> high card '5' (rank 3)
                straight_hi[m] = 3
        v, k, mm = 0, 0, m
        while mm and k < 5:
            b = mm.bit_length() - 1
            v |= b << (4 * (4 - k))
            mm &= ~(1 << b)
            k += 1
        top5[m] = v
    return straight_hi, hibit, popcnt, top5


STRAIGHT_HI, HIBIT, POPCNT, TOP5 = _build_mask_tables()


@njit(cache=True, inline="always")
def _add_card(m1, m2, m3, m4, s0, s1, s2, s3, c):
    b = 1 << (c >> 2)
    if m1 & b == 0:
        m1 |= b
    elif m2 & b == 0:
        m2 |= b
    elif m3 & b == 0:
        m3 |= b
    else:
        m4 |= b
    su = c & 3
    if su == 0:
        s0 |= b
    elif su == 1:
        s1 |= b
    elif su == 2:
        s2 |= b
    else:
        s3 |= b
    return m1, m2, m3, m4, s0, s1, s2, s3


@njit(cache=True)
def _packed(m1, m2, m3, m4, s0, s1, s2, s3, SH, HB, PC, T5):
    """Packed comparable value: category<<20 | r1<<16 | r2<<12 | r3<<8 | r4<<4 | r5."""
    fm = 0
    if PC[s0] >= 5:
        fm = s0
    elif PC[s1] >= 5:
        fm = s1
    elif PC[s2] >= 5:
        fm = s2
    elif PC[s3] >= 5:
        fm = s3
    if fm != 0:
        sh = SH[fm]
        if sh >= 0:
            return (8 << 20) | (np.int32(sh) << 16)
    if m4 != 0:
        q = np.int32(HB[m4])
        k = np.int32(HB[m1 & ~(1 << q)])
        return (7 << 20) | (q << 16) | (k << 12)
    if m3 != 0:
        t = np.int32(HB[m3])
        rest = m2 & ~(1 << t)
        if rest != 0:
            return (6 << 20) | (t << 16) | (np.int32(HB[rest]) << 12)
    if fm != 0:
        return (5 << 20) | T5[fm]
    sh = SH[m1]
    if sh >= 0:
        return (4 << 20) | (np.int32(sh) << 16)
    if m3 != 0:
        t = np.int32(HB[m3])
        kick = m1 & ~(1 << t)
        k1 = np.int32(HB[kick])
        kick &= ~(1 << k1)
        k2 = np.int32(HB[kick])
        return (3 << 20) | (t << 16) | (k1 << 12) | (k2 << 8)
    if m2 != 0:
        p1 = np.int32(HB[m2])
        rest = m2 & ~(1 << p1)
        if rest != 0:
            p2 = np.int32(HB[rest])
            kick = m1 & ~(1 << p1) & ~(1 << p2)
            return (2 << 20) | (p1 << 16) | (p2 << 12) | (np.int32(HB[kick]) << 8)
        kick = m1 & ~(1 << p1)
        k1 = np.int32(HB[kick])
        kick &= ~(1 << k1)
        k2 = np.int32(HB[kick])
        kick &= ~(1 << k2)
        k3 = np.int32(HB[kick])
        return (1 << 20) | (p1 << 16) | (k1 << 12) | (k2 << 8) | (k3 << 4)
    return T5[m1]


@njit(cache=True)
def _packed_cards(cards, n, SH, HB, PC, T5):
    m1 = m2 = m3 = m4 = s0 = s1 = s2 = s3 = 0
    for i in range(n):
        m1, m2, m3, m4, s0, s1, s2, s3 = _add_card(m1, m2, m3, m4, s0, s1, s2, s3, cards[i])
    return _packed(m1, m2, m3, m4, s0, s1, s2, s3, SH, HB, PC, T5)


@njit(cache=True)
def _all_packed5(SH, HB, PC, T5):
    out = np.empty(2598960, np.int32)
    buf = np.empty(5, np.int8)
    k = 0
    for a in range(48):
        buf[0] = a
        for b in range(a + 1, 49):
            buf[1] = b
            for c in range(b + 1, 50):
                buf[2] = c
                for d in range(c + 1, 51):
                    buf[3] = d
                    for e in range(d + 1, 52):
                        buf[4] = e
                        out[k] = _packed_cards(buf, 5, SH, HB, PC, T5)
                        k += 1
    return out


def _build_dense():
    vals = np.unique(_all_packed5(STRAIGHT_HI, HIBIT, POPCNT, TOP5))
    assert len(vals) == N_CLASSES, len(vals)
    dense = np.full(9 << 20, -1, np.int16)
    dense[vals] = np.arange(N_CLASSES, dtype=np.int16)
    cat = (vals >> 20).astype(np.int8)
    return dense, cat, vals


DENSE, CAT_OF_RANK, PACKED_OF_RANK = _build_dense()
CAT_START = np.searchsorted(CAT_OF_RANK, np.arange(9)).astype(np.int16)  # first rank of each category


# ----------------------------------------------------------------------------------------------
# rank evaluation
# ----------------------------------------------------------------------------------------------
@njit(cache=True)
def _rank_state(m1, m2, m3, m4, s0, s1, s2, s3, SH, HB, PC, T5, DN):
    return DN[_packed(m1, m2, m3, m4, s0, s1, s2, s3, SH, HB, PC, T5)]


@njit(cache=True, parallel=True)
def _rank_batch(cards, counts, SH, HB, PC, T5, DN):
    n = cards.shape[0]
    out = np.empty(n, np.int16)
    for i in prange(n):
        k = counts[i]
        if k < 5:
            out[i] = -1
            continue
        m1 = m2 = m3 = m4 = s0 = s1 = s2 = s3 = 0
        for j in range(k):
            m1, m2, m3, m4, s0, s1, s2, s3 = _add_card(m1, m2, m3, m4, s0, s1, s2, s3, cards[i, j])
        out[i] = DN[_packed(m1, m2, m3, m4, s0, s1, s2, s3, SH, HB, PC, T5)]
    return out


def rank_batch(cards2d, counts=None) -> np.ndarray:
    """cards2d: (n, k<=7) codes. counts: cards used per row (default k). Rows with <5 cards -> -1."""
    cards2d = np.ascontiguousarray(cards2d, dtype=np.int8)
    if counts is None:
        counts = np.full(cards2d.shape[0], cards2d.shape[1], np.int8)
    return _rank_batch(cards2d, np.ascontiguousarray(counts, dtype=np.int8), STRAIGHT_HI, HIBIT, POPCNT, TOP5, DENSE)


def rank_cards(cards) -> int:
    a = np.asarray(cards, dtype=np.int8).reshape(1, -1)
    return int(rank_batch(a)[0])


@njit(cache=True, parallel=True)
def _table_ranks(holes, board, nboard, upto, SH, HB, PC, T5, DN):
    n, P = holes.shape[0], holes.shape[1]
    out = np.full((n, P), -1, np.int16)
    for i in prange(n):
        if nboard[i] < upto:
            continue
        b1 = b2 = b3 = b4 = t0 = t1 = t2 = t3 = 0
        for j in range(upto):
            b1, b2, b3, b4, t0, t1, t2, t3 = _add_card(b1, b2, b3, b4, t0, t1, t2, t3, board[i, j])
        for p in range(P):
            c1 = holes[i, p, 0]
            c2 = holes[i, p, 1]
            if c1 < 0 or c2 < 0:
                continue
            m1, m2, m3, m4, s0, s1, s2, s3 = _add_card(b1, b2, b3, b4, t0, t1, t2, t3, c1)
            m1, m2, m3, m4, s0, s1, s2, s3 = _add_card(m1, m2, m3, m4, s0, s1, s2, s3, c2)
            out[i, p] = DN[_packed(m1, m2, m3, m4, s0, s1, s2, s3, SH, HB, PC, T5)]
    return out


def table_ranks(holes, board5, nboard, upto: int) -> np.ndarray:
    """holes (n,P,2) codes (-1 = empty seat), board5 (n,5) codes (-1 pad), nboard (n,). -> int16 (n,P)."""
    holes = np.ascontiguousarray(holes, dtype=np.int8)
    board5 = np.ascontiguousarray(board5, dtype=np.int8)
    nboard = np.ascontiguousarray(nboard, dtype=np.int8)
    return _table_ranks(holes, board5, nboard, np.int64(upto), STRAIGHT_HI, HIBIT, POPCNT, TOP5, DENSE)


def rank_hole_board(c1, c2, board5, nboard, upto: int) -> np.ndarray:
    holes = np.stack([np.asarray(c1, np.int8), np.asarray(c2, np.int8)], axis=1)[:, None, :]
    return table_ranks(holes, board5, nboard, upto)[:, 0]


def hand_category(rank):
    """dense rank (scalar or array) -> category 0..8; -1 stays -1."""
    r = np.asarray(rank)
    out = np.where(r >= 0, CAT_OF_RANK[np.clip(r, 0, N_CLASSES - 1)], -1).astype(np.int8)
    return int(out) if out.ndim == 0 else out


def describe_rank(rank: int) -> str:
    if rank < 0:
        return "n/a"
    p = int(PACKED_OF_RANK[rank])
    cat = p >> 20
    nsig = (5, 4, 3, 3, 1, 5, 2, 2, 1)[cat]  # significant rank slots per category
    rs = [(p >> s) & 15 for s in (16, 12, 8, 4, 0)][:nsig]
    return f"{CATEGORY_NAMES[cat]} " + "".join(RANK_CHARS[x] for x in rs)


# ----------------------------------------------------------------------------------------------
# preflop 169 classes
# ----------------------------------------------------------------------------------------------
def preflop_class(c1, c2):
    """13x13 grid index (row/col 0 = Ace). Pairs diagonal, suited upper (row=hi), offsuit lower (row=lo)."""
    c1 = np.asarray(c1).astype(np.int16)
    c2 = np.asarray(c2).astype(np.int16)
    r1, r2 = c1 >> 2, c2 >> 2
    hi = 12 - np.maximum(r1, r2)
    lo = 12 - np.minimum(r1, r2)  # note: grid coordinate, so hi <= lo
    suited = (c1 & 3) == (c2 & 3)
    out = np.where(suited, hi * 13 + lo, lo * 13 + hi).astype(np.int16)
    return int(out) if out.ndim == 0 else out


def preflop_class_name(idx: int) -> str:
    row, col = divmod(int(idx), 13)
    a, b = RANK_CHARS[12 - min(row, col)], RANK_CHARS[12 - max(row, col)]
    if row == col:
        return a + b
    return a + b + ("s" if row < col else "o")


PREFLOP_NAMES = [preflop_class_name(i) for i in range(169)]
PREFLOP_COMBOS = np.array([6 if i // 13 == i % 13 else (4 if i // 13 < i % 13 else 12) for i in range(169)], np.int16)


# ----------------------------------------------------------------------------------------------
# equity
# ----------------------------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _splitmix(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15))
    z = x
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x, z ^ (z >> np.uint64(31))


@njit(cache=True)
def _n_choose_k(n, k):
    if k < 0 or k > n:
        return 0
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


@njit(cache=True)
def _equity_one(hole_c1, hole_c2, active, board, nb_, deadmask, max_exact, n_mc, seed, SH, HB, PC, T5, DN, eq):
    """Core: fills eq[p] for active seats. hole arrays length P. deadmask: 52-bit mask of known cards."""
    P = hole_c1.shape[0]
    deck = np.empty(52, np.int8)
    nd = 0
    for c in range(52):
        if (deadmask >> np.uint64(c)) & np.uint64(1) == 0:
            deck[nd] = c
            nd += 1
    need = 5 - nb_
    b1 = b2 = b3 = b4 = t0 = t1 = t2 = t3 = 0
    for j in range(nb_):
        b1, b2, b3, b4, t0, t1, t2, t3 = _add_card(b1, b2, b3, b4, t0, t1, t2, t3, board[j])
    acc = np.zeros(P, np.float64)
    ranks = np.empty(P, np.int32)
    total = _n_choose_k(nd, need)
    exact = total <= max_exact
    idx = np.arange(need)
    it = 0
    state = np.uint64(seed)
    n_iter = total if exact else n_mc
    while it < n_iter:
        m1, m2, m3, m4, s0, s1, s2, s3 = b1, b2, b3, b4, t0, t1, t2, t3
        if exact:
            for j in range(need):
                m1, m2, m3, m4, s0, s1, s2, s3 = _add_card(m1, m2, m3, m4, s0, s1, s2, s3, deck[idx[j]])
        else:
            # partial Fisher-Yates on deck copy
            for j in range(need):
                state, r = _splitmix(state)
                k = j + np.int64(r % np.uint64(nd - j))
                tmp = deck[j]
                deck[j] = deck[k]
                deck[k] = tmp
                m1, m2, m3, m4, s0, s1, s2, s3 = _add_card(m1, m2, m3, m4, s0, s1, s2, s3, deck[j])
        best = -1
        nwin = 0
        for p in range(P):
            if not active[p]:
                ranks[p] = -2
                continue
            a1, a2, a3, a4, u0, u1, u2, u3 = _add_card(m1, m2, m3, m4, s0, s1, s2, s3, hole_c1[p])
            a1, a2, a3, a4, u0, u1, u2, u3 = _add_card(a1, a2, a3, a4, u0, u1, u2, u3, hole_c2[p])
            rk = np.int32(DN[_packed(a1, a2, a3, a4, u0, u1, u2, u3, SH, HB, PC, T5)])
            ranks[p] = rk
            if rk > best:
                best = rk
                nwin = 1
            elif rk == best:
                nwin += 1
        share = 1.0 / nwin
        for p in range(P):
            if ranks[p] == best:
                acc[p] += share
        it += 1
        if exact and need > 0:
            # next combination
            j = need - 1
            while j >= 0 and idx[j] == nd - need + j:
                j -= 1
            if j < 0:
                break
            idx[j] += 1
            for q in range(j + 1, need):
                idx[q] = idx[q - 1] + 1
        elif exact:
            break
    for p in range(P):
        eq[p] = acc[p] / it if active[p] else np.nan
    return it


@njit(cache=True, parallel=True)
def _equity_batch(holes, active, board, nboard, max_exact, n_mc, seed, SH, HB, PC, T5, DN):
    n, P = holes.shape[0], holes.shape[1]
    out = np.full((n, P), np.nan, np.float32)
    for i in prange(n):
        dead = np.uint64(0)
        c1 = np.empty(P, np.int8)
        c2 = np.empty(P, np.int8)
        act = np.zeros(P, np.bool_)
        nact = 0
        for p in range(P):
            c1[p] = holes[i, p, 0]
            c2[p] = holes[i, p, 1]
            if c1[p] >= 0 and c2[p] >= 0:
                dead |= (np.uint64(1) << np.uint64(c1[p])) | (np.uint64(1) << np.uint64(c2[p]))
                if active[i, p]:
                    act[p] = True
                    nact += 1
        if nact == 0:
            continue
        nb_ = nboard[i]
        for j in range(nb_):
            dead |= np.uint64(1) << np.uint64(board[i, j])
        eq = np.empty(P, np.float64)
        _equity_one(c1, c2, act, board[i], nb_, dead, max_exact, n_mc,
                    np.uint64(seed) * np.uint64(1000003) + np.uint64(i), SH, HB, PC, T5, DN, eq)
        for p in range(P):
            if act[p]:
                out[i, p] = eq[p]
    return out


def equity_batch(holes, active, board5, nboard, max_exact: int = 20000, n_mc: int = 2000, seed: int = 0):
    """holes (n,P,2) int8 (-1 = empty), active (n,P) bool, board5 (n,5) int8 (-1 pad), nboard (n,) 0..5.
    Returns float32 (n,P) equity (win + split share) for active seats; NaN elsewhere."""
    return _equity_batch(np.ascontiguousarray(holes, np.int8), np.ascontiguousarray(active, np.bool_),
                         np.ascontiguousarray(board5, np.int8), np.ascontiguousarray(nboard, np.int64),
                         np.int64(max_exact), np.int64(n_mc), np.int64(seed),
                         STRAIGHT_HI, HIBIT, POPCNT, TOP5, DENSE)


def _as_codes(x):
    if isinstance(x, str):
        return parse_cards(x)
    return [card_code(c) if isinstance(c, str) else int(c) for c in x]


def equity_multi(holes, board=(), dead=(), max_exact: int = 10**9, n_mc: int = 200000, seed: int = 0) -> np.ndarray:
    """Exact (default) all-in equity for k<=6 known hands. holes: list of 2-card lists/strings."""
    hs = [_as_codes(h) for h in holes]
    bd = _as_codes(board)
    P = len(hs)
    c1 = np.array([h[0] for h in hs], np.int8)
    c2 = np.array([h[1] for h in hs], np.int8)
    b = np.full(5, -1, np.int8)
    b[:len(bd)] = bd
    deadmask = 0
    for c in list(c1) + list(c2) + bd + _as_codes(dead):
        deadmask |= 1 << int(c)
    eq = np.empty(P, np.float64)
    _equity_one(c1, c2, np.ones(P, np.bool_), b, len(bd), np.uint64(deadmask), np.int64(max_exact),
                np.int64(n_mc), np.uint64(seed), STRAIGHT_HI, HIBIT, POPCNT, TOP5, DENSE, eq)
    return eq


@njit(cache=True)
def _hu_win_tie(h1a, h1b, h2a, h2b, board, nb_, SH, HB, PC, T5, DN):
    dead = np.uint64(0)
    for c in (h1a, h1b, h2a, h2b):
        dead |= np.uint64(1) << np.uint64(c)
    for j in range(nb_):
        dead |= np.uint64(1) << np.uint64(board[j])
    deck = np.empty(52, np.int8)
    nd = 0
    for c in range(52):
        if (dead >> np.uint64(c)) & np.uint64(1) == 0:
            deck[nd] = c
            nd += 1
    need = 5 - nb_
    b1 = b2 = b3 = b4 = t0 = t1 = t2 = t3 = 0
    for j in range(nb_):
        b1, b2, b3, b4, t0, t1, t2, t3 = _add_card(b1, b2, b3, b4, t0, t1, t2, t3, board[j])
    win = 0
    tie = 0
    tot = 0
    idx = np.arange(need)
    while True:
        m1, m2, m3, m4, s0, s1, s2, s3 = b1, b2, b3, b4, t0, t1, t2, t3
        for j in range(need):
            m1, m2, m3, m4, s0, s1, s2, s3 = _add_card(m1, m2, m3, m4, s0, s1, s2, s3, deck[idx[j]])
        a1, a2, a3, a4, u0, u1, u2, u3 = _add_card(m1, m2, m3, m4, s0, s1, s2, s3, h1a)
        a1, a2, a3, a4, u0, u1, u2, u3 = _add_card(a1, a2, a3, a4, u0, u1, u2, u3, h1b)
        r1 = DN[_packed(a1, a2, a3, a4, u0, u1, u2, u3, SH, HB, PC, T5)]
        a1, a2, a3, a4, u0, u1, u2, u3 = _add_card(m1, m2, m3, m4, s0, s1, s2, s3, h2a)
        a1, a2, a3, a4, u0, u1, u2, u3 = _add_card(a1, a2, a3, a4, u0, u1, u2, u3, h2b)
        r2 = DN[_packed(a1, a2, a3, a4, u0, u1, u2, u3, SH, HB, PC, T5)]
        if r1 > r2:
            win += 1
        elif r1 == r2:
            tie += 1
        tot += 1
        if need == 0:
            break
        j = need - 1
        while j >= 0 and idx[j] == nd - need + j:
            j -= 1
        if j < 0:
            break
        idx[j] += 1
        for q in range(j + 1, need):
            idx[q] = idx[q - 1] + 1
    return win, tie, tot


def equity_hu(h1, h2, board=()):
    """Exact heads-up equity. Returns (equity_h1, p_win_h1, p_tie)."""
    a, b = _as_codes(h1), _as_codes(h2)
    bd = _as_codes(board)
    barr = np.full(5, -1, np.int8)
    barr[:len(bd)] = bd
    w, t, n = _hu_win_tie(np.int8(a[0]), np.int8(a[1]), np.int8(b[0]), np.int8(b[1]), barr, len(bd),
                          STRAIGHT_HI, HIBIT, POPCNT, TOP5, DENSE)
    return (w + 0.5 * t) / n, w / n, t / n


@njit(cache=True, parallel=True)
def _preflop_vs_random(reps, n_opp, n_samples, seed, SH, HB, PC, T5, DN):
    """reps (169,2): MC equity of each rep hand vs n_opp uniformly random hands (all-in to river)."""
    out = np.zeros(169, np.float64)
    for ci in prange(169):
        h1 = reps[ci, 0]
        h2 = reps[ci, 1]
        deck = np.empty(50, np.int8)
        nd = 0
        for c in range(52):
            if c != h1 and c != h2:
                deck[nd] = c
                nd += 1
        state = np.uint64(seed) * np.uint64(7919) + np.uint64(ci)
        acc = 0.0
        need = 5 + 2 * n_opp
        for s in range(n_samples):
            for j in range(need):
                state, r = _splitmix(state)
                k = j + np.int64(r % np.uint64(nd - j))
                tmp = deck[j]
                deck[j] = deck[k]
                deck[k] = tmp
            m1 = m2 = m3 = m4 = s0 = s1 = s2 = s3 = 0
            for j in range(5):
                m1, m2, m3, m4, s0, s1, s2, s3 = _add_card(m1, m2, m3, m4, s0, s1, s2, s3, deck[j])
            a1, a2, a3, a4, u0, u1, u2, u3 = _add_card(m1, m2, m3, m4, s0, s1, s2, s3, h1)
            a1, a2, a3, a4, u0, u1, u2, u3 = _add_card(a1, a2, a3, a4, u0, u1, u2, u3, h2)
            me = DN[_packed(a1, a2, a3, a4, u0, u1, u2, u3, SH, HB, PC, T5)]
            best_opp = -1
            nties = 0
            for o in range(n_opp):
                a1, a2, a3, a4, u0, u1, u2, u3 = _add_card(m1, m2, m3, m4, s0, s1, s2, s3, deck[5 + 2 * o])
                a1, a2, a3, a4, u0, u1, u2, u3 = _add_card(a1, a2, a3, a4, u0, u1, u2, u3, deck[6 + 2 * o])
                ro = DN[_packed(a1, a2, a3, a4, u0, u1, u2, u3, SH, HB, PC, T5)]
                if ro > best_opp:
                    best_opp = ro
                    nties = 1 if ro == me else 0
                elif ro == best_opp and ro == me:
                    nties += 1
            if me > best_opp:
                acc += 1.0
            elif me == best_opp:
                acc += 1.0 / (nties + 1)
        out[ci] = acc / n_samples
    return out


def preflop_equity_table(n_samples: int = 400000, seed: int = 12345):
    """MC all-in equity of each of the 169 classes vs 1 and vs 5 random hands. Returns (eq_vs1, eq_vs5)."""
    reps = np.zeros((169, 2), np.int8)
    for i in range(169):
        row, col = divmod(i, 13)
        if row == col:
            r = 12 - row
            reps[i] = (r * 4, r * 4 + 1)
        elif row < col:  # suited
            reps[i] = ((12 - row) * 4, (12 - col) * 4)
        else:  # offsuit
            reps[i] = ((12 - col) * 4, (12 - row) * 4 + 1)
    assert (preflop_class(reps[:, 0], reps[:, 1]) == np.arange(169)).all()
    args = (STRAIGHT_HI, HIBIT, POPCNT, TOP5, DENSE)
    return (_preflop_vs_random(reps, 1, n_samples, seed, *args),
            _preflop_vs_random(reps, 5, n_samples, seed + 1, *args))


# ----------------------------------------------------------------------------------------------
# self-test / benchmark:  python src/handeval.py [n_verify]
# ----------------------------------------------------------------------------------------------
def _brute_equity(holes, board):
    """reference equity via itertools + phevaluator (slow, for tests only)."""
    from phevaluator import evaluate_cards
    used = set(c for h in holes for c in h) | set(board)
    deck = [c for c in range(52) if c not in used]
    acc = np.zeros(len(holes))
    n = 0
    for ro in combinations(deck, 5 - len(board)):
        bd = list(board) + list(ro)
        rs = [evaluate_cards(*(bd + list(h))) for h in holes]  # lower better
        m = min(rs)
        w = [i for i, r in enumerate(rs) if r == m]
        for i in w:
            acc[i] += 1.0 / len(w)
        n += 1
    return acc / n


if __name__ == "__main__":
    import sys
    import time

    from phevaluator import evaluate_cards

    nver = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
    rng = np.random.default_rng(0)
    print(f"numba threads = {nb.get_num_threads()}")
    print("category starts:", dict(zip(CATEGORY_NAMES, CAT_START.tolist())))

    for k in (5, 6, 7):
        cards = np.argsort(rng.random((nver, 52)), axis=1)[:, :k].astype(np.int8)
        ours = rank_batch(cards)
        t = time.perf_counter()
        ref = np.array([evaluate_cards(*row) for row in cards.tolist()], np.int32)
        tp = time.perf_counter() - t
        mism = int((ours.astype(np.int32) != (N_CLASSES - ref)).sum())
        # ordering check on random pairs (redundant given exact equality, kept explicit)
        i, j = rng.integers(0, nver, (2, nver))
        ordmis = int((np.sign(ours[i].astype(np.int32) - ours[j]) != np.sign(ref[j] - ref[i])).sum())
        print(f"{k}-card: {nver:,} hands  rank mismatches={mism}  ordering mismatches={ordmis}  "
              f"(phevaluator python loop {nver / tp:,.0f} hands/s)")

    # throughput
    n = 12_000_000
    cards = np.argsort(rng.random((n // 12, 52)), axis=1)[:, :7].astype(np.int8)
    cards = np.tile(cards, (12, 1))
    rank_batch(cards[:1000])
    for th in sorted({1, nb.get_num_threads()}):
        nb.set_num_threads(th)
        t = time.perf_counter()
        rank_batch(cards)
        dt = time.perf_counter() - t
        print(f"rank_batch 7-card, {th} thread(s): {n / dt / 1e6:.1f} M hands/s")

    # equity correctness vs brute force
    tests = [(["As", "Ks"], ["Qd", "Qc"], []), (["Ah", "Ad"], ["7c", "2d"], ["Ac", "7d", "7h"]),
             (["9s", "8s"], ["Ac", "Kd"], ["Ts", "7s", "2c", "2h"])]
    for h1, h2, bd in tests:
        e = equity_hu(h1, h2, bd)
        t = time.perf_counter()
        e2 = equity_hu(h1, h2, bd)
        dt = time.perf_counter() - t
        hs = [parse_cards("".join(h1)), parse_cards("".join(h2))]
        ref = _brute_equity(hs, parse_cards("".join(bd))) if bd else None
        em = equity_multi([hs[0], hs[1]], parse_cards("".join(bd)))
        print(f"HU {h1} vs {h2} board {bd}: eq={e[0]:.5f} win={e[1]:.5f} tie={e[2]:.5f} "
              f"({dt * 1e3:.1f} ms)  multi={em[0]:.5f}" + (f"  brute={ref[0]:.5f}" if ref is not None else ""))
    six = [parse_cards(x) for x in ("AsKs", "QdQc", "JhTh", "7c7d", "Ad5d", "9s8s")]
    t = time.perf_counter()
    e6 = equity_multi(six)
    dt = time.perf_counter() - t
    print(f"6-way preflop exact: {np.round(e6, 4).tolist()} sum={e6.sum():.6f} ({dt * 1e3:.0f} ms)")
    fl = parse_cards("2s7hKd")
    e6f = equity_multi(six, fl)
    ref = _brute_equity(six, fl)
    print(f"6-way flop exact max|diff| vs brute = {np.abs(e6f - ref).max():.2e}")

    # batch equity throughput on random 6-handed spots
    m = 20000
    perm = np.argsort(rng.random((m, 52)), axis=1).astype(np.int8)
    holes = perm[:, :12].reshape(m, 6, 2)
    board5 = perm[:, 12:17]
    active = rng.random((m, 6)) < 0.5
    active[:, :2] = True
    equity_batch(holes[:10], active[:10], board5[:10], np.full(10, 3))
    for nbd in (4, 3, 0):
        nboard = np.full(m, nbd)
        mm = m if nbd else 2000
        t = time.perf_counter()
        eq = equity_batch(holes[:mm], active[:mm], board5[:mm], nboard[:mm], max_exact=20000, n_mc=2000)
        dt = time.perf_counter() - t
        s = np.nansum(eq, axis=1)
        print(f"equity_batch nboard={nbd}: {mm / dt:,.0f} spots/s ({'exact' if nbd else 'MC 2000'}), "
              f"row-sum range [{s.min():.4f},{s.max():.4f}]")
