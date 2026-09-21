"""Fast NLHE hand evaluation + exact / Monte-Carlo equity with full hole-card visibility (numba).

Card code = rank*4 + suit, rank 0..12 = '23456789TJQKA', suit 0..3 = 'cdhs'
(identical to phevaluator's Card ids, so results are cross-checkable).

Core API (all njit, arrays are int64 unless noted):
  eval_cards(cards, n)            -> int score of best 5-of-n (5..7) cards, HIGHER = stronger.
                                     category = score >> 20 (0 high,1 pair,2 two pair,3 trips,
                                     4 straight,5 flush,6 full house,7 quads,8 straight flush)
  equity_exact(holes, nplayers, board, nboard, dead, ndead, out)
                                  -> fills out[i] = pot-share equity of player i (ties split)
                                     by enumerating every remaining runout. Cost:
                                     river 1 runout, turn <=46, flop <=1081, preflop C(n,5)
                                     (1.7M heads-up: use equity_mc preflop).
  equity_mc(..., nsim, seed, out) -> same, sampling runouts (use preflop).
  hand_strength_vs_random(h1,h2,board,nboard,dead,ndead)
                                  -> (HS now vs one random unseen hand, share of combos beaten/tied)
                                     = what the DECIDER can know (no opponents' cards).
  class_id(c1, c2)                -> 0..168 starting-hand class (see class_name()).

Python helpers: class_name(cid), preflop_class_table(), card_str(code), parse_cards("AhKd").
Build of data/derived/preflop_equity_169.parquet: `python src/poker_equity.py build_preflop`.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from numba import njit, prange

RANKS = "23456789TJQKA"
SUITS = "cdhs"

# ----------------------------------------------------------------------------- lookup tables


def _build_tables():
    straight = np.full(8192, -1, dtype=np.int64)
    top5 = np.zeros(8192, dtype=np.int64)
    highbit = np.full(8192, -1, dtype=np.int64)
    popc = np.zeros(8192, dtype=np.int64)
    for m in range(8192):
        bits = [r for r in range(12, -1, -1) if m >> r & 1]
        popc[m] = len(bits)
        if bits:
            highbit[m] = bits[0]
        enc = 0
        for i in range(5):
            r = bits[i] if i < len(bits) else 0
            enc = (enc << 4) | r
        top5[m] = enc
        for hi in range(12, 3, -1):
            if (m >> (hi - 4)) & 0x1F == 0x1F:
                straight[m] = hi
                break
        else:
            if m & 0x100F == 0x100F:  # wheel A2345
                straight[m] = 3
    return straight, top5, highbit, popc


STRAIGHT_HIGH, TOP5, HIGHBIT, POPCOUNT = _build_tables()

# ----------------------------------------------------------------------------- evaluator


@njit(cache=True, inline="always")
def _score(m1, m2, m3, m4, sm0, sm1, sm2, sm3, sc0, sc1, sc2, sc3):
    fm = 0
    if sc0 >= 5:
        fm = sm0
    elif sc1 >= 5:
        fm = sm1
    elif sc2 >= 5:
        fm = sm2
    elif sc3 >= 5:
        fm = sm3
    if fm != 0:
        sf = STRAIGHT_HIGH[fm]
        if sf >= 0:
            return (8 << 20) | (sf << 16)
    if m4 != 0:
        q = HIGHBIT[m4]
        k = HIGHBIT[m1 & ~(1 << q)]
        return (7 << 20) | (q << 16) | (k << 12)
    if m3 != 0:
        t = HIGHBIT[m3]
        rest = m2 & ~(1 << t)
        if rest != 0:
            p = HIGHBIT[rest]
            return (6 << 20) | (t << 16) | (p << 12)
    if fm != 0:
        return (5 << 20) | TOP5[fm]
    st = STRAIGHT_HIGH[m1]
    if st >= 0:
        return (4 << 20) | (st << 16)
    if m3 != 0:
        t = HIGHBIT[m3]
        return (3 << 20) | (t << 16) | ((TOP5[m1 & ~(1 << t)] >> 12) << 8)
    if m2 != 0:
        p1 = HIGHBIT[m2]
        rem = m2 & ~(1 << p1)
        if rem != 0:
            p2 = HIGHBIT[rem]
            k = HIGHBIT[m1 & ~(1 << p1) & ~(1 << p2)]
            if k < 0:
                k = 0
            return (2 << 20) | (p1 << 16) | (p2 << 12) | (k << 8)
        return (1 << 20) | (p1 << 16) | ((TOP5[m1 & ~(1 << p1)] >> 8) << 4)
    return TOP5[m1]


@njit(cache=True)
def eval_cards(cards, n):
    """Best-5 score of cards[0:n] (5<=n<=7). Higher is stronger."""
    m1 = 0
    m2 = 0
    m3 = 0
    m4 = 0
    sm0 = 0
    sm1 = 0
    sm2 = 0
    sm3 = 0
    sc0 = 0
    sc1 = 0
    sc2 = 0
    sc3 = 0
    for i in range(n):
        c = cards[i]
        r = c >> 2
        s = c & 3
        b = 1 << r
        if m3 & b:
            m4 |= b
        elif m2 & b:
            m3 |= b
        elif m1 & b:
            m2 |= b
        else:
            m1 |= b
        if s == 0:
            sm0 |= b
            sc0 += 1
        elif s == 1:
            sm1 |= b
            sc1 += 1
        elif s == 2:
            sm2 |= b
            sc2 += 1
        else:
            sm3 |= b
            sc3 += 1
    return _score(m1, m2, m3, m4, sm0, sm1, sm2, sm3, sc0, sc1, sc2, sc3)


@njit(cache=True)
def eval7(a, b, c, d, e, f, g):
    arr = np.empty(7, dtype=np.int64)
    arr[0] = a
    arr[1] = b
    arr[2] = c
    arr[3] = d
    arr[4] = e
    arr[5] = f
    arr[6] = g
    return eval_cards(arr, 7)


# ----------------------------------------------------------------------------- equity


@njit(cache=True)
def _remaining_deck(holes, nplayers, board, nboard, dead, ndead):
    used = np.zeros(52, dtype=np.bool_)
    for i in range(nplayers):
        used[holes[i, 0]] = True
        used[holes[i, 1]] = True
    for i in range(nboard):
        used[board[i]] = True
    for i in range(ndead):
        if dead[i] >= 0:
            used[dead[i]] = True
    deck = np.empty(52, dtype=np.int64)
    k = 0
    for c in range(52):
        if not used[c]:
            deck[k] = c
            k += 1
    return deck, k


@njit(cache=True)
def _settle(holes, nplayers, full_board, acc, scores, cards7):
    best = -1
    nbest = 0
    for i in range(nplayers):
        cards7[0] = holes[i, 0]
        cards7[1] = holes[i, 1]
        for j in range(5):
            cards7[2 + j] = full_board[j]
        s = eval_cards(cards7, 7)
        scores[i] = s
        if s > best:
            best = s
            nbest = 1
        elif s == best:
            nbest += 1
    share = 1.0 / nbest
    for i in range(nplayers):
        if scores[i] == best:
            acc[i] += share


@njit(cache=True)
def equity_exact(holes, nplayers, board, nboard, dead, ndead, out):
    """Exact pot-share equity for live players holes[0:nplayers] (shape [k,2]).
    board[0:nboard] known community cards (0,3,4,5); dead[0:ndead] = other known cards
    (e.g. folded players' hole cards; -1 entries ignored). Returns number of runouts."""
    deck, nd = _remaining_deck(holes, nplayers, board, nboard, dead, ndead)
    need = 5 - nboard
    fb = np.empty(5, dtype=np.int64)
    for j in range(nboard):
        fb[j] = board[j]
    acc = np.zeros(nplayers, dtype=np.float64)
    scores = np.empty(nplayers, dtype=np.int64)
    cards7 = np.empty(7, dtype=np.int64)
    cnt = 0
    if need == 0:
        _settle(holes, nplayers, fb, acc, scores, cards7)
        cnt = 1
    elif need == 1:
        for a in range(nd):
            fb[4] = deck[a]
            _settle(holes, nplayers, fb, acc, scores, cards7)
            cnt += 1
    elif need == 2:
        for a in range(nd):
            fb[3] = deck[a]
            for b in range(a + 1, nd):
                fb[4] = deck[b]
                _settle(holes, nplayers, fb, acc, scores, cards7)
                cnt += 1
    else:  # need 5 (preflop) -- 1.7M runouts heads-up; prefer equity_mc
        for a in range(nd):
            fb[0] = deck[a]
            for b in range(a + 1, nd):
                fb[1] = deck[b]
                for c in range(b + 1, nd):
                    fb[2] = deck[c]
                    for d in range(c + 1, nd):
                        fb[3] = deck[d]
                        for e in range(d + 1, nd):
                            fb[4] = deck[e]
                            _settle(holes, nplayers, fb, acc, scores, cards7)
                            cnt += 1
    for i in range(nplayers):
        out[i] = acc[i] / cnt
    return cnt


@njit(cache=True)
def equity_mc(holes, nplayers, board, nboard, dead, ndead, nsim, seed, out):
    """Monte-Carlo pot-share equity (sampled runouts); SE ~ 0.5/sqrt(nsim)."""
    np.random.seed(seed)
    deck, nd = _remaining_deck(holes, nplayers, board, nboard, dead, ndead)
    need = 5 - nboard
    fb = np.empty(5, dtype=np.int64)
    for j in range(nboard):
        fb[j] = board[j]
    acc = np.zeros(nplayers, dtype=np.float64)
    scores = np.empty(nplayers, dtype=np.int64)
    cards7 = np.empty(7, dtype=np.int64)
    for _ in range(nsim):
        for j in range(need):  # partial Fisher-Yates
            k = j + np.random.randint(nd - j)
            t = deck[j]
            deck[j] = deck[k]
            deck[k] = t
            fb[nboard + j] = deck[j]
        _settle(holes, nplayers, fb, acc, scores, cards7)
    for i in range(nplayers):
        out[i] = acc[i] / nsim
    return nsim


@njit(cache=True)
def hand_strength_vs_random(h1, h2, board, nboard, dead, ndead):
    """Current (no runout) strength vs ONE random unseen 2-card hand on a board of 3..5 cards:
    returns (P(win)+0.5 P(tie)). dead = cards the decider could NOT hold against (normally only
    own cards + board; pass other players' cards only for an 'omniscient' variant)."""
    holes = np.empty((1, 2), dtype=np.int64)
    holes[0, 0] = h1
    holes[0, 1] = h2
    deck, nd = _remaining_deck(holes, 1, board, nboard, dead, ndead)
    cards = np.empty(7, dtype=np.int64)
    cards[0] = h1
    cards[1] = h2
    for j in range(nboard):
        cards[2 + j] = board[j]
    me = eval_cards(cards, 2 + nboard)
    win = 0.0
    tot = 0.0
    for a in range(nd):
        cards[0] = deck[a]
        for b in range(a + 1, nd):
            cards[1] = deck[b]
            s = eval_cards(cards, 2 + nboard)
            if me > s:
                win += 1.0
            elif me == s:
                win += 0.5
            tot += 1.0
    return win / tot


@njit(cache=True, parallel=True)
def preflop_vs_random_mc(nsim, max_opp, seed):
    """For each of 169 classes and k=1..max_opp random opponents (no folding): pot-share
    equity and outright win prob. Returns (eq[169,max_opp], win[169,max_opp])."""
    eq = np.zeros((169, max_opp))
    win = np.zeros((169, max_opp))
    for cid in prange(169):
        np.random.seed(seed + cid * 7919)
        hi = cid // 13
        lo = cid % 13
        # class_id convention: pair r*13+r; suited hi*13+lo (hi>lo); offsuit lo*13+hi
        if hi == lo:
            h1 = hi * 4
            h2 = hi * 4 + 1
        elif hi > lo:
            h1 = hi * 4
            h2 = lo * 4
        else:
            h1 = lo * 4
            h2 = hi * 4 + 1
        deck0 = np.empty(50, dtype=np.int64)
        k = 0
        for c in range(52):
            if c != h1 and c != h2:
                deck0[k] = c
                k += 1
        cards = np.empty(7, dtype=np.int64)
        oscore = np.empty(max_opp, dtype=np.int64)
        for nopp in range(1, max_opp + 1):
            deck = deck0.copy()
            need = 5 + 2 * nopp
            acc = 0.0
            wacc = 0.0
            for _ in range(nsim):
                for j in range(need):
                    kk = j + np.random.randint(50 - j)
                    t = deck[j]
                    deck[j] = deck[kk]
                    deck[kk] = t
                for j in range(5):
                    cards[2 + j] = deck[j]
                cards[0] = h1
                cards[1] = h2
                me = eval_cards(cards, 7)
                best = me
                nbest = 1
                beaten = False
                for o in range(nopp):
                    cards[0] = deck[5 + 2 * o]
                    cards[1] = deck[6 + 2 * o]
                    s = eval_cards(cards, 7)
                    if s > best:
                        beaten = True
                        best = s
                    elif s == best:
                        nbest += 1
                if not beaten and best == me:
                    # recount ties among players at 'me'
                    acc += 1.0 / nbest
                    if nbest == 1:
                        wacc += 1.0
            eq[cid, nopp - 1] = acc / nsim
            win[cid, nopp - 1] = wacc / nsim
    return eq, win


# ----------------------------------------------------------------------------- classes


@njit(cache=True)
def class_id(c1, c2):
    r1 = c1 >> 2
    r2 = c2 >> 2
    if r1 == r2:
        return r1 * 13 + r1
    hi = r1 if r1 > r2 else r2
    lo = r2 if r1 > r2 else r1
    if (c1 & 3) == (c2 & 3):
        return hi * 13 + lo  # suited: first digit > second
    return lo * 13 + hi  # offsuit: first digit < second


def class_name(cid: int) -> str:
    a, b = divmod(int(cid), 13)
    if a == b:
        return RANKS[a] * 2
    if a > b:
        return RANKS[a] + RANKS[b] + "s"
    return RANKS[b] + RANKS[a] + "o"


def card_str(code: int) -> str:
    return RANKS[code >> 2] + SUITS[code & 3]


def parse_cards(s: str) -> list[int]:
    s = s.replace(" ", "")
    return [RANKS.index(s[i]) * 4 + SUITS.index(s[i + 1]) for i in range(0, len(s), 2)]


def chen_score(cid: int) -> float:
    """Bill Chen's formula (rounded up to whole number as in the original)."""
    import math

    a, b = divmod(int(cid), 13)
    hi, lo = max(a, b), min(a, b)
    pts = {12: 10, 11: 8, 10: 7, 9: 6}.get(hi, (hi + 2) / 2)
    if a == b:
        return max(5.0, pts * 2) if hi >= 0 else pts
    score = pts
    if a > b:
        score += 2
    gap = hi - lo - 1
    score -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and hi < 10:  # both below Q with 0/1 gap
        score += 1
    return float(math.ceil(score))


_SKLANSKY = {
    1: "AA KK QQ JJ AKs",
    2: "TT AQs AJs KQs AKo",
    3: "99 JTs QJs KJs ATs AQo",
    4: "T9s KQo 88 QTs 98s J9s AJo KTs",
    5: "77 87s Q9s T8s KJo QJo JTo 76s 97s A9s A8s A7s A6s A5s A4s A3s A2s 65s",
    6: "66 ATo 55 86s KTo QTo 54s K9s J8s 75s",
    7: "44 J9o 64s T9o 53s 33 98o 43s 22 K8s K7s K6s K5s K4s K3s K2s T7s Q8s",
    8: "87o A9o Q9o 76o 42s 32s 96s 85s J8o J7s 65o 54o 74s K9o T8o",
}


def sklansky_group(name: str) -> int:
    for g, hands in _SKLANSKY.items():
        if name in hands.split():
            return g
    return 9


def preflop_class_table():
    import polars as pl

    rows = []
    for c1 in range(52):
        for c2 in range(52):
            if c1 != c2:
                cid = int(class_id(c1, c2))
                rows.append((c1, c2, cid, class_name(cid)))
    return pl.DataFrame(rows, schema=["c1", "c2", "class_id", "hand_class"], orient="row").with_columns(
        pl.col("c1").cast(pl.Int8), pl.col("c2").cast(pl.Int8), pl.col("class_id").cast(pl.Int16)
    )


def build_preflop(nsim: int = 400_000):
    import time

    import polars as pl

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "derived")
    t = time.time()
    eq, win = preflop_vs_random_mc(nsim, 5, 20260915)
    print(f"MC done in {time.time() - t:.1f}s (nsim={nsim})")
    recs = []
    for cid in range(169):
        name = class_name(cid)
        a, b = divmod(cid, 13)
        combos = 6 if a == b else (4 if a > b else 12)
        hi_, lo_ = max(a, b), min(a, b)
        grid = (12 - hi_) * 13 + (12 - lo_) if a >= b else (12 - lo_) * 13 + (12 - hi_)
        rec = {
            "class_id": cid,
            "grid_idx": grid,  # == src/handeval.py preflop_class() convention (AA=0, AKs=1, AKo=13, 22=168)
            "hand_class": name,
            "hi_rank": max(a, b),
            "lo_rank": min(a, b),
            "pair": a == b,
            "suited": a > b,
            "combos": combos,
        }
        for k in range(5):
            rec[f"equity_vs{k + 1}"] = float(eq[cid, k])
        for k in range(5):
            rec[f"win_vs{k + 1}"] = float(win[cid, k])
        rec["chen"] = chen_score(cid)
        rec["sklansky_group"] = sklansky_group(name)
        recs.append(rec)
    df = pl.DataFrame(recs)
    for k in range(1, 6):  # combo-weighted percentile rank (1.0 = best), i.e. "top x% of hands"
        df = df.sort(f"equity_vs{k}", descending=True).with_columns(
            (pl.col("combos").cum_sum() / 1326.0).alias(f"top_pct_vs{k}")
        )
    df = df.sort("class_id").with_columns(
        pl.col("class_id").cast(pl.Int16), pl.col("hi_rank").cast(pl.Int8), pl.col("lo_rank").cast(pl.Int8),
        pl.col("combos").cast(pl.Int8), pl.col("sklansky_group").cast(pl.Int8), pl.lit(nsim).alias("nsim"),
    )
    df.write_parquet(os.path.join(base, "preflop_equity_169.parquet"))
    preflop_class_table().join(df.select("class_id", pl.col("grid_idx").cast(pl.Int16)), on="class_id").select(
        "c1", "c2", "class_id", "grid_idx", "hand_class"
    ).write_parquet(os.path.join(base, "preflop_class_map.parquet"))
    print(df.sort("equity_vs1", descending=True).select("hand_class", "equity_vs1", "equity_vs2", "equity_vs5", "chen", "sklansky_group").head(10))
    return df


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build_preflop":
        build_preflop(int(sys.argv[2]) if len(sys.argv) > 2 else 400_000)
