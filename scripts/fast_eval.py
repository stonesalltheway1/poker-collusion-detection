"""Numba 7-card poker evaluator + exact/MC equity (card code = rank*4+suit, rank 2..A = 0..12).
score(): higher = better (category<<20 | kickers). Verified against phevaluator in __main__.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _straight_top(mask):
    # returns top rank index of best straight in 13-bit rank mask, -1 if none
    for top in range(12, 3, -1):
        need = 0x1F << (top - 4)
        if (mask & need) == need:
            return top
    # wheel A2345
    if (mask & 0x100F) == 0x100F:
        return 3
    return -1


@njit(cache=True)
def score7(cards, n):
    rc = np.zeros(13, np.int64)
    sc = np.zeros(4, np.int64)
    sm = np.zeros(4, np.int64)
    rmask = 0
    for i in range(n):
        c = cards[i]
        r = c // 4
        s = c % 4
        rc[r] += 1
        sc[s] += 1
        sm[s] |= 1 << r
        rmask |= 1 << r
    fs = -1
    for s in range(4):
        if sc[s] >= 5:
            fs = s
    if fs >= 0:
        t = _straight_top(sm[fs])
        if t >= 0:
            return (8 << 20) | (t << 16)
    q = -1
    t1 = -1
    t2 = -1
    p1 = -1
    p2 = -1
    for r in range(12, -1, -1):
        if rc[r] == 4:
            q = r
        elif rc[r] == 3:
            if t1 < 0:
                t1 = r
            elif t2 < 0:
                t2 = r
        elif rc[r] == 2:
            if p1 < 0:
                p1 = r
            elif p2 < 0:
                p2 = r
    if q >= 0:
        k = -1
        for r in range(12, -1, -1):
            if r != q and rc[r] > 0:
                k = r
                break
        return (7 << 20) | (q << 16) | (k << 12)
    if t1 >= 0 and (t2 >= 0 or p1 >= 0):
        pp = t2 if t2 > p1 else p1
        return (6 << 20) | (t1 << 16) | (pp << 12)
    if fs >= 0:
        v = 0
        cnt = 0
        for r in range(12, -1, -1):
            if (sm[fs] >> r) & 1:
                v = (v << 4) | r
                cnt += 1
                if cnt == 5:
                    break
        return (5 << 20) | v
    t = _straight_top(rmask)
    if t >= 0:
        return (4 << 20) | (t << 16)
    if t1 >= 0:
        v = 0
        cnt = 0
        for r in range(12, -1, -1):
            if r != t1 and rc[r] > 0:
                v = (v << 4) | r
                cnt += 1
                if cnt == 2:
                    break
        return (3 << 20) | (t1 << 16) | (v << 8)
    if p1 >= 0 and p2 >= 0:
        k = -1
        for r in range(12, -1, -1):
            if r != p1 and r != p2 and rc[r] > 0:
                k = r
                break
        return (2 << 20) | (p1 << 16) | (p2 << 12) | (k << 8)
    if p1 >= 0:
        v = 0
        cnt = 0
        for r in range(12, -1, -1):
            if r != p1 and rc[r] > 0:
                v = (v << 4) | r
                cnt += 1
                if cnt == 3:
                    break
        return (1 << 20) | (p1 << 16) | (v << 4)
    v = 0
    cnt = 0
    for r in range(12, -1, -1):
        if rc[r] > 0:
            v = (v << 4) | r
            cnt += 1
            if cnt == 5:
                break
    return v


@njit(cache=True)
def equity(holes, nh, board, nb, deadmask, mc, seed):
    """holes: int64[nh,2]; board int64[5] with nb dealt; deadmask: bool[52] extra dead cards.
    Exact enumeration when <=2 cards to come, else MC `mc` samples. Returns float64[nh] pot share."""
    np.random.seed(seed)
    used = deadmask.copy()
    for i in range(nh):
        used[holes[i, 0]] = True
        used[holes[i, 1]] = True
    for i in range(nb):
        used[board[i]] = True
    rest = np.empty(52, np.int64)
    nr = 0
    for c in range(52):
        if not used[c]:
            rest[nr] = c
            nr += 1
    need = 5 - nb
    eq = np.zeros(nh)
    cards = np.empty(7, np.int64)
    b = np.empty(5, np.int64)
    for i in range(nb):
        b[i] = board[i]
    sc = np.empty(nh, np.int64)
    total = 0
    if need == 0:
        runs = 1
    elif need == 1:
        runs = nr
    elif need == 2:
        runs = nr * (nr - 1) // 2
    else:
        runs = mc
    if need == 2:
        i1 = 0
        i2 = 1
    for it in range(runs):
        if need == 1:
            b[4] = rest[it]
        elif need == 2:
            b[3] = rest[i1]
            b[4] = rest[i2]
            i2 += 1
            if i2 >= nr:
                i1 += 1
                i2 = i1 + 1
        elif need >= 3:
            # partial Fisher-Yates
            for k in range(need):
                j = k + np.random.randint(nr - k)
                tmp = rest[k]
                rest[k] = rest[j]
                rest[j] = tmp
            for k in range(need):
                b[nb + k] = rest[k]
        best = -1
        nbest = 0
        for h in range(nh):
            for k in range(5):
                cards[k] = b[k]
            cards[5] = holes[h, 0]
            cards[6] = holes[h, 1]
            s = score7(cards, 7)
            sc[h] = s
            if s > best:
                best = s
                nbest = 1
            elif s == best:
                nbest += 1
        for h in range(nh):
            if sc[h] == best:
                eq[h] += 1.0 / nbest
        total += 1
    return eq / total


@njit(cache=True)
def equity_vs_random(h0, h1, board, nb, nopp, mc, seed):
    """Own-information hand strength: pot share of (h0,h1) vs `nopp` random opponent hands,
    random runout, MC `mc` samples (known cards = own hole + dealt board only)."""
    np.random.seed(seed)
    used = np.zeros(52, np.bool_)
    used[h0] = True
    used[h1] = True
    for i in range(nb):
        used[board[i]] = True
    rest = np.empty(52, np.int64)
    nr = 0
    for c in range(52):
        if not used[c]:
            rest[nr] = c
            nr += 1
    need = 5 - nb + 2 * nopp
    cards = np.empty(7, np.int64)
    b = np.empty(5, np.int64)
    for i in range(nb):
        b[i] = board[i]
    tot = 0.0
    for it in range(mc):
        for k in range(need):
            j = k + np.random.randint(nr - k)
            tmp = rest[k]
            rest[k] = rest[j]
            rest[j] = tmp
        for k in range(5 - nb):
            b[nb + k] = rest[k]
        off = 5 - nb
        for k in range(5):
            cards[k] = b[k]
        cards[5] = h0
        cards[6] = h1
        me = score7(cards, 7)
        best_opp = -1
        nties = 0
        for o in range(nopp):
            cards[5] = rest[off + 2 * o]
            cards[6] = rest[off + 2 * o + 1]
            s = score7(cards, 7)
            if s > best_opp:
                best_opp = s
        if me > best_opp:
            tot += 1.0
        elif me == best_opp:
            # split approx (ignores multi-way tie count)
            tot += 0.5
    return tot / mc


@njit(cache=True)
def made_score(hole0, hole1, board, nb):
    cards = np.empty(7, np.int64)
    for k in range(nb):
        cards[k] = board[k]
    cards[nb] = hole0
    cards[nb + 1] = hole1
    return score7(cards, nb + 2)


if __name__ == "__main__":
    import time
    from phevaluator import evaluate_cards
    rng = np.random.default_rng(0)
    N = 200000
    S = np.empty(N, np.int64)
    P = np.empty(N, np.int64)
    for i in range(N):
        c = rng.choice(52, 7, replace=False)
        S[i] = score7(c.astype(np.int64), 7)
        P[i] = evaluate_cards(*c.tolist())
    # consistency: order of S must be reverse of P (ties equal)
    idx = rng.integers(0, N, (500000, 2))
    a, b = idx[:, 0], idx[:, 1]
    ok = np.sign(S[a] - S[b]) == -np.sign(P[a] - P[b])
    print("pairwise order agreement", ok.mean())
    holes = np.array([[48, 49], [44, 45]], np.int64)
    board = np.zeros(5, np.int64)
    t = time.time()
    e = equity(holes, 2, board, 0, np.zeros(52, np.bool_), 20000, 1)
    print("AA vs KK preflop", e, time.time() - t)
    board[:3] = [0, 13, 30]
    t = time.time()
    for _ in range(1000):
        e = equity(holes, 2, board, 3, np.zeros(52, np.bool_), 0, 1)
    print("flop exact x1000", e, time.time() - t)
