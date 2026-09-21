"""Per-action omniscient equity replay (all hole cards visible) -> data/derived/action_equity_<tag>.parquet

For every action row it computes, with EVERY dealt hole card known (folded hands = dead cards):
  eq_pre_s0..s5   pot-share equity of each LIVE seat just before the action on the current board
                  (NaN = seat already folded). Exact enumeration postflop, Monte-Carlo preflop.
  eq_post_s0..s5  same just after the action (differs from eq_pre only when the action is a fold)
  hs_actor        actor's PERCEIVED strength postflop = P(win)+P(tie)/2 of his made hand vs ONE random
                  unseen hand on the current board (no knowledge of other hands; NaN preflop -> use
                  preflop_equity_169.parquet equity_vsN instead)
and per hand/street a heads-up matrix  hu_s{street}_{i}{j}  = equity of seat i vs seat j alone
(all other dealt cards dead), for seat pairs both live at the start of that street.

Value accounting (Mazrooei, Archibald & Bowling AAAI-13 "collusion table", with V_j = eq_j * pot):
  impact of action t (actor k, chips a, pot P before) on seat j:
      dV_j = (eq_post_j - eq_pre_j) * P + eq_post_j * a - [j == k] * a
  sums to 0 over seats; chance (board cards) impacts are excluded => luck-free "who gave value to whom".

Usage: python src/hand_replay.py <tag> [hand_idx_list.parquet]   (default: dev hands shared by labelled pairs)
Threads: NUMBA_NUM_THREADS (keep <=4 on the shared box).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poker_equity import equity_exact, equity_mc, hand_strength_vs_random  # noqa: E402

NB_BY_STREET = np.array([0, 3, 4, 5], dtype=np.int64)
PAIR_I = np.array([i for i in range(6) for j in range(i + 1, 6)], dtype=np.int64)
PAIR_J = np.array([j for i in range(6) for j in range(i + 1, 6)], dtype=np.int64)


@njit(cache=True)
def _eqvec(holes6, board5, nb, mask, nsim_pf, seed, out6):
    k = 0
    for s in range(6):
        out6[s] = np.nan
        if mask >> s & 1:
            k += 1
    if k == 1:
        for s in range(6):
            if mask >> s & 1:
                out6[s] = 1.0
        return
    hl = np.empty((k, 2), dtype=np.int64)
    dead = np.empty(12, dtype=np.int64)
    nd = 0
    idx = np.empty(k, dtype=np.int64)
    q = 0
    for s in range(6):
        if mask >> s & 1:
            hl[q, 0] = holes6[s, 0]
            hl[q, 1] = holes6[s, 1]
            idx[q] = s
            q += 1
        elif holes6[s, 0] >= 0:
            dead[nd] = holes6[s, 0]
            dead[nd + 1] = holes6[s, 1]
            nd += 2
    res = np.zeros(k, dtype=np.float64)
    if nb == 0:
        equity_mc(hl, k, board5, 0, dead, nd, nsim_pf, seed, res)
    else:
        equity_exact(hl, k, board5, nb, dead, nd, res)
    for q in range(k):
        out6[idx[q]] = res[q]


@njit(cache=True, parallel=True)
def replay_batch(h_start, a_street, a_slot, a_type, holes, board, nboard, nsim_pf, nsim_pf_hu, seed,
                 eq_pre, eq_post, hs_actor, hu):
    nh = h_start.shape[0] - 1
    for h in prange(nh):
        holes6 = holes[h]
        b5 = board[h]
        mask = 0
        for s in range(6):
            if holes6[s, 0] >= 0:
                mask |= 1 << s
        ck = np.full(40, -1, dtype=np.int64)
        cv = np.empty((40, 6), dtype=np.float64)
        nc = 0
        tmp = np.empty(6, dtype=np.float64)
        street_mask = np.full(4, -1, dtype=np.int64)
        hs_cache = np.full((4, 6), -1.0, dtype=np.float64)
        empty = np.zeros(1, dtype=np.int64)
        for t in range(h_start[h], h_start[h + 1]):
            st = a_street[t]
            nb = NB_BY_STREET[st]
            if nb > nboard[h]:
                nb = nboard[h]
            if street_mask[st] < 0:
                street_mask[st] = mask
            for phase in range(2):
                m = mask
                if phase == 1 and a_type[t] == 0:
                    m = mask & ~(1 << a_slot[t])
                key = st * 64 + m
                hit = -1
                for c in range(nc):
                    if ck[c] == key:
                        hit = c
                        break
                if hit < 0:
                    _eqvec(holes6, b5, nb, m, nsim_pf, seed + h * 131 + key, tmp)
                    if nc < 40:
                        ck[nc] = key
                        for s in range(6):
                            cv[nc, s] = tmp[s]
                        hit = nc
                        nc += 1
                    else:
                        for s in range(6):
                            if phase == 0:
                                eq_pre[t, s] = tmp[s]
                            else:
                                eq_post[t, s] = tmp[s]
                        continue
                for s in range(6):
                    if phase == 0:
                        eq_pre[t, s] = cv[hit, s]
                    else:
                        eq_post[t, s] = cv[hit, s]
            sl = a_slot[t]
            if nb >= 3:
                if hs_cache[st, sl] < 0:
                    hs_cache[st, sl] = hand_strength_vs_random(holes6[sl, 0], holes6[sl, 1], b5, nb, empty, 0)
                hs_actor[t] = hs_cache[st, sl]
            else:
                hs_actor[t] = np.nan
            if a_type[t] == 0:
                mask = mask & ~(1 << sl)
        # heads-up matrix per street for pairs live at street start
        for st in range(4):
            m = street_mask[st]
            if m < 0:
                continue
            nb = NB_BY_STREET[st]
            if nb > nboard[h]:
                continue
            for p in range(15):
                i = PAIR_I[p]
                j = PAIR_J[p]
                if (m >> i & 1) == 0 or (m >> j & 1) == 0:
                    continue
                hl = np.empty((2, 2), dtype=np.int64)
                hl[0, 0] = holes6[i, 0]
                hl[0, 1] = holes6[i, 1]
                hl[1, 0] = holes6[j, 0]
                hl[1, 1] = holes6[j, 1]
                dead = np.empty(8, dtype=np.int64)
                nd = 0
                for s in range(6):
                    if s != i and s != j and holes6[s, 0] >= 0:
                        dead[nd] = holes6[s, 0]
                        dead[nd + 1] = holes6[s, 1]
                        nd += 2
                res = np.zeros(2, dtype=np.float64)
                if nb == 0:
                    equity_mc(hl, 2, b5, 0, dead, nd, nsim_pf_hu, seed + h * 7 + p, res)
                else:
                    equity_exact(hl, 2, b5, nb, dead, nd, res)
                hu[h, st, p] = res[0]


def run(hand_ids: np.ndarray, tag: str, nsim_pf: int = 400, nsim_pf_hu: int = 800):
    import polars as pl

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "derived")
    hand_ids = np.unique(hand_ids.astype(np.int32))
    t0 = time.time()
    hl = pl.Series("hand_idx", hand_ids)
    A = (pl.scan_parquet(f"{base}/actions.parquet").filter(pl.col("hand_idx").is_in(hl.implode()))
         .select("hand_idx", "action_no", "street", "player_idx", "action", "amount", "pot_before", "to_call")
         .collect().sort("hand_idx", "action_no"))
    S = (pl.scan_parquet(f"{base}/seats.parquet").filter(pl.col("hand_idx").is_in(hl.implode()))
         .select("hand_idx", "player_idx", "seat_no", "c1", "c2").collect())
    H = (pl.scan_parquet(f"{base}/hands.parquet").filter(pl.col("hand_idx").is_in(hl.implode()))
         .select("hand_idx", "board").collect().sort("hand_idx"))
    hand_ids = H["hand_idx"].to_numpy()
    pos = {int(h): i for i, h in enumerate(hand_ids)}
    nh = len(hand_ids)
    holes = np.full((nh, 6, 2), -1, dtype=np.int64)
    hrow = np.searchsorted(hand_ids, S["hand_idx"].to_numpy())
    seat = S["seat_no"].to_numpy().astype(np.int64)
    holes[hrow, seat, 0] = S["c1"].to_numpy()
    holes[hrow, seat, 1] = S["c2"].to_numpy()
    board = np.zeros((nh, 5), dtype=np.int64)
    nboard = np.zeros(nh, dtype=np.int64)
    for i, b in enumerate(H["board"].to_list()):
        nboard[i] = len(b)
        board[i, : len(b)] = b
    A = A.join(S.select("hand_idx", "player_idx", "seat_no"), on=["hand_idx", "player_idx"], how="left").sort("hand_idx", "action_no")
    arow = np.searchsorted(hand_ids, A["hand_idx"].to_numpy())
    h_start = np.searchsorted(arow, np.arange(nh + 1)).astype(np.int64)
    na = A.height
    eq_pre = np.full((na, 6), np.nan, dtype=np.float32)
    eq_post = np.full((na, 6), np.nan, dtype=np.float32)
    hs = np.full(na, np.nan, dtype=np.float32)
    hu = np.full((nh, 4, 15), np.nan, dtype=np.float32)
    print(f"loaded {nh} hands / {na} actions in {time.time() - t0:.1f}s", flush=True)
    t1 = time.time()
    replay_batch(h_start, A["street"].to_numpy().astype(np.int64), A["seat_no"].to_numpy().astype(np.int64),
                 A["action"].to_numpy().astype(np.int64), holes, board, nboard, nsim_pf, nsim_pf_hu, 20260915,
                 eq_pre, eq_post, hs, hu)
    print(f"replay {time.time() - t1:.1f}s", flush=True)
    cols = {"hand_idx": A["hand_idx"], "action_no": A["action_no"], "seat_no": A["seat_no"]}
    for s in range(6):
        cols[f"eq_pre_s{s}"] = eq_pre[:, s]
    for s in range(6):
        cols[f"eq_post_s{s}"] = eq_post[:, s]
    cols["hs_actor"] = hs
    out = pl.DataFrame(cols)
    out.write_parquet(f"{base}/action_equity_{tag}.parquet")
    hucols = {"hand_idx": hand_ids}
    for st in range(4):
        for p in range(15):
            hucols[f"hu_s{st}_{PAIR_I[p]}{PAIR_J[p]}"] = hu[:, st, p]
    pl.DataFrame(hucols).write_parquet(f"{base}/hu_equity_{tag}.parquet")
    print(f"wrote action_equity_{tag}.parquet ({na} rows) and hu_equity_{tag}.parquet ({nh} rows); total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    os.environ.setdefault("POLARS_MAX_THREADS", "3")
    import duckdb
    import polars as pl

    tag = sys.argv[1] if len(sys.argv) > 1 else "labeled_dev"
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "derived").replace("\\", "/")
    if len(sys.argv) > 2 and sys.argv[2]:
        ids = pl.read_parquet(sys.argv[2])["hand_idx"].to_numpy()
    else:
        con = duckdb.connect()
        con.sql("SET threads=3; SET memory_limit='4GB'")
        ids = con.sql(f"""select distinct s1.hand_idx from '{base}/labels.parquet' l
            join '{base}/seats.parquet' s1 on s1.player_idx=l.p1
            join '{base}/seats.parquet' s2 on s2.player_idx=l.p2 and s2.hand_idx=s1.hand_idx
            join '{base}/hands.parquet' h on h.hand_idx=s1.hand_idx where h.phase=0""").pl()["hand_idx"].to_numpy()
    if len(sys.argv) > 3:
        ids = ids[: int(sys.argv[3])]
    run(ids, tag)
