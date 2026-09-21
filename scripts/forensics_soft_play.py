"""Evidence-hand forensics for behavior family 'soft_play'.

Re-runnable. Stages (python scripts/forensics_soft_play.py <stage>):
  load      build the study subset (pairs x dev shared hands; actions/seats/hands rows) -> data/derived/forensics_sp/
  print     print evidence hands + non-evidence shared hands of SP pairs in full detail
  features  compute hand-level (pair, hand) features / candidate signatures for every study pair-hand
  analyze   coverage / rates / temporal / pair-level separability -> research/forensics/sp_*.csv
  all       everything

Study pair groups:
  SP  = confirmed soft_play positives (132)     DT / CI = other positive families
  NEG = confirmed_non_target (1488)             UNK = 3000 random unknown dev pairs (seed 42)
Roles: player_1/player_2 are ID-sorted (arbitrary); forensics labels them A (=player_1) and B (=player_2)
and derives behavioural roles afterwards.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "3")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import polars as pl

BASE = Path(__file__).resolve().parent.parent
DER = BASE / "data" / "derived"
WORK = DER / "forensics_sp"
OUT = BASE / "research" / "forensics"
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE / "src"))

RANKS, SUITS = "23456789TJQKA", "cdhs"
ACT = ["fold", "check", "call", "bet", "raise", "allin"]
STREETS = ["PRE", "FLOP", "TURN", "RIVER"]
FAM = "soft_play"
N_UNK = 3000


def cs(c):
    return RANKS[c // 4] + SUITS[c % 4]


def scan(name):
    return pl.scan_parquet(DER / f"{name}.parquet")


# ============================================================================ load
def stage_load():
    lab = pl.read_parquet(DER / "labels.parquet")
    ev = pl.read_parquet(DER / "evidence.parquet")
    hands_meta = scan("hands").select("hand_idx", "table_idx", "hand_seq", "phase").collect()
    dev_ids = hands_meta.filter(pl.col("phase") == 0)["hand_idx"]

    pos = lab.filter(pl.col("label") == 1)
    grp = {"soft_play": "SP", "directed_transfer": "DT", "coordinated_isolation": "CI"}
    pairs = pl.concat([
        pos.select("pair_id", "p1", "p2", pl.col("behavior_family").replace_strict(grp).alias("grp")),
        lab.filter(pl.col("label") == 0).select("pair_id", "p1", "p2", pl.lit("NEG").alias("grp")),
    ])
    # player -> table
    seats_dev = (scan("seats").select("hand_idx", "player_idx")
                 .join(scan("hands").select("hand_idx", "table_idx", "phase"), on="hand_idx")
                 .filter(pl.col("phase") == 0))
    ptab = seats_dev.select("player_idx", "table_idx").unique().collect()
    # unknown pairs: random same-table player pairs with >=1 dev shared hand, not labelled
    rng = np.random.default_rng(42)
    tab_players = ptab.group_by("table_idx").agg(pl.col("player_idx").sort()).sort("table_idx")
    cand = []
    for t, pl_list in tab_players.iter_rows():
        arr = np.array(pl_list)
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                cand.append((arr[i], arr[j]))
    cand = np.array(cand)
    labelled = set(zip(lab["p1"].to_list(), lab["p2"].to_list()))
    pid = pl.read_parquet(DER / "players.parquet").select("player_idx", "player_id")
    order = rng.permutation(len(cand))
    unk = []
    for k in order:
        a, b = int(cand[k, 0]), int(cand[k, 1])
        if (a, b) in labelled or (b, a) in labelled:
            continue
        unk.append((a, b))
        if len(unk) >= int(N_UNK * 1.3):
            break
    unk = pl.DataFrame({"p1": [u[0] for u in unk], "p2": [u[1] for u in unk]}, schema={"p1": pl.Int32, "p2": pl.Int32})
    # orient p1/p2 by player_id string order (same convention as labels)
    unk = (unk.join(pid.rename({"player_idx": "p1", "player_id": "id1"}), on="p1")
           .join(pid.rename({"player_idx": "p2", "player_id": "id2"}), on="p2")
           .with_columns(pl.when(pl.col("id1") < pl.col("id2")).then(pl.col("p1")).otherwise(pl.col("p2")).alias("q1"),
                         pl.when(pl.col("id1") < pl.col("id2")).then(pl.col("p2")).otherwise(pl.col("p1")).alias("q2"))
           .select(pl.format("U{}_{}", "q1", "q2").alias("pair_id"), pl.col("q1").alias("p1"), pl.col("q2").alias("p2"),
                   pl.lit("UNK").alias("grp")))
    pairs = pl.concat([pairs, unk])
    players = pl.concat([pairs["p1"], pairs["p2"]]).unique()

    sd = seats_dev.filter(pl.col("player_idx").is_in(players.implode())).select("hand_idx", "player_idx").collect()
    ph = (pairs.join(sd.rename({"player_idx": "p1"}), on="p1")
          .join(sd.rename({"player_idx": "p2"}), on=["p2", "hand_idx"]))
    # unknown: keep pairs with >= 1 dev shared hand, first N_UNK
    cnt = ph.group_by("pair_id").len()
    unk_keep = (pairs.filter(pl.col("grp") == "UNK").join(cnt, on="pair_id").head(N_UNK)["pair_id"])
    pairs = pairs.filter((pl.col("grp") != "UNK") | pl.col("pair_id").is_in(unk_keep.implode()))
    ph = ph.filter(pl.col("pair_id").is_in(pairs["pair_id"].implode()))
    ph = ph.join(ev.select("pair_id", "hand_idx", "evidence_rank"), on=["pair_id", "hand_idx"], how="left")
    pairs = pairs.join(ph.group_by("pair_id").len("n_shared"), on="pair_id", how="left")
    # sanity: evidence hands all shared?
    evj = ev.join(pairs.select("pair_id", "grp"), on="pair_id")
    n_ev_found = ph.filter(pl.col("evidence_rank").is_not_null()).height
    print(f"pairs: {pairs.group_by('grp').len().sort('grp').rows()}")
    print(f"pair-hands: {ph.height:,}; evidence rows {ev.height} found in shared dev hands {n_ev_found}")
    hand_set = ph["hand_idx"].unique()
    print(f"distinct hands: {hand_set.len():,}")
    pairs.write_parquet(WORK / "pairs.parquet")
    ph.write_parquet(WORK / "pair_hands.parquet")
    hs = hand_set.implode()
    scan("hands").filter(pl.col("hand_idx").is_in(hs)).collect().write_parquet(WORK / "hands.parquet")
    scan("seats").filter(pl.col("hand_idx").is_in(hs)).collect().write_parquet(WORK / "seats.parquet")
    scan("actions").filter(pl.col("hand_idx").is_in(hs)).collect().write_parquet(WORK / "actions.parquet")
    print("saved", WORK)


# ============================================================================ printing
class HandDB:
    def __init__(self):
        self.hands = pl.read_parquet(WORK / "hands.parquet")
        self.seats = pl.read_parquet(WORK / "seats.parquet")
        self.actions = pl.read_parquet(WORK / "actions.parquet")
        self.h = {r["hand_idx"]: r for r in self.hands.iter_rows(named=True)}
        self.s = {}
        for r in self.seats.sort("hand_idx", "seat_no").iter_rows(named=True):
            self.s.setdefault(r["hand_idx"], []).append(r)
        self.a = {}
        for r in self.actions.sort("hand_idx", "action_no").iter_rows(named=True):
            self.a.setdefault(r["hand_idx"], []).append(r)


def hand_text(db, hidx, pA, pB, title=""):
    from poker_equity import eval_cards
    h = db.h[hidx]
    seats = db.s[hidx]
    acts = db.a.get(hidx, [])
    board = list(h["board"]) if h["board"] is not None else []
    role = {}
    k = 0
    for s in seats:
        if s["player_idx"] == pA:
            role[s["player_idx"]] = "A"
        elif s["player_idx"] == pB:
            role[s["player_idx"]] = "B"
        else:
            k += 1
            role[s["player_idx"]] = f"o{k}"
    L = [f"--- hand {h['hand_id']} idx={hidx} seq={h['hand_seq']} {title} blinds {h['sb']}/{h['bb']} btn_seat={h['button_seat']} "
         f"board={' '.join(cs(c) for c in board)} final_pot={h['final_pot']} sd={h['players_at_showdown']}"]
    for s in seats:
        hole = [s["c1"], s["c2"]]
        st = ""
        if len(board) >= 3:
            cat = int(eval_cards(np.array(hole + board, dtype=np.int64), 2 + len(board))) >> 20
            st = ["high", "pair", "2pair", "trips", "str", "flush", "FH", "quads", "SF"][cat]
        L.append(f"  seat{s['seat_no']} {role[s['player_idx']]:>3} {cs(s['c1'])}{cs(s['c2'])} stack={s['starting_stack']:>5} "
                 f"contrib={s['total_contribution']:>5} net={s['net_chips']:>6} fold={int(s['folded'])} sd={int(s['went_to_showdown'])} "
                 f"won={s['won_share']:.2f} final={st}")
    cur = -1
    for a in acts:
        if a["street"] != cur:
            cur = a["street"]
            nb = {0: 0, 1: 3, 2: 4, 3: 5}[cur]
            L.append(f"  [{STREETS[cur]}] {' '.join(cs(c) for c in board[:nb])}")
        L.append(f"    {role.get(a['player_idx'], '?'):>3} {ACT[a['action']]:<6} amt={a['amount']:>5} to={a['amount_to']:>5} "
                 f"pot={a['pot_before']:>5} stk={a['stack_before']:>5} tocall={a['to_call']:>5} active={a['players_active']}")
    return "\n".join(L)


def stage_print(n_ev=30, n_non=15, fam="soft_play", seed=0, out_name=None):
    db = HandDB()
    pairs = pl.read_parquet(WORK / "pairs.parquet")
    ph = pl.read_parquet(WORK / "pair_hands.parquet")
    g = {"soft_play": "SP", "directed_transfer": "DT", "coordinated_isolation": "CI"}.get(fam, fam)
    pp = pairs.filter(pl.col("grp") == g)
    evh = ph.filter(pl.col("pair_id").is_in(pp["pair_id"].implode()) & pl.col("evidence_rank").is_not_null())
    evh = evh.sample(n=min(n_ev, evh.height), seed=seed, shuffle=True)
    lines = []
    for r in evh.iter_rows(named=True):
        lines.append(hand_text(db, r["hand_idx"], r["p1"], r["p2"], f"[EVIDENCE {g} pair={r['pair_id']} rank={r['evidence_rank']}]"))
    non = ph.filter(pl.col("pair_id").is_in(pp["pair_id"].implode()) & pl.col("evidence_rank").is_null())
    non = non.sample(n=min(n_non, non.height), seed=seed, shuffle=True)
    for r in non.iter_rows(named=True):
        lines.append(hand_text(db, r["hand_idx"], r["p1"], r["p2"], f"[NON-EV {g} pair={r['pair_id']}]"))
    txt = "\n".join(lines)
    if out_name:
        (OUT / out_name).write_text(txt, encoding="utf-8")
    print(txt)


# ============================================================================ action context + strength
def _numba_ctx():
    from numba import njit, prange
    import poker_equity as pe

    @njit(cache=False)
    def context(hptr, a_street, a_pl, a_act, a_amt, a_tocall, seat_pl):
        """Per action: actor seat, facing aggressor player (-1 none), active seat mask BEFORE the action,
        aggressive flag, number of aggressive actions earlier on the street, street-start active mask."""
        n = a_street.shape[0]
        seat = np.full(n, -1, np.int8)
        facing = np.full(n, -1, np.int64)
        mask = np.zeros(n, np.int16)
        aggr = np.zeros(n, np.bool_)
        naggr = np.zeros(n, np.int8)
        smask = np.zeros(n, np.int16)
        for h in range(hptr.shape[0] - 1):
            lo, hi = hptr[h], hptr[h + 1]
            m = 63
            cur = -1
            last = -1
            cnt = 0
            sm = 63
            for i in range(lo, hi):
                if a_street[i] != cur:
                    cur = a_street[i]
                    last = -1
                    cnt = 0
                    sm = m
                p = a_pl[i]
                s = -1
                for k in range(6):
                    if seat_pl[h, k] == p:
                        s = k
                seat[i] = s
                facing[i] = last
                mask[i] = m
                naggr[i] = cnt
                smask[i] = sm
                ac = a_act[i]
                ag = ac == 3 or ac == 4 or (ac == 5 and a_amt[i] > a_tocall[i])
                aggr[i] = ag
                if ag:
                    last = p
                    cnt += 1
                if ac == 0 and s >= 0:
                    m = m & ~(1 << s)
        return seat, facing, mask, aggr, naggr, smask

    @njit(parallel=True, cache=False)
    def strength(idx_h, idx_seat, street, mask, seat_c, boards):
        """hs_rand: made-hand strength vs 1 random hand (decider view, no runout);
        eq_omni: exact pot-share equity vs actual active opponents' hole cards (folded cards dead);
        cat: made-hand category (0 high .. 8 SF)."""
        n = idx_h.shape[0]
        hs = np.full(n, np.nan)
        eq = np.full(n, np.nan)
        cat = np.full(n, -1, np.int8)
        for j in prange(n):
            h = idx_h[j]
            st = street[j]
            if st == 0:
                continue
            nb = 3 if st == 1 else (4 if st == 2 else 5)
            me = idx_seat[j]
            board = np.empty(5, np.int64)
            for q in range(nb):
                board[q] = boards[h, q]
            holes = np.empty((6, 2), np.int64)
            dead = np.full(12, -1, np.int64)
            holes[0, 0] = seat_c[h, me, 0]
            holes[0, 1] = seat_c[h, me, 1]
            k = 1
            nd = 0
            for s in range(6):
                if s == me:
                    continue
                if mask[j] & (1 << s):
                    holes[k, 0] = seat_c[h, s, 0]
                    holes[k, 1] = seat_c[h, s, 1]
                    k += 1
                else:
                    dead[nd] = seat_c[h, s, 0]
                    dead[nd + 1] = seat_c[h, s, 1]
                    nd += 2
            cards = np.empty(7, np.int64)
            cards[0] = holes[0, 0]
            cards[1] = holes[0, 1]
            for q in range(nb):
                cards[2 + q] = board[q]
            cat[j] = pe.eval_cards(cards, 2 + nb) >> 20
            nodead = np.full(1, -1, np.int64)
            hs[j] = pe.hand_strength_vs_random(holes[0, 0], holes[0, 1], board, nb, nodead, 0)
            out = np.zeros(6)
            if k >= 2:
                pe.equity_exact(holes, k, board, nb, dead, nd, out)
                eq[j] = out[0]
        return hs, eq, cat

    return context, strength


def preflop_table():
    f = WORK / "preflop169.npy"
    if f.exists():
        return np.load(f)
    import poker_equity as pe
    eq, _ = pe.preflop_vs_random_mc(40000, 1, 7)
    np.save(f, eq[:, 0])
    return eq[:, 0]


def stage_actions():
    """Action-level context table for all study hands -> WORK/actx.parquet"""
    import time
    import poker_equity as pe
    t0 = time.time()
    hands = pl.read_parquet(WORK / "hands.parquet").sort("hand_idx")
    seats = pl.read_parquet(WORK / "seats.parquet").sort("hand_idx", "seat_no")
    acts = pl.read_parquet(WORK / "actions.parquet").sort("hand_idx", "action_no")
    hid = hands["hand_idx"].to_numpy()
    H = len(hid)
    assert seats.height == 6 * H
    seat_pl = seats["player_idx"].to_numpy().reshape(H, 6).astype(np.int64)
    seat_c = np.stack([seats["c1"].to_numpy(), seats["c2"].to_numpy()], 1).reshape(H, 6, 2).astype(np.int64)
    boards = np.full((H, 5), -1, np.int64)
    for i, b in enumerate(hands["board"].to_list()):
        if b:
            boards[i, :len(b)] = b
    a_h = acts["hand_idx"].to_numpy()
    hrow = np.searchsorted(hid, a_h)
    hptr = np.searchsorted(a_h, hid).astype(np.int64)
    hptr = np.append(hptr, len(a_h))
    context, strength = _numba_ctx()
    seat, facing, mask, aggr, naggr, smask = context(
        hptr, acts["street"].to_numpy().astype(np.int64), acts["player_idx"].to_numpy().astype(np.int64),
        acts["action"].to_numpy().astype(np.int64), acts["amount"].to_numpy().astype(np.int64),
        acts["to_call"].to_numpy().astype(np.int64), seat_pl)
    print(f"context {time.time()-t0:.1f}s")
    street = acts["street"].to_numpy().astype(np.int64)
    hs, eq, cat = strength(hrow.astype(np.int64), seat.astype(np.int64), street, mask.astype(np.int64), seat_c, boards)
    print(f"strength {time.time()-t0:.1f}s")
    pf = preflop_table()
    c1 = seat_c[hrow, seat.astype(np.int64), 0]
    c2 = seat_c[hrow, seat.astype(np.int64), 1]
    # vectorised class id
    r1, r2 = c1 >> 2, c2 >> 2
    hi_, lo_ = np.maximum(r1, r2), np.minimum(r1, r2)
    suited = (c1 & 3) == (c2 & 3)
    cid = np.where(r1 == r2, r1 * 13 + r1, np.where(suited, hi_ * 13 + lo_, lo_ * 13 + hi_))
    pre_eq = pf[cid]
    actx = acts.with_columns(
        pl.Series("seat", seat), pl.Series("facing", facing.astype(np.int32)), pl.Series("mask", mask),
        pl.Series("aggr", aggr), pl.Series("naggr", naggr), pl.Series("smask", smask),
        pl.Series("hs", hs.astype(np.float32)), pl.Series("eq", eq.astype(np.float32)), pl.Series("cat", cat),
        pl.Series("pre_eq", pre_eq.astype(np.float32)))
    actx.write_parquet(WORK / "actx.parquet")
    print(f"saved actx {actx.height:,} rows, {time.time()-t0:.1f}s")


def stage_features():
    """(pair, hand) feature table -> WORK/phf.parquet. Role-specific (_A/_B) and symmetric columns."""
    import time
    t0 = time.time()
    ph = pl.read_parquet(WORK / "pair_hands.parquet").select("pair_id", "grp", "p1", "p2", "hand_idx", "evidence_rank")
    seats = pl.read_parquet(WORK / "seats.parquet").select("hand_idx", "player_idx", "seat_no", "c1", "c2", "net_chips",
                                                            "went_to_showdown", "folded", "total_contribution")
    ph = (ph.join(seats.rename({"player_idx": "p1", "seat_no": "sA", "net_chips": "netA", "went_to_showdown": "sdA",
                                "c1": "a1", "c2": "a2", "folded": "fA", "total_contribution": "cA"}), on=["hand_idx", "p1"])
          .join(seats.rename({"player_idx": "p2", "seat_no": "sB", "net_chips": "netB", "went_to_showdown": "sdB",
                              "c1": "b1", "c2": "b2", "folded": "fB", "total_contribution": "cB"}), on=["hand_idx", "p2"]))
    bits = {k: 1 << k for k in range(6)}
    ph = ph.with_row_index("ph_id").with_columns(bA=pl.col("sA").replace_strict(bits, return_dtype=pl.Int32),
                                                 bB=pl.col("sB").replace_strict(bits, return_dtype=pl.Int32))
    actx = pl.read_parquet(WORK / "actx.parquet").select(
        "hand_idx", "action_no", "street", "player_idx", "action", "amount", "to_call", "pot_before", "stack_before",
        "facing", "mask", "aggr", "naggr", "smask", "hs", "eq", "cat", "pre_eq", "players_active")
    j = ph.select("ph_id", "hand_idx", "p1", "p2", "bA", "bB").join(actx, on="hand_idx")
    bitA = pl.col("bA")
    bitB = pl.col("bB")
    j = j.with_columns(
        role=pl.when(pl.col("player_idx") == pl.col("p1")).then(pl.lit("A"))
        .when(pl.col("player_idx") == pl.col("p2")).then(pl.lit("B")).otherwise(pl.lit("O")),
        frole=pl.when(pl.col("facing") == pl.col("p1")).then(pl.lit("A"))
        .when(pl.col("facing") == pl.col("p2")).then(pl.lit("B"))
        .when(pl.col("facing") >= 0).then(pl.lit("O")).otherwise(pl.lit("-")),
        A_act=(pl.col("mask").cast(pl.Int32) & bitA) > 0,
        B_act=(pl.col("mask").cast(pl.Int32) & bitB) > 0,
        hu_start=(pl.col("smask").cast(pl.Int32) == (bitA | bitB)),
        both_start=((pl.col("smask").cast(pl.Int32) & (bitA | bitB)) == (bitA | bitB)),
        strg=pl.when(pl.col("street") == 0).then(pl.col("pre_eq")).otherwise(pl.col("hs")),
        is_call=(pl.col("action") == 2) | ((pl.col("action") == 5) & (pl.col("amount") <= pl.col("to_call"))),
    )
    j = j.with_columns(
        partner_act=pl.when(pl.col("role") == "A").then(pl.col("B_act")).when(pl.col("role") == "B").then(pl.col("A_act")).otherwise(False),
        vs_p=((pl.col("role") == "A") & (pl.col("frole") == "B")) | ((pl.col("role") == "B") & (pl.col("frole") == "A")),
        vs_o=pl.col("role").is_in(["A", "B"]) & (pl.col("frole") == "O"),
        pre=pl.col("street") == 0, post=pl.col("street") > 0,
    )
    nan = float("nan")
    aggs = []
    for R in ("A", "B", "X"):
        rm = pl.col("role").is_in(["A", "B"]) if R == "X" else (pl.col("role") == R)
        def c(cond, name):
            aggs.append((rm & cond).sum().cast(pl.Int16).alias(f"{name}_{R}"))
        def mx(cond, val, name):
            aggs.append(pl.when(rm & cond).then(pl.col(val)).otherwise(None).max().alias(f"{name}_{R}"))
        fold = pl.col("action") == 0
        c(fold & pl.col("vs_p") & pl.col("pre"), "fold_vs_p_pre")
        c(fold & pl.col("vs_p") & pl.col("post"), "fold_vs_p_post")
        mx(fold & pl.col("vs_p") & pl.col("pre"), "strg", "fold_vs_p_pre_str")
        mx(fold & pl.col("vs_p") & pl.col("post"), "hs", "fold_vs_p_post_hs")
        mx(fold & pl.col("vs_p") & pl.col("post"), "eq", "fold_vs_p_post_eq")
        mx(fold & pl.col("vs_p") & pl.col("post"), "cat", "fold_vs_p_post_cat")
        c(fold & pl.col("vs_o") & pl.col("post"), "fold_vs_o_post")
        c(fold & pl.col("vs_o") & pl.col("pre"), "fold_vs_o_pre")
        c(pl.col("is_call") & pl.col("vs_p") & pl.col("pre"), "call_vs_p_pre")
        c(pl.col("is_call") & pl.col("vs_p") & pl.col("post"), "call_vs_p_post")
        mx(pl.col("is_call") & pl.col("vs_p") & pl.col("post"), "hs", "call_vs_p_post_hs")
        mx(pl.col("is_call") & pl.col("vs_p") & pl.col("post"), "eq", "call_vs_p_post_eq")
        mx(pl.col("is_call") & pl.col("vs_p") & pl.col("pre"), "strg", "call_vs_p_pre_str")
        c(pl.col("aggr") & pl.col("vs_p") & pl.col("pre"), "raise_vs_p_pre")
        c(pl.col("aggr") & pl.col("vs_p") & pl.col("post"), "raise_vs_p_post")
        c(pl.col("aggr") & pl.col("vs_o"), "raise_vs_o")
        # checks / bets with partner still active postflop
        chk = (pl.col("action") == 1) & pl.col("post") & pl.col("partner_act")
        c(chk, "check_pact_post")
        mx(chk, "hs", "check_pact_hs")
        mx(chk, "eq", "check_pact_eq")
        c(chk & pl.col("hu_start"), "check_hu")
        mx(chk & pl.col("hu_start"), "hs", "check_hu_hs")
        mx(chk & pl.col("hu_start"), "eq", "check_hu_eq")
        mx(chk & pl.col("hu_start") & (pl.col("street") == 3), "hs", "check_hu_river_hs")
        c(pl.col("aggr") & pl.col("post") & pl.col("partner_act"), "aggr_pact_post")
        c(pl.col("aggr") & pl.col("post") & pl.col("hu_start"), "aggr_hu_post")
        c(pl.col("aggr") & pl.col("pre") & pl.col("partner_act"), "aggr_pact_pre")
        c(pl.col("vs_p"), "faced_p")
        c(pl.col("vs_o"), "faced_o")
    # street-level structure
    aggs += [
        pl.col("street").max().alias("max_street"),
        pl.when(pl.col("post") & pl.col("hu_start")).then(pl.col("street")).otherwise(None).drop_nulls().n_unique().alias("n_hu_streets"),
        pl.when(pl.col("post") & pl.col("hu_start") & pl.col("aggr")).then(pl.col("street")).otherwise(None).drop_nulls().n_unique().alias("n_hu_aggr_streets"),
        pl.when(pl.col("post") & pl.col("both_start")).then(pl.col("street")).otherwise(None).drop_nulls().n_unique().alias("n_both_streets"),
        pl.when(pl.col("post") & pl.col("both_start") & pl.col("aggr")).then(pl.col("street")).otherwise(None).drop_nulls().n_unique().alias("n_both_aggr_streets"),
        pl.when(pl.col("post") & pl.col("both_start") & pl.col("aggr") & pl.col("role").is_in(["A", "B"])).then(pl.col("street")).otherwise(None).drop_nulls().n_unique().alias("n_both_pairaggr_streets"),
        (pl.col("pre") & pl.col("aggr")).sum().alias("n_pre_raises"),
        pl.len().alias("n_actions"),
    ]
    f = j.group_by("ph_id").agg(aggs)
    ph = ph.join(f, on="ph_id", how="left")
    # hole-card strength at showdown-independent level: final made hand of each partner and preflop class equity
    pf = preflop_table()
    def cid(c1, c2):
        r1, r2 = pl.col(c1).cast(pl.Int32) // 4, pl.col(c2).cast(pl.Int32) // 4
        hi_, lo_ = pl.max_horizontal(r1, r2), pl.min_horizontal(r1, r2)
        su = (pl.col(c1).cast(pl.Int32) % 4) == (pl.col(c2).cast(pl.Int32) % 4)
        return pl.when(r1 == r2).then(r1 * 13 + r1).when(su).then(hi_ * 13 + lo_).otherwise(lo_ * 13 + hi_)
    ph = ph.with_columns(cidA=cid("a1", "a2"), cidB=cid("b1", "b2"))
    mp = {i: float(v) for i, v in enumerate(pf)}
    ph = ph.with_columns(preA=pl.col("cidA").replace_strict(mp, return_dtype=pl.Float32),
                         preB=pl.col("cidB").replace_strict(mp, return_dtype=pl.Float32))
    hands = pl.read_parquet(WORK / "hands.parquet").select("hand_idx", "hand_seq", "table_idx", "bb", "final_pot", "players_at_showdown")
    ph = ph.join(hands, on="hand_idx", how="left")
    ph.write_parquet(WORK / "phf.parquet")
    print(f"saved phf {ph.shape}, {time.time()-t0:.1f}s")


ACLS = {0: "fold", 1: "check", 2: "call", 3: "raise", 4: "raise", 5: "raise"}


def action_table():
    """Partner (A/B) actions with context, for every study pair-hand -> WORK/pact.parquet"""
    ph = pl.read_parquet(WORK / "phf.parquet").select("ph_id", "pair_id", "grp", "p1", "p2", "hand_idx", "evidence_rank",
                                                      "bA", "bB", "table_idx", "hand_seq")
    actx = pl.read_parquet(WORK / "actx.parquet").select(
        "hand_idx", "action_no", "street", "player_idx", "action", "amount", "to_call", "pot_before", "stack_before",
        "facing", "mask", "aggr", "hs", "eq", "cat", "pre_eq", "players_active")
    j = ph.join(actx, on="hand_idx")
    j = j.filter((pl.col("player_idx") == pl.col("p1")) | (pl.col("player_idx") == pl.col("p2")))
    j = j.with_columns(
        role=pl.when(pl.col("player_idx") == pl.col("p1")).then(pl.lit("A")).otherwise(pl.lit("B")),
        pbit=pl.when(pl.col("player_idx") == pl.col("p1")).then(pl.col("bB")).otherwise(pl.col("bA")),
        partner=pl.when(pl.col("player_idx") == pl.col("p1")).then(pl.col("p2")).otherwise(pl.col("p1")))
    j = j.with_columns(
        pact=((pl.col("mask").cast(pl.Int32) & pl.col("pbit")) > 0).cast(pl.Int8),
        fac=pl.when(pl.col("facing") == pl.col("partner")).then(pl.lit(1)).when(pl.col("facing") >= 0).then(pl.lit(2)).otherwise(pl.lit(0)).cast(pl.Int8),
        hu=(pl.col("mask").cast(pl.Int32) == (pl.col("bA") | pl.col("bB"))).cast(pl.Int8),
        acl=pl.when(pl.col("action") == 0).then(pl.lit(0)).when(pl.col("action") == 1).then(pl.lit(1))
        .when((pl.col("action") == 2) | ((pl.col("action") == 5) & (pl.col("amount") <= pl.col("to_call")))).then(pl.lit(2))
        .otherwise(pl.lit(3)).cast(pl.Int8),
        sb=pl.when(pl.col("street") == 0).then(((pl.col("pre_eq") - 0.3) / 0.1).floor().clip(0, 4))
        .otherwise((pl.col("hs") * 5).floor().clip(0, 4)).cast(pl.Int8),
    ).sort("ph_id", "role", "action_no")
    j = j.with_columns(
        vol=((((pl.col("street") == 0) & pl.col("acl").is_in([2, 3])).cast(pl.Int32).cum_sum().over("ph_id", "role")
              - ((pl.col("street") == 0) & pl.col("acl").is_in([2, 3])).cast(pl.Int32)) > 0).cast(pl.Int8))
    j = j.with_columns(vol=pl.when(pl.col("street") == 0).then(pl.col("vol")).otherwise(0).cast(pl.Int8),
                       st=pl.col("street").clip(0, 2).cast(pl.Int8))
    folds = pl.read_csv(BASE / "data" / "folds_tables_5.csv", schema_overrides={"table_idx": pl.Int16, "fold": pl.Int8})
    j = j.join(folds, on="table_idx", how="left")
    j.write_parquet(WORK / "pact.parquet")
    return j


def fit_llr(tr, alpha=8.0, beta=8.0):
    """Count-based smoothed log P(action|ctx, SP evidence) - log P(action|ctx, NEG) with backoff."""
    full = ["st", "fac", "pact", "hu", "sb", "vol"]
    back = ["st", "fac", "pact", "vol"]
    sp = tr.filter((pl.col("grp") == "SP") & pl.col("evidence_rank").is_not_null())
    ng = tr.filter(pl.col("grp") == "NEG")
    def probs(df, keys, name):
        c = df.group_by(keys + ["acl"]).len("n")
        return c.with_columns(tot=pl.col("n").sum().over(keys)).rename({"n": f"n_{name}", "tot": f"t_{name}"})
    grid = (tr.select(full).unique().join(pl.DataFrame({"acl": [0, 1, 2, 3]}, schema={"acl": pl.Int8}), how="cross"))
    # NEG backoff (very well estimated) -> NEG full -> SP backoff -> SP full
    g = grid.join(probs(ng, back, "nb"), on=back + ["acl"], how="left").join(probs(ng, full, "nf"), on=full + ["acl"], how="left")
    g = g.join(probs(sp, back, "sb_"), on=back + ["acl"], how="left").join(probs(sp, full, "sf"), on=full + ["acl"], how="left")
    g = g.fill_null(0)
    g = g.with_columns(t_nb=pl.col("t_nb").max().over(back), t_nf=pl.col("t_nf").max().over(full),
                       t_sb_=pl.col("t_sb_").max().over(back), t_sf=pl.col("t_sf").max().over(full))
    g = g.with_columns(p_nb=(pl.col("n_nb") + 0.5) / (pl.col("t_nb") + 2.0))
    g = g.with_columns(p_nf=(pl.col("n_nf") + alpha * pl.col("p_nb")) / (pl.col("t_nf") + alpha),
                       p_sb=(pl.col("n_sb_") + beta * pl.col("p_nb")) / (pl.col("t_sb_") + beta))
    g = g.with_columns(p_sf=(pl.col("n_sf") + alpha * pl.col("p_sb") * pl.col("p_nf") / pl.col("p_nb")) / (pl.col("t_sf") + alpha))
    # renormalise p_sf within ctx
    g = g.with_columns(p_sf=pl.col("p_sf") / pl.col("p_sf").sum().over(full), p_nf=pl.col("p_nf") / pl.col("p_nf").sum().over(full))
    return g.select(full + ["acl", (pl.col("p_sf").log() - pl.col("p_nf").log()).alias("llr"), "t_sf", "t_nf"])


def stage_llr():
    """Out-of-fold (table folds) action LLR -> hand score (sum over partner actions) -> WORK/hand_scores.parquet"""
    j = action_table() if not (WORK / "pact.parquet").exists() else pl.read_parquet(WORK / "pact.parquet")
    outs = []
    full = ["st", "fac", "pact", "hu", "sb", "vol"]
    for k in range(5):
        tr = j.filter(pl.col("fold") != k)
        te = j.filter(pl.col("fold") == k)
        m = fit_llr(tr)
        outs.append(te.join(m.select(full + ["acl", "llr"]), on=full + ["acl"], how="left"))
    a = pl.concat(outs).with_columns(pl.col("llr").fill_null(0.0))
    a.select("ph_id", "role", "action_no", "street", "acl", "fac", "pact", "hu", "sb", "vol", "llr").write_parquet(WORK / "pact_llr.parquet")
    hs = a.group_by("ph_id").agg(pl.col("llr").sum().alias("llr_sum"),
                                 pl.col("llr").filter(pl.col("street") > 0).sum().alias("llr_post"),
                                 pl.col("llr").filter(pl.col("street") == 0).sum().alias("llr_pre"),
                                 pl.col("llr").max().alias("llr_max"))
    full_m = fit_llr(j)  # in-sample table for inspection / reuse on eval
    full_m.write_parquet(OUT / "sp_action_llr_table.parquet")
    phf = pl.read_parquet(WORK / "phf.parquet")
    out = phf.join(hs, on="ph_id", how="left").with_columns(pl.col("llr_sum", "llr_post", "llr_pre", "llr_max").fill_null(0.0))
    out.write_parquet(WORK / "hand_scores.parquet")
    print("saved hand_scores", out.shape)


# ============================================================================ signatures / analysis
def build_signature_table():
    """Hand-level signature booleans + response-to-partner features -> WORK/sig.parquet"""
    f = pl.read_parquet(WORK / "hand_scores.parquet")
    a = pl.read_parquet(WORK / "pact.parquet")
    # G1: both partners' FIRST preflop action is not a fold; G0: no partner folds first unless facing partner aggression
    first = (a.filter(pl.col("street") == 0).sort("ph_id", "role", "action_no").group_by("ph_id", "role")
             .agg(pl.col("acl").first().alias("first_acl"), pl.col("fac").first().alias("first_fac")))
    g = first.group_by("ph_id").agg((pl.col("first_acl") != 0).all().alias("G1"),
                                    ~((pl.col("first_acl") == 0) & (pl.col("first_fac") != 1)).any().alias("G0"))
    # responses to partner aggression, with NEG-calibrated P(raise | ctx, strength) -> "missed raise"
    ctx = ["st", "hu", "sb", "vol"]
    r = a.filter(pl.col("fac") == 1)
    negd = (r.filter(pl.col("grp") == "NEG").group_by(ctx)
            .agg((pl.col("acl") == 3).mean().alias("p_raise"), (pl.col("acl") == 0).mean().alias("p_fold")))
    r = r.join(negd, on=ctx, how="left").with_columns(
        miss=pl.when(pl.col("acl") != 3).then(pl.col("p_raise")).otherwise(0.0),
        fsur=pl.when(pl.col("acl") == 0).then(1 - pl.col("p_fold")).otherwise(0.0))
    resp = r.group_by("ph_id").agg(
        (pl.col("acl") == 3).sum().alias("n_rr"), pl.len().alias("n_resp"), (pl.col("acl") == 2).sum().alias("n_cr"),
        (pl.col("acl") == 0).sum().alias("n_fr"), pl.col("miss").max().alias("max_miss"), pl.col("fsur").max().alias("max_fsur"))
    f = (f.join(g, on="ph_id", how="left").join(resp, on="ph_id", how="left")
         .with_columns(pl.col("G1", "G0").fill_null(False), pl.col("n_rr", "n_resp", "n_cr", "n_fr").fill_null(0),
                       pl.col("max_miss", "max_fsur").fill_null(0.0)))
    Z = lambda c: pl.col(c).fill_nan(None).fill_null(-1)
    noraise = (pl.col("raise_vs_p_pre_X") + pl.col("raise_vs_p_post_X")) == 0
    f = f.with_columns(
        isev=pl.col("evidence_rank").is_not_null(),
        S_G0_no_first_fold=pl.col("G0"),
        S_G1_both_enter=pl.col("G1"),
        S_faced_partner_aggr=pl.col("faced_p_X") > 0,
        S_no_raise_vs_partner=noraise,
        S_PL=pl.col("G1") & (pl.col("faced_p_X") > 0) & noraise,
        S_strong_check_pact=Z("check_pact_hs_X") >= 0.8,
        S_check_pact_hs60=Z("check_pact_hs_X") >= 0.6,
        S_fold_to_partner_post=pl.col("fold_vs_p_post_X") > 0,
        S_fold_to_partner_post_hs60=Z("fold_vs_p_post_hs_X") >= 0.6,
        S_call_partner_post_hs80=Z("call_vs_p_post_hs_X") >= 0.8,
        S_call_partner_pre_eq55=Z("call_vs_p_pre_str_X") >= 0.55,
        S_fold_to_partner_pre=pl.col("fold_vs_p_pre_X") > 0,
        S_hu_checked_street=(pl.col("n_hu_streets") - pl.col("n_hu_aggr_streets")) > 0,
        S_both_see_flop=pl.col("n_both_streets") > 0,
        S_missed_raise30=pl.col("max_miss") >= 0.3,
        S_U1=(Z("check_pact_hs_X") >= 0.8) | (pl.col("fold_vs_p_post_X") > 0) | (Z("call_vs_p_post_hs_X") >= 0.8),
        S_llr_gt2=pl.col("llr_sum") > 2,
        S_llr_gt4=pl.col("llr_sum") > 4,
    )
    f = f.with_columns(
        S_PL_and_U1=pl.col("S_PL") & pl.col("S_U1"),
        S_PL_and_softresp=pl.col("S_PL") & (pl.col("S_U1") | pl.col("S_missed_raise30") | pl.col("S_call_partner_pre_eq55")),
        soft_post=(pl.col("S_strong_check_pact").cast(pl.Int8) + pl.col("S_fold_to_partner_post").cast(pl.Int8)
                   + pl.col("S_call_partner_post_hs80").cast(pl.Int8)),
        g=pl.col("grp") + pl.when(pl.col("evidence_rank").is_not_null()).then(pl.lit("_ev")).otherwise(pl.lit("_non")),
    )
    folds = pl.read_csv(BASE / "data" / "folds_tables_5.csv", schema_overrides={"table_idx": pl.Int16, "fold": pl.Int8})
    f = f.join(folds, on="table_idx", how="left")
    f = f.sort("pair_id", "hand_seq").with_columns(k=pl.int_range(pl.len()).over("pair_id"), n=pl.len().over("pair_id"))
    f = f.with_columns(q=pl.col("k") / pl.col("n"))
    f.write_parquet(WORK / "sig.parquet")
    return f


def _map5(df, col):
    aps = []
    for _, g in df.group_by("pair_id"):
        g = g.sort(col, descending=True)
        top = g["isev"].to_numpy()[:5]
        nrel = int(g["isev"].sum())
        hits, s = 0, 0.0
        for k, h in enumerate(top, 1):
            if h:
                hits += 1
                s += hits / k
        aps.append(s / min(nrel, 5))
    return float(np.mean(aps))


def stage_analyze():
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb
    import warnings
    warnings.filterwarnings("ignore")
    pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(250)
    log = []
    def P(*x):
        s = " ".join(str(v) for v in x)
        print(s)
        log.append(s)
    f = build_signature_table()
    sigs = [c for c in f.columns if c.startswith("S_")]
    # ---------------- 1. signature rates by group
    order = ["SP_ev", "SP_non", "NEG_non", "UNK_non", "DT_ev", "CI_ev", "DT_non", "CI_non"]
    rates = f.group_by("g").agg([pl.col(s).mean().alias(s) for s in sigs] + [pl.len().alias("n_hands")])
    rates = rates.with_columns(pl.col("g").replace_strict({k: i for i, k in enumerate(order)}).alias("o")).sort("o").drop("o")
    rates.write_csv(OUT / "sp_signatures.csv")
    P("=== 1. hand-level signature rates (fraction of pair-hands) ===")
    P(rates.transpose(include_header=True, column_names="g"))
    # ---------------- 1b. action shares by context x strength bin (SP_ev / SP_non / NEG)
    a = pl.read_parquet(WORK / "pact.parquet").with_columns(
        g=pl.col("grp") + pl.when(pl.col("evidence_rank").is_not_null()).then(pl.lit("_ev")).otherwise(pl.lit("_non")),
        post=(pl.col("street") > 0).cast(pl.Int8))
    a = a.filter(pl.col("g").is_in(["SP_ev", "SP_non", "NEG_non"]) & ((pl.col("post") == 0) | (pl.col("pact") == 1)))
    shares = (a.group_by("g", "post", "fac", "sb", "acl").len()
              .with_columns(share=pl.col("len") / pl.col("len").sum().over("g", "post", "fac", "sb"),
                            n_ctx=pl.col("len").sum().over("g", "post", "fac", "sb"))
              .with_columns(pl.col("acl").replace_strict({0: "fold", 1: "check", 2: "call", 3: "bet_raise"}).alias("action"),
                            pl.col("fac").replace_strict({0: "none", 1: "partner", 2: "other"}).alias("facing"))
              .sort("post", "facing", "sb", "g", "action"))
    shares.select("g", "post", "facing", "sb", "action", "share", "n_ctx").write_csv(OUT / "sp_action_shares.csv")
    P("=== 1b. action shares (post=1 rows require partner active; sb = strength bin: postflop hs*5, preflop (pre_eq-0.3)/0.1) -> sp_action_shares.csv")
    # ---------------- 2. roles
    ev = f.filter(pl.col("g") == "SP_ev")
    rng = np.random.default_rng(0)
    for R in "AB":
        ev = ev.with_columns(**{f"soft_{R}": ((pl.col(f"fold_vs_p_post_{R}") + pl.col(f"fold_vs_p_pre_{R}") + pl.col(f"call_vs_p_pre_{R}")
                                              + pl.col(f"call_vs_p_post_{R}")) > 0).cast(pl.Int32)})
    pp = ev.group_by("pair_id").agg(pl.col("soft_A").sum(), pl.col("soft_B").sum(), (pl.col("netA") > 0).sum().alias("wA"),
                                     (pl.col("netB") > 0).sum().alias("wB"), pl.col("netA").sum().alias("netA"), pl.col("netB").sum().alias("netB"))
    for a_, b_, name in [("soft_A", "soft_B", "passive responder (call/fold facing partner aggression)"), ("wA", "wB", "chip winner")]:
        q = pp.with_columns(tot=pl.col(a_) + pl.col(b_)).filter(pl.col("tot") >= 3)
        dom = (pl.max_horizontal(a_, b_) / pl.col("tot"))
        obs = q.select(dom.mean()).item()
        exp = np.mean([np.mean(np.maximum(x, n - x) / n) for n in q["tot"].to_list() for x in [rng.binomial(n, 0.5, 2000)]])
        P(f"role asymmetry [{name}]: pairs={q.height} mean dominant-role share={obs:.3f} vs 50/50 expectation {exp:.3f}")
    P(f"player_1 share of passive responses in SP evidence: {pp['soft_A'].sum() / (pp['soft_A'].sum() + pp['soft_B'].sum()):.3f}")
    P(f"mean net chips per SP evidence hand: player_1 {ev['netA'].mean():.1f}, player_2 {ev['netB'].mean():.1f}")
    # ---------------- 3. temporal
    P("=== 3. temporal ===")
    from scipy.stats import spearmanr
    for G in ["SP", "DT", "CI"]:
        s = f.filter(pl.col("grp") == G)
        e = s.filter(pl.col("isev"))
        rs = []
        for _, grp in e.group_by("pair_id"):
            rr = grp.sort("hand_seq")["evidence_rank"].to_numpy()
            if len(rr) >= 3:
                rs.append(spearmanr(np.arange(len(rr)), rr).statistic)
        P(f"{G}: evidence position within pair shared-hand timeline, deciles: {np.histogram(e['q'].to_numpy(), bins=10, range=(0, 1))[0].tolist()} mean q={e['q'].mean():.3f}")
        P(f"{G}: evidence hand_seq deciles over dev 0..2999: {np.histogram(e['hand_seq'].to_numpy(), bins=10, range=(0, 3000))[0].tolist()}")
        P(f"{G}: spearman(chronological order, evidence_rank) mean={np.nanmean(rs):.3f}")
        tl = s.with_columns(qq=(pl.col("q") * 5).floor()).group_by("qq").agg(pl.col("S_PL").mean()).sort("qq")
        P(f"{G}: S_PL rate (all shared hands incl. evidence) by pair-timeline quintile: {[round(x, 4) for x in tl['S_PL'].to_list()]}")
    s = f.filter(pl.col("grp") == "NEG").with_columns(qq=(pl.col("q") * 5).floor()).group_by("qq").agg(pl.col("S_PL").mean()).sort("qq")
    P(f"NEG: S_PL rate by quintile: {[round(x, 4) for x in s['S_PL'].to_list()]}")
    # distance (hand_seq) to nearest evidence hand vs S_PL rate among non-evidence hands of SP pairs
    rows = {}
    for _, g in f.filter(pl.col("grp") == "SP").group_by("pair_id"):
        seq = g["hand_seq"].to_numpy(); isev = g["isev"].to_numpy(); pl_ = g["S_PL"].to_numpy()
        d = np.min(np.abs(seq[:, None] - seq[isev][None, :]), axis=1)
        for lo, hi in [(1, 10), (11, 30), (31, 100), (101, 300), (301, 1000), (1001, 3000)]:
            m = (~isev) & (d >= lo) & (d <= hi)
            rr = rows.setdefault((lo, hi), [0, 0]); rr[0] += m.sum(); rr[1] += pl_[m].sum()
    P("SP non-evidence S_PL rate by hand_seq distance to nearest evidence hand: " + str({k: (int(v[0]), round(v[1] / max(v[0], 1), 4)) for k, v in rows.items()}))
    base = f.filter(pl.col("grp") == "NEG")["S_PL"].mean()
    pp = f.filter(pl.col("grp") == "SP").group_by("pair_id").agg(pl.len().alias("n"), pl.col("S_PL").sum().alias("npl"),
                                                               (pl.col("S_PL") & pl.col("isev")).sum().alias("npl_ev"))
    P(f"SP pairs: mean dev shared hands {pp['n'].mean():.1f}; S_PL hands {pp['npl'].mean():.2f}; NEG-rate expectation {base * pp['n'].mean():.2f}; "
      f"excess (~planted) {(pp['npl'] - base * pp['n']).mean():.2f}; listed evidence that are S_PL {pp['npl_ev'].mean():.2f}")
    h = (f.filter((pl.col("grp") == "SP") & ~pl.col("isev")).with_columns(half=pl.col("q") >= 0.5)
         .group_by("pair_id", "half").agg(pl.col("S_PL").mean()).pivot(on="half", index="pair_id", values="S_PL"))
    P(f"SP split-half S_PL rate (excl evidence) corr={np.corrcoef(h['false'].to_numpy(), h['true'].to_numpy())[0, 1]:.3f}; "
      f"1st half {h['false'].mean():.4f} 2nd half {h['true'].mean():.4f}")
    # ---------------- 5. pair level
    P("=== 5. pair-level separability ===")
    lab = pl.read_parquet(DER / "labels.parquet").filter(pl.col("label") == 1)
    posp = set(lab["p1"].to_list()) | set(lab["p2"].to_list())
    bases = {c: f.filter(pl.col("grp") == "NEG")[c].mean() for c in ["S_PL", "S_U1", "S_both_see_flop", "S_llr_gt2", "S_strong_check_pact", "S_G1_both_enter"]}
    agg = [pl.len().alias("n")]
    for c in bases:
        agg += [pl.col(c).sum().alias("c_" + c), pl.col(c).mean().alias("r_" + c)]
    agg += [pl.col("n_rr").sum(), pl.col("n_resp").sum(), pl.col("llr_sum").top_k(5).mean().alias("top5_llr"),
            pl.col("llr_sum").mean().alias("mean_llr"), pl.col("max_miss").top_k(5).mean().alias("top5_miss"),
            (pl.col("raise_vs_p_pre_X") + pl.col("raise_vs_p_post_X")).sum().alias("n_raise_p"), pl.col("S_both_see_flop").sum().alias("n_both")]
    PP = f.group_by("pair_id", "grp", "p1", "p2").agg(agg)
    for c, b in bases.items():
        PP = PP.with_columns(((pl.col("c_" + c) - b * pl.col("n")) / np.sqrt(b * (1 - b) * pl.col("n"))).alias("z_" + c))
    PP = PP.with_columns(neg_raise_resp_rate=-(pl.col("n_rr") + 0.5) / (pl.col("n_resp") + 5),
                         neg_raise_p_per_both=-(pl.col("n_raise_p") + 0.5) / (pl.col("n_both") + 5),
                         unk_clean=~(pl.col("p1").is_in(list(posp)) | pl.col("p2").is_in(list(posp))))
    PP.write_parquet(WORK / "pair_stats.parquet")
    feats = ["n"] + [f"{p}_{c}" for c in bases for p in ("r", "z")] + ["top5_llr", "mean_llr", "top5_miss", "neg_raise_resp_rate", "neg_raise_p_per_both"]
    grp = {k: PP.filter(pl.col("grp") == k) for k in ["SP", "NEG", "UNK", "DT", "CI"]}
    grp["UNKc"] = grp["UNK"].filter("unk_clean")
    def auc(a_, b_, c):
        return roc_auc_score(np.r_[np.ones(a_.height), np.zeros(b_.height)], np.r_[a_[c].to_numpy(), b_[c].to_numpy()])
    res = []
    for c in feats:
        res.append({"feature": c, "SP_vs_NEG": auc(grp["SP"], grp["NEG"], c), "SP_vs_UNK": auc(grp["SP"], grp["UNK"], c),
                    "SP_vs_UNKclean": auc(grp["SP"], grp["UNKc"], c), "SP_vs_DT": auc(grp["SP"], grp["DT"], c),
                    "SP_vs_CI": auc(grp["SP"], grp["CI"], c), "DT_vs_NEG": auc(grp["DT"], grp["NEG"], c), "CI_vs_NEG": auc(grp["CI"], grp["NEG"], c)})
    res = pl.DataFrame(res).with_columns(pl.exclude("feature").round(4))
    res.write_csv(OUT / "sp_pair_auc.csv")
    P(res)
    P(f"mean dev shared hands: SP {grp['SP']['n'].mean():.1f} NEG {grp['NEG']['n'].mean():.1f} UNK {grp['UNK']['n'].mean():.1f} UNKclean {grp['UNKc']['n'].mean():.1f}")
    cuts = np.quantile(grp["NEG"]["n"].to_numpy(), [1 / 3, 2 / 3])
    for lo, hi in [(0, cuts[0]), (cuts[0], cuts[1]), (cuts[1], 1e9)]:
        sel = lambda d: d.filter((pl.col("n") > lo) & (pl.col("n") <= hi))
        s_, n_, u_ = sel(grp["SP"]), sel(grp["NEG"]), sel(grp["UNKc"])
        P(f"shared in ({lo:.0f},{hi:.0f}]: SP {s_.height} NEG {n_.height} UNKc {u_.height} | AUC vs NEG: z_S_PL {auc(s_, n_, 'z_S_PL'):.3f} "
          f"c_S_PL {auc(s_, n_, 'c_S_PL'):.3f} z_S_llr_gt2 {auc(s_, n_, 'z_S_llr_gt2'):.3f} top5_llr {auc(s_, n_, 'top5_llr'):.3f} | vs UNKc top5_llr {auc(s_, u_, 'top5_llr'):.3f}")
    # ---------------- 6. evidence ranking recipes (proxy: dev listed evidence within SP pairs)
    P("=== 6. evidence ranking (proxy MAP@5 on dev shared hands of SP pairs; random = see first row) ===")
    sp = f.filter(pl.col("grp") == "SP").with_columns(rnd=pl.lit(np.random.default_rng(0).random(f.filter(pl.col("grp") == "SP").height)))
    Zf = lambda c: pl.col(c).fill_nan(None).fill_null(0)
    recipes = {
        "random": pl.col("rnd"),
        "S_U1 (binary)": pl.col("S_U1").cast(pl.Float64) + 1e-3 * pl.col("rnd"),
        "S_PL (binary)": pl.col("S_PL").cast(pl.Float64) + 1e-3 * pl.col("rnd"),
        "llr_sum": pl.col("llr_sum"),
        "llr_post": pl.col("llr_post"),
        "RULE: 10*PL + 4*max_miss + 1.5*strong_check + n_call_resp + 0.3*llr_post": pl.col("S_PL").cast(pl.Float64) * 10 + 4 * Zf("max_miss")
        + 1.5 * pl.col("S_strong_check_pact").cast(pl.Float64) + pl.col("n_cr") + 0.3 * pl.col("llr_post"),
    }
    rec = []
    for k, e in recipes.items():
        d = sp.with_columns(s=e)
        rec.append({"recipe": k, "map5": _map5(d, "s"), "map5_with_minus3q": _map5(d.with_columns(s=pl.col("s") - 3 * pl.col("q")), "s")})
    feat = [c for c in f.columns if c.endswith("_X")] + ["n_hu_streets", "n_hu_aggr_streets", "n_both_streets", "n_both_aggr_streets",
            "n_both_pairaggr_streets", "n_pre_raises", "n_actions", "max_street", "preA", "preB", "llr_sum", "llr_post", "llr_pre",
            "llr_max", "G1", "G0", "players_at_showdown", "n_rr", "n_resp", "n_cr", "n_fr", "max_miss", "max_fsur", "soft_post"]
    def oof_rank(tr, te, fe):
        oof = np.zeros(te.height)
        Xt = tr.select([pl.col(c).cast(pl.Float64).fill_nan(None) for c in fe]).to_numpy(); yt = tr["isev"].to_numpy().astype(int)
        ft = tr["fold"].to_numpy()
        Xe = te.select([pl.col(c).cast(pl.Float64).fill_nan(None) for c in fe]).to_numpy(); fe_ = te["fold"].to_numpy()
        for kk in range(5):
            m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=3)
            m.fit(Xt[ft != kk], yt[ft != kk]); oof[fe_ == kk] = m.predict_proba(Xe[fe_ == kk])[:, 1]
        return oof
    o = oof_rank(sp, sp, feat)
    rec.append({"recipe": "LGBM hand ranker trained within SP pairs (listed vs unlisted), OOF table folds", "map5": _map5(sp.with_columns(s=pl.Series(o)), "s"),
                "map5_with_minus3q": _map5(sp.with_columns(s=pl.Series(o) - 0.3 * pl.col("q")), "s")})
    allpos = f.filter(pl.col("grp").is_in(["SP", "DT", "CI"]))
    o2 = oof_rank(allpos, sp, feat)
    rec.append({"recipe": "LGBM hand ranker trained on all positive pairs (3 families), OOF", "map5": _map5(sp.with_columns(s=pl.Series(o2)), "s"),
                "map5_with_minus3q": _map5(sp.with_columns(s=pl.Series(o2) - 0.3 * pl.col("q")), "s")})
    o3 = oof_rank(sp, sp, feat + ["q"])
    rec.append({"recipe": "LGBM within SP + timeline position q as feature", "map5": _map5(sp.with_columns(s=pl.Series(o3)), "s"), "map5_with_minus3q": None})
    tr = f.filter(((pl.col("grp") == "SP") & pl.col("isev")) | (pl.col("grp") == "NEG"))
    o4 = oof_rank(tr, sp, feat)
    rec.append({"recipe": "LGBM SP evidence vs NEG hands (planted-vs-normal detector)", "map5": _map5(sp.with_columns(s=pl.Series(o4)), "s"),
                "map5_with_minus3q": _map5(sp.with_columns(s=pl.Series(o4) - 0.3 * pl.col("q")), "s")})
    rec = pl.DataFrame(rec, schema={"recipe": pl.String, "map5": pl.Float64, "map5_with_minus3q": pl.Float64}).with_columns(pl.col("map5", "map5_with_minus3q").round(4))
    rec.write_csv(OUT / "sp_evidence_ranking.csv")
    P(rec)
    # hand-level AUCs
    spev, spnon, neg = f.filter(pl.col("g") == "SP_ev"), f.filter(pl.col("g") == "SP_non"), f.filter(pl.col("grp") == "NEG")
    for c in ["llr_sum", "llr_post", "llr_pre", "max_miss"]:
        P(f"hand AUC {c}: SP_ev vs NEG {auc(spev, neg, c):.4f} | SP_ev vs SP_non {auc(spev, spnon, c):.4f}")
    (OUT / "sp_analysis_output.txt").write_text("\n".join(log), encoding="utf-8")


ARCHETYPES = ["H13B5DD9B5BB68A", "H067FA4AFCA9CEF", "HC87612371A8222", "HCDF38FB6CD488F", "H5BFF709DC36AFA", "HA532C1A81DF5BC"]


def stage_archetypes():
    db = HandDB()
    ph = pl.read_parquet(WORK / "sig.parquet")
    hid = pl.read_parquet(WORK / "hands.parquet").select("hand_idx", "hand_id")
    sel = ph.join(hid, on="hand_idx").filter(pl.col("hand_id").is_in(ARCHETYPES) & pl.col("isev"))
    txt = []
    for r in sel.iter_rows(named=True):
        txt.append(hand_text(db, r["hand_idx"], r["p1"], r["p2"], f"[EVIDENCE {r['grp']} pair={r['pair_id']} rank={r['evidence_rank']} "
                             f"llr_sum={r['llr_sum']:.2f} max_miss={r['max_miss']:.2f} PL={r['S_PL']} U1={r['S_U1']}]"))
    (OUT / "sp_archetypes.txt").write_text("\n\n".join(txt), encoding="utf-8")
    print("\n\n".join(txt))


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st in ("load", "all"):
        stage_load()
    if st in ("actions", "all"):
        stage_actions()
    if st in ("features", "all"):
        stage_features()
    if st in ("llr", "all"):
        if (WORK / "pact.parquet").exists() and st == "all":
            (WORK / "pact.parquet").unlink()
        stage_llr()
    if st in ("analyze", "all"):
        stage_analyze()
    if st in ("archetypes", "all"):
        stage_archetypes()
    if st == "print":
        fam = sys.argv[2] if len(sys.argv) > 2 else "soft_play"
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        stage_print(n_ev=n, n_non=15, fam=fam, seed=seed)
