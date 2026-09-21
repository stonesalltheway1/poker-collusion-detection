"""Evidence-hand forensics for behavior family 'directed_transfer' (DT).

Re-runnable, stage-based:
  python scripts/forensics_directed_transfer.py extract            # subset parquet files -> research/forensics/dt_sub_*.parquet
  python scripts/forensics_directed_transfer.py print 40 directed_transfer 15   # print evidence + non-evidence shared hands
  python scripts/forensics_directed_transfer.py features           # per pair-hand + per partner-action features (numba equity)
  python scripts/forensics_directed_transfer.py policy             # normal-play action model (LightGBM) on all subset actions
  python scripts/forensics_directed_transfer.py policy_cf          # cross-fitted (2 table halves) version of the policy
  python scripts/forensics_directed_transfer.py signatures         # hand signatures: coverage / rates per group -> dt_handsig.parquet
  python scripts/forensics_directed_transfer.py policy_signatures  # policy-surprise signatures + pair-level AUC
  python scripts/forensics_directed_transfer.py temporal           # time pattern of evidence / signature hands, listing rule
  python scripts/forensics_directed_transfer.py pairlevel          # pair-level separability (flow/gift signatures)
  python scripts/forensics_directed_transfer.py ranker             # evidence-hand ranker (OOF) + dev MAP@5 of recipes
  python scripts/forensics_directed_transfer.py evalcheck          # does the DT signature persist in the eval phase?
Order: extract -> features -> policy -> policy_cf -> signatures -> policy_signatures -> temporal -> pairlevel -> ranker -> evalcheck.
Needs scripts/fast_eval.py (numba 7-card evaluator, verified 100% order-agreement with phevaluator).
Memory-light: polars scan with is_in filters on hand_idx; POLARS_MAX_THREADS=3.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "3")
import numpy as np
import polars as pl

BASE = Path(__file__).resolve().parent.parent
DER = BASE / "data" / "derived"
OUT = BASE / "research" / "forensics"
OUT.mkdir(parents=True, exist_ok=True)
FAM = "directed_transfer"
RANKS, SUITS = "23456789TJQKA", "cdhs"
ANAME = {0: "fold", 1: "check", 2: "call", 3: "bet", 4: "raise", 5: "all_in"}
SNAME = {0: "PRE", 1: "FLOP", 2: "TURN", 3: "RIVER"}
SEED = 42


def scan(n):
    return pl.scan_parquet(DER / f"{n}.parquet")


def cs(c):
    return RANKS[c // 4] + SUITS[c % 4]


# ============================================================================ extract
def stage_extract():
    import duckdb
    labels = pl.read_parquet(DER / "labels.parquet")
    ev = pl.read_parquet(DER / "evidence.parquet")
    con = duckdb.connect()
    con.sql("SET threads=3; SET memory_limit='4GB'; SET preserve_insertion_order=false")
    seats_p = (DER / "seats.parquet").as_posix()
    hands_p = (DER / "hands.parquet").as_posix()
    # all dev co-seated pairs with shared-hand counts
    dev_pairs = con.sql(f"""
        WITH s AS (SELECT s.hand_idx, s.player_idx FROM '{seats_p}' s JOIN '{hands_p}' h USING (hand_idx) WHERE h.phase = 0)
        SELECT a.player_idx AS p1, b.player_idx AS p2, count(*)::INT AS dev_shared
        FROM s a JOIN s b ON a.hand_idx = b.hand_idx AND a.player_idx < b.player_idx
        GROUP BY ALL""").pl()
    print("dev pairs", dev_pairs.shape)
    lab = labels.with_columns(pl.min_horizontal("p1", "p2").alias("lo"), pl.max_horizontal("p1", "p2").alias("hi"))
    dev_pairs = dev_pairs.join(lab.select(pl.col("lo").alias("p1"), pl.col("hi").alias("p2"), "pair_id", "label",
                                          "behavior_family"), on=["p1", "p2"], how="left")
    pos_players = set(labels.filter(pl.col("label") == 1).select(pl.concat_list("p1", "p2").explode()).to_series().to_list())
    unk = dev_pairs.filter(pl.col("pair_id").is_null() & (pl.col("dev_shared") >= 38)
                           & ~pl.col("p1").is_in(pos_players) & ~pl.col("p2").is_in(pos_players))
    unk = unk.sample(n=3000, seed=SEED).with_columns(
        (pl.lit("U") + pl.col("p1").cast(pl.Utf8) + "_" + pl.col("p2").cast(pl.Utf8)).alias("pair_id"),
        pl.lit(-1).cast(pl.Int64).alias("label"), pl.lit("unknown").alias("behavior_family"))
    # selected pairs keep labels' player_1/player_2 orientation (A = player_1, B = player_2)
    sel_lab = labels.select("pair_id", pl.col("p1").alias("A"), pl.col("p2").alias("B"), "label", "behavior_family")
    sel_unk = unk.select("pair_id", pl.col("p1").alias("A"), pl.col("p2").alias("B"), "label", "behavior_family")
    sel = pl.concat([sel_lab, sel_unk])
    sel = sel.join(dev_pairs.select(pl.min_horizontal("p1", "p2").alias("lo"), pl.max_horizontal("p1", "p2").alias("hi"),
                                    "dev_shared"),
                   left_on=[pl.min_horizontal("A", "B"), pl.max_horizontal("A", "B")], right_on=["lo", "hi"], how="left")
    sel = sel.select("pair_id", "A", "B", "label", "behavior_family", "dev_shared")
    sel.write_parquet(OUT / "dt_pairs.parquet")
    dev_pairs.select("p1", "p2", "dev_shared", "label").write_parquet(OUT / "dt_all_dev_pair_counts.parquet")
    print(sel.group_by("behavior_family").agg(pl.len(), pl.col("dev_shared").mean().alias("mean_sh"), pl.col("dev_shared").median().alias("med_sh")))
    # pair-hands (dev) for all selected pairs
    players = set(sel["A"].to_list()) | set(sel["B"].to_list())
    s = (scan("seats").select("hand_idx", "player_idx")
         .filter(pl.col("player_idx").is_in(list(players)))
         .join(scan("hands").filter(pl.col("phase") == 0).select("hand_idx"), on="hand_idx").collect())
    sa = s.rename({"player_idx": "A"})
    sb = s.rename({"player_idx": "B"})
    ph = (sel.lazy().select("pair_id", "A", "B").join(sa.lazy(), on="A")
          .join(sb.lazy(), on=["B", "hand_idx"]).collect())
    ph = ph.join(ev.select("pair_id", "hand_idx", "evidence_rank"), on=["pair_id", "hand_idx"], how="left")
    ph.write_parquet(OUT / "dt_pair_hands.parquet")
    print("pair-hands", ph.shape, "evidence matched", ph["evidence_rank"].is_not_null().sum(), "of", ev.height)
    hidx = sorted(set(ph["hand_idx"].to_list()) | set(ev["hand_idx"].to_list()))
    print("distinct hands", len(hidx))
    for name in ("hands", "seats", "actions"):
        df = scan(name).filter(pl.col("hand_idx").is_in(hidx)).collect()
        df.write_parquet(OUT / f"dt_sub_{name}.parquet")
        print(name, df.shape)


# ============================================================================ hand access
CAT_BOUNDS = [(10, "StrFlush"), (166, "Quads"), (322, "FullHouse"), (1599, "Flush"), (1609, "Straight"),
              (2467, "Trips"), (3325, "TwoPair"), (6185, "Pair"), (7462, "HighCard")]


def cat_name(r):
    for b, n in CAT_BOUNDS:
        if r <= b:
            return n
    return "?"


def hand_rank(cards):
    from phevaluator import evaluate_cards
    return evaluate_cards(*cards)


def equity(holes, board, dead=(), mc=3000):
    """Exact all-in equity (share of pot, ties split) for each hole pair given board (0-5 cards),
    enumerating all runouts over unseen cards (all seated hole cards known -> pass as dead).
    Preflop (3+ cards to come) uses Monte Carlo `mc` samples."""
    from itertools import combinations
    from phevaluator import evaluate_cards
    used = set(board) | set(dead)
    for h in holes:
        used |= set(h)
    rest = [c for c in range(52) if c not in used]
    need = 5 - len(board)
    eq = np.zeros(len(holes))
    if need == 0:
        runs = [()]
    elif need <= 2:
        runs = combinations(rest, need)
    else:
        rng = np.random.default_rng(0)
        runs = (tuple(int(x) for x in rng.choice(rest, need, replace=False)) for _ in range(mc))
    n = 0
    for r in runs:
        b = list(board) + list(r)
        rk = [evaluate_cards(*b, h[0], h[1]) for h in holes]
        m = min(rk)
        w = [i for i, x in enumerate(rk) if x == m]
        for i in w:
            eq[i] += 1.0 / len(w)
        n += 1
    return eq / n


class Data:
    def __init__(self):
        self.hands = pl.read_parquet(OUT / "dt_sub_hands.parquet").sort("hand_idx")
        self.seats = pl.read_parquet(OUT / "dt_sub_seats.parquet").sort("hand_idx", "seat_no")
        self.actions = pl.read_parquet(OUT / "dt_sub_actions.parquet").sort("hand_idx", "action_no")
        self.pairs = pl.read_parquet(OUT / "dt_pairs.parquet")
        self.ph = pl.read_parquet(OUT / "dt_pair_hands.parquet")
        self.ev = pl.read_parquet(DER / "evidence.parquet")
        self.h_idx = {h: i for i, h in enumerate(self.hands["hand_idx"].to_list())}
        self.hrows = self.hands.to_dicts()
        self.S = {c: self.seats[c].to_numpy() for c in self.seats.columns}
        self.A = {c: self.actions[c].to_numpy() for c in self.actions.columns}
        self.s_off = self._offsets(self.S["hand_idx"])
        self.a_off = self._offsets(self.A["hand_idx"])

    @staticmethod
    def _offsets(arr):
        u, st = np.unique(arr, return_index=True)
        en = np.append(st[1:], len(arr))
        return {int(h): (int(s), int(e)) for h, s, e in zip(u, st, en)}

    def hand(self, h):
        hr = self.hrows[self.h_idx[h]]
        s0, s1 = self.s_off[h]
        a0, a1 = self.a_off.get(h, (0, 0))
        seats = {k: v[s0:s1] for k, v in self.S.items()}
        acts = {k: v[a0:a1] for k, v in self.A.items()}
        return hr, seats, acts


NB = {0: 0, 1: 3, 2: 4, 3: 5}


def fmt_hand(D, h, A, B, equity_on=True):
    hr, se, ac = D.hand(h)
    role = {}
    for p in se["player_idx"]:
        role[int(p)] = "A" if p == A else ("B" if p == B else "o")
    board = list(hr["board"])
    bb = hr["bb"]
    lines = [f"hand_idx={h} seq={hr['hand_seq']} phase={hr['phase']} sb/bb={hr['sb']}/{bb} button_seat={hr['button_seat']} "
             f"dealt={hr['players_dealt']} sd={hr['players_at_showdown']} final_pot={hr['final_pot']} "
             f"board={' '.join(cs(c) for c in board)}"]
    holes = {}
    for i in range(len(se["player_idx"])):
        p = int(se["player_idx"][i])
        c1, c2 = int(se["c1"][i]), int(se["c2"][i])
        holes[p] = (c1, c2)
        fr = ""
        if len(board) == 5:
            r = hand_rank([c1, c2] + board)
            fr = f"final={cat_name(r)}({r})"
        lab = role[p] if role[p] != "o" else f"o{p}"
        lines.append(f"   seat{se['seat_no'][i]} {lab:>6} stack={se['starting_stack'][i]:5d} ({se['starting_stack'][i]/bb:6.1f}bb) "
                     f"{cs(c1)}{cs(c2)} contrib={se['total_contribution'][i]:5d} net={se['net_chips'][i]:6d} "
                     f"fold={int(se['folded'][i])} sd={int(se['went_to_showdown'][i])} won={se['won_share'][i]:.2f} {fr}")
    active = set(holes)
    all_holes = [c for hh in holes.values() for c in hh]
    cur_street = -1
    for j in range(len(ac["action_no"])):
        stt = int(ac["street"][j])
        nb = NB[stt]
        if stt != cur_street:
            cur_street = stt
            lines.append(f"  -- {SNAME[stt]} {' '.join(cs(c) for c in board[:nb])}   active={len(active)}")
        p = int(ac["player_idx"][j])
        a = int(ac["action"][j])
        extra = ""
        if equity_on and role[p] in "AB" and p in active and len(active) >= 2 and nb <= len(board):
            act_list = sorted(active)
            dead = [c for q in holes if q not in active for c in holes[q]]
            eq = equity([holes[q] for q in act_list], board[:nb], dead=dead, mc=1000)
            e_me = eq[act_list.index(p)] if p in act_list else float("nan")
            other = B if p == A else A
            eqs = f"eq={e_me:.2f}"
            if other in active and len(active) > 2 and p in active:
                dead2 = [c for q in holes if q not in (p, other) for c in holes[q]]
                hu = equity([holes[p], holes[other]], board[:nb], dead=dead2, mc=1000)
                eqs += f" huVsPartner={hu[0]:.2f}"
            if nb >= 3:
                eqs += " " + cat_name(hand_rank(list(holes[p]) + board[:nb]))
            extra = eqs
        lab = role[p] if role[p] != "o" else f"o{p}"
        agg_flag = ""
        if a == 5:
            agg_flag = "(ai-call)" if ac["amount"][j] <= ac["to_call"][j] else "(ai-raise)"
        potodds = ""
        if ac["to_call"][j] > 0:
            potodds = f"odds={ac['to_call'][j]/(ac['pot_before'][j]+ac['to_call'][j]):.2f}"
        lines.append(f"     #{ac['action_no'][j]:2d} {lab:>6} {ANAME[a]:6s}{agg_flag:10s} amt={ac['amount'][j]:5d} to={ac['amount_to'][j]:5d} "
                     f"pot={ac['pot_before'][j]:5d} tocall={ac['to_call'][j]:5d} stack={ac['stack_before'][j]:5d} "
                     f"act={ac['players_active'][j]} {potodds:9s} {extra}")
        if a == 0:
            active.discard(p)
    return "\n".join(lines)


def stage_print(n="25", fam=FAM, n_non="15", out_name=None):
    n, n_non = int(n), int(n_non)
    D = Data()
    pairs = D.pairs.filter(pl.col("behavior_family") == fam).sample(fraction=1.0, shuffle=True, seed=SEED)
    buf = []
    k = 0
    for pr in pairs.iter_rows(named=True):
        evh = D.ev.filter(pl.col("pair_id") == pr["pair_id"]).sort("evidence_rank")
        for er in evh.iter_rows(named=True):
            if k >= n:
                break
            buf.append(f"\n===== EVIDENCE pair={pr['pair_id']} fam={fam} rank={er['evidence_rank']} A={pr['A']} B={pr['B']} dev_shared={pr['dev_shared']}")
            buf.append(fmt_hand(D, er["hand_idx"], pr["A"], pr["B"]))
            k += 1
        if k >= n:
            break
    for pr in pairs.head(n_non).iter_rows(named=True):
        ph = D.ph.filter((pl.col("pair_id") == pr["pair_id"]) & pl.col("evidence_rank").is_null())
        for h in ph.sample(n=1, seed=SEED)["hand_idx"].to_list():
            buf.append(f"\n===== NON-EVIDENCE shared pair={pr['pair_id']} fam={fam} A={pr['A']} B={pr['B']}")
            buf.append(fmt_hand(D, h, pr["A"], pr["B"]))
    txt = "\n".join(buf)
    (OUT / (out_name or f"print_{fam}.txt")).write_text(txt, encoding="utf-8")
    print(f"wrote {len(txt.splitlines())} lines")


# ============================================================================ features
MAX_RUNS = 300


def _preflop_strength_table():
    """HU equity of each 169 hole class vs a random hand (MC); cached."""
    p = OUT / "preflop_strength169.npy"
    if p.exists():
        return np.load(p)
    from phevaluator import evaluate_cards
    rng = np.random.default_rng(1)
    tab = np.zeros((13, 13))  # [hi, lo] suited if hi>lo index order swapped: tab[r1,r2] r1>r2 suited, r1<r2 offsuit, diag pair
    for r1 in range(13):
        for r2 in range(13):
            if r1 == r2:
                h = (r1 * 4, r1 * 4 + 1)
            elif r1 > r2:
                h = (r1 * 4, r2 * 4)  # suited (same suit 0)
            else:
                h = (r2 * 4, r1 * 4 + 1)  # offsuit
            rest = np.array([c for c in range(52) if c not in h])
            w = 0.0
            N = 3000
            for _ in range(N):
                s = rng.choice(rest, 7, replace=False).tolist()
                a = evaluate_cards(h[0], h[1], *s[:5])
                b = evaluate_cards(s[5], s[6], *s[:5])
                w += 1.0 if a < b else (0.5 if a == b else 0.0)
            tab[r1, r2] = w / N
    np.save(p, tab)
    return tab


def pf_strength(tab, c1, c2):
    r1, r2 = c1 // 4, c2 // 4
    if r1 == r2:
        return tab[r1, r1]
    hi, lo = max(r1, r2), min(r1, r2)
    if c1 % 4 == c2 % 4:
        return tab[hi, lo]
    return tab[lo, hi]


def fast_equity(holes, board, dead, rng):
    """Equity with exact enumeration when <= MAX_RUNS runouts, else MC MAX_RUNS samples."""
    from itertools import combinations
    from phevaluator import evaluate_cards
    used = set(board) | set(dead)
    for h in holes:
        used.update(h)
    rest = [c for c in range(52) if c not in used]
    need = 5 - len(board)
    eq = np.zeros(len(holes))
    if need == 0:
        runs = [()]
    elif need == 1:
        runs = [(c,) for c in rest]
    elif need == 2 and len(rest) * (len(rest) - 1) // 2 <= MAX_RUNS:
        runs = combinations(rest, 2)
    else:
        ra = np.array(rest)
        runs = (tuple(int(x) for x in rng.choice(ra, need, replace=False)) for _ in range(MAX_RUNS))
    n = 0
    for r in runs:
        b = list(board) + list(r)
        rk = [evaluate_cards(*b, h[0], h[1]) for h in holes]
        m = min(rk)
        nw = rk.count(m)
        for i, x in enumerate(rk):
            if x == m:
                eq[i] += 1.0 / nw
        n += 1
    return eq / max(n, 1)


def _feat_worker(args):
    """Compute action-level records + pair-hand summaries for a chunk of hands (numba equity)."""
    hands_chunk, pairs_by_hand = args
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fast_eval import equity as nequity, made_score
    tab = _preflop_strength_table()
    hh = pl.read_parquet(OUT / "dt_sub_hands.parquet").filter(pl.col("hand_idx").is_in(hands_chunk))
    se = pl.read_parquet(OUT / "dt_sub_seats.parquet").filter(pl.col("hand_idx").is_in(hands_chunk))
    ac = pl.read_parquet(OUT / "dt_sub_actions.parquet").filter(pl.col("hand_idx").is_in(hands_chunk))
    hrow = {r["hand_idx"]: r for r in hh.select("hand_idx", "bb", "board", "hand_seq").iter_rows(named=True)}
    seats_by = {}
    for r in se.iter_rows(named=True):
        seats_by.setdefault(r["hand_idx"], []).append(r)
    acts_by = {}
    for r in ac.sort("hand_idx", "action_no").iter_rows(named=True):
        acts_by.setdefault(r["hand_idx"], []).append(r)
    act_rows, sum_rows = [], []
    zdead = np.zeros(52, np.bool_)
    for h in hands_chunk:
        hr = hrow[h]
        bb = hr["bb"]
        board = list(hr["board"])
        barr = np.zeros(5, np.int64)
        barr[:len(board)] = board
        seats = {r["player_idx"]: r for r in seats_by[h]}
        holes = {p: (r["c1"], r["c2"]) for p, r in seats.items()}
        acts = acts_by.get(h, [])
        active = set(holes)
        last_agg = None
        last_agg_street = -1
        states = []
        vol = {p: False for p in holes}
        agg_cnt = {p: 0 for p in holes}
        eq_cache = {}
        for j, a in enumerate(acts):
            st = a["street"]
            if st != last_agg_street:
                last_agg, last_agg_street = None, st
            states.append((frozenset(active), last_agg))
            p, an = a["player_idx"], a["action"]
            is_agg = an in (3, 4) or (an == 5 and a["amount"] > a["to_call"])
            if an in (2, 3, 4, 5) and a["amount"] > 0:
                vol[p] = True
            if is_agg:
                last_agg = p
                agg_cnt[p] += 1
            if an == 0:
                active.discard(p)
        final_active = active

        def eq_of(players, nb, X):
            key = (tuple(sorted(players)), nb)
            if key not in eq_cache:
                pls = key[0]
                ho = np.array([holes[q] for q in pls], np.int64)
                dead = zdead.copy()
                for q in holes:
                    if q not in players:
                        dead[holes[q][0]] = True
                        dead[holes[q][1]] = True
                e = nequity(ho, len(pls), barr, nb, dead, 1000, h % 100000)
                eq_cache[key] = dict(zip(pls, e))
            return eq_cache[key][X]

        for (pair_id, A, B) in pairs_by_hand[h]:
            sA, sB = seats[A], seats[B]
            last_two = set(holes) == {A, B}
            act_now = set(holes)
            for a in acts:
                if a["action"] == 0:
                    act_now.discard(a["player_idx"])
                    if act_now == {A, B}:
                        last_two = True
            sm = dict(pair_id=pair_id, hand_idx=h, hand_seq=hr["hand_seq"], bb=bb,
                      netA=sA["net_chips"] / bb, netB=sB["net_chips"] / bb,
                      contA=sA["total_contribution"] / bb, contB=sB["total_contribution"] / bb,
                      foldA=sA["folded"], foldB=sB["folded"], sdA=sA["went_to_showdown"], sdB=sB["went_to_showdown"],
                      wonA=sA["won_share"], wonB=sB["won_share"], volA=vol[A], volB=vol[B],
                      aggA=agg_cnt[A], aggB=agg_cnt[B], last_two=last_two,
                      n_final=len(final_active), board_len=len(board), n_dealt=len(holes),
                      stackA=sA["starting_stack"] / bb, stackB=sB["starting_stack"] / bb,
                      pfA=pf_strength(tab, *holes[A]), pfB=pf_strength(tab, *holes[B]))
            sum_rows.append(sm)
            for j, a in enumerate(acts):
                X = a["player_idx"]
                if X not in (A, B):
                    continue
                Y = B if X == A else A
                act_set, la = states[j]
                if Y not in act_set:
                    continue
                an = a["action"]
                st = a["street"]
                is_agg = an in (3, 4) or (an == 5 and a["amount"] > a["to_call"])
                facing_Y = la == Y
                y_resp = -1
                if is_agg:
                    for a2 in acts[j + 1:]:
                        if a2["player_idx"] == Y:
                            y_resp = a2["action"]
                            break
                        if a2["player_idx"] == X:
                            break
                if not (st > 0 or facing_Y or (is_agg and y_resp >= 0)):
                    continue
                nb = NB[st]
                if nb > len(board):
                    continue
                eq_mw = eq_of(act_set, nb, X)
                eq_hu = eq_mw if len(act_set) == 2 else eq_of({X, Y}, nb, X)
                eq_hu_final = eq_of({X, Y}, len(board), X) if len(board) >= nb else -1.0
                own = int(made_score(holes[X][0], holes[X][1], barr, nb)) if nb >= 3 else -1
                act_rows.append(dict(
                    pair_id=pair_id, hand_idx=h, action_no=a["action_no"], actor=("A" if X == A else "B"),
                    street=st, action=an, is_agg=is_agg, facing_partner=facing_Y, partner_resp=y_resp,
                    amount_bb=a["amount"] / bb, to_call_bb=a["to_call"] / bb, pot_bb=a["pot_before"] / bb,
                    stack_bb=a["stack_before"] / bb, n_active=len(act_set), eq_mw=float(eq_mw), eq_hu=float(eq_hu), eq_hu_final=float(eq_hu_final),
                    own_cat=(own >> 20) if own >= 0 else -1, own_score=own, pf_str=pf_strength(tab, *holes[X])))
    return act_rows, sum_rows


def stage_features(nproc="3", limit=None):
    from multiprocessing import Pool
    _preflop_strength_table()
    ph = pl.read_parquet(OUT / "dt_pair_hands.parquet")
    if limit:
        keep = ph.select("pair_id").unique().sample(n=int(limit), seed=SEED)
        ph = ph.join(keep, on="pair_id")
    pairs_by_hand = {}
    for r in ph.select("hand_idx", "pair_id", "A", "B").iter_rows():
        pairs_by_hand.setdefault(r[0], []).append((r[1], r[2], r[3]))
    hands = sorted(pairs_by_hand)
    chunks = [hands[i:i + 4000] for i in range(0, len(hands), 4000)]
    jobs = [(c, {h: pairs_by_hand[h] for h in c}) for c in chunks]
    A_all, S_all = [], []
    import time
    t0 = time.time()
    with Pool(int(nproc)) as pool:
        for k, (ar, sr) in enumerate(pool.imap_unordered(_feat_worker, jobs)):
            A_all.append(pl.DataFrame(ar) if ar else None)
            S_all.append(pl.DataFrame(sr))
            if k % 10 == 0:
                print(f"chunk {k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    acts = pl.concat([x for x in A_all if x is not None], how="vertical_relaxed")
    summ = pl.concat(S_all, how="vertical_relaxed")
    ev = pl.read_parquet(DER / "evidence.parquet").select("pair_id", "hand_idx", "evidence_rank")
    summ = summ.join(ev, on=["pair_id", "hand_idx"], how="left")
    acts.write_parquet(OUT / "dt_feat_actions.parquet")
    summ.write_parquet(OUT / "dt_feat_pairhands.parquet")
    print(acts.shape, summ.shape, f"{time.time() - t0:.0f}s")


# ============================================================================ normal-play policy model
POLICY_FEATS = ["street", "facing", "to_call_bb", "pot_bb", "stack_bb", "spr", "call_frac", "players_active",
                "n_agg_street_before", "n_agg_hand_before", "put_street_bb", "put_hand_bb", "pos", "is_pf_aggr",
                "hs1", "hsN", "own_cat", "pf_str", "players_dealt", "n_act_street_before"]


def build_action_states(actions, seats, hands):
    """Per-action public/own-information state features (everything a normal agent could condition on)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fast_eval import equity_vs_random, made_score
    tab = _preflop_strength_table()
    a = actions.sort("hand_idx", "action_no")
    a = a.join(hands.select("hand_idx", "bb", "button_seat", "players_dealt", "board"), on="hand_idx")
    a = a.join(seats.select("hand_idx", "player_idx", "seat_no", "c1", "c2"), on=["hand_idx", "player_idx"])
    agg = (pl.col("action").is_in([3, 4]) | ((pl.col("action") == 5) & (pl.col("amount") > pl.col("to_call"))))
    a = a.with_columns(agg.alias("is_agg"))
    a = a.with_columns(
        (pl.col("is_agg").cast(pl.Int32).cum_sum().over(["hand_idx", "street"]) - pl.col("is_agg").cast(pl.Int32)).alias("n_agg_street_before"),
        (pl.col("is_agg").cast(pl.Int32).cum_sum().over("hand_idx") - pl.col("is_agg").cast(pl.Int32)).alias("n_agg_hand_before"),
        (pl.col("action_no").rank("ordinal").over(["hand_idx", "street"]) - 1).alias("n_act_street_before"),
        ((pl.col("amount_to") - pl.col("amount")) / pl.col("bb")).alias("put_street_bb"),
        ((pl.col("amount").cum_sum().over(["hand_idx", "player_idx"]) - pl.col("amount")) / pl.col("bb")).alias("put_hand_bb"),
        (pl.col("to_call") > 0).alias("facing"),
        (pl.col("to_call") / pl.col("bb")).alias("to_call_bb"),
        (pl.col("pot_before") / pl.col("bb")).alias("pot_bb"),
        (pl.col("stack_before") / pl.col("bb")).alias("stack_bb"),
        (pl.col("stack_before") / pl.col("pot_before").clip(1)).alias("spr"),
        (pl.col("to_call") / (pl.col("pot_before") + pl.col("to_call")).clip(1)).alias("call_frac"),
        ((pl.col("seat_no") - pl.col("button_seat")) % 6).alias("pos"),
    )
    # last preflop aggressor per hand
    pfa = (a.filter((pl.col("street") == 0) & pl.col("is_agg")).group_by("hand_idx")
           .agg(pl.col("player_idx").last().alias("pf_aggr")))
    a = a.join(pfa, on="hand_idx", how="left").with_columns(
        ((pl.col("street") > 0) & (pl.col("player_idx") == pl.col("pf_aggr"))).fill_null(False).alias("is_pf_aggr"))
    st = a["street"].to_numpy()
    c1 = a["c1"].to_numpy().astype(np.int64)
    c2 = a["c2"].to_numpy().astype(np.int64)
    pa = a["players_active"].to_numpy()
    boards = a["board"].to_list()
    n = a.height
    hs1 = np.empty(n)
    hsN = np.empty(n)
    own = np.full(n, -1, np.int64)
    pfs = np.empty(n)
    bar = np.zeros(5, np.int64)
    for i in range(n):
        pfs[i] = pf_strength(tab, c1[i], c2[i])
        nb = NB[int(st[i])]
        if nb == 0:
            hs1[i] = pfs[i]
            hsN[i] = np.nan
            continue
        b = boards[i]
        if len(b) < nb:
            hs1[i] = np.nan
            hsN[i] = np.nan
            continue
        bar[:nb] = b[:nb]
        hs1[i] = equity_vs_random(c1[i], c2[i], bar, nb, 1, 250, i % 100000)
        hsN[i] = equity_vs_random(c1[i], c2[i], bar, nb, max(int(pa[i]) - 1, 1), 120, (i + 7) % 100000)
        own[i] = made_score(c1[i], c2[i], bar, nb) >> 20
    a = a.with_columns(pl.Series("hs1", hs1), pl.Series("hsN", hsN), pl.Series("own_cat", own), pl.Series("pf_str", pfs),
                       pl.when(pl.col("action") == 0).then(0).when(pl.col("action") == 1).then(1)
                       .when(pl.col("is_agg")).then(3).otherwise(2).alias("y"))
    return a.select("hand_idx", "action_no", "player_idx", "action", "amount", "is_agg", "y", *POLICY_FEATS)


def stage_policy(max_train="1500000"):
    """Train a normal-play action model on hands that contain NO labelled-positive pair (both seated),
    then score every action in the subset: p(fold/check/call/agg) and surprise = -log p(actual)."""
    import lightgbm as lgb
    import time
    t0 = time.time()
    hands = pl.read_parquet(OUT / "dt_sub_hands.parquet")
    seats = pl.read_parquet(OUT / "dt_sub_seats.parquet")
    acts = pl.read_parquet(OUT / "dt_sub_actions.parquet")
    parts = []
    hidx = hands["hand_idx"].to_numpy()
    for k in range(0, len(hidx), 60000):
        hs = hidx[k:k + 60000]
        parts.append(build_action_states(acts.filter(pl.col("hand_idx").is_in(hs)), seats.filter(pl.col("hand_idx").is_in(hs)),
                                         hands.filter(pl.col("hand_idx").is_in(hs))))
        print("states", k, f"{time.time() - t0:.0f}s", flush=True)
    S = pl.concat(parts)
    del parts
    # hands with any labelled-positive pair seated together (dev) -> excluded from training
    lab = pl.read_parquet(DER / "labels.parquet").filter(pl.col("label") == 1)
    ph = pl.read_parquet(OUT / "dt_pair_hands.parquet").join(lab.select("pair_id"), on="pair_id")
    bad = set(ph["hand_idx"].to_list())
    ev_all = set(pl.read_parquet(DER / "evidence.parquet")["hand_idx"].to_list())
    bad |= ev_all
    S = S.with_columns(pl.col("hand_idx").is_in(list(bad)).alias("pos_hand"))
    tr = S.filter(~pl.col("pos_hand") & pl.col("hs1").is_not_nan())
    rng = np.random.default_rng(SEED)
    m = min(int(max_train), tr.height)
    tr = tr[rng.choice(tr.height, m, replace=False)]
    X = tr.select(POLICY_FEATS).to_pandas().astype(float)
    y = tr["y"].to_numpy()
    nv = len(y) // 10
    dtr = lgb.Dataset(X.iloc[nv:], y[nv:])
    dva = lgb.Dataset(X.iloc[:nv], y[:nv])
    params = dict(objective="multiclass", num_class=4, learning_rate=0.1, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, num_threads=3, verbose=-1)
    bst = lgb.train(params, dtr, 600, valid_sets=[dva], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)])
    bst.save_model(str(OUT / "policy_lgb.txt"))
    P = bst.predict(S.select(POLICY_FEATS).to_pandas().astype(float))
    yy = S["y"].to_numpy()
    p_act = P[np.arange(len(yy)), yy]
    S = S.with_columns(pl.Series("p_fold", P[:, 0]), pl.Series("p_check", P[:, 1]), pl.Series("p_call", P[:, 2]),
                       pl.Series("p_agg", P[:, 3]), pl.Series("p_act", p_act),
                       pl.Series("surprise", -np.log(np.clip(p_act, 1e-6, 1))))
    S.select("hand_idx", "action_no", "player_idx", "action", "amount", "is_agg", "y", *POLICY_FEATS, "p_fold", "p_check", "p_call",
             "p_agg", "p_act", "surprise", "pos_hand").with_columns(pl.col(pl.Float64).cast(pl.Float32)).write_parquet(OUT / "dt_policy_actions.parquet")
    print("done", S.height, f"{time.time() - t0:.0f}s")


def stage_policy_cf(max_train="1200000"):
    """Cross-fitted normal-play policy: 2 folds by table (frozen table folds, fold%2); train on hands without a
    labelled-positive pair seated together, predict the other half. Overwrites p_*/surprise columns."""
    import lightgbm as lgb
    S = pl.read_parquet(OUT / "dt_policy_actions.parquet").drop("p_fold", "p_check", "p_call", "p_agg", "p_act", "surprise")
    hands = pl.read_parquet(OUT / "dt_sub_hands.parquet").select("hand_idx", "table_idx")
    folds = pl.read_csv(BASE / "data" / "folds_tables_5.csv").with_columns(pl.col("table_idx").cast(pl.Int16))
    S = S.join(hands.join(folds, on="table_idx"), on="hand_idx").with_columns((pl.col("fold") % 2).alias("half"))
    P = np.zeros((S.height, 4))
    params = dict(objective="multiclass", num_class=4, learning_rate=0.1, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, num_threads=3, verbose=-1)
    rng = np.random.default_rng(SEED)
    for h in (0, 1):
        tr = S.filter((pl.col("half") != h) & ~pl.col("pos_hand") & pl.col("hs1").is_not_nan())
        tr = tr[rng.choice(tr.height, min(int(max_train), tr.height), replace=False)]
        X = tr.select(POLICY_FEATS).to_pandas().astype(float)
        y = tr["y"].to_numpy()
        nv = len(y) // 10
        bst = lgb.train(params, lgb.Dataset(X.iloc[nv:], y[nv:]), 600, valid_sets=[lgb.Dataset(X.iloc[:nv], y[:nv])],
                        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)])
        idx = np.flatnonzero((S["half"] == h).to_numpy())
        P[idx] = bst.predict(S[idx].select(POLICY_FEATS).to_pandas().astype(float))
        print("half", h, "done", flush=True)
    yy = S["y"].to_numpy()
    p_act = P[np.arange(len(yy)), yy]
    S = S.with_columns(pl.Series("p_fold", P[:, 0]), pl.Series("p_check", P[:, 1]), pl.Series("p_call", P[:, 2]),
                       pl.Series("p_agg", P[:, 3]), pl.Series("p_act", p_act),
                       pl.Series("surprise", -np.log(np.clip(p_act, 1e-6, 1))))
    S.drop("table_idx", "fold", "half").with_columns(pl.col(pl.Float64).cast(pl.Float32)).write_parquet(OUT / "dt_policy_actions.parquet")
    print("cross-fitted policy written", S.height)


def stage_policy_signatures():
    """Signatures built on the (cross-fitted) normal-play policy: donor actions that a normal agent rarely takes."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(400)
    P = pl.read_parquet(OUT / "dt_policy_actions.parquet").select("hand_idx", "action_no", "p_act", "surprise")
    A = pl.read_parquet(OUT / "dt_feat_actions.parquet").join(P, on=["hand_idx", "action_no"], how="left")
    T = pl.read_parquet(OUT / "dt_handsig.parquet").select("pair_id", "hand_idx", "grp", "donor", "isev", "behavior_family")
    A = A.with_columns(pl.when(pl.col("action") == 0).then(pl.lit("fold")).when(pl.col("is_agg")).then(pl.lit("agg"))
                       .when(pl.col("action").is_in([2, 5])).then(pl.lit("call")).otherwise(pl.lit("check")).alias("atype"))
    fp = pl.col("facing_partner")
    defs = {
        "pfcall_weak": (pl.col("street") == 0) & (pl.col("atype") == "call") & fp & (pl.col("p_act") <= 0.15),
        "fold_surpr": (pl.col("atype") == "fold") & fp & (pl.col("p_act") <= 0.2),
        "postcall_surpr": (pl.col("street") > 0) & (pl.col("atype") == "call") & fp & (pl.col("p_act") <= 0.2),
        "agg_surpr": pl.col("is_agg") & (pl.col("partner_resp") >= 2) & (pl.col("p_act") <= 0.2),
        "any_surpr": pl.col("surprise") >= 1.6,
    }
    g = A.group_by("pair_id", "hand_idx", "actor").agg([v.any().alias(k) for k, v in defs.items()]
                                                       + [pl.col("surprise").max().alias("smax")])
    cols = list(defs) + ["smax"]
    X = T
    for s in "AB":
        X = X.join(g.filter(pl.col("actor") == s).drop("actor").rename({c: f"{c}_{s}" for c in cols}), on=["pair_id", "hand_idx"], how="left")
    X = X.with_columns([pl.col(f"{c}_{s}").fill_null(False if c != "smax" else 0.0) for c in cols for s in "AB"])
    fAB, fBA = pl.col("donor") == "A", pl.col("donor") == "B"
    X = X.with_columns([((fAB & pl.col(f"{c}_A")) | (fBA & pl.col(f"{c}_B"))).alias(f"SIG_{c}") for c in defs]
                       + [pl.when(fAB).then(pl.col("smax_A")).when(fBA).then(pl.col("smax_B")).otherwise(0.0).alias("donor_smax")])
    X = X.with_columns((pl.col("SIG_pfcall_weak") | pl.col("SIG_fold_surpr") | pl.col("SIG_postcall_surpr") | pl.col("SIG_agg_surpr")).alias("SIG_policy_union"))
    sig = [c for c in X.columns if c.startswith("SIG_")]
    cov = X.group_by("grp").agg([pl.len()] + [pl.col(c).mean().round(4) for c in sig]).sort("grp")
    long = cov.unpivot(index=["grp", "len"], variable_name="sig", value_name="rate").pivot(on="grp", index="sig", values="rate")
    print(long)
    long.write_csv(OUT / "dt_signature_coverage_policy.csv")
    pos = X.filter(pl.col("grp") == f"{FAM}:ev")
    for other in ["none", f"{FAM}:nonev"]:
        neg = X.filter(pl.col("grp") == other)
        y = np.r_[np.ones(pos.height), np.zeros(neg.height)]
        s = np.r_[pos["donor_smax"].to_numpy(), neg["donor_smax"].to_numpy()]
        print(f"hand AUC donor max surprise: DT evidence vs {other}: {roc_auc_score(y, s):.4f}")
    X.select("pair_id", "hand_idx", "donor_smax", *sig).write_parquet(OUT / "dt_handsig_policy.parquet")
    # pair level (orientation-free: max over the two players)
    PP = X.group_by("pair_id", "behavior_family").agg(
        pl.len().alias("shared"), pl.col("pfcall_weak_A").sum().alias("pfA"), pl.col("pfcall_weak_B").sum().alias("pfB"),
        (fAB & pl.col("SIG_policy_union")).sum().alias("puAB"), (fBA & pl.col("SIG_policy_union")).sum().alias("puBA"))
    PP = PP.with_columns((pl.max_horizontal("pfA", "pfB") / pl.col("shared")).alias("pfcall_weak_max_rate"),
                         pl.max_horizontal("pfA", "pfB").alias("pfcall_weak_max"),
                         (pl.max_horizontal("puAB", "puBA") / pl.col("shared")).alias("policy_union_max_rate"))
    PP.write_parquet(OUT / "dt_pairlevel_policy_stats.parquet")
    for c in ["pfcall_weak_max_rate", "pfcall_weak_max", "policy_union_max_rate"]:
        out = []
        for pos_f, neg_f in [([FAM], ["none"]), ([FAM], ["unknown"]), (["soft_play", "coordinated_isolation"], ["none"]), ([FAM], ["soft_play", "coordinated_isolation"])]:
            x = PP.filter(pl.col("behavior_family").is_in(pos_f + neg_f))
            yv = x["behavior_family"].is_in(pos_f).to_numpy()
            sv = x[c].to_numpy().astype(float)
            out.append(f"{'+'.join(pos_f)} vs {'+'.join(neg_f)} AUC {roc_auc_score(yv, sv):.4f} AP {average_precision_score(yv, sv):.4f}")
        print(c, " | ".join(out))


# ============================================================================ signatures
def _grp_expr():
    fam = pl.col("behavior_family")
    return (pl.when(fam.is_in(["directed_transfer", "soft_play", "coordinated_isolation"]))
            .then(fam + pl.when(pl.col("isev")).then(pl.lit(":ev")).otherwise(pl.lit(":nonev")))
            .otherwise(fam).alias("grp"))


def build_hand_table():
    """Pair-hand table with donor-oriented signature flags + gift EV. Written to dt_handsig.parquet."""
    pairs = pl.read_parquet(OUT / "dt_pairs.parquet")
    S = (pl.read_parquet(OUT / "dt_feat_pairhands.parquet").join(pairs.select("pair_id", "behavior_family"), on="pair_id")
         .with_columns(pl.col("evidence_rank").is_not_null().alias("isev")))
    S = S.with_columns(_grp_expr())
    A = pl.read_parquet(OUT / "dt_feat_actions.parquet")
    A = A.with_columns(
        pl.when(pl.col("action") == 0).then(pl.lit("fold")).when(pl.col("is_agg")).then(pl.lit("agg"))
        .when(pl.col("action").is_in([2, 5])).then(pl.lit("call")).otherwise(pl.lit("check")).alias("atype"),
        pl.when(pl.col("to_call_bb") > 0).then(pl.col("to_call_bb") / (pl.col("pot_bb") + pl.col("to_call_bb"))).otherwise(0.0).alias("odds"))
    fold, call, agg, fp = pl.col("atype") == "fold", pl.col("atype") == "call", pl.col("is_agg"), pl.col("facing_partner")
    eq = pl.col("eq_hu")
    A = A.with_columns(
        # EV (perfect-information, heads-up vs partner) the actor gives up by this action, in bb
        pl.when(fold & fp).then((pl.col("eq_mw") * (pl.col("pot_bb") + pl.col("to_call_bb")) - pl.col("to_call_bb")).clip(0))
        .when(call & fp).then((pl.col("amount_bb") - pl.col("eq_mw") * (pl.col("pot_bb") + pl.col("amount_bb"))).clip(0))
        .when(agg & (pl.col("partner_resp") >= 2)).then((pl.col("amount_bb") * (1 - 2 * eq)).clip(0))
        .otherwise(0.0).alias("gift"))
    flags = {
        "fold_vs_partner": fold & fp,
        "fold_best50": fold & fp & (eq >= 0.5),
        "fold_best50_post": fold & fp & (eq >= 0.5) & (pl.col("street") > 0),
        "fold_best70_post": fold & fp & (eq >= 0.7) & (pl.col("street") > 0),
        "call_vs_partner": call & fp,
        "call_weak25": call & fp & (eq <= 0.25),
        "call_below_odds": call & fp & (eq < pl.col("odds")),
        "call_weak15_turnriver": call & fp & (eq <= 0.15) & (pl.col("street") >= 2),
        "bluff_into_partner30": agg & (pl.col("partner_resp") >= 2) & (eq <= 0.3),
    }
    g = A.group_by("pair_id", "hand_idx", "actor").agg([v.any().alias(k) for k, v in flags.items()]
                                                        + [pl.col("gift").sum().alias("gift")])
    cols = list(flags) + ["gift"]
    T = S
    for side in ("A", "B"):
        T = T.join(g.filter(pl.col("actor") == side).drop("actor").rename({c: f"{c}_{side}" for c in cols}),
                   on=["pair_id", "hand_idx"], how="left")
    T = T.with_columns([pl.col(f"{c}_{s}").fill_null(False if c != "gift" else 0.0) for c in cols for s in "AB"])
    fAB = (pl.col("netA") < 0) & (pl.col("netB") > 0)
    fBA = (pl.col("netB") < 0) & (pl.col("netA") > 0)
    T = T.with_columns(
        pl.when(fAB).then(pl.lit("A")).when(fBA).then(pl.lit("B")).otherwise(None).alias("donor"),
        pl.max_horizontal(pl.min_horizontal(-pl.col("netA"), pl.col("netB")), pl.min_horizontal(-pl.col("netB"), pl.col("netA"))).clip(0).alias("transfer"))
    ex = [(fAB | fBA).alias("SIG_flow"), ((fAB | fBA) & pl.col("last_two")).alias("SIG_flow_last2")]
    for c in cols:
        if c == "gift":
            ex.append(pl.when(fAB).then(pl.col("gift_A")).when(fBA).then(pl.col("gift_B")).otherwise(0.0).alias("gift"))
        else:
            ex.append(((fAB & pl.col(f"{c}_A")) | (fBA & pl.col(f"{c}_B"))).alias(f"SIG_{c}"))
    T = T.with_columns(ex)
    T = T.with_columns(
        (pl.col("SIG_fold_best50") | pl.col("SIG_call_weak25") | pl.col("SIG_bluff_into_partner30")).alias("SIG_union"),
        (pl.col("SIG_fold_vs_partner") | pl.col("SIG_call_vs_partner") | pl.col("SIG_bluff_into_partner30")).alias("SIG_anyconf"),
        (pl.col("SIG_fold_best70_post") | pl.col("SIG_call_weak15_turnriver")).alias("SIG_strong"),
        (pl.col("gift") > 5).alias("SIG_gift5"))
    # pair donor from evidence (analysis only) and inferred donor (usable at eval time: larger summed gift)
    inf = T.group_by("pair_id").agg(
        pl.when(pl.col("donor") == "A").then(pl.col("gift")).otherwise(0.0).sum().alias("gAtot"),
        pl.when(pl.col("donor") == "B").then(pl.col("gift")).otherwise(0.0).sum().alias("gBtot"))
    inf = inf.select("pair_id", pl.when(pl.col("gAtot") >= pl.col("gBtot")).then(pl.lit("A")).otherwise(pl.lit("B")).alias("inf_donor"))
    evd = T.filter(pl.col("isev")).group_by("pair_id").agg(pl.col("donor").mode().first().alias("ev_donor"))
    T = T.join(inf, on="pair_id").join(evd, on="pair_id", how="left")
    T = T.with_columns((pl.col("donor") == pl.col("inf_donor")).fill_null(False).alias("samedir_inf"),
                       (pl.col("donor") == pl.col("ev_donor")).fill_null(False).alias("samedir_ev"))
    T.write_parquet(OUT / "dt_handsig.parquet")
    return T, A


def stage_signatures():
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(400)
    T, A = build_hand_table()
    sigs = [c for c in T.columns if c.startswith("SIG_")]
    cov = T.group_by("grp").agg([pl.len()] + [pl.col(c).mean().round(4) for c in sigs]).sort("grp")
    cov.write_csv(OUT / "dt_signature_coverage.csv")
    long = cov.unpivot(index=["grp", "len"], variable_name="sig", value_name="rate").pivot(on="grp", index="sig", values="rate")
    print(long)
    long.write_csv(OUT / "dt_signature_coverage_long.csv")
    # donor orientation
    ev = T.filter(pl.col("grp") == "directed_transfer:ev")
    print("DT evidence: donor=player_1(A)", (ev["donor"] == "A").sum(), "donor=player_2(B)", (ev["donor"] == "B").sum(),
          "no flow", ev["donor"].is_null().sum())
    pp = ev.group_by("pair_id").agg((pl.col("donor") == "A").sum().alias("a"), (pl.col("donor") == "B").sum().alias("b"))
    print("pairs with both donor directions among evidence:", pp.filter((pl.col("a") > 0) & (pl.col("b") > 0)).height, "/", pp.height,
          "| pairs donor A:", (pp["a"] > pp["b"]).sum(), "donor B:", (pp["b"] > pp["a"]).sum())
    fam_inf = T.select("pair_id", "behavior_family", "inf_donor", "ev_donor").unique().filter(pl.col("ev_donor").is_not_null())
    print(fam_inf.group_by("behavior_family").agg((pl.col("inf_donor") == pl.col("ev_donor")).mean().alias("inferred_donor_acc")))
    # non-evidence same-direction excess (unlisted planted hands)
    x = T.filter(pl.col("behavior_family").is_in(["directed_transfer", "soft_play", "coordinated_isolation"]) & ~pl.col("isev")
                 & pl.col("donor").is_not_null())
    print(x.with_columns(pl.col("gift").cut([0, 5, 20]).alias("gift_bin")).group_by("behavior_family", "gift_bin")
          .agg(pl.len(), pl.col("samedir_ev").mean().round(3).alias("same_dir_as_evidence")).sort("behavior_family", "gift_bin"))
    # donor action profile
    Ad = A.join(T.select("pair_id", "hand_idx", "grp", "donor"), on=["pair_id", "hand_idx"]).with_columns(
        (pl.col("actor") == pl.col("donor")).alias("is_donor"))
    prof = (Ad.filter(pl.col("grp").is_in(["directed_transfer:ev", "none"]))
            .group_by("grp", "is_donor", "atype", "facing_partner").agg(pl.len(), pl.col("eq_hu").mean().round(3), pl.col("odds").mean().round(3),
                                                                       pl.col("street").mean().round(2)).sort("grp", "is_donor", "atype", "facing_partner"))
    print(prof)
    prof.write_csv(OUT / "dt_action_profile.csv")


def _evmap(df, score):
    out = []
    for _, g in df.group_by("pair_id"):
        g = g.sort([score, "hand_idx"], descending=[True, False])
        rel = set(g.filter(pl.col("isev"))["hand_idx"].to_list())
        hits, ps = 0, 0.0
        for r, h in enumerate(g["hand_idx"].to_list()[:5], 1):
            if h in rel:
                hits += 1
                ps += hits / r
        out.append(ps / min(len(rel), 5))
    return float(np.mean(out))


def stage_temporal():
    pl.Config.set_tbl_rows(60)
    T = pl.read_parquet(OUT / "dt_handsig.parquet")
    dt = T.filter(pl.col("behavior_family") == FAM)
    ev = dt.filter(pl.col("isev"))
    bins = [0, 600, 1200, 1800, 2400, 3000]
    he, _ = np.histogram(ev["hand_seq"].to_numpy(), bins=bins)
    hs, _ = np.histogram(dt["hand_seq"].to_numpy(), bins=bins)
    print("evidence per 600-hand bin", he, "shared hands", hs, "evidence/shared", np.round(he / hs, 4))
    rng = np.random.default_rng(0)
    g_ev, g_rand = [], []
    for _, g in dt.group_by("pair_id"):
        seqs = np.sort(g["hand_seq"].to_numpy())
        e = np.sort(g.filter(pl.col("isev"))["hand_seq"].to_numpy())
        g_ev += list(np.diff(e))
        for _ in range(20):
            g_rand += list(np.diff(np.sort(rng.choice(seqs, len(e), replace=False))))
    g_ev, g_rand = np.array(g_ev), np.array(g_rand)
    print(f"gap consecutive evidence: median {np.median(g_ev):.0f} <=20: {(g_ev <= 20).mean():.3f} | random 5 shared: median {np.median(g_rand):.0f} <=20: {(g_rand <= 20).mean():.3f}")
    for grp in [FAM, "none", "unknown"]:
        out = []
        for _, g in T.filter(pl.col("behavior_family") == grp).group_by("pair_id"):
            s = np.sort(g.filter(pl.col("gift") > 5)["hand_seq"].to_numpy())
            if len(s) >= 2:
                d = np.diff(s)
                out += list(np.minimum(np.r_[d, 99999], np.r_[99999, d]))
        out = np.array(out)
        print(f"{grp}: gift>5 hands nearest-neighbour gap median {np.median(out):.0f}, frac<=30 {(out <= 30).mean():.3f}, n={len(out)}")
    print(T.filter(pl.col("grp").is_in([f"{FAM}:ev", f"{FAM}:nonev", "none", "unknown"]))
          .group_by("grp", (pl.col("hand_seq") >= 1500).alias("late")).agg(pl.len(), pl.col("SIG_union").mean().round(4), pl.col("SIG_gift5").mean().round(4))
          .sort("grp", "late"))
    # listing rule: within-pair chronological order among planted-like candidates (same dir as evidence + strong anomaly)
    cc = (dt.filter(pl.col("isev") | (pl.col("samedir_ev") & pl.col("SIG_strong"))).sort("pair_id", "hand_seq")
          .with_columns(pl.col("hand_seq").rank("ordinal").over("pair_id").alias("order"), pl.len().over("pair_id").alias("N")))
    print("P(listed) by chronological order among candidates:")
    print(cc.with_columns(pl.col("order").clip(1, 12)).group_by("order").agg(pl.len(), pl.col("isev").mean().round(3)).sort("order"))
    cc = cc.with_columns(((pl.col("order") - 1) / pl.col("N")).alias("rel"))
    for lo, hi in [(5, 7), (8, 10), (11, 15), (16, 99)]:
        x = cc.filter((pl.col("N") >= lo) & (pl.col("N") <= hi))
        r = x.group_by(pl.col("rel").cut([0.25, 0.5, 0.75])).agg(pl.len(), pl.col("isev").mean().round(3)).sort("rel")
        print(f"N in [{lo},{hi}] pairs={x['pair_id'].n_unique()} P(listed) by relative position quartile:", r["isev"].to_list())
    print("candidates per pair by #evidence:", dt.group_by("pair_id").agg(pl.col("isev").sum().alias("nev"),
          (pl.col("samedir_ev") & pl.col("SIG_strong")).sum().alias("ncand")).group_by("nev").agg(pl.len(), pl.col("ncand").mean()).sort("nev").to_dicts())


def stage_pairlevel():
    from sklearn.metrics import roc_auc_score, average_precision_score
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(300)
    T = pl.read_parquet(OUT / "dt_handsig.parquet")
    fAB = pl.col("donor") == "A"
    fBA = pl.col("donor") == "B"
    base = {"union": pl.col("SIG_union"), "strong": pl.col("SIG_strong"), "gift5": pl.col("SIG_gift5")}
    ex = []
    for k, v in base.items():
        ex += [(v & fAB).cast(pl.Int64).alias(f"{k}AB"), (v & fBA).cast(pl.Int64).alias(f"{k}BA")]
    ex += [pl.when(fAB).then(pl.col("gift")).otherwise(0.0).alias("giftAB"), pl.when(fBA).then(pl.col("gift")).otherwise(0.0).alias("giftBA"),
           pl.when(fAB).then(pl.col("transfer")).otherwise(0.0).alias("trAB"), pl.when(fBA).then(pl.col("transfer")).otherwise(0.0).alias("trBA")]
    T = T.with_columns(ex)
    agg_cols = [f"{k}{d}" for k in list(base) + ["gift", "tr"] for d in ("AB", "BA")]
    P = T.group_by("pair_id", "behavior_family").agg([pl.len().cast(pl.Int64).alias("shared")] + [pl.col(c).sum() for c in agg_cols])
    ex = []
    for k in list(base) + ["gift"]:
        ex += [pl.max_horizontal(f"{k}AB", f"{k}BA").alias(f"{k}_max"), (pl.col(f"{k}AB") - pl.col(f"{k}BA")).abs().alias(f"{k}_asym")]
    ex.append((pl.col("trAB") - pl.col("trBA")).abs().alias("netflow"))
    P = P.with_columns(ex)
    feats = [c for c in P.columns if c.endswith(("_max", "_asym")) or c == "netflow"]
    P = P.with_columns([(pl.col(c) / pl.col("shared")).alias(c + "_rate") for c in feats])
    feats = feats + [c + "_rate" for c in feats] + ["shared"]
    P.write_parquet(OUT / "dt_pairlevel_stats.parquet")

    def sc(pos, neg, c):
        x = P.filter(pl.col("behavior_family").is_in(pos + neg))
        y = x["behavior_family"].is_in(pos).to_numpy().astype(int)
        s = x[c].to_numpy().astype(float)
        return roc_auc_score(y, s), average_precision_score(y, s)
    rows = []
    for c in feats:
        a1, p1 = sc([FAM], ["none"], c)
        a2, p2 = sc([FAM], ["unknown"], c)
        a3, _ = sc(["soft_play", "coordinated_isolation"], ["none"], c)
        a4, _ = sc([FAM], ["soft_play", "coordinated_isolation"], c)
        rows.append((c, a1, p1, a2, p2, a3, a4))
    R = pl.DataFrame(rows, schema=["feat", "AUC_DT_vs_neg", "AP_DT_vs_neg", "AUC_DT_vs_unk", "AP_DT_vs_unk", "AUC_SPCI_vs_neg", "AUC_DT_vs_SPCI"],
                     orient="row").sort("AUC_DT_vs_neg", descending=True)
    print(R.with_columns(pl.exclude("feat").round(4)))
    R.write_csv(OUT / "dt_pairlevel_auc.csv")
    best = R["feat"][0]
    x = P.filter(pl.col("behavior_family").is_in([FAM, "none", "unknown"])).with_columns(pl.col("shared").qcut(4, labels=["q1", "q2", "q3", "q4"]).alias("shq"))
    for q in ["q1", "q2", "q3", "q4"]:
        g = x.filter(pl.col("shq") == q)
        ys = g["behavior_family"].to_numpy()
        m1 = np.isin(ys, [FAM, "none"])
        m2 = np.isin(ys, [FAM, "unknown"])
        s = g[best].to_numpy().astype(float)
        print(f"shared quartile {q} (shared {g['shared'].min()}-{g['shared'].max()}): n_DT={int((ys == FAM).sum())} AUC({best}) vs neg "
              f"{roc_auc_score(ys[m1] == FAM, s[m1]):.3f} vs unk {roc_auc_score(ys[m2] == FAM, s[m2]):.3f}; "
              f"count version {roc_auc_score(ys[m1] == FAM, g[best.replace('_rate', '')].to_numpy()[m1]):.3f}")
    print("spearman(best, shared) among negatives:",
          P.filter(pl.col("behavior_family") == "none").select(pl.corr(best, "shared", method="spearman")).item(),
          "| count version:", P.filter(pl.col("behavior_family") == "none").select(pl.corr(best.replace("_rate", ""), "shared", method="spearman")).item())


def build_oriented_table():
    """One row per (pair, hand, candidate donor X in {A,B}); X-actions vs partner Y aggregated."""
    P = pl.read_parquet(OUT / "dt_policy_actions.parquet").select("hand_idx", "action_no", "surprise", "p_act")
    A = pl.read_parquet(OUT / "dt_feat_actions.parquet").join(P, on=["hand_idx", "action_no"], how="left")
    T = pl.read_parquet(OUT / "dt_handsig.parquet")
    A = A.with_columns(pl.when(pl.col("action") == 0).then(pl.lit("fold")).when(pl.col("is_agg")).then(pl.lit("agg"))
                       .when(pl.col("action").is_in([2, 5])).then(pl.lit("call")).otherwise(pl.lit("check")).alias("atype"))
    fold, call, agg, fp = pl.col("atype") == "fold", pl.col("atype") == "call", pl.col("is_agg"), pl.col("facing_partner")
    A = A.with_columns(
        pl.when(fold & fp).then((pl.col("eq_mw") * (pl.col("pot_bb") + pl.col("to_call_bb")) - pl.col("to_call_bb")).clip(0))
        .when(call & fp).then((pl.col("amount_bb") - pl.col("eq_mw") * (pl.col("pot_bb") + pl.col("amount_bb"))).clip(0))
        .when(agg & (pl.col("partner_resp") >= 2)).then((pl.col("amount_bb") * (1 - 2 * pl.col("eq_hu"))).clip(0)).otherwise(0.0).alias("gift"))
    g = A.group_by("pair_id", "hand_idx", "actor").agg(
        (fold & fp).sum().alias("n_fold_fp"), pl.when(fold & fp).then(pl.col("eq_hu")).otherwise(None).max().alias("eq_fold_max"),
        (fold & fp & (pl.col("street") > 0)).sum().alias("n_fold_fp_post"),
        (call & fp).sum().alias("n_call_fp"), pl.when(call & fp).then(pl.col("eq_hu")).otherwise(None).min().alias("eq_call_min"),
        pl.when(call & fp).then(pl.col("amount_bb")).otherwise(0).sum().alias("call_amt"),
        (agg & (pl.col("partner_resp") >= 2)).sum().alias("n_agg_called"),
        pl.when(agg & (pl.col("partner_resp") >= 2)).then(pl.col("eq_hu")).otherwise(None).min().alias("eq_agg_min"),
        pl.col("gift").sum().alias("gift"), pl.col("gift").max().alias("gift_max"),
        pl.col("surprise").max().alias("s_max"), pl.col("surprise").sum().alias("s_sum"),
        pl.when((pl.col("street") == 0) & call & fp).then(pl.col("surprise")).otherwise(None).max().alias("s_pfcall"),
        pl.when(fold & fp).then(pl.col("surprise")).otherwise(None).max().alias("s_fold"),
        pl.when((pl.col("street") > 0) & call & fp).then(pl.col("surprise")).otherwise(None).max().alias("s_postcall"),
        pl.col("street").max().alias("max_street"), pl.len().alias("n_act"), (agg & ~fp).sum().alias("n_agg"),
        pl.col("eq_hu").mean().alias("eq_mean"))
    fc = [c for c in g.columns if c not in ("pair_id", "hand_idx", "actor")]
    rows = []
    for X, Y in (("A", "B"), ("B", "A")):
        t = T.select("pair_id", "hand_idx", "hand_seq", "behavior_family", "isev", "donor", "inf_donor", "last_two", "n_final", "board_len",
                     pl.col("net" + X).alias("netX"), pl.col("net" + Y).alias("netY"), pl.col("cont" + X).alias("contX"),
                     pl.col("cont" + Y).alias("contY"), pl.col("vol" + X).alias("volX"), pl.col("vol" + Y).alias("volY"),
                     pl.col("stack" + X).alias("stackX"), pl.col("pf" + X).alias("pfX"), pl.col("pf" + Y).alias("pfY")).with_columns(pl.lit(X).alias("X"))
        t = (t.join(g.filter(pl.col("actor") == X).drop("actor").rename({c: "x_" + c for c in fc}), on=["pair_id", "hand_idx"], how="left")
             .join(g.filter(pl.col("actor") == Y).drop("actor").rename({c: "y_" + c for c in fc}), on=["pair_id", "hand_idx"], how="left"))
        rows.append(t)
    R = pl.concat(rows).with_columns(
        ((pl.col("netX") < 0) & (pl.col("netY") > 0)).alias("flow"), pl.min_horizontal(-pl.col("netX"), pl.col("netY")).clip(0).alias("transfer"),
        (pl.col("X") == pl.col("inf_donor")).alias("is_inf_donor"),
        (pl.col("isev") & (pl.col("donor") == pl.col("X")) & (pl.col("behavior_family") == FAM)).alias("y"))
    hands = pl.read_parquet(OUT / "dt_sub_hands.parquet").select("hand_idx", "table_idx")
    folds = pl.read_csv(BASE / "data" / "folds_tables_5.csv").with_columns(pl.col("table_idx").cast(pl.Int16))
    pf = pl.read_parquet(OUT / "dt_pair_hands.parquet").select("pair_id", "hand_idx").unique("pair_id").join(hands, on="hand_idx").join(folds, on="table_idx").select("pair_id", "fold")
    R = R.join(pf, on="pair_id")
    feats = [c for c in R.columns if c.startswith(("x_", "y_"))] + ["flow", "transfer", "last_two", "n_final", "board_len", "netX", "netY",
                                                                    "contX", "contY", "volX", "volY", "stackX", "pfX", "pfY"]
    return R, feats


def stage_ranker():
    """Evidence-hand ranker for DT pairs: OOF (table folds) hand model + ordering recipes, scored by dev MAP@5."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    R, F = build_oriented_table()
    R = R.filter(pl.col("behavior_family").is_in([FAM, "none", "soft_play", "coordinated_isolation"]))
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=3, verbose=-1)
    schemes = {"S1": pl.col("behavior_family").is_in(["none"]) | pl.col("y"),
               "S2": pl.col("behavior_family").is_in(["none", FAM])}
    for name, cond in schemes.items():
        oof = np.zeros(R.height)
        for k in range(5):
            tr = R.filter((pl.col("fold") != k) & cond)
            bst = lgb.train(params, lgb.Dataset(tr.select(F).to_pandas().astype(float), tr["y"].to_numpy().astype(int)), 300)
            te = np.flatnonzero((R["fold"] == k).to_numpy())
            oof[te] = bst.predict(R[te].select(F).to_pandas().astype(float))
        R = R.with_columns(pl.Series("p_" + name, oof))
        if name == "S1":
            imp = sorted(zip(bst.feature_importance("gain"), F), reverse=True)[:15]
            print("S1 top features (gain):", [(f, int(v)) for v, f in imp])
    R.select("pair_id", "hand_idx", "hand_seq", "X", "behavior_family", "isev", "y", "is_inf_donor", "p_S1", "p_S2").write_parquet(OUT / "dt_evidence_oof.parquet")
    dt = R.filter(pl.col("behavior_family") == FAM)
    neg = R.filter(pl.col("behavior_family") == "none")
    for name in ("S1", "S2"):
        pos = dt.filter(pl.col("y"))
        y = np.r_[np.ones(pos.height), np.zeros(neg.height)]
        s = np.r_[pos["p_" + name].to_numpy(), neg["p_" + name].to_numpy()]
        print(f"{name}: hand AUC DT-evidence vs negative-pair hands {roc_auc_score(y, s):.4f}; "
              f"vs same-pair non-evidence {roc_auc_score(dt['y'].to_numpy(), dt['p_' + name].to_numpy()):.4f}")
    base = T0 = pl.read_parquet(OUT / "dt_handsig.parquet").filter(pl.col("behavior_family") == FAM)
    out = [("gift (flow donor EV gift, desc)", _evmap(base, "gift")),
           ("transfer desc", _evmap(base, "transfer"))]
    x = base.with_columns((pl.col("SIG_anyconf") & pl.col("samedir_inf") & (pl.col("transfer") >= 2)).cast(float).alias("c"))
    out.append(("candidates(anyconf & inferred donor & transfer>=2) earliest-first", _evmap(x.with_columns((pl.col("c") * 1e4 - pl.col("hand_seq")).alias("s")), "s")))
    for name in ("S1", "S2"):
        h = dt.group_by("pair_id", "hand_idx", "hand_seq", "isev").agg(pl.col("p_" + name).max().alias("p")).sort("pair_id", "hand_seq")
        out.append((f"{name} raw", _evmap(h, "p")))
        for tau in (0.05, 0.35):
            out.append((f"{name} p>{tau} earliest-first", _evmap(h.with_columns(pl.when(pl.col("p") > tau).then(1e4 - pl.col("hand_seq")).otherwise(pl.col("p")).alias("s")), "s")))
        for lam in (2, 4, 8):
            hh = h.with_columns((pl.col("p").cum_sum().over("pair_id") - pl.col("p")).alias("K")).with_columns((pl.col("p") * (-pl.col("K") / lam).exp()).alias("s"))
            out.append((f"{name} soft time-decay p*exp(-K/{lam})", _evmap(hh, "s")))
    res = pl.DataFrame(out, schema=["recipe", "devMAP5_DT"], orient="row")
    print(res)
    res.write_csv(OUT / "dt_evidence_recipes.csv")


def stage_evalcheck():
    """Does the persistent DT signature (partner flat-calls the other's preflop raise) continue in the EVAL phase?
    Uses labelled dev pairs' eval-phase shared hands (labels only used to group; nothing is fitted)."""
    pl.Config.set_tbl_rows(40)
    pairs = pl.read_parquet(OUT / "dt_pairs.parquet").filter(pl.col("behavior_family") != "unknown")
    evd = pl.read_parquet(OUT / "dt_handsig.parquet").select("pair_id", "ev_donor").unique()
    players = list(set(pairs["A"].to_list()) | set(pairs["B"].to_list()))
    s = (scan("seats").select("hand_idx", "player_idx").filter(pl.col("player_idx").is_in(players))
         .join(scan("hands").filter(pl.col("phase") == 1).select("hand_idx"), on="hand_idx").collect())
    ph = (pairs.select("pair_id", "A", "B").join(s.rename({"player_idx": "A"}), on="A")
          .join(s.rename({"player_idx": "B"}), on=["B", "hand_idx"]))
    hidx = ph["hand_idx"].unique().to_list()
    ac = scan("actions").filter(pl.col("hand_idx").is_in(hidx) & (pl.col("street") == 0)).collect().sort("hand_idx", "action_no")
    ac = ac.with_columns((pl.col("action").is_in([3, 4]) | ((pl.col("action") == 5) & (pl.col("amount") > pl.col("to_call")))).alias("agg"))
    ac = ac.with_columns(pl.when(pl.col("agg")).then(pl.col("player_idx")).otherwise(None).forward_fill().shift(1).over("hand_idx").alias("last_agg"))
    rows = []
    for X, Y in (("A", "B"), ("B", "A")):
        x = ph.join(ac, left_on=["hand_idx", X], right_on=["hand_idx", "player_idx"]).filter(pl.col("last_agg") == pl.col(Y))
        rows.append(x.select("pair_id", pl.lit(X).alias("actor"), "action", "agg"))
    R = pl.concat(rows).join(pairs.select("pair_id", "behavior_family"), on="pair_id").join(evd, on="pair_id", how="left")
    R = R.with_columns(pl.when(pl.col("ev_donor").is_null()).then(pl.lit("n/a")).when(pl.col("actor") == pl.col("ev_donor"))
                       .then(pl.lit("donor")).otherwise(pl.lit("receiver")).alias("role"),
                       (pl.col("action").is_in([2, 5]) & ~pl.col("agg")).alias("call"))
    print("EVAL phase: response when facing partner's preflop raise")
    print(R.group_by("behavior_family", "role").agg(pl.len(), pl.col("call").mean().round(3).alias("call_rate"),
                                                    pl.col("agg").mean().round(3).alias("reraise_rate"),
                                                    (pl.col("action") == 0).mean().round(3).alias("fold_rate")).sort("behavior_family", "role"))
    pp = R.group_by("pair_id", "behavior_family", "actor").agg(pl.len(), pl.col("call").mean().alias("cr")).filter(pl.col("len") >= 5)
    pp = pp.group_by("pair_id", "behavior_family").agg(pl.col("cr").max().alias("max_call_rate"))
    from sklearn.metrics import roc_auc_score
    x = pp.filter(pl.col("behavior_family").is_in([FAM, "none"]))
    print("EVAL-phase pair AUC (max over players of call-rate vs partner preflop raise, >=5 situations) DT vs neg:",
          round(roc_auc_score((x["behavior_family"] == FAM).to_numpy(), x["max_call_rate"].to_numpy()), 4), "n", x.group_by("behavior_family").len().to_dicts())


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "extract"
    globals()[f"stage_{st}"](*sys.argv[2:])
