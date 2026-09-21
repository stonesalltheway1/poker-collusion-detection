"""Case reviews for the prize submission -- family-diverse, with the omniscient read and a
case-specific benign alternative.  Replacement for scripts/case_review.py.

What it adds over the old script (which dumped five all-directed_transfer cases of raw actions):
  * SELECTION: the highest-risk pair of EACH predicted family first, then the next highest-risk
    pairs overall.  The shipped submissions' top five are all directed_transfer, so the old output
    showed a reviewer one behaviour out of three.
  * The omniscient equity of both partners at every partner decision (all hole cards are public in
    this dataset), which is what the detector actually prices.
  * A per-hand READ: which listing-rule signature fired, stated as an observation, not a verdict.
  * A per-hand BENIGN alternative built from that hand's own numbers (pot odds, equity, position),
    instead of one canned sentence per family repeated five times.
  * A pair-level summary: shared hands in the scored period, net chip flow, showdown record.

Usage: python case_review2.py <submission.csv> [n_cases] [out.md]
Read-only with respect to the repository; writes only <out.md>.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

BASE = Path(__file__).resolve().parents[0]
# allow running from the scratchpad against the repo
REPO = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import common as C  # noqa: E402

SUB = Path(sys.argv[1])
NCASE = int(sys.argv[2]) if len(sys.argv) > 2 else 5
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "research" / "CASE_REVIEWS.md"

ACT = {0: "fold", 1: "check", 2: "call", 3: "bet", 4: "raise", 5: "all-in"}
ST = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
FAMNAME = {"directed_transfer": "directed transfer", "soft_play": "soft play",
           "coordinated_isolation": "coordinated isolation", "other_coordination": "other coordination"}


def cards(*cs):
    return " ".join(C.card_str(int(c)) for c in cs if c is not None and int(c) >= 0)


# --------------------------------------------------------------------------- selection
def pick_cases(sub, n):
    """Highest-risk pair of each predicted family, then the next highest-risk pairs overall."""
    s = sub.sort("risk_score", descending=True)
    chosen, seen = [], set()
    for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
        f = s.filter(pl.col("predicted_behavior") == fam)
        if f.height:
            r = f.row(0, named=True)
            chosen.append(r)
            seen.add(r["pair_id"])
    for r in s.iter_rows(named=True):
        if len(chosen) >= n:
            break
        if r["pair_id"] not in seen:
            chosen.append(r)
            seen.add(r["pair_id"])
    return chosen[:n]


# --------------------------------------------------------------------------- per-hand read
def hand_read(fam, acts, seats, role, eq_pre, n_fold_before_pair_raise, bb=1.0, first_raise_pos=None):
    """Return (read, benign) for one evidence hand. Observational language only."""
    seat_of = {v: k for k, v in role.items()}          # 'A'/'B' -> seat_no
    sa, sb = seat_of.get("A"), seat_of.get("B")
    net = {r["seat_no"]: r["net_chips"] for r in seats}
    pair_acts = [a for a in acts if a["seat_no"] in (sa, sb)]

    # last aggressive action by each partner, and any fold facing the other
    def eq(seat, i):
        return eq_pre[i][seat] if eq_pre is not None and i < len(eq_pre) else None

    fold_ahead = None      # (folder, aggressor, eq_folder, eq_aggr, street)
    passive_call = None    # (caller, aggressor, to_call, pot, eq_caller)
    last_aggr = {}
    for i, a in enumerate(acts):
        if a["seat_no"] in (sa, sb) and a["action"] in (3, 4, 5):
            last_aggr[a["seat_no"]] = i
        if a["seat_no"] in (sa, sb) and a["action"] == 0 and a["to_call"] > 0:
            other = sb if a["seat_no"] == sa else sa
            if other in last_aggr and last_aggr[other] < i:
                ef, ea = eq(a["seat_no"], i), eq(other, i)
                if ef is not None and ea is not None and ef == ef and ea == ea and ef > ea:
                    fold_ahead = (a["seat_no"], other, ef, ea, a["street"])
        if a["seat_no"] in (sa, sb) and a["action"] == 2:
            other = sb if a["seat_no"] == sa else sa
            if other in last_aggr and last_aggr[other] < i:
                passive_call = (a["seat_no"], other, a["to_call"], a["pot_before"], eq(a["seat_no"], i))

    raised_back = any(a["seat_no"] in (sa, sb) and a["action"] in (4, 5)
                      and any(b["seat_no"] == (sb if a["seat_no"] == sa else sa) and b["action"] in (3, 4, 5)
                              for b in acts[:acts.index(a)]) for a in acts)

    L = lambda s: role.get(s, "?")  # noqa: E731

    if fam == "coordinated_isolation":
        pos = "" if first_raise_pos is None else f" (it was action #{first_raise_pos + 1} of the hand)"
        kind = "type-1" if n_fold_before_pair_raise == 0 else "type-2"
        read = (f"The pair's first preflop raise came with **{n_fold_before_pair_raise} player(s) already folded** "
                f"ahead of it{pos}, so under the host's own listing convention this is a {kind} "
                f"coordinated-isolation event. The point of the pattern is that the raise is made into a field "
                f"that still contains the partner, who then gets out of the way cheaply.")
        benign = ("A wide early-position opening range is a real (if unprofitable) style, and the partner folding "
                  "behind is ordinary tight play. Nothing here is visible to the raiser at the table: the two of "
                  "them cannot see each other's cards, so the pattern only becomes notable across many hands.")
        return read, benign

    if fold_ahead:
        f, o, ef, ea, st = fold_ahead
        read = (f"**{L(f)} folds on the {ST[st]} facing {L(o)}'s aggression while holding the better hand** "
                f"({L(f)} {ef:.0%} equity vs {L(o)} {ea:.0%} at that moment, all cards visible). "
                f"That is the type-1 signature for {FAMNAME[fam]}: the value goes to the partner without a showdown.")
        benign = (f"{L(f)} cannot see {L(o)}'s cards. Folding {ef:.0%} equity to a large bet is a normal, if tight, "
                  f"laydown -- the equity edge is only visible to us. A single such fold is unremarkable; what the "
                  f"model actually scores is how often it recurs between these two and not with their other 28 "
                  f"table-mates.")
        return read, benign

    if passive_call and fam == "soft_play":
        cl, ag, tc, pot, ec = passive_call
        read = (f"**{L(cl)} only calls {L(ag)}'s bet** ({tc / bb:.1f}bb into {pot / bb:.1f}bb) and never raises"
                + (", and no raise between the partners appears anywhere in the hand" if not raised_back else "")
                + (f"; {L(cl)}'s equity at that point was {ec:.0%}." if ec is not None else "."))
        benign = ("Loose-passive is the single most common recreational style: calling stations under-raise "
                  "everyone, not just one player. This only becomes evidence if the same player raises other "
                  "opponents in the same spots.")
        return read, benign

    # chip-flow fallback (type-2 directed transfer: call-downs and bets that lose)
    if sa is not None and sb is not None:
        flow = net.get(sa, 0), net.get(sb, 0)
        donor, recv = ("A", "B") if flow[0] < flow[1] else ("B", "A")
        read = (f"No fold-ahead in this hand: the chips move by **{donor} paying {recv} off** "
                f"({donor} {min(flow) / 1.0:+.0f} chips, {recv} {max(flow):+.0f}). Under the listing rule this is a "
                f"type-2 event -- a call-down or bet that transfers value without the fold signature.")
        benign = ("Paying off a better hand is the most ordinary loss in poker. One hand of this shape is pure "
                  "variance; only the direction being consistently one-way across the pair's shared hands, and "
                  "absent against everyone else at the table, makes it notable.")
        return read, benign
    return "", ""


# --------------------------------------------------------------------------- main
def main():
    sub = pl.read_csv(SUB)
    ep = C.load("eval_pairs").select("pair_id", "p1", "p2", "player_1", "player_2", "shared_hands")
    sub = sub.join(ep, on="pair_id")
    n_pairs = sub.height
    rank = {p: i + 1 for i, p in enumerate(sub.sort("risk_score", descending=True)["pair_id"].to_list())}
    cases = pick_cases(sub, NCASE)

    hands = C.load("hands").select("hand_id", "hand_idx", "table_idx", "hand_seq", "bb", "board", "final_pot")
    seats_lf = C.scan("seats")
    acts_lf = C.scan("actions")

    ev_idx = []
    for cse in cases:
        for c in C.EV_COLS:
            if cse[c] != C.NO_EV:
                ev_idx.append(cse[c])
    hsel = hands.filter(pl.col("hand_id").is_in(ev_idx))
    hidx = hsel["hand_idx"].to_list()
    EQ = (pl.scan_parquet(C.DER / "action_equity_eval.parquet")
          .filter(pl.col("hand_idx").is_in(hidx))
          .select("hand_idx", "action_no", *[f"eq_pre_s{s}" for s in range(6)]).collect())
    eqmap = defaultdict(dict)
    for r in EQ.iter_rows(named=True):
        eqmap[r["hand_idx"]][r["action_no"]] = [r[f"eq_pre_s{s}"] for s in range(6)]

    L = [f"# Case reviews — `{SUB.name}`", "",
         "Five pairs from the submitted ranking, chosen to cover **all three disclosed behaviour families** "
         "(the top of the ranking is dominated by directed transfer, so a plain top-5 would show only one "
         "pattern). For every submitted evidence hand this lists both partners' hole cards, every other seat's "
         "cards, the board, every action with pot and price, the omniscient equity at each partner decision, "
         "the signature that fired, and a benign alternative for that specific hand.", "",
         "The data is synthetic and the labels are a benchmark, not an adjudication. A high score is a prompt "
         "for human review, never a finding of guilt.", ""]

    for cse in cases:
        pid = cse["pair_id"]
        fam = cse["predicted_behavior"]
        ev = [cse[c] for c in C.EV_COLS if cse[c] != C.NO_EV]
        # pair-level summary over the scored period
        both = (seats_lf.select("hand_idx", "player_idx", "net_chips", "went_to_showdown")
                .filter(pl.col("player_idx").is_in([cse["p1"], cse["p2"]])).collect()
                .join(hands.select("hand_idx", "hand_seq", "bb"), on="hand_idx", how="inner"))
        shared = (both.group_by("hand_idx").agg(pl.len().alias("k")).filter(pl.col("k") == 2)["hand_idx"])
        bs = both.filter(pl.col("hand_idx").is_in(shared))
        netA = float((bs.filter(pl.col("player_idx") == cse["p1"])["net_chips"]
                      / bs.filter(pl.col("player_idx") == cse["p1"])["bb"]).sum())
        netB = float((bs.filter(pl.col("player_idx") == cse["p2"])["net_chips"]
                      / bs.filter(pl.col("player_idx") == cse["p2"])["bb"]).sum())

        L += [f"## Case {cases.index(cse) + 1} — pair `{pid}` · predicted **{FAMNAME[fam]}**", "",
              f"- Rank **{rank[pid]} of {n_pairs:,}** by submitted risk score ({cse['risk_score']:.5f}); the score "
              f"is a rank within the evaluation set, so the top of the list is where review effort should go.",
              f"- Players: **A = `{cse['player_1']}`**, **B = `{cse['player_2']}`**, sharing "
              f"{len(shared):,} hands in the scored period.",
              f"- Net over those shared hands: A {netA:+.0f} bb, B {netB:+.0f} bb.",
              f"- Submitted evidence: {', '.join('`' + h + '`' for h in ev)}", ""]

        for k, h in enumerate(ev, 1):
            hrow = hsel.filter(pl.col("hand_id") == h)
            if not hrow.height:
                continue
            hr = hrow.row(0, named=True)
            bb = hr["bb"]
            s = seats_lf.filter(pl.col("hand_idx") == hr["hand_idx"]).collect().sort("seat_no")
            a = acts_lf.filter(pl.col("hand_idx") == hr["hand_idx"]).collect().sort("action_no")
            role = {}
            for r in s.iter_rows(named=True):
                if r["player_idx"] == cse["p1"]:
                    role[r["seat_no"]] = "A"
                elif r["player_idx"] == cse["p2"]:
                    role[r["seat_no"]] = "B"
            seat_by_player = {r["player_idx"]: r["seat_no"] for r in s.iter_rows(named=True)}
            arows = [{**r, "seat_no": seat_by_player[r["player_idx"]]} for r in a.iter_rows(named=True)]
            srows = list(s.iter_rows(named=True))

            # players folded before the pair's first preflop raise (the exact CI type key)
            nfold = 0
            for r in arows:
                if r["street"] != 0:
                    break
                if r["seat_no"] in role and r["action"] in (4, 5):
                    break
                if r["action"] == 0:
                    nfold += 1

            eqs = eqmap.get(hr["hand_idx"], {})
            eq_pre = [eqs.get(i) for i in range(len(arows))]
            frp = next((i for i, r in enumerate(arows)
                        if r["street"] == 0 and r["seat_no"] in role and r["action"] in (4, 5)), None)
            read, benign = hand_read(fam, arows, srows, role,
                                     [eqs.get(r["action_no"]) for r in arows], nfold, bb, frp)

            L += [f"### Evidence {k}: `{h}` — table {hr['table_idx']}, hand #{hr['hand_seq']}, "
                  f"blinds {hr['bb'] // 2}/{hr['bb']}", "",
                  "| seat | who | hole cards | net (bb) | showdown |", "|---|---|---|---|---|"]
            for r in srows:
                who = role.get(r["seat_no"], "·")
                L.append(f"| {r['seat_no']} | {who} | {cards(r['c1'], r['c2'])} | "
                         f"{r['net_chips'] / bb:+.1f} | {'yes' if r['went_to_showdown'] else 'no'} |")
            L += ["", f"Board: **{cards(*hr['board']) or '(no flop)'}** · final pot "
                      f"{hr['final_pot'] / bb:.1f} bb", ""]
            cur = -1
            for i, r in enumerate(arows):
                if r["street"] != cur:
                    cur = r["street"]
                    L.append(f"*{ST[cur]}*")
                who = role.get(r["seat_no"], "other")
                amt = f" {r['amount'] / bb:.1f}bb" if r["amount"] else ""
                eqtxt = ""
                if who in ("A", "B") and eq_pre[i] is not None:
                    sa = [k2 for k2, v in role.items() if v == "A"]
                    sb = [k2 for k2, v in role.items() if v == "B"]
                    if sa and sb:
                        ea_, eb_ = eq_pre[i][sa[0]], eq_pre[i][sb[0]]
                        fa = "folded" if ea_ != ea_ else f"{ea_:.0%}"
                        fb = "folded" if eb_ != eb_ else f"{eb_:.0%}"
                        eqtxt = f"  _[equity now: A {fa}, B {fb}]_"
                L.append(f"- **{who}**: {ACT[r['action']]}{amt} (pot {r['pot_before'] / bb:.1f}bb, "
                         f"to call {r['to_call'] / bb:.1f}bb){eqtxt}"
                         if who in ("A", "B") else
                         f"- {who}: {ACT[r['action']]}{amt} (pot {r['pot_before'] / bb:.1f}bb, "
                         f"to call {r['to_call'] / bb:.1f}bb)")
            L += ["", f"**What the model saw.** {read}", "",
                  f"**Benign alternative.** {benign}", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT} ({len(L)} lines, {len(cases)} cases: "
          f"{[c['predicted_behavior'] for c in cases]})")


if __name__ == "__main__":
    main()
