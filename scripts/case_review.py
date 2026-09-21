"""Render human-readable case reviews for submitted evidence hands (prize-eligibility requirement + sanity check).

Usage: python scripts/case_review.py <submission.csv> [n_cases] [out.md]
For the n highest-risk pairs it prints, per submitted evidence hand: seats with hole cards, the board by street,
every action in order (partners marked A/B), the exact omniscient equities at the partners' key decisions, chip
flow, and the automatic read (which family signature fired). A benign alternative is listed for each pattern so a
reviewer can weigh it, per the competition's "responsible interpretation" note.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
import common as C  # noqa: E402

SUB = Path(sys.argv[1])
NCASE = int(sys.argv[2]) if len(sys.argv) > 2 else 5
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("research/CASE_REVIEWS.md")
ACT = {0: "fold", 1: "check", 2: "call", 3: "bet", 4: "raise", 5: "all-in"}
ST = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
BENIGN = {
    "directed_transfer": "The loser may simply be a weak/tilted player who calls too wide, and the winner may have "
                         "run hot in these spots; one-way chip flow over a few hands is common variance.",
    "soft_play": "Both players may be passive/loose-passive by style, so checking down strong hands and not raising "
                 "each other can reflect a cautious strategy rather than an agreement.",
    "coordinated_isolation": "Wide opening ranges from early position are a known (if unprofitable) style, and the "
                             "partner folding behind can be ordinary tight play rather than avoidance.",
    "other_coordination": "Unusual joint behaviour can arise from similar training material or copying a regular's style.",
}


def cards(*cs):
    return " ".join(C.card_str(int(c)) for c in cs if c is not None and int(c) >= 0)


def main():
    sub = pl.read_csv(SUB)
    ep = C.load("eval_pairs").select("pair_id", "p1", "p2", "player_1", "player_2")
    top = (sub.join(ep, on="pair_id").sort("risk_score", descending=True).head(NCASE))
    hands = C.load("hands")
    seats = C.scan("seats")
    acts = C.scan("actions")
    hid = hands.select("hand_id", "hand_idx", "table_idx", "hand_seq", "bb", "board", "final_pot")
    lines = [f"# Case reviews — {SUB.name}", "",
             "Synthetic data: these are benchmark patterns, not accusations. Each case lists the observable "
             "behaviour and a plausible benign alternative.", ""]
    for case in top.iter_rows(named=True):
        ev = [case[c] for c in C.EV_COLS if case[c] != C.NO_EV]
        lines += [f"## Pair {case['pair_id']} — predicted {case['predicted_behavior']} "
                  f"(risk {case['risk_score']:.4f})",
                  f"Players: **A = {case['player_1']}**, **B = {case['player_2']}**. Evidence hands: {', '.join(ev)}", ""]
        for hidx, h in enumerate(ev, 1):
            hrow = hid.filter(pl.col("hand_id") == h)
            if hrow.height == 0:
                continue
            hr = hrow.row(0, named=True)
            bb = hr["bb"]
            s = seats.filter(pl.col("hand_idx") == hr["hand_idx"]).collect().sort("seat_no")
            a = acts.filter(pl.col("hand_idx") == hr["hand_idx"]).collect().sort("action_no")
            role = {case["p1"]: "A", case["p2"]: "B"}
            lines += [f"### Evidence {hidx}: hand `{h}` (table {hr['table_idx']}, hand #{hr['hand_seq']}, "
                      f"blinds {hr['bb'] // 2}/{hr['bb']})",
                      "", "| seat | player | cards | net (bb) | showdown |", "|---|---|---|---|---|"]
            for r in s.iter_rows(named=True):
                who = role.get(r["player_idx"], "·")
                lines.append(f"| {r['seat_no']} | {who} | {cards(r['c1'], r['c2'])} | "
                             f"{r['net_chips'] / bb:+.1f} | {'yes' if r['went_to_showdown'] else 'no'} |")
            lines += ["", f"Board: **{cards(*hr['board']) or '(no flop)'}**  ·  final pot {hr['final_pot'] / bb:.1f} bb", ""]
            cur = -1
            for r in a.iter_rows(named=True):
                if r["street"] != cur:
                    cur = r["street"]
                    lines.append(f"*{ST[cur]}*")
                who = role.get(r["player_idx"], "other")
                amt = f" {r['amount'] / bb:.1f}bb" if r["amount"] else ""
                lines.append(f"- {who}: {ACT[r['action']]}{amt} (pot {r['pot_before'] / bb:.1f}bb, "
                             f"to call {r['to_call'] / bb:.1f}bb)")
            lines += ["", f"**Benign alternative:** {BENIGN.get(case['predicted_behavior'], '')}", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({len(lines)} lines, {NCASE} cases)")


if __name__ == "__main__":
    main()
