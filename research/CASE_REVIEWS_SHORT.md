# Five case reviews

Submission reviewed: `sub028_famspecw15_exp042ramp.csv` (private 0.92817). Each review gives the pair ID, the
evidence hand IDs actually submitted, the observable in-gameplay behaviour that drove the flag, and a plausible
benign alternative. Cases were chosen to cover all three disclosed families, because the top of the ranking is
dominated by directed transfer and a plain top-5 would show a reviewer one pattern out of three.

Two notes on method. First, **the detector scores a pattern across a pair's shared hands, not any single hand.**
Every hand below has an ordinary explanation on its own; what the model keys on is how often the pattern recurs
between these two players and not between either of them and their other 28 table-mates. Second, this dataset
exposes every hole card, including folded players, so the "observable behaviour" below includes equities that a
real investigator would not have. Each review therefore also states **what would overturn the reading**, which is
the check a real review should run.

Full hand-by-hand replays of all 25 evidence hands, with every action, pot, price and per-decision equity, are in
[`CASE_REVIEWS.md`](CASE_REVIEWS.md).

---

## Case 1 of 5 · directed transfer · pair `PE4D799DCE834`

**Players:** `U1503AA4813BD` (A), `U3C54D3A546E2` (B) · 222 shared hands in the scored period · net A +307 bb, B −399 bb
**Evidence hands:** `H9F6E6EAB15A72D`, `HEADF53E86AF271`, `HF745F4A26D300F`, `HEA5A04F5B8A9B8`, `H328A8672EA1EE3`

**Observable behaviour.** Chips move one way across the whole period, and they move by B paying A off rather than
by B folding: in the listed hands B calls down or bets into A and loses the pot (B −69 and A +74 in one,
B −188 and A +195 in another). Gross flow is one-directional over 222 hands, which is the rarest shape in the
data; a symmetric win-rate between two players of similar strength is the norm.

**Benign alternative.** B may simply be the weaker player. A calling station who pays off a stronger regular for
200+ hands produces exactly this one-way flow, and the two happen to be seated together for most of the period.
Losing 399 bb over 222 hands is a perfectly ordinary bad run for a loose-passive style.

**What would overturn it.** If B loses at a similar rate to the other 28 players at the table, the directedness
disappears and this is just a losing player.

---

## Case 2 of 5 · soft play · pair `PC178304F6F93`

**Players:** `U2B73D38F0F64` (A), `U54EBF17596CB` (B) · 195 shared hands · net A +46 bb, B +200 bb
**Evidence hands:** `HF745A6904D873E`, `H22311A667D6951`, `H3965E8228C0661`, `H8EEE93EC74FFB5`, `H90FC271477B452`

**Observable behaviour.** When these two are the only players left in a pot, aggression between them stops. In
the listed hands A only ever calls B's bet and never raises, and no raise between the partners appears anywhere
in the hand. In one of them A holds 100% equity at the moment of the call and still does not raise. Across
planted soft-play hands in the labelled data a raise-back between partners appears in only 1.8% of hands, against
a much higher rate for the same players against everyone else.

**Benign alternative.** Pot control with a medium-strength holding is correct poker. Two tight-passive players
seated together will naturally produce many low-aggression shared pots, and flat-calling rather than raising out
of position is standard. The 100%-equity call is only visible because we can see both hands.

**What would overturn it.** If A's raise rate facing B matches A's raise rate facing the rest of the table, this
is a passive style rather than a directed one.

---

## Case 3 of 5 · coordinated isolation · pair `P88F46B09F1D2`

**Players:** `UA7C9B1557591` (A), `UFC3467ABF7FB` (B) · 231 shared hands · net A +181 bb, B −63 bb
**Evidence hands:** `HA0B5BC8A1D70CD`, `H7ADA9BC5C64558`, `H6D0600BDF290CD`, `H3B9ADE5F4DD8B7`, `H23A864724614E6`

**Observable behaviour.** The pair opens the pot with a raise that the population makes in under 2% of that exact
spot (same position, same number of prior raisers and callers, same hand class), with the partner still to act
behind. In every listed hand the pair's first preflop raise comes before anyone has folded, which is the exact
condition the host uses to mark a type-1 coordinated-isolation event. The effect is to clear the field and leave
the two of them heads-up against one outsider.

**Benign alternative.** A loose-aggressive player opens far wider than the population average, and any two
aggressive players seated together will mechanically squeeze the table and generate frequent isolation spots.
A sub-2% opening frequency is unusual, not impossible, and blind-steal ranges are legitimately very wide.

**What would overturn it.** If A opens equally wide when B has already folded or is not seated, the "partner
still to act" conditioning is coincidental and this is just an aggressive opener.

---

## Case 4 of 5 · directed transfer · pair `P8A8E7B5DF6FF`

**Players:** `U5264E27D18AE` (A), `U812E0DB2B2FD` (B) · 145 shared hands · net A −156 bb, B +126 bb
**Evidence hands:** `H1B3D95F3C4FAC2`, `HD8893051FF28C9`, `HAB446445ECC1E1`, `H306218EB171AAC`, `HA74C86D575A096`

**Observable behaviour.** A repeatedly folds to B's aggression while holding the better hand. In
`H1B3D95F3C4FAC2` A folds the turn with 64% equity against B's 36%; in another listed hand A folds with 72%
against 28%. A also flat-calls B's preflop raises in spots where the label-free normal-play policy gives
P(call) below 0.15. Folding the best hand to the same opponent, repeatedly, while over-calling that opponent
preflop, is the fold-ahead signature the host marks as a type-1 directed-transfer event.

**Benign alternative.** A cannot see B's cards. Folding 64% equity to a large turn bet is a normal, if tight,
laydown; the equity edge is only visible to us. Over-calling a specific opponent preflop is consistent with
tilt or with a read that the opponent is bluffing too often.

**What would overturn it.** If A folds the better hand at a similar rate against all opponents, this is a
generally over-folding player rather than a directed one.

---

## Case 5 of 5 · directed transfer · pair `P8E67334B0E04`

**Players:** `U56785476817A` (A), `UF5BDAA08411D` (B) · 211 shared hands · net A +274 bb, B −444 bb
**Evidence hands:** `HC0F810438324E0`, `H9F305E58AD5929`, `HC6F0775882C4AA`, `H945017B5E5C347`, `H5FFA82F65F824E`

**Observable behaviour.** The same fold-ahead shape as Case 4, but at the extreme: B folds the turn holding
100% equity against A's 0%, and folds the flop in another listed hand with 99% against 1%. These are not close
decisions. Over 211 shared hands the net flow is 444 bb from B to A, and the folds cluster on hands where B is
already the certain winner.

**Benign alternative.** Even a 100%-equity fold can be honest if the player misreads the board, for example
failing to notice a counterfeited two pair or a completed straight on a coordinated runout. A player on tilt,
or one who has decided an opponent never bluffs, can fold very strong hands repeatedly.

**What would overturn it.** Board-reading errors of this kind should appear against every opponent. If B's folds
with over 90% equity occur only against A, misreading does not explain the pattern.

---

*The data is synthetic and these are benchmark patterns, not accusations. A high score is a prompt for human
review, never a finding of guilt.*
