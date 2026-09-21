**Final placement: 8th of 370 teams. Private 0.92817, public 0.91954.**
Selected submissions: `sub025_fusebags_exp042ramp.csv` (private 0.92777) and `sub028_famspecw15_exp042ramp.csv` (private 0.92817).

This write-up is **1,496 words**. The five required case reviews follow as an appendix and are the separate required item, not counted against that limit.

| required item | where |
|---|---|
| Solution write-up | this page, and as a notebook: https://www.kaggle.com/code/erickeller2/8th-place-detecting-coordinated-poker-pairs |
| Public code reproducing the selected submission | **https://github.com/stonesalltheway1/poker-collusion-detection** |
| Five case reviews | the appendix below, and https://github.com/stonesalltheway1/poker-collusion-detection/blob/main/research/CASE_REVIEWS_SHORT.md |

`src/run_all.py` rebuilds both selected submissions from `data/raw` in 49 stages. `python src/run_all.py --list` prints every stage and whether its output already exists; `--dry-run` prints the plan without executing. Roughly 10 to 14 hours on one machine (i5-10400F, 32 GB, GTX 1660S); set `XGB_DEVICE=cpu` to run without a GPU. Setup instructions are in the README and every package version is pinned in `requirements.txt`. The competition data is not redistributed; download it from Kaggle into `data/raw/`.

**Rule compliance.** Every feature derives from poker activity: actions, amounts, hole cards, board, and hand order within the scored period. Nothing uses ID formats, file or row ordering, generator internals, or evaluation-file membership, and no `players.parquet` metadata reaches any model. One disclosure, repeated in section 8: the evidence ranker uses chronological order inside the scored period, because the host's lists are ordered that way. The ordering key is the hand start timestamp, and there are zero tied `(table, timestamp)` groups across all 2,000,000 hands, so no identifier is ever load-bearing.

---

Two findings did almost all of the work. First, a colluding pair is active **only** in the period it is being
scored in, so every feature has to be computed inside that period. Second, the host's evidence list is not a
ranking at all, it is `sorted(EVENTS, key=(type, hand_seq))[:5]`, and reconstructing that rule took evidence
MAP@5 from 0.666 to 0.705 on the public board.

## 1. Read the metric before writing any code

`score = 0.70 · pairAP + 0.20 · evidenceMAP@5 + 0.10 · behaviourMacroAP`: three components, three different
problems. You can measure all three exactly by submitting copies of one scored file with a single field blanked,
one with every evidence cell set to `NO_EVIDENCE` and one with every `predicted_behavior` set to
`other_coordination`. Two probes and some algebra pin the decomposition to four decimals.

![score anatomy](https://raw.githubusercontent.com/stonesalltheway1/poker-collusion-detection/main/research/figures/fig5_score_anatomy.png)

One trap, which cost me two wrong versions of this table. Blanking the behaviour column does **not** give AP = 0.
The host ranks tied scores with a stable sort, so an all-zero score vector is ranked in `pair_id` order and
returns the average precision of a *random* ranking. That floor is scored **per family**, over roughly a third of
the positives, so it is 0.0018 at public size, not 0.0047. Add it back or your pair AP is wrong by 0.001.

Probe the model you ship: behaviour AP is risk-model dependent, so borrowing it from an older probe was off
by 0.005.

## 2. Coordination is phase-local

Each table plays 5,000 hands: the first 3,000 carry the labels, the last 2,000 are scored. Take the 281 labelled
colluding pairs sharing 38+ hands in the evaluation period and recompute the same statistics there. They collapse
to chance.

![phase locality](https://raw.githubusercontent.com/stonesalltheway1/poker-collusion-detection/main/research/figures/fig1_phase_locality.png)

The colluding player sets of the two periods are disjoint (0 overlap against 4.5 expected by chance), and the
correlation of our own pair scores across periods is 0.16. So a pair is scored **only** on hands from the period
being judged, with the other period used purely as a clean per-player baseline. Every public notebook I read
mixed the periods, and the best public score was 0.695.

## 3. The evidence list follows a rule

Evidence is 20% of the score and was my weakest component all week. The breakthrough came from reading listed
hands against unlisted hands that looked identical. Each list is **two chronologically sorted segments
concatenated**: exactly one time inversion in 91 of 91 directed-transfer lists, 11 of 11 coordinated-isolation
lists, and 93 of 96 soft-play lists.

The rule is: take every type-1 event of the period in chronological order, then the earliest type-2 events, and
cut at 5.

![the listing rule](https://raw.githubusercontent.com/stonesalltheway1/poker-collusion-detection/main/research/figures/fig2_listing_rule.png)

Two pieces of evidence pin it down. Segment 1 sits at mean position 0.50 of the pair's timeline, so it is not
selected by time, it is the *complete* type-1 set; segment 2 sits early, so it holds the first few type-2 events.
And for coordinated isolation the type key is exact and mechanical: the number of players who folded before the
pair's first preflop raise, 0 for type 1 and 1 for type 2, holding on all 460 listed hands.

For directed transfer, type 1 is "the donor folds to the receiver while holding the better made hand" (221 of 236
segment-1 hands against 32 of 213 segment-2 hands); for soft play it is the same fold-ahead (182 of 213), with
type 2 a passive call (223 of 232). Those two types are only partly visible, since identical action strings occur
in both segments, so I learn them (out-of-fold AUC 0.972 and 0.945) and compute the probability that a hand is
listed with an exact Poisson-binomial dynamic program over each pair's hands. Blended 0.7 with a learned in-pair
ranker, development MAP@5 went 0.711 to 0.775.

## 4. Pipeline

![pipeline](https://raw.githubusercontent.com/stonesalltheway1/poker-collusion-detection/main/research/figures/fig3_pipeline.png)

Every hole card is visible, including folded players, so each decision can be priced exactly. Directed transfer
shows one-way chip flow (100% of planted hands) plus a donor flat-calling the partner's raise where a label-free
normal-play policy gives P(call) below 0.15 (pair AUC 0.993). Soft play shows a partner answering the other's
aggression passively, with a raise-back in only 1.8% of planted hands. Coordinated isolation shows a preflop
raise the population makes under 2% of the time in that exact spot (pair AUC 1.000).

**Directedness beats raw rates.** The strongest features compare a player's behaviour *with this partner* against
*the same player with their other 28 table-mates*, as shrunk log-lift, excess counts and mutual-top flags. That
removes the tilt, weak and loose personas the generator simulates: per-player fold rates span 0.42 to 0.79 where
the population policy predicts a band of 0.61 to 0.67.

**PU learning.** Only 372 pairs are labelled positive, and the 1,488 confirmed-clean pairs are indistinguishable
from random unlabelled pairs (AUC 0.494), so no hard-negative machinery is needed. Two self-training steps help:
highly scored non-listed hands of positive pairs become weak positives, and ~154 unlabelled pairs carrying full
family signatures are promoted.

## 5. Validation

Five folds by table, frozen on day zero: a player sits at exactly one table, so table folds share no players.
Nothing was accepted without a paired table-bootstrap at P(better) above 0.8, then a submission that changed
**exactly one component**, which is what makes the attribution below possible.

![attribution](https://raw.githubusercontent.com/stonesalltheway1/poker-collusion-detection/main/research/figures/fig4_attribution.png)

## 6. What limits the score, and what I would do next

Hand the scorer the true *type* and development MAP@5 goes 0.775 to 0.793. Hand it the true *event*, meaning
which hands the generator actually altered, and it goes to 0.932.

![what limits it](https://raw.githubusercontent.com/stonesalltheway1/poker-collusion-detection/main/research/figures/fig6_what_limits.png)

Retrieval is already solved: recall@10 is 0.970 and recall@5 is 0.820 against an AP@5 of 0.775, so our ordering
of a retrieved set is 94.5% efficient. The entire residual is a binary discrimination over roughly ten candidate
hands per pair, and it is genuinely latent: the generator activates a scenario and a hand counts only if the
colluder's action actually changed, so look-alikes with the same action string, position and hand strength sit
unlisted *between* listed hands. The best deterministic predicate reproduces a pair's exact list for only 7% to
25% of pairs.

Solving `0.7P + 0.2E + 0.1B` for third place (0.94196) against my private components: even a perfect pair ranking
and perfect behaviour routing would still require evidence near 0.710, just below my 0.722. So the top three beat
me on both axes, with evidence around 0.75 to 0.77 and pair AP around 0.985 to 0.990.

## 7. What did not work

* **Six attacks on the event bit**, all neutral or negative: self-trained detectors, sequence-model margins, an
  EM step on the censored tail, player-by-situation normal-play baselines, a within-pair ranking objective, and
  per-pair rate recalibration. Better *detection* reduces listing MAP, because better detectors also fire on
  planted-but-unlisted hands.
* **65 label-free directed statistics** hunting an undisclosed fourth mechanism, against three nulls. Documented
  null, from an instrument that recovers the three known families at 0.30 to 0.74 recall at zero false positives.
* **A validation view that deleted the population I was scored on.** My decision view excluded 137 suspected
  hidden positives, but every evaluation positive is by construction an *unlabelled* positive. A candidate at
  P(better) = 1.000 on that view, positive in 5 of 5 folds and replicated at a second seed, delivered
  **minus 0.00007** on the leaderboard. Build the adverse view too, and ship only what wins on both.
* **An unidentified fit that closed my best lane for three days.** I fitted a two-parameter transfer model to
  three leaderboard points at RMSE 0.0006 and concluded evidence was capped. It is linear in a reparameterisation,
  so the interesting parameter ranged from 0.065 to 0.218 at *identical* RMSE. Always state free parameters
  against data points before letting a fit close a lane.

## 8. Rules and responsible interpretation

Every feature derives from poker activity: actions, amounts, hole cards, board, and hand order within the scored
period. Nothing uses ID format, file or row order, or evaluation-file membership, and no player metadata reaches
any model. **Disclosure:** the evidence ranker uses chronological order inside the scored period, because the
host's lists are ordered that way. The ordering key is the hand start timestamp, and there are zero tied
(table, timestamp) groups across all 2,000,000 hands, so no identifier is ever load-bearing.

The data is synthetic and these are benchmark patterns, not accusations. Each case review gives the hand, both
players' cards, the board, every action, the statistic that fired with its population baseline, a benign
alternative, and the observation that would overturn the reading. A high score is a prompt for human review.

Thanks to the organisers for an unusually well-designed task.

---

# Appendix: the five required case reviews



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
