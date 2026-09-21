# 8th place: coordination is phase-local, and the evidence list is a sorting rule

**Private 0.92817 · Public 0.91954 · 29 submissions · joined five days before the deadline**

Two findings did almost all of the work. First, a colluding pair is active **only** in the period it is being
scored in, so every feature has to be computed inside that period. Second, the host's evidence list is not a
ranking at all, it is `sorted(EVENTS, key=(type, hand_seq))[:5]`, and reconstructing that rule took evidence
MAP@5 from 0.666 to 0.705 on the public board.

* **Code (reproduces both selected submissions):** https://github.com/stonesalltheway1/poker-collusion-detection
* **Five case reviews (pair ID, hand IDs, observable behaviour, benign alternative):** https://github.com/stonesalltheway1/poker-collusion-detection/blob/main/research/CASE_REVIEWS_SHORT.md
* Full experiment log including every null: `experiments/LEDGER.md`, `research/CLOSED_LANES.md`

---

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
