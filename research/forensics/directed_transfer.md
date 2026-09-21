# Directed transfer (DT): evidence-hand forensics

Script: `scripts/forensics_directed_transfer.py` (stages: extract, print, features, policy, policy_cf, signatures,
policy_signatures, temporal, pairlevel, ranker, evalcheck). Helper: `scripts/fast_eval.py`, a numba 7-card evaluator. Its hand
ordering matched phevaluator on 500k random 7-card comparisons. Exact flop equity takes 0.5 ms.
Outputs: `research/forensics/dt_*` (tables, CSVs, logs), printed hands in `print_directed_transfer.txt` and
`print_dt_lowscore_evidence.txt`.

Scope: 148 DT positive pairs and their 725 evidence hands (137 pairs list 5, 7 list 4, 4 list 3). Every evidence hand is
a dev-phase shared hand. Controls: all 1,488 confirmed negatives, all SP (132) and CI (92) positives, and 3,000 random
unknown dev pairs (dev shared ≥ 38, no labelled-positive player). In total 560,059 dev pair-hands and 454,769 distinct hands.
Equity in this note is perfect-information equity using all seated hole cards: exact enumeration postflop, Monte Carlo
preflop. Throughout, "donor" = the partner with net_chips < 0 and "receiver" = the partner with net_chips > 0.

---------------------------------------------------------------------------------------------------------------------
## TL;DR

1. **Every DT evidence hand moves chips from one partner to the other: 725/725.** The donor has net < 0 and the receiver
   net > 0. The direction never flips inside a pair (0/148 pairs have both directions). The donor is player_1 in 79 pairs
   and player_2 in 69, so pair orientation carries no signal. At eval time you can infer the donor as the player with the
   larger summed EV gift across shared hands. That rule matches the evidence donor in **97.3%** of DT pairs (SP 69%, CI 59%).
2. **The planted action is the donor giving up value to the receiver's aggression.** The donor's last action in the
   evidence hand splits as follows:
   - **fold 74.5%**: preflop 9.4%, flop 41.9%, turn 14.5%, river 8.7%. These folds usually come after the donor put
     chips in, often with the best hand. Mean HU equity at the fold is 0.714, against 0.339 for negatives.
   - **call 17.9%**: calling down with a weak hand. Turn calls average 0.155 equity and river calls lose outright.
   - **bet/raise 6.8%**: the donor bets into the receiver, who calls. Donor bets and raises while the receiver is active
     average 0.27 equity.
   - **check 0.8%**.

   The receiver is the aggressor: 2.29 aggressive actions per hand, against 0.42 for the donor. The hand ends heads-up
   between the two in 78.8% of cases (negatives' flow hands 41.5%). Transfer median is 6.25bb, mean 25.7bb, p90 89.8bb
   (negatives' flow hands: median 1bb, mean 5.9bb).
3. **The strongest DT-specific tell is persistent, not episodic: the donor flat-calls the receiver's preflop raise.**
   Facing the partner's preflop raise, the DT donor calls 71.1% of the time and re-raises 2.8%. The receiver does this
   20.9% of the time, and negative pairs 19.4% (re-raise 9.7%). The donor also calls with trash that the normal-play
   model calls with p ≤ 0.15: 42.0% of these spots, against 2.7% for negatives. The rate is flat across dev time
   (69–72% in every 500-hand bin).
   - Hand signature `SIG_pfcall_weak`: 49.1% of evidence hands, 0.24% of negative-pair hands.
   - Pair statistic `pfcall_weak_max`, the max over the two players of the number of shared hands in which that player
     calls the partner's preflop raise with p_call ≤ 0.15:
     **AUC 0.994 / AP 0.950 vs confirmed negatives, AUC 0.994 / AP 0.903 vs unknown**, and AUC 0.991 DT vs SP+CI.
     It scores AUC 0.556 for SP+CI vs negatives, so it is DT-specific.
4. **Unlisted planted hands exist, roughly as many as the listed ones.**
   - Among non-evidence DT shared hands with an EV gift > 20bb, 85.8% flow in the evidence donor's direction. Under the
     null the rate is 50%, and SP and CI sit at 42–53%.
   - That excess implies about 780 unlisted planted-like hands, against 725 listed.
   - Donor-direction excess is concentrated in large call-downs: an estimated 18–26% of those are listed, against 54–63%
     of postflop folds.
5. **The listing is biased toward the chronologically first planted hands. It is not strictly the first 5.**
   - Among planted-like candidates (evidence donor direction and a strong anomaly), P(listed) by chronological order
     within the pair is 0.87, 0.80, 0.73, 0.68, 0.57, then 0.38, 0.41, 0.34, 0.28, 0.29, 0.19, and 0.14 for the 12th
     candidate onward.
   - With the candidate count N held fixed (8–10), P(listed) by relative-position quartile is 0.85 / 0.55 / 0.39 / 0.26.
   - Evidence per shared hand falls across the dev phase: 0.055 → 0.047 → 0.038 → 0.033 → 0.025 per 600 hands. The
     non-evidence signature rate rises over the same span (0.074 early, 0.091 late).
6. **Activation is phase-specific.** In the eval phase, the labelled DT donors call the partner's preflop raise at 20.7%,
   the population rate. A dev-trained signature therefore does not carry over pair by pair: **eval pairs must be scored
   on eval-phase hands.**
7. **Evidence ranking, dev MAP@5 on DT pairs** (all dev shared hands as candidates):

   | Recipe | Dev MAP@5 |
   |---|---|
   | EV gift, descending | 0.273 |
   | Transfer size | 0.192 |
   | Signature candidates, earliest first | 0.508 |
   | Out-of-fold LightGBM hand model S2 | 0.525 |
   | S2 with chronological decay p·exp(−K/2) | **0.554** |

   Here K is the summed p of the pair's earlier hands. The hand model alone reaches AUC 0.998 against negative-pair
   hands and 0.987 against the same pair's non-evidence hands. The listing rule is the bottleneck, not detection.

---------------------------------------------------------------------------------------------------------------------
## 1. Actor roles and what the planted hand looks like

| Evidence hand stat | DT evidence (725) | Negative-pair flow hands (27,952) |
|---|---|---|
| Donor put money in voluntarily | 94.2% | 43.4% |
| Receiver put money in voluntarily | 100% | 96.6% |
| A and B were the last two players | 78.8% | 41.5% |
| Both went to showdown | 25.0% | 7.0% |
| Hand ended by a fold | 72.3% | 81.9% |
| Transfer min(−net_donor, net_receiver): median / mean / p90 (bb) | 6.25 / 25.7 / 89.8 | 1.0 / 5.9 / 8.0 |
| Donor preflop strength (HU vs random) | 0.496 | 0.504 |
| Receiver preflop strength | 0.567 | 0.599 |
| Aggressive actions per hand, donor / receiver | 0.42 / 2.29 | 0.28 / 1.43 |

Donor actions taken while the receiver is still active, in DT evidence hands (from `dt_action_profile.csv`):

| Donor action | Count | Mean HU equity vs receiver | Mean pot odds | Mean street |
|---|---|---|---|---|
| Fold facing receiver | 536 | 0.714 | 0.305 | 1.30 |
| Call facing receiver | 1,055 | 0.292 | 0.297 | 0.69 |
| Bet/raise while receiver active | 298 | 0.270 | — | 1.25 |
| Check | 425 | 0.571 | — | — |

Donor folds facing the partner in negative-pair flow hands average HU equity 0.339 at street 0.45. The same folds in
DT evidence average 0.714 at street 1.30, so DT folds happen later and with the better hand.

By street, donor folds facing the receiver in evidence average equity 0.60 preflop, 0.72 flop, 0.72 turn and 0.80 river.
Negative-pair donors fold at 0.35, 0.22, 0.16 and 0.05. Calls on turn and river average 0.155 and 0.00 equity. The
normal-policy model finds DT evidence donors' preflop calls most surprising: surprise 2.48, p(call) 0.15, against 1.25
and 0.33 for negatives.

Archetypes, all with one-directional chip flow. Full hands are printed in section 9.
- **A. Invest, then fold the best hand to the partner's bet.** Typical is a flop or turn fold with equity ≥ 0.7, including
  straights and two pair on the river. It covers 46% of evidence (postflop, equity ≥ 0.7).
- **B. Call down with a weak or dead hand.** Flat-call the preflop raise, then call flop, turn and river bets with a high
  card or weak pair. Turn/river calls with equity ≤ 0.15 cover 17% of evidence.
- **C. Bet or raise into the partner and get called.** The donor bluffs with equity ≤ 0.3 and the receiver calls or
  shoves. Covers 19% of evidence.
- **D. Preflop squeeze and fold.** The donor calls the partner's 3-bet, the partner 4-bets, and the donor folds. This
  is small money.
- **E. Min-raise, then fold** to the partner's small bet after the partner calls.

The preflop flat-call of the receiver's raise appears in 49% of evidence hands, whichever archetype follows.

---------------------------------------------------------------------------------------------------------------------
## 2. Candidate signatures and their measured rates

All flags are hand-level and oriented. SIG_x is true if either partner, as the donor (net < 0 while the other's net > 0),
satisfies x against the other. "Facing partner" means the last aggressive action on the street was the partner's. The
EV gift is chips the donor gives up relative to the alternative, in bb, using HU equity against the partner:
- fold: max(0, eq·(pot + to_call) − to_call)
- call: max(0, amt − eq·(pot + amt))
- bet or raise that the partner calls: max(0, amt·(1 − 2·eq))

Rates are fractions of hands. Columns:
- **DT ev**: DT evidence hands (725).
- **DT non-ev**: the same pairs' other dev shared hands (17,425).
- **Neg**: confirmed-negative pair hands (178,620).
- **Unk**: unknown-sample pair hands (336,310).
- **SP ev / CI ev**: evidence hands of those families (632 / 460).

| Signature | Definition | DT ev | DT non-ev | Neg | Unk | SP ev | CI ev |
|---|---|---|---|---|---|---|---|
| SIG_flow | net_donor < 0 and net_receiver > 0 | **1.000** | 0.189 | 0.157 | 0.158 | 0.837 | 0.439 |
| SIG_flow_last2 | flow, and the two were the last two active players | 0.788 | 0.098 | 0.065 | 0.065 | 0.574 | 0.230 |
| SIG_anyconf | flow, and the donor folds or calls facing the receiver, or bluffs into a receiver who calls (eq ≤ 0.3) | **1.000** | 0.150 | 0.119 | 0.120 | 0.801 | 0.424 |
| SIG_fold_best50 | flow, and the donor folds facing the receiver with HU eq ≥ 0.5 | 0.611 | 0.031 | 0.020 | 0.020 | 0.331 | 0.152 |
| SIG_fold_best70_post | same, postflop, eq ≥ 0.7 | 0.462 | 0.015 | 0.0054 | 0.0057 | 0.263 | 0.022 |
| SIG_call_weak25 | flow, and the donor calls facing the receiver with eq ≤ 0.25 | 0.317 | 0.042 | 0.011 | 0.012 | 0.120 | 0.009 |
| SIG_call_below_odds | flow, and the donor calls facing the receiver with eq < pot odds | 0.446 | 0.048 | 0.013 | 0.014 | 0.150 | 0.015 |
| SIG_call_weak15_turnriver | flow, and a turn/river call facing the receiver with eq ≤ 0.15 | 0.170 | 0.028 | 0.0059 | 0.0059 | 0.065 | 0.004 |
| SIG_bluff_into_partner30 | flow, and the donor bets/raises, the receiver calls or raises, donor eq ≤ 0.3 | 0.190 | 0.031 | 0.013 | 0.013 | 0.199 | 0.083 |
| **SIG_union** | flow and (fold_best50 or call_weak25 or bluff30) | **0.897** | 0.083 | 0.039 | 0.040 | 0.614 | 0.237 |
| **SIG_strong** | flow and (fold_best70_post or call_weak15_turnriver) | 0.623 | 0.042 | **0.011** | 0.012 | 0.328 | 0.026 |
| **SIG_gift5** | flow, and donor EV gift to the receiver > 5bb | 0.790 | 0.067 | 0.019 | 0.019 | 0.380 | 0.117 |
| **SIG_pfcall_weak** | flow, and the donor calls the receiver's preflop raise with cross-fitted normal-policy p(call) ≤ 0.15 | 0.491 | 0.037 | **0.0024** | 0.0026 | 0.033 | 0.011 |
| SIG_fold_surpr | flow, and the donor folds facing the receiver with policy p(fold) ≤ 0.2 | 0.171 | 0.004 | 0.0019 | 0.0020 | 0.084 | 0.002 |
| SIG_postcall_surpr | flow, and a postflop call facing the receiver with p(call) ≤ 0.2 | 0.170 | 0.021 | 0.0018 | 0.0019 | 0.011 | 0.000 |
| SIG_agg_surpr | flow, and a bet/raise into a receiver who calls, with p(agg) ≤ 0.2 | 0.109 | 0.020 | 0.0066 | 0.0065 | 0.067 | 0.297 |
| SIG_policy_union | flow and (pfcall_weak or fold_surpr or postcall_surpr or agg_surpr) | 0.686 | 0.062 | 0.012 | 0.012 | 0.197 | 0.311 |
| SIG_union or SIG_pfcall_weak | — | **0.956** | 0.091 | 0.040 | 0.041 | 0.627 | 0.242 |
| SIG_strong or SIG_pfcall_weak | — | 0.799 | 0.061 | 0.013 | 0.014 | 0.347 | 0.037 |

Continuous hand scores (AUC for DT evidence against negative-pair hands / against the same pair's non-evidence hands):

| Score | AUC vs negative-pair hands | AUC vs same-pair non-evidence |
|---|---|---|
| Transfer | 0.973 | 0.934 |
| EV gift | 0.977 | 0.943 |
| Donor max normal-policy surprise | 0.989 | 0.956 |
| LightGBM hand model S1 (positives = DT evidence, negatives = negative-pair hands; out-of-fold by frozen table folds) | 0.9983 | 0.983 |
| LightGBM hand model S2 (negatives also include DT non-evidence hands) | 0.9977 | 0.987 |

S1's top features by gain:

| Feature | What it measures |
|---|---|
| x_s_pfcall | surprise of the donor's preflop call vs the partner |
| transfer | chips moved from donor to receiver |
| x_eq_fold_max | donor's highest equity at a fold facing the partner |
| x_gift_max | donor's largest EV gift in one action |
| y_s_max, y_s_sum | receiver's max and summed surprise |
| x_s_fold | surprise of the donor's fold |
| pfX | donor's preflop strength |

Normal-play policy model: LightGBM with 4 classes (fold/check/call/agg). It sees only public state and own cards:
street, facing, to_call, pot, stack, SPR, pot odds, players active, aggression counts, amount put in, position,
preflop aggressor, own HS vs 1 and vs N random hands, made-hand category, preflop strength. It trains on 1.2M actions
from hands with no labelled-positive pair seated. It is cross-fitted over two table halves (fold % 2), with validation
logloss 0.554–0.560. The non-cross-fitted and cross-fitted versions give the same signature rates to 3 decimals.

---------------------------------------------------------------------------------------------------------------------
## 3. Persistent pair behaviour, used for pair detection and behaviour classification

When a partner has made the last preflop raise:

| Role (dev phase) | Spots | Call | Re-raise | Weak call (policy p ≤ 0.15) | Median preflop strength of calls |
|---|---|---|---|---|---|
| DT donor | 3,754 | **0.711** | 0.028 | **0.420** | 0.458 |
| DT receiver | 1,817 | 0.209 | 0.111 | 0.040 | 0.570 |
| Confirmed-negative pair members | 36,444 | 0.194 | 0.097 | 0.027 | 0.569 |
| SP pair members (either role) | ~2,100 | 0.39 | 0.03 | 0.05 | 0.57 |
| CI pair members | ~4,600 | 0.13–0.20 | 0.23–0.25 | 0.02–0.04 | 0.55 |

- The DT donor's call rate by 500-hand dev bin runs 0.723, 0.716, 0.689, 0.713, 0.714, 0.711. It is not episodic within
  the dev phase.
- Per-pair donor call rates range from 0.29 to 1.00 (median 0.73).
- SP pairs share the "don't re-raise the partner" pattern, but call at 0.39 rather than 0.71 and without weak holdings.

**Eval phase** (`evalcheck`): the labelled DT donors call the partner's preflop raise at 0.207, against a population
0.200. The pair-level AUC of the max call rate in eval hands, DT vs negatives, is 0.476. The CI and structure notes
reach the same conclusion: labelled pairs are inactive in eval, so eval features must come from eval-phase hands.

A caution against pooling phases:
- In the dev phase, 13 confirmed negatives and 26 unknown pairs have ≥ 5 weak preflop calls against the partner.
- In the eval phase, 38% of those negatives and 68% of those unknowns still call the partner's raise ≥ 50% of the time.
  The baseline is 11–12%.
- A trait that persists across both phases looks more like a hard negative than like phase-local collusion. Treat a
  dev-and-eval match with suspicion rather than as confirmation.

---------------------------------------------------------------------------------------------------------------------
## 4. Unlisted planted hands and the listing rule

- **DT non-evidence hands flow in the evidence donor's direction far more often than chance**, and the excess grows
  with the size of the gift. SP and CI controls stay flat:

  | EV gift of the non-evidence hand | Share in the evidence donor's direction |
  |---|---|
  | ≤ 0 | 41.8% |
  | 0–5bb | 53.9% |
  | 5–20bb | 78.4% |
  | > 20bb | **85.8%** |

  SP: 45%, 48%, 45%, 42%. CI is flat too. The implied excess is about 780 unlisted planted-like DT hands, against 725
  listed.
- **Share listed = listed / (listed + estimated unlisted)**, by the donor's last action:

  | Donor's last action | Share listed |
  |---|---|
  | Flop fold | 0.54 |
  | Turn fold | 0.63 |
  | River fold | 0.62 |
  | Preflop call | 0.39 |
  | Flop call | 0.26 |
  | Turn call | 0.18 |
  | River call | 0.23 |
  | Turn bet/raise | 0.30 |
  | River bet/raise | 0.35 |

  Large call-down hands are planted-like but mostly unlisted. By transfer bin, 2–5bb is 61% listed, 5–10bb 60%,
  50–100bb 26%, and >100bb 24%.
- **Chronology.** Define candidates as SIG_strong in the evidence donor's direction, plus the evidence hands themselves.
  - P(listed) by the candidate's chronological order within the pair is 0.865, 0.804, 0.730, 0.681, 0.572, 0.381,
    0.414, 0.341, 0.282, 0.293, 0.186, and 0.144 for 12th onward.
  - Controlled for N (the pair's candidate count), P(listed) falls monotonically by relative-position quartile:

    | N | Pairs | Q1 | Q2 | Q3 | Q4 |
    |---|---|---|---|---|---|
    | 5–7 | 47 | 0.94 | 0.89 | 0.72 | 0.53 |
    | 8–10 | 48 | 0.85 | 0.55 | 0.39 | 0.26 |
    | 11–15 | 27 | 0.59 | 0.41 | 0.30 | 0.20 |
    | 16+ | 16 | 0.41 | 0.35 | 0.17 | 0.11 |

  - Among the first 3 candidates, P(listed) does not depend on absolute hand_seq (0.63–0.67 in every 750-hand bin).
    The rule is about order within the pair, not absolute time.
  - Pairs with 3 or 4 evidence hands have only 1.25–2.0 candidates, against 8.0 for pairs with 5. This is consistent
    with "list min(5, #planted), preferring the earliest".
- evidence_rank correlates with hand_seq at Spearman 0.44 (SP 0.37, CI 0.91) and with transfer at 0.33. Rank 1 has a
  mean hand_seq of 935, against about 1,300 for ranks 2–5.
- Unlisted strong same-direction candidates still occur before the pair's last listed hand: 3.3% of those hands,
  against a 0.46% opposite-direction background. The rule is not a strict first-5.

---------------------------------------------------------------------------------------------------------------------
## 5. Temporal pattern

- Evidence per 600-hand dev bin: 188, 208, 127, 114, 88. Shared hands per bin: 3,426, 4,415, 3,305, 3,467, 3,537.
  Per shared hand: 0.055, 0.047, 0.038, 0.033, 0.025.
- The planted-like signal itself is not front-loaded. Non-evidence SIG_union runs 0.074 in dev hands 0–1499 and 0.091
  in 1500–2999. Negatives sit at 0.039 in both halves.
- **Episodes.** For hands with EV gift > 5bb, the median gap to the nearest other such hand in the same pair is
  **4 table hands** for DT pairs (79% within 30). Negative pairs: 39 (48%). Unknown pairs: 74 (44%).
  Consecutive evidence hands have a median gap of 120 table hands (41% within 20). Five random shared hands of the same
  pairs: median 192 (34% within 20).
  - Timelines show 2–5 bursts per pair lasting roughly 10–60 hands. Example, P603CAEBF9999: a burst over hands
    1614–1665 holds about 10 same-direction transfer hands with evidence at 1644 and 1652. A second burst over 2307–2336
    has evidence at 2317, 2327 and 2329.
- Sessions, defined as shared hands with gaps ≤ 15 table hands: 9.3 per DT pair at 13.2 hands each. 34% of sessions
  contain evidence, 1.54 evidence hands per such session, spread over 3.2 sessions per pair (median 3). The same-direction
  signature rate in non-evidence hands is equal in sessions with and without evidence (6.6% vs 6.4%). Planted hands are
  spread over many sessions, not only the listed ones.
- Early vs late dev: evidence coverage by signatures is stable (SIG_union 0.894 early, 0.902 late).

---------------------------------------------------------------------------------------------------------------------
## 6. Pair-level separability

Groups: 148 DT, 1,488 confirmed negatives, 3,000 unknown, plus 224 SP+CI. The "unknown" sample contains hidden
positives, so figures against it are lower bounds.

| Pair statistic (dev shared hands) | AUC DT vs neg | AP DT vs neg | AUC DT vs unk | AP DT vs unk | AUC SP+CI vs neg | AUC DT vs SP+CI |
|---|---|---|---|---|---|---|
| **pfcall_weak_max** (count) | **0.9943** | 0.950 | **0.9938** | 0.903 | 0.556 | **0.991** |
| pfcall_weak_max_rate (per shared hand) | 0.9942 | **0.957** | 0.9930 | **0.914** | 0.555 | 0.990 |
| policy_union_max_rate | 0.9855 | 0.906 | 0.9837 | 0.813 | 0.735 | 0.900 |
| gift5_max_rate (max over direction of #hands with gift > 5bb / shared) | 0.9770 | 0.880 | 0.9753 | 0.802 | 0.754 | 0.939 |
| gift5_max | 0.9716 | 0.846 | 0.9725 | 0.761 | 0.739 | 0.927 |
| gift_max_rate (summed gift) | 0.9667 | 0.845 | 0.9641 | 0.751 | 0.680 | 0.936 |
| union_max_rate | 0.9638 | 0.805 | 0.9618 | 0.713 | 0.779 | 0.870 |
| strong_max_rate | 0.9632 | 0.822 | 0.9597 | 0.732 | 0.683 | 0.922 |
| gift5_asym_rate (abs(A→B − B→A) / shared) | 0.9582 | 0.877 | 0.9537 | 0.774 | 0.617 | 0.940 |
| netflow (abs net transfer) | 0.9251 | 0.792 | 0.9298 | 0.726 | 0.604 | 0.907 |
| shared hands | 0.531 | 0.096 | 0.582 | 0.055 | 0.515 | 0.518 |

- **Threshold rules:**

  | Rule | DT | Negatives | Unknown |
  |---|---|---|---|
  | pfcall_weak_max ≥ 5 | 89.9% | 0.94% | 0.87% |
  | pfcall_weak_max ≥ 3 | 96% | 5.0% | 4.7% |
  | pfcall_weak_max ≥ 6 and gift5_max ≥ 3 | 83.8% | 0.34% | 0.40% |
- **Shared-hand confound.** Count statistics correlate with shared hands (negatives: Spearman 0.34 for gift5_max).
  Rate statistics do not (−0.03). AUC of gift5_max_rate within shared-hand quartiles (38–75, 76–105, 106–144, 145–465):
  0.974, 0.985, 0.973, 0.985 against negatives, and 0.970, 0.988, 0.967, 0.983 against unknown. Use rates, or counts
  with shared hands as a feature.
- **Simulated eval prevalence.** With unknown pairs as negatives, reweighted to 0.25% positives,
  pfcall_weak_max_rate alone reaches AP ≈ 0.61. A rank-average with gift5_max_rate and policy_union_max_rate reaches
  0.54. Both figures are depressed by hidden positives among the unknown pairs.

---------------------------------------------------------------------------------------------------------------------
## 7. Evidence ranking recipe for DT-predicted eval pairs (eval-phase shared hands only)

Dev MAP@5 on the 148 DT pairs, out-of-fold (`dt_evidence_recipes.csv`):

| Recipe | Dev MAP@5 |
|---|---|
| Transfer size, descending | 0.192 |
| EV gift, descending | 0.273 |
| S1 raw probability | 0.395 |
| Candidates (SIG_anyconf, inferred donor, transfer ≥ 2bb), earliest first | 0.508 |
| S1 p > 0.05, earliest first | 0.523 |
| S2 raw probability | 0.525 |
| S2 p > 0.35, earliest first | 0.538 |
| S2 with p·exp(−K/8) | 0.547 |
| S2 with p·exp(−K/4) | 0.553 |
| **S2 with p·exp(−K/2)** | **0.554** |

Recipe:
1. **Orient.** For each shared hand compute both orientations. Donor = the player with the larger summed EV gift over
   the pair's phase hands (97% accurate on DT). Or take the max over orientations, which performs the same.
2. **Hand probability p.** Use a LightGBM trained on dev with positives = DT evidence (donor orientation) and negatives
   = negative-pair hands plus DT non-evidence hands (S2).
   - Features: donor's max and summed normal-policy surprise; surprise of the donor's preflop call vs the partner;
     surprise of the donor's fold and of postflop calls; transfer (bb).
   - Also: max HU equity at the donor's folds facing the partner; min equity at the donor's calls and bets into the
     partner; EV gift (sum and max).
   - Also: nets, contributions, voluntary-money flags, last_two, n_final, board length, both preflop strengths, stack.
3. **Chronology.** Sort the pair's eval-phase shared hands by hand_seq. Set K_i = the sum of p over earlier hands and
   score_i = p_i · exp(−K_i / 2). Submit the top 5 by score. Fallback without a model: SIG_strong or SIG_pfcall_weak in
   the inferred donor's direction, earliest first, padded with SIG_gift5 hands.
4. Never submit a hand without flow. Every evidence hand has donor net < 0 and receiver net > 0, which removes 81–84% of
   shared hands at no recall cost.
5. **Hedge on chronology.** Whether the eval list is also early-biased cannot be checked locally. The decay adds about
   +0.03 over S2 raw and +0.16 over S1 raw. An LB A/B of the same risk and behaviour with and without the decay settles
   it; the CI note proposes the same test.

---------------------------------------------------------------------------------------------------------------------
## 8. Recommended pair-level features, computed on phase-local hands

For risk:
- **pfcall_weak_max and pfcall_weak_max_rate** (max over players), plus the raw call and re-raise rates facing the
  partner's preflop raise, asymmetric between the two players.
- policy_union_max_rate.
- gift5_max_rate, gift5_asym_rate and gift_max_rate.
- strong_max_rate, union_max_rate, netflow_rate.
- Burstiness: the median nearest-neighbour gap between gift > 5 hands.

For behaviour (DT vs SP vs CI):
- pfcall_weak_max_rate: AUC 0.990 DT vs SP+CI.
- gift5_asym_rate: 0.940.
- Donor-direction consistency: the share of flow-signature hands in the dominant direction.
- SP shares "don't 3-bet the partner" but lacks weak calls and the large one-way flow.
- CI shows SIG_agg_surpr (30% of CI evidence), not folds or calls to the partner.

For evidence: the hand model p, chronology position within the phase, and the inferred donor.

---------------------------------------------------------------------------------------------------------------------
## 9. Archetypal evidence hands (full detail)

How to read these printouts:
- **Roles:** A = player_1, B = player_2, oNNN = another player.
- **Per action:** `amt` = chips added; `to` = street total after the action; `tocall` = chips needed to call before
  the action; `odds` = to_call / (pot + to_call).
- **Equity columns:** `eq` = multiway perfect-information equity at the decision; `huVsPartner` = heads-up equity
  against the partner. The made-hand label is the actor's current category.

### Archetype A: invest, call down, then fold the nuts on the river to the partner's bet (pair P27D0D8DAE097, rank 1)
- Preflop and flop: donor B (5hTd, 13% equity) flat-calls A's open, then calls a flop bet with ten-high.
- Turn: B calls again with 19% equity.
- River: B makes a straight (equity 1.00), checks, and folds to A's 44-chip bet with one pair. B loses 29 and A wins 36.
- Verified flags: SIG_fold_best70_post, SIG_call_weak25, SIG_pfcall_weak, SIG_fold_surpr, SIG_strong, SIG_gift5
  (gift 56bb, transfer 14.5bb).

```
===== EVIDENCE pair=P27D0D8DAE097 fam=directed_transfer rank=1 A=1196 B=7256 dev_shared=114
hand_idx=1635981 seq=981 phase=0 sb/bb=1/2 button_seat=3 dealt=6 sd=0 final_pot=109 board=6h Qd 3d 4s 7s
   seat0      A stack=  247 ( 123.5bb) 7hTh contrib=   73 net=    36 fold=0 sd=0 won=1.00 final=Pair(4976)
   seat1  o6160 stack=  148 (  74.0bb) 8s4c contrib=    0 net=     0 fold=1 sd=0 won=0.00 final=Pair(5647)
   seat2  o4984 stack=  282 ( 141.0bb) TsKh contrib=    0 net=     0 fold=1 sd=0 won=0.00 final=HighCard(6727)
   seat3  o1669 stack=  153 (  76.5bb) Ks8d contrib=    5 net=    -5 fold=1 sd=0 won=0.00 final=HighCard(6763)
   seat4      B stack=  395 ( 197.5bb) 5hTd contrib=   29 net=   -29 fold=1 sd=0 won=0.00 final=Straight(1607)
   seat5  o1997 stack=  199 (  99.5bb) Ac9s contrib=    2 net=    -2 fold=1 sd=0 won=0.00 final=HighCard(6420)
  -- PRE    active=6
     # 0      A raise            amt=    5 to=    5 pot=    3 tocall=    2 stack=  247 act=6 odds=0.40 eq=0.19 huVsPartner=0.65
     # 1  o6160 fold             amt=    0 to=    0 pot=    8 tocall=    5 stack=  148 act=6 odds=0.38 
     # 2  o4984 fold             amt=    0 to=    0 pot=    8 tocall=    5 stack=  282 act=5 odds=0.38 
     # 3  o1669 call             amt=    5 to=    5 pot=    8 tocall=    5 stack=  153 act=4 odds=0.38 
     # 4      B call             amt=    4 to=    5 pot=   13 tocall=    4 stack=  394 act=4 odds=0.24 eq=0.13 huVsPartner=0.35
     # 5  o1997 fold             amt=    0 to=    2 pot=   17 tocall=    3 stack=  197 act=4 odds=0.15 
  -- FLOP 6h Qd 3d   active=3
     # 6      B check            amt=    0 to=    0 pot=   17 tocall=    0 stack=  390 act=3           eq=0.19 huVsPartner=0.29 HighCard
     # 7      A bet              amt=   10 to=   10 pot=   17 tocall=    0 stack=  242 act=3           eq=0.22 huVsPartner=0.71 HighCard
     # 8  o1669 fold             amt=    0 to=    0 pot=   27 tocall=   10 stack=  148 act=3 odds=0.27 
     # 9      B call             amt=   10 to=   10 pot=   27 tocall=   10 stack=  390 act=2 odds=0.27 eq=0.29 HighCard
  -- TURN 6h Qd 3d 4s   active=2
     #10      B check            amt=    0 to=    0 pot=   37 tocall=    0 stack=  380 act=2           eq=0.19 HighCard
     #11      A bet              amt=   14 to=   14 pot=   37 tocall=    0 stack=  232 act=2           eq=0.81 HighCard
     #12      B call             amt=   14 to=   14 pot=   51 tocall=   14 stack=  380 act=2 odds=0.22 eq=0.19 HighCard
  -- RIVER 6h Qd 3d 4s 7s   active=2
     #13      B check            amt=    0 to=    0 pot=   65 tocall=    0 stack=  366 act=2           eq=1.00 Straight
     #14      A bet              amt=   44 to=   44 pot=   65 tocall=    0 stack=  218 act=2           eq=0.00 Pair
     #15      B fold             amt=    0 to=    0 pot=  109 tocall=   44 stack=  366 act=2 odds=0.29 eq=1.00 Straight
```

### Archetype B: weak call-down for a stack (pair P0AE72A0C606C, rank 2)
- Preflop: donor B (9h6h) calls A's 3-bet.
- Flop: B leads 58 into 86 with nine-high (21% equity).
- Turn and river: B calls a 103-chip turn bet (14%) and a 173-chip river shove (0%) with nine-high, against A's
  ace-high, which is barely better. B loses 370 chips (92bb) to A. Every B action is a gift.
- Verified flags: SIG_call_weak15_turnriver, SIG_bluff_into_partner30, SIG_pfcall_weak, SIG_postcall_surpr,
  SIG_agg_surpr, SIG_strong, SIG_gift5 (gift 65.5bb, transfer 92.5bb).

```
===== EVIDENCE pair=P0AE72A0C606C fam=directed_transfer rank=2 A=4902 B=6255 dev_shared=138
hand_idx=223 seq=223 phase=0 sb/bb=2/4 button_seat=5 dealt=6 sd=2 final_pot=754 board=Kh Js 2c 7d 5d
   seat0 o10170 stack=  476 ( 119.0bb) Jd2d contrib=    2 net=    -2 fold=1 sd=0 won=0.00 final=TwoPair(2920)
   seat1      B stack=  455 ( 113.8bb) 9h6h contrib=  370 net=  -370 fold=0 sd=1 won=0.00 final=HighCard(6832)
   seat2  o5757 stack=  817 ( 204.2bb) 6d4d contrib=    0 net=     0 fold=1 sd=0 won=0.00 final=HighCard(6862)
   seat3  o1525 stack=  703 ( 175.8bb) Ah7h contrib=   12 net=   -12 fold=1 sd=0 won=0.00 final=Pair(4867)
   seat4      A stack=  370 (  92.5bb) AsQs contrib=  370 net=   384 fold=0 sd=1 won=1.00 final=HighCard(6188)
   seat5  o6027 stack=  435 ( 108.8bb) Jc8s contrib=    0 net=     0 fold=1 sd=0 won=0.00 final=Pair(4065)
  -- PRE    active=6
     # 0  o5757 fold             amt=    0 to=    0 pot=    6 tocall=    4 stack=  817 act=6 odds=0.40 
     # 1  o1525 raise            amt=   12 to=   12 pot=    6 tocall=    4 stack=  703 act=5 odds=0.40 
     # 2      A raise            amt=   36 to=   36 pot=   18 tocall=   12 stack=  370 act=5 odds=0.40 eq=0.34 huVsPartner=0.64
     # 3  o6027 fold             amt=    0 to=    0 pot=   54 tocall=   36 stack=  435 act=5 odds=0.40 
     # 4 o10170 fold             amt=    0 to=    2 pot=   54 tocall=   34 stack=  474 act=4 odds=0.39 
     # 5      B call             amt=   32 to=   36 pot=   54 tocall=   32 stack=  451 act=3 odds=0.37 eq=0.26 huVsPartner=0.36
     # 6  o1525 fold             amt=    0 to=   12 pot=   86 tocall=   24 stack=  691 act=3 odds=0.22 
  -- FLOP Kh Js 2c   active=2
     # 7      B bet              amt=   58 to=   58 pot=   86 tocall=    0 stack=  419 act=2           eq=0.21 HighCard
     # 8      A call             amt=   58 to=   58 pot=  144 tocall=   58 stack=  334 act=2 odds=0.29 eq=0.79 HighCard
  -- TURN Kh Js 2c 7d   active=2
     # 9      B check            amt=    0 to=    0 pot=  202 tocall=    0 stack=  361 act=2           eq=0.14 HighCard
     #10      A bet              amt=  103 to=  103 pot=  202 tocall=    0 stack=  276 act=2           eq=0.86 HighCard
     #11      B call             amt=  103 to=  103 pot=  305 tocall=  103 stack=  361 act=2 odds=0.25 eq=0.14 HighCard
  -- RIVER Kh Js 2c 7d 5d   active=2
     #12      B check            amt=    0 to=    0 pot=  408 tocall=    0 stack=  258 act=2           eq=0.00 HighCard
     #13      A all_in(ai-raise) amt=  173 to=  173 pot=  408 tocall=    0 stack=  173 act=2           eq=1.00 HighCard
     #14      B call             amt=  173 to=  173 pot=  581 tocall=  173 stack=  258 act=2 odds=0.23 eq=0.00 HighCard
```

### Archetype C: bet and raise into the partner with nothing, partner calls (pair P05C8D8034969, rank 5)
- Flop: donor B (3dTh) leads the flop and calls A's raise with ten-high (20% equity).
- Turn: B bets 101 into 143 with a board pair only (12%), and A calls all-in with a better kicker. B loses 125 and A wins 128.
- Verified flags: SIG_bluff_into_partner30, SIG_call_weak25, SIG_pfcall_weak, SIG_agg_surpr, SIG_gift5
  (gift 53bb, transfer 62.5bb).

```
===== EVIDENCE pair=P05C8D8034969 fam=directed_transfer rank=5 A=751 B=988 dev_shared=142
hand_idx=1705885 seq=885 phase=0 sb/bb=1/2 button_seat=2 dealt=6 sd=2 final_pot=299 board=8c Ad Jc 8d 5h
   seat0  o7061 stack=  218 ( 109.0bb) 3sAc contrib=    0 net=     0 fold=1 sd=0 won=0.00 final=TwoPair(2525)
   seat1  o1834 stack=  185 (  92.5bb) 7sTd contrib=    2 net=    -2 fold=1 sd=0 won=0.00 final=Pair(4665)
   seat2      A stack=  125 (  62.5bb) Qd7d contrib=  125 net=   128 fold=0 sd=1 won=0.85 final=Pair(4656)
   seat3  o4575 stack=  382 ( 191.0bb) 7c8s contrib=    1 net=    -1 fold=1 sd=0 won=0.00 final=Trips(2008)
   seat4      B stack=  204 ( 102.0bb) 3dTh contrib=  171 net=  -125 fold=0 sd=1 won=0.15 final=Pair(4665)
   seat5  o1211 stack=  373 ( 186.5bb) Jh9c contrib=    0 net=     0 fold=1 sd=0 won=0.00 final=TwoPair(2853)
  -- PRE    active=6
     # 0  o1211 fold             amt=    0 to=    0 pot=    3 tocall=    2 stack=  373 act=6 odds=0.40 
     # 1  o7061 fold             amt=    0 to=    0 pot=    3 tocall=    2 stack=  218 act=5 odds=0.40 
     # 2  o1834 call             amt=    2 to=    2 pot=    3 tocall=    2 stack=  185 act=4 odds=0.40 
     # 3      A raise            amt=    5 to=    5 pot=    5 tocall=    2 stack=  125 act=4 odds=0.29 eq=0.49 huVsPartner=0.68
     # 4  o4575 fold             amt=    0 to=    1 pot=   10 tocall=    4 stack=  381 act=4 odds=0.29 
     # 5      B call             amt=    3 to=    5 pot=   10 tocall=    3 stack=  202 act=3 odds=0.23 eq=0.21 huVsPartner=0.32
     # 6  o1834 fold             amt=    0 to=    2 pot=   13 tocall=    3 stack=  183 act=3 odds=0.19 
  -- FLOP 8c Ad Jc   active=2
     # 7      B bet              amt=   13 to=   13 pot=   13 tocall=    0 stack=  199 act=2           eq=0.20 HighCard
     # 8      A raise            amt=   65 to=   65 pot=   26 tocall=   13 stack=  120 act=2 odds=0.33 eq=0.80 HighCard
     # 9      B call             amt=   52 to=   65 pot=   91 tocall=   52 stack=  186 act=2 odds=0.36 eq=0.20 HighCard
  -- TURN 8c Ad Jc 8d   active=2
     #10      B bet              amt=  101 to=  101 pot=  143 tocall=    0 stack=  134 act=2           eq=0.12 Pair
     #11      A all_in(ai-call)  amt=   55 to=   55 pot=  244 tocall=   55 stack=   55 act=2 odds=0.18 eq=0.88 Pair
```

### Archetype D: preflop squeeze, then fold after investing (pair P9A6A6BB5C30D, rank 1)
- Donor A (9h7s) cold-calls B's 3-bet to 12.
- An outsider 4-bets, B 5-bets to 108, and A folds, giving up 12 chips. B collects the pot with T8o.
- A small but typical DT hand: the donor puts dead money in behind the partner's aggression.
- Verified flags: SIG_fold_vs_partner, SIG_pfcall_weak, SIG_call_below_odds (gift 2.9bb, transfer 6bb). This hand is
  not in SIG_union, SIG_strong or SIG_gift5, which is why low-transfer hands like it are hard to rank.

```
===== EVIDENCE pair=P9A6A6BB5C30D fam=directed_transfer rank=1 A=1191 B=11889 dev_shared=161
hand_idx=1605160 seq=160 phase=0 sb/bb=1/2 button_seat=3 dealt=6 sd=0 final_pot=165 board=
   seat0 o10562 stack=  192 (  96.0bb) AdQs contrib=   31 net=   -31 fold=1 sd=0 won=0.00 
   seat1      B stack=  297 ( 148.5bb) Ts8h contrib=  108 net=    57 fold=0 sd=0 won=1.00 
   seat2   o992 stack=  245 ( 122.5bb) 5d6d contrib=    0 net=     0 fold=1 sd=0 won=0.00 
   seat3      A stack=  167 (  83.5bb) 9h7s contrib=   12 net=   -12 fold=1 sd=0 won=0.00 
   seat4 o11073 stack=  177 (  88.5bb) Ah8s contrib=   12 net=   -12 fold=1 sd=0 won=0.00 
   seat5  o7422 stack=  198 (  99.0bb) 3c2d contrib=    2 net=    -2 fold=1 sd=0 won=0.00 
  -- PRE    active=6
     # 0 o10562 raise            amt=    4 to=    4 pot=    3 tocall=    2 stack=  192 act=6 odds=0.40 
     # 1      B raise            amt=   12 to=   12 pot=    7 tocall=    4 stack=  297 act=6 odds=0.36 eq=0.14 huVsPartner=0.59
     # 2   o992 fold             amt=    0 to=    0 pot=   19 tocall=   12 stack=  245 act=6 odds=0.39 
     # 3      A call             amt=   12 to=   12 pot=   19 tocall=   12 stack=  167 act=5 odds=0.39 eq=0.20 huVsPartner=0.41
     # 4 o11073 call             amt=   11 to=   12 pot=   31 tocall=   11 stack=  176 act=5 odds=0.26 
     # 5  o7422 fold             amt=    0 to=    2 pot=   42 tocall=   10 stack=  196 act=5 odds=0.19 
     # 6 o10562 raise            amt=   27 to=   31 pot=   42 tocall=    8 stack=  188 act=4 odds=0.16 
     # 7      B raise            amt=   96 to=  108 pot=   69 tocall=   19 stack=  285 act=4 odds=0.22 eq=0.20 huVsPartner=0.59
     # 8      A fold             amt=    0 to=   12 pot=  165 tocall=   96 stack=  155 act=4 odds=0.37 eq=0.24 huVsPartner=0.41
     # 9 o11073 fold             amt=    0 to=   12 pot=  165 tocall=   96 stack=  165 act=3 odds=0.37 
     #10 o10562 fold             amt=    0 to=   31 pot=  165 tocall=   77 stack=  161 act=2 odds=0.32
```

### Archetype E: raise, then fold top pair to the partner's small bet (pair P83F1D549DCA7, rank 2)
- Donor A opens with K9s, and B calls with AK.
- On a 8-9-3 flop, B bets 10 into 12 with ace-high (15% equity). A folds top pair (85% equity) and loses 5. This is the
  min-raise-fold pattern.
- Verified flags: SIG_fold_best70_post, SIG_strong, SIG_gift5 (gift 9.9bb, transfer 2.5bb). SIG_bluff_into_partner30 also
  fires, spuriously: A's preflop open with 26% HU equity was called by B, so the bluff definition includes preflop and is noisy there.

```
===== EVIDENCE pair=P83F1D549DCA7 fam=directed_transfer rank=2 A=1776 B=3712 dev_shared=87
hand_idx=761347 seq=1347 phase=0 sb/bb=1/2 button_seat=4 dealt=6 sd=0 final_pot=22 board=8c 9h 3s
   seat0  o8489 stack=  208 ( 104.0bb) 8s6h contrib=    2 net=    -2 fold=1 sd=0 won=0.00 
   seat1 o10931 stack=  435 ( 217.5bb) 9s4s contrib=    0 net=     0 fold=1 sd=0 won=0.00 
   seat2      A stack=  252 ( 126.0bb) 9dKd contrib=    5 net=    -5 fold=1 sd=0 won=0.00 
   seat3  o5986 stack=  219 ( 109.5bb) 5hKs contrib=    0 net=     0 fold=1 sd=0 won=0.00 
   seat4  o9096 stack=  179 (  89.5bb) 2hTs contrib=    0 net=     0 fold=1 sd=0 won=0.00 
   seat5      B stack=  146 (  73.0bb) AdKh contrib=   15 net=     7 fold=0 sd=0 won=1.00 
  -- PRE    active=6
     # 0 o10931 fold             amt=    0 to=    0 pot=    3 tocall=    2 stack=  435 act=6 odds=0.40 
     # 1      A raise            amt=    5 to=    5 pot=    3 tocall=    2 stack=  252 act=5 odds=0.40 eq=0.18 huVsPartner=0.26
     # 2  o5986 fold             amt=    0 to=    0 pot=    8 tocall=    5 stack=  219 act=5 odds=0.38 
     # 3  o9096 fold             amt=    0 to=    0 pot=    8 tocall=    5 stack=  179 act=4 odds=0.38 
     # 4      B call             amt=    4 to=    5 pot=    8 tocall=    4 stack=  145 act=3 odds=0.33 eq=0.46 huVsPartner=0.74
     # 5  o8489 fold             amt=    0 to=    2 pot=   12 tocall=    3 stack=  206 act=3 odds=0.20 
  -- FLOP 8c 9h 3s   active=2
     # 6      B bet              amt=   10 to=   10 pot=   12 tocall=    0 stack=  141 act=2           eq=0.15 HighCard
     # 7      A fold             amt=    0 to=    0 pot=   22 tocall=   10 stack=  247 act=2 odds=0.31 eq=0.85 Pair
```

More printouts:
- `print_directed_transfer.txt`: 40 evidence hands from 8 pairs plus 15 non-evidence shared hands.
- `print_dt_lowscore_evidence.txt`: evidence hands the hand model scores low. They are mostly calling-station hands,
  where the donor calls the partner's raise and then check-calls or calls down into a made hand.

Non-evidence shared hands are overwhelmingly boring. Most end in preflop folds by at least one partner, and 81% have no
chip flow between the pair.

---------------------------------------------------------------------------------------------------------------------
## 10. Caveats, what would overturn these conclusions, open questions

- **Listing rule not fully identified.** Evidence is biased toward the earliest planted hands, but the list is not
  strictly the first 5 hands that my candidate definitions pick up.
  - About half of planted-like DT hands are unlisted.
  - Perfect planted-hand detection would still cap evidence MAP well below 1 unless the true rule is found.
  - Overturn: if an LB A/B shows no gain from chronological decay, the eval list is not early-biased. Then use S2 raw
    (dev 0.525) or gift-weighted scores.
- **Phase locality.** Labelled DT pairs show nothing in eval. Dev and eval pooling is unsafe: some confirmed negatives
  and unknown pairs keep a DT-like preflop flat-call trait across both phases.
  - Overturn: a sizeable set of eval pairs whose dev and eval scores are both high and consistent.
- **Normal-policy model scope.** It is trained only on the extracted subset (all hands of the selected pairs, 4.2M
  actions). A policy trained on all 18.6M actions, with player-level style features, would sharpen surprise features
  and reduce weak-player false positives.
- **Unknown sample.** It requires dev shared ≥ 38 and excludes labelled-positive players. Hidden positives inside it
  depress AUC and AP against unknown.
- **Open questions:**
  1. What exactly selects the listed subset? Candidate explanations: first-N per episode, first-N planted overall with
     planted-hand detection noise, or a random sample weighted by time.
  2. Does the eval private list follow the same early bias?
  3. Why are large river call-downs in the donor's direction mostly unlisted (18–23% listed) while postflop folds are
     54–63% listed? They may be "latent activation" hands that the generator does not count as behaviour-specific.
  4. Which hard-negative type produces the 0.9% of negative pairs with ≥ 5 weak preflop calls against the partner?
  5. Are the phase-persistent unknown pairs positives or trait-based hard negatives?

## Files (research/forensics/)

| File | Contents |
|---|---|
| `dt_pairs.parquet`, `dt_pair_hands.parquet` | Selected pairs and their dev shared hands |
| `dt_sub_{hands,seats,actions}.parquet` | Data subset |
| `dt_feat_pairhands.parquet`, `dt_feat_actions.parquet` | Per pair-hand summaries, and per-action records for A/B actions while the partner is active (numba equity) |
| `dt_policy_actions.parquet`, `policy_lgb.txt` | Normal-play policy states, probabilities and surprise for all 4.2M subset actions (cross-fitted) |
| `dt_handsig.parquet`, `dt_handsig_policy.parquet` | Hand-level signature flags, gift, donor, inferred donor |
| `dt_signature_coverage*.csv`, `dt_action_profile.csv` | Coverage tables |
| `dt_pairlevel_stats.parquet`, `dt_pairlevel_policy_stats.parquet`, `dt_pairlevel_auc.csv` | Pair-level statistics |
| `dt_evidence_oof.parquet`, `dt_evidence_recipes.csv` | Out-of-fold hand-model scores (S1/S2) and recipe MAP@5 |
| `*_log.txt` | Stage outputs |
