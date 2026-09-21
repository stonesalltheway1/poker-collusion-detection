> **STATUS NOTE (2026-09-18).** This is the DAY-0 plan and remains accurate about the data, the metric and the
> forensics. Several of its *strategic* conclusions have since been superseded by measurement — notably the evidence
> estimates (§4, §5) and the "what limits us" reasoning. For current state read `PLAN.md`, for the evidence listing
> rule `research/forensics/listing_rule.md`, and for every measured dead end `research/CLOSED_LANES.md`.
> Superseded here: eval evidence MAP is .699 (not ~.28); pair AP is .973; ~14% of eval positives are not served by
> 3-family routing; chronology-aware evidence was LB-confirmed (+.0223) and the full listing rule added +.0066.

# SYNTHESIS: attack plan for "Detect Suspicious Value Transfers in Poker"

Written 2026-09-15 (D0, about 20:00 UTC). Deadline **2026-09-20 22:00 UTC**. 5 subs/day; pick 2 finals by hand on the site.

Sources: the recon outputs, 6 notebook NOTES files, LITERATURE_poker/ml, DOMAIN_signals, TOOLS, and `forensics/{structure,directed_transfer,soft_play,coordinated_isolation}.md`.

Spot-check scripts: `C:\Users\Eric\AppData\Local\Temp\claude\F--kaggle-competitions-suspicious-poker\24eb27e2-d32b-410a-971a-6518069075a2\scratchpad\spot{1,2,3}.py` (scratchpad, not in the repo). Each runs in under 10 s with duckdb on 3 threads. Their numbers are marked **VERIFIED** below.

LB context (snapshot 19:56 UTC, 253 teams):
- #1 0.938, #2 0.930, #10 0.906, #29 0.884.
- The best public notebook scores **0.695**.

---------------------------------------------------------------------------------------------------------------------

## 0. The campaign in one paragraph

The planted behaviours can be detected almost perfectly with **hand-level, poker-semantic signatures computed inside the phase being scored**:
- dev pair AUC 0.99 to 1.00 for each of the three disclosed families;
- the public notebooks sit 0.25 below the frontier because they use aggregate chip and action counts and ignore phase-locality and exposure.

Collusion is **phase-local**:
- dev-labelled pairs are inert in eval;
- eval positives are active only in eval.

So the whole pipeline follows one rule: *train on dev-phase features of labelled pairs, score eval pairs on eval-phase features only, and use the other phase only as a clean baseline*.

How the 30 points above the public notebooks split up:
1. **Pair AP (70%)**: family detectors turned into count-aware pair statistics, directedness against each player's own baseline, and one pooled GBDT.
2. **Evidence MAP@5 (20%)**: gated, family-routed hand rankers trained on listed evidence. Scored for *every* eval pair, because the metric weights every true positive equally whatever its risk rank.
3. **Behaviour (10%)**: argmax family for every pair, never `none`.
4. **other_coordination**: counts in 90% of the weight and is unknown. It gets a family-agnostic branch and is hunted by unsupervised mining on the eval phase.

---------------------------------------------------------------------------------------------------------------------

## 1. The 15 most important facts

| # | Fact (numbers) | Status |
|---|---|---|
| 1 | **Metric mechanics** (host code, `src/metric.py`). final = 0.70·pairAP + 0.20·evidence MAP@5 + 0.10·behaviour macro-AP over DT/SP/CI.<br>• AP sorts with a stable mergesort on a frame sorted by pair_id, so **tied scores are broken by pair_id**.<br>• Behaviour class score = risk if predicted==family, else 0. Zeros tie and are ordered by pair_id.<br>• Evidence AP runs over **every true positive pair, regardless of its risk rank**. The denominator is min(\|relevant\|,5), and a true positive with no hits scores 0.<br>• risk must be in [0,1]; evidence hand IDs must not repeat within a pair. | VERIFIED (code read) |
| 2 | **Evidence file.** 1,817 hands for 372 positives, 100% in the dev phase. Pairs listing 5 / 4 / 3 hands: 340 / 21 / 11. | VERIFIED |
| 3 | **Collusion is phase-local.** Labelled positives vs negatives, dev → eval (eval requires eval shared ≥ 38; 281 positives):<br>• both-VPIP rate AUC 0.882 → **0.470**<br>• SP both-enter rate 0.926 → **0.467**<br>• CI weak-preflop-raise rate 0.999 → **0.500**<br>Also reported: Spearman of dev vs eval pair score 0.16; CI rate r = 0.08; labelled DT donors call the partner's raise 0.207 in eval vs a population rate of 0.200. | VERIFIED (+ REPORTED) |
| 4 | **Eval positives are active in eval; hidden positives also sit in dev "unknown".** Crude CI flag = a pair member raises preflop with a bottom-30% class, rate ≥ 0.05 per shared hand and ≥ 5 hands:<br>• 150 eval pairs, measured on eval hands (82 at rate ≥ 0.08);<br>• in dev: 67 of 92 labelled CI pairs **plus 59 unlabelled pairs**, measured on dev hands.<br>Both crude counts include colluder-with-non-partner spill-over. The forensic lift-based version, which removes it: 108 eval-only CI-like pairs, 39 unlabelled dev CI-like pairs. | VERIFIED (crude) / REPORTED |
| 5 | **Directed transfer is always a one-way chip flow.** 725/725 DT evidence hands have donor net < 0 and receiver net > 0, against SP 83.7% and CI 43.9%. The direction flips in 0/148 pairs. The donor is player_1 in 79/148, so pair orientation carries no information. | VERIFIED |
| 6 | **Soft play is forced joint participation.** Both partners' first preflop action is not a fold in **97.6%** of SP evidence hands, against **10.0%** of confirmed-negative shared hands (DT evidence 94.9%, CI 49.6%). | VERIFIED |
| 7 | **Coordinated isolation is trash raising with the partner behind.**<br>• Crude definition (preflop aggression with a bottom-30% class): 39.6% of CI evidence vs 0.32% of negative hands; pair AUC 0.999.<br>• Forensic definition (population P(raise \| class, position, prior raisers, prior callers) < 0.02): 69.4% vs 0.39% (178×). First pair aggression comes from UTG in 73.7%.<br>• excess_trash AP is 0.992 at 2% prevalence. | VERIFIED (crude) / REPORTED |
| 8 | **The evidence list follows a structured, early-biased rule.** evidence_rank equals chronological order in **81/92 CI**, 57/148 DT and 36/132 SP pairs; chance is about 1/120 for 5 items.<br>For CI, P(listed) by chronological rank inside the "UTG weak open" set is 0.92, 0.85, 0.69, 0.34, 0.13, then 0 (reported). | VERIFIED / REPORTED |
| 9 | **Exposure.**<br>• Every labelled pair has dev shared ≥ 57 (positive median 112, negative 108.5). The same pairs share a median of 66.5 / 65 hands in eval.<br>• Eval pairs: min 38, median 76, eval/dev median ratio 0.642.<br>• At fixed thresholds, clean pairs cross high cutoffs about 3× more often in eval (FPR 0.61% vs 0.13%). | VERIFIED (dev/labelled) / REPORTED (eval) |
| 10 | **Hand time is legitimate gameplay data.** `hand_seq` is the rank of `started_at` within the table. All 2,000,000 (table, ts) values are distinct, so the hand_id tie-break is never used. The median gap between hands is 76 s. | VERIFIED |
| 11 | **The "hard negatives" are effectively a random clean sample.** On 115 pair, player and metadata features, max \|AUC(neg vs unknown) − 0.5\| = 0.035; a LightGBM gets neg-vs-unknown AUC 0.494. Tilt, weak and similar-strategy personas occur population-wide. However, 190/1,488 negatives contain a positive player: colluder-with-non-partner spill-over. | REPORTED |
| 12 | **The eval universe rule reproduces exactly:** within-table pair, eval shared ≥ 38, not a labelled pair, no labelled-positive player. That gives 112,569 candidates for 112,540 actual pairs; the 29 missing ones show no pattern. 37.8% of eval pairs contain a labelled-negative player. **These are file-construction artifacts: never use them as features.** | REPORTED |
| 13 | **Estimated eval positives: about 350 to 600** among the three known families only (prevalence 0.3–0.5%), plus an unknown other_coordination share.<br>• Public 30% ≈ 100–180 positives, so public pair-AP SE ≈ 0.02.<br>• Paired evidence-only A/B tests are much sharper, SE ≈ 0.005 in final score, because the same positives are scored in both runs. | REPORTED (estimate) |
| 14 | **Seats, co-presence, sessions and metadata carry no signal** (AUC 0.48–0.53). Seats are redrawn every hand and partners are adjacent at the random rate (0.41 vs 0.40). Metadata is host-allowed but useless, so include it at most as a cheap feature. | REPORTED |
| 15 | **More planted hands exist than are listed.**<br>• DT: about 780 unlisted planted-like hands vs 725 listed.<br>• SP: about 9.4 excess planted-like hands per pair, 4.5 of them listed.<br>• CI: about 15 signature hands per pair in dev vs 5 listed.<br>Detection is not the bottleneck: the DT hand model has AUC 0.998 vs negative hands. **Guessing which planted hands were listed is.** Dev evidence proxies: DT 0.525–0.554, SP 0.558–0.638, CI 0.464–0.686; the higher number of each pair uses chronology. | REPORTED |

Supporting facts used below:
- Boards are revealed only up to the street reached (host-confirmed); 46.7% of hands have no flop.
- Players sit at exactly one table, so table folds are also player folds.
- Per-table positives look Poisson (λ ≈ 0.9).
- Host getting-started notebook: "`other_coordination` is reserved for evaluation". Expect it **only in eval**; mining dev unknowns will not find it.

---------------------------------------------------------------------------------------------------------------------

## 2. What the planted behaviours are (hand-level definitions)

### 2.0 Common generator model (inferred)

A positive pair is activated in **one phase**. Inside that phase a per-pair scenario fires on a roughly steady fraction of shared hands:
- CI ≈ 13%;
- SP ≈ 8% excess;
- DT persistent (the flat-call trait), with additional 2–5 bursts of gift hands.

The scenario **forces the actions whatever the cards**: trash raises, forced entry plus passive responses, folding winners and calling with dead hands. The listed evidence is up to 5 of those planted hands, chosen with an early bias (CI ≈ "first ~5 scenario hands").

Roles:
- **DT** is directional: a fixed donor gives to a fixed receiver.
- **SP and CI** are symmetric: either partner can take either role.

Coverage columns in the tables below:
- **ev** = coverage of the family's listed evidence hands.
- **neg** = rate on confirmed-negative pair-hands.
- **same-pair non-ev** = rate on the same pairs' other dev hands (a high value means unlisted planted hands).

### 2.1 directed_transfer (148 pairs, 725 evidence hands)

The donor gives up value to the receiver's aggression.
- Donor's last action: fold 74.5% (flop 41.9%), call 17.9%, bet/raise 6.8%.
- Donor folds facing the receiver have mean HU equity 0.714 (negatives 0.339). River folds average 0.80; river calls lose outright.
- The hand ends heads-up between them in 78.8%. Median transfer 6.25bb, mean 25.7bb.
- **Persistent trait:** facing the receiver's preflop raise, the donor flat-calls 71.1% and 3-bets 2.8% (negatives 19.4% / 9.7%). The rate is flat over dev time.

| signature | definition | ev | neg | same-pair non-ev |
|---|---|---|---|---|
| SIG_flow (gate) | net(donor) < 0 and net(receiver) > 0 | **1.000** | 0.157 | 0.189 |
| SIG_union | flow and (donor folds to partner with HU eq ≥ .5, or calls with eq ≤ .25, or bets/raises with eq ≤ .30 and gets called) | 0.897 | 0.039 | 0.083 |
| SIG_strong | flow and (postflop fold to partner with eq ≥ .70, or turn/river call with eq ≤ .15) | 0.623 | 0.011 | 0.042 |
| SIG_gift5 | flow and summed donor EV gift > 5bb (definitions in `forensics/directed_transfer.md` §2) | 0.790 | 0.019 | 0.067 |
| SIG_pfcall_weak | flow and donor calls partner's preflop raise with normal-policy p(call) ≤ .15 | 0.491 | **0.0024** | 0.037 |
| HandModel S2 (LightGBM, OOF) | continuous | AUC 0.998 vs neg hands | | AUC 0.987 vs same-pair |

Pair level (dev):
- **pfcall_weak_max**: AUC 0.994 / AP 0.950 vs negatives; AUC 0.991 DT vs SP+CI.
- gift5_max_rate: AUC 0.977.
- Mazrooei max-direction impact: AUC 0.968.
- Donor inference (larger summed EV gift) matches the evidence donor in 97.3% of DT pairs.

Unexplained:
1. **The listing rule.** Evidence is early-biased but not strictly the first 5. Large call-downs are only 18–26% listed; postflop folds are 54–63% listed.
2. Whether the eval list follows the same early bias.
3. Why 0.9% of negatives show ≥ 5 weak flat-calls, and why that trait persists across phases (hard-negative personas).

### 2.2 soft_play (132 pairs, 632 evidence hands)

Two parts:
1. **Forced joint participation.** Neither partner's first preflop action is a fold (100%); both enter voluntarily in 97.6%; both see the flop in 93%.
2. **A direct confrontation answered passively, regardless of strength.** A partner bets or raises and the other calls or folds; only 1.8% of evidence hands contain a raise-back. Separately, strong hands are checked while the partner is active.

| signature | definition | ev | neg | same-pair non-ev |
|---|---|---|---|---|
| S_G0 (gate) | no partner's first preflop action is a fold (unless facing the partner's raise) | **1.000** | 0.219 | 0.263 |
| S_G1 | both partners' first preflop action is call, raise, all-in or BB check | 0.976 | 0.100 | 0.204 |
| S_PL | G1 and ≥ 1 action facing the partner's aggression and no raise against the partner in the hand | 0.946 | 0.060 | 0.105 |
| S_strong_check_pact | postflop check (to_call = 0) with partner active and made-hand hs ≥ .8 | 0.459 | **0.0079** | 0.045 |
| S_PL_and_softresp | S_PL and (strong check, or postflop fold/call to partner, or missed-raise ≥ .3, or preflop call of partner raise with eq ≥ .55) | 0.881 | 0.039 | 0.081 |
| S_llr_gt4 | OOF hand action-LLR (SP evidence vs NEG contexts) > 4 | 0.495 | 0.005 | 0.054 |

Pair level:
- **top5_llr**: AUC 0.991 vs negatives, 0.993 vs unknown, 0.923 vs DT, 0.971 vs CI. AP ≈ 0.83 at simulated 1% prevalence.
- strong_check rate: SP vs DT 0.943.
- Raises vs partner per both-in-pot hand: SP vs CI 1.000.

Unexplained:
1. The listing criterion. The best listed-vs-unlisted cues are the size of the passed-up raise (`max_miss`, AUC 0.744), a direct confrontation, and earliness.
2. 5% of evidence falls outside S_PL.
3. The generator's strength variable (hs thresholds are proxies).
4. 1.4% of negatives look SP-like at pair level.

### 2.3 coordinated_isolation (92 pairs, 460 evidence hands)

A partner raises preflop with a hand the population almost never raises, usually as first raiser from UTG with the partner still to act (97.2%). The partner then folds (61.3%; preflop in 75% of those) or re-raises over it (36.3%, whipsaw/squeeze). Third players fold (all four fold in 49%).
- CI members trash-raise in 6.97% of hands with the partner seated vs 0.25% without (negatives 0.20% / 0.20%).
- Activation is uniform over the phase, not episodic.

| signature | definition | ev | neg | same-pair non-ev |
|---|---|---|---|---|
| S1 trash raise | pair preflop aggression with population p_aggr(class, position, #prior aggr, #prior callers) < .02 (`ci_baseline_pre_pos.parquet`) | 0.694 | **0.0039** | 0.097 |
| S7 | S1 and (partner folds after partner aggression, or partner re-raises) | 0.678 | **0.0026** | 0.094 |
| S14 UTG weak open | first pair aggression by UTG with p_aggr < .3 | 0.626 | 0.011 | **0.029** |
| S15 (structural template) | first pair aggression from UTG/HJ/CO, partner still to act, then partner fold or re-raise | 0.939 | 0.131 | 0.202 |
| S6 sandwich | member raises, a third player calls or raises, the other member re-raises | 0.185 | 0.0056 | 0.031 |

Pair level:
- rate_C02 AUC 1.000 vs negatives and unknowns.
- **excess_trash** = Σ over directions of (trash with partner seated − n_with × own rate without partner). AP 0.992.
- mutual_top_lift: true for 90.2% of CI pairs and 0% of negatives.
- CI vs DT: rate_Cfap AUC 0.986.
- At 38 shared hands, AP falls to 0.88–0.90, so use count-aware statistics.

Unexplained:
1. Why listing prefers UTG (74% of evidence vs 14% of unlisted trash raises).
2. The 26% of evidence that is not UTG.
3. Whether the 39–59 unlabelled dev CI-like pairs are hidden CI positives. They almost surely are; exclude them from negatives.

### 2.4 other_coordination (hidden; never in public labels)

Nothing is known about it beyond the description: "a 4th undisclosed mechanism". Per the host notebook it is **reserved for evaluation**.

Candidate mechanisms from the literature and domain notes:
- information sharing / ghosting: an action that depends on the partner's hidden cards;
- best-hand-playing: the weaker partner folds early in multiway pots;
- blind-steal agreement;
- mutual pot-building against a third player.

It is scored in pair AP and in evidence. If a fraction φ of eval positives is missed, AP ≈ (1−φ)·AP_known.

---------------------------------------------------------------------------------------------------------------------

## 3. Architecture

```
actions/seats/hands ──► equity replay (eval done; dev full: exp002)
                     ──► label-free population baselines: preflop p_aggr table (CI), normal-play policy (exp003)
                               │
                               ▼
         PAIR-HAND ENGINE (numba, per phase/window): one row per (unordered pair, shared hand)  [exp001]
           DT/SP/CI signature flags + continuous scores, generic impact/surprise, gates, pot_bb, hand_seq
                │                                              │
                ▼                                              ▼
   PAIR FEATURES (per phase; count-aware)          EVIDENCE RANKERS (dev positives, target = listed)
   rates, binomial LLR/z vs population,            gate → family rankers → posterior mixture → top-5
   top-k of OOF hand-model scores, lift vs         (all 112,540 eval pairs)
   own/other-co-player baseline, other-phase DiD
                │
                ▼
   PAIR MODEL (LightGBM + LR-on-LLR, table-fold OOF) ─► risk = rank/(N+1), tie-break = secondary score
   BEHAVIOUR MODEL (multinomial on positives) ─► argmax family for all; other_coordination override (generic-only)
```

### 3(a) Hand-level detectors, rules first, then learned

1. **Gates**, used both as candidate filters and as features. Measure recall on dev evidence first; it must be ≥ 99.5%.
   - SP/DT gate G0 (100% SP and DT evidence; removes 78% of hands).
   - DT flow gate (100%; removes 81–84%).
   - CI template S15 (93.9%) or S3∧S4.
   - Union for family-agnostic candidates: G0 ∨ S15 ∨ "one partner acts facing the other's aggression".
2. **Rule scores**, deterministic and label-free except for thresholds read off dev:
   - DT: SIG_union, SIG_strong, SIG_gift5, SIG_pfcall_weak, plus EV-gift in bb and the inferred donor.
   - SP: S_PL, S_strong_check_pact, S_U1, missed-raise (NEG-calibrated P(raise) passed up).
   - CI: S1, S7, S14, S16, sandwich, partner re-raise, third-party folds facing the pair.
   - Generic: Mazrooei per-action value impact dV (luck-free, sums to 0), both directions; donor and receiver policy surprise (−log p(action)) against the partner.
3. **Label-free baselines**, built on all 2M hands:
   - preflop P(aggr | class, position, #prior aggr, #prior callers), already built;
   - a normal-play LightGBM policy (fold/check/call/agg from public state plus own cards; 2-fold table cross-fit) applied to both phases (exp003).
   Label-free means no leakage and identical treatment of dev and eval.
4. **Learned hand models**, all cross-fitted on the frozen table folds:
   - *Planted-hand detectors* per family and pooled. Positives = listed evidence; negatives = negative-pair hands plus same-pair non-listed hands (DT "S2" style). Their OOF scores feed pair aggregation.
   - The *SP action-LLR table* (smoothed counts: SP evidence vs NEG per context) must be OOF for training rows and refit on all dev for eval.

### 3(b) Pair-level features (computed per phase; eval pairs use eval hands only)

1. **Count-aware family statistics.** For each signature: count n, rate n/shared, and **binomial LLR / z against the population base rate** p0 (e.g. S1 p0 = 0.004; S_PL p0 = 0.060; SIG_strong p0 = 0.011). Eval pairs have 38–419 shared hands, so raw rates are noisy at low exposure and LLR handles that.
2. **Top-k of hand-model scores** (k = 1, 3, 5), top-5 minus the pair median, and a mixture-GLR. Never the mean of hand scores (simulated AP 0.02–0.43).
   - SP top5_llr (0.991), DT pfcall_weak_max and rate (0.994), CI excess_trash (AP 0.992).
3. **Directedness against baselines.** These kill the spill-over negatives (a colluder with a non-partner) and population personas (tilt, weak, similar strategies).
   - *Own baseline*: X's event rate with Y vs X's rate with the other co-players in the same phase (excess_trash style), with EB shrinkage `(a+O)/(a+E0)`.
   - *Double-centring*: S(X,Y) − median over Z of S(X,Z), and the same for Y; mutual-top-lift flags; the rank of Y among X's 29 co-players.
   - *Other-phase DiD* (phase-locality is VERIFIED): stat_phase − stat_other_phase(w2000-matched).
     - For dev training rows the other phase is eval; for eval pairs it is dev. Symmetric and legitimate, since it uses only poker activity.
     - Positives are active in one phase only. Phase-persistent traits (38% of weak-call negatives keep the trait) look like hard negatives.
   - *Player persona* from the other phase (vpip, pfr, aggression; dev↔eval Spearman 0.71–0.88) as context.
4. **Episodic.** Max rolling-window LLR over shared-hand windows (25/50/100) and the nearest-neighbour gap of gift > 5bb hands (DT median 4 vs negatives 39). Mainly for DT; CI and SP are uniform.
5. **Generic, family-agnostic.**
   - Mazrooei total/marginal impact per 100 co-live hands (DT AUC 0.968 raw).
   - Policy-surprise excess against the partner vs others.
   - Both-enter z, faced-partner-aggression passivity z.
   - Conditional dependence of X's continue/raise on Y's hidden HU equity (score test z, double-centred).
6. **Exposure.** log(shared_hands) as a feature. **No raw counts without their exposure.**
7. **Metadata** same_* and age gap: one cheap ablation, expected ≈ 0.
8. **Banned.** Anything from ID strings, file or row order, eval-file membership (e.g. "contains a labelled-negative player", eval pairs per table, the 29 missing pairs), or labels-file graph structure. See §7.

### 3(c) Pair model: PU with near-clean negatives, not "hard-negative" machinery

The labelled negatives equal the unknown pairs in distribution (fact 11), so the telescoping P-vs-HN × (P+HN)-vs-U construction is unnecessary.

**Training set (dev phase):**
- 372 positives;
- 1,488 labelled negatives (weight 1);
- a **U-sample**: all dev pairs with shared ≥ 38 and no labelled-positive player, about 40–60 per table (≈ 16–24k), weight 0.3–0.5, **after spy-removal** of U pairs flagged by strong family rules (CI mutual-top-lift ∧ excess ≥ 5; pfcall_weak_max ≥ 6 ∧ gift5 ≥ 3; top5_llr above the SP p10). About 140–220 hidden dev positives exist.
- Optional: the eval-phase rows of labelled pairs (281 inert positives + 1,149 negatives) as clean, eval-exposure negatives (exp011).

**Exposure augmentation:** training rows from 2000-hand dev windows (0–1999 and 1000–2999) in addition to or instead of full-dev rows, grouped by table (exp010).

**Models:**
- **R1** LightGBM binary: num_leaves 15, min_data 20, feature_fraction 0.5, bagging, lr 0.03, a fixed number of rounds chosen on the CV mean (no early stopping on the validation fold), 5 seeds.
- **R2** L2 logistic regression on ≈ 20 standardized family LLR/z statistics. Robust to exposure shift; it acts as a guard.
- Fusion: rank-average or a cross-fitted LR on clipped logits. **No nonlinear stackers.**

**Output:** risk = rank/(N+1). Break ties with a secondary legitimate score (R2 or the generic branch), **never pair_id**.

### 3(d) Behaviour

- A multinomial (LightGBM depth ≤ 4, or LR) trained on the 372 positives only, with class-balanced weights, on family-discriminating features:
  - pfcall_weak_max_rate (DT vs SP+CI 0.991), gift5_asym_rate;
  - strong_check rate, raises vs partner per both-in-pot hand;
  - rate_Cfap, sandwich rate, rate_rr, S14/S16 rates.
- **predicted_behavior = argmax for all 112,540 pairs.** Gating to `none` is a strict loss, as every public notebook shows.
- **other_coordination override:** assign it only when risk is high because of the generic branch while every family posterior×detector is weak. A true other_coordination pair labelled with a family becomes a high-ranked false positive in that family's list. Tune on a leave-one-family-out simulation; the expected effect is small but positive.

### 3(e) Evidence selection (for every eval pair)

1. **Candidates** = eval-phase shared hands passing the union gate. Always output 5 distinct hands, padding with the best non-gated hands.
2. **Rankers**, per family: LightGBM binary on dev positive pairs' shared hands, target = listed. The in-pair objective (listed vs unlisted hands of positive pairs) beats "evidence vs negative hands": SP 0.558 vs 0.409.
   - Features: family hand features, within-pair percentiles (the eval candidate pool is ≈ 0.64× the dev pool), pot bb, the inferred donor (DT), position of the first pair aggressor (CI: fpa_ord, fpa_p), max_miss (SP).
   - A lambdarank/xendcg rank-average and a graded target `6 − evidence_rank` are ablations (exp016).
3. **Family mixture:** score(h) = Σ_f P(f | pair)·pct_f(h). Validate with OOF *predicted* posteriors so family errors are priced in.
4. **Chronology within the phase** (rule-sensitive, see §7): prior_cand = number of earlier candidate hands of the pair in the phase, relative position q, and DT's decay p·exp(−K/2).
   - Dev proxy gains: CI 0.464 → 0.686, SP 0.558 → 0.638, DT 0.525 → 0.554. About +0.09 MAP, i.e. ≈ +0.018 final score.
   - Implement it as learned features, not as a hard-coded "first 5" rule. Adopt it only after an LB A/B (sub002 vs sub003) and Eric's sign-off.
5. **Fallbacks** with no model:
   - DT: SIG_strong ∨ SIG_pfcall_weak in the inferred donor's direction, then SIG_gift5.
   - SP: RULE = 10·S_PL + 4·max_miss + 1.5·strong_check + n_call_resp + 0.3·llr_post (0.436).
   - CI: S14 hands, then S16, then S1 (0.586 with earliest-first; 0.320 without).
6. For **other_coordination-routed pairs**, rank by the generic hand anomaly: |dV impact| between the partners × policy surprise.

### 3(f) other_coordination handling

1. **Generic branch** (exp014): a pair model on family-agnostic features only (impact, surprise excess, both-enter / passivity z, conditional dependence on the partner's hidden equity, pooled planted-hand top-k).
   - **Leave-one-family-out (LOFO) validation:** train on two families, score the held-out family on the pool. Admit to fusion only if LOFO AP is meaningfully above chance and fusion does not drop view-C AP by more than 0.005.
2. **Unsupervised eval-phase mining** (exp013), using the phase-contrast instrument:
   - For every statistic S, N_excess(t) = #eval pairs with S_eval > t − #same pairs with S_dev,w2000 > t. Each eval pair's dev phase is a clean null.
   - Excess not covered by the three family detectors' flagged pairs is the other_coordination footprint. It also estimates how many eval positives each detector captures (a label-free validation).
   - Read the hands of the top 30 unexplained pairs and write a signature if a cluster appears.
3. **LB probe:** sub007 = best model + generic branch fused. Keep it if LB pair-AP gain > 0.01; an uncovered 4th family would show up as a large gain.

---------------------------------------------------------------------------------------------------------------------

## 4. Validation protocol and CV↔LB calibration

### Folds and hygiene
- `data/folds_tables_5.csv` (md5 322a34c8…) stays frozen: 80 tables per fold; positives per fold 91/75/58/62/86; CI 24/12/15/17/24.
- Every learned component is cross-fitted on these folds: hand models, SP LLR table, pair model, behaviour model, evidence rankers. Label-free baselines (p_aggr table, policy, equity) need no cross-fitting.
- No early stopping on the validation fold; pick round counts on the CV mean. Repeat 3×5 seeds for decisions below +0.01.

### Scoring views (log all, decide on the composite)

| view | what | use |
|---|---|---|
| A | labelled-only pair AP (372 positives vs 1,488 negatives, 20% prevalence) | sanity check only; saturates around 0.98 |
| B | pool AP: positives vs negatives + all U pairs with shared ≥ 57 on held-out tables (0.27% prevalence) | biased down by hidden dev positives; a perfect detector ≈ 0.63–0.70 |
| **C** | **pool AP on held-out tables using `dev_w2000` features, shared ≥ 38** (328 positives / 1,319 negatives / 119k U) | **primary pair-AP decision metric**; carries both the exposure and prevalence shifts |
| C′ | view C with spy-flagged U removed | report next to C; it is circular, so never the only metric |
| PU-est | Jain–White–Radivojac corrected AP on the **eval** score distribution (`pu_ap_estimate` in LITERATURE_ml §3.4) with the mixture-proportion α | CV↔LB bridge for each submission |
| phase-contrast | excess counts of eval over dev(w2000) per detector (§3f) | label-free estimate of eval positives captured |
| evidence | host MAP@5 on OOF over the 372 positives, per family; full-dev candidates, plus a w2000 variant restricted to hand_seq < 2000 (363 pairs) | lower bound for eval (smaller candidate pools in eval) |
| behaviour | host macro-AP on view C with OOF posteriors | |

**Composite CV** = 0.7·AP_C + 0.2·MAP5_OOF + 0.1·BehAP_C.

**Accept rule:** composite improves by at least +0.003 with a stable sign across 3 seeds, **and** a paired table-bootstrap P(better) ≥ 0.8.

### What to trust
- Trust: view C, the per-family OOF MAP@5, and **paired LB A/B tests that change one component at a time**.
  - Evidence-only changes are measured exactly on the fixed public positive set; SE ≈ 0.005 in final score.
- Distrust: labelled-only AP, single-fold numbers (CI fold 1 is thin), and LB deltas below 0.015 on pair-AP changes (SE ≈ 0.02).

### Calibration plan
The ledger gets one row per submission:

| field | source |
|---|---|
| exp | experiment id |
| AP_A, AP_C, PU-est eval AP, fitted α | CV |
| MAP5_OOF (DT/SP/CI) | CV |
| BehAP_C | CV |
| predicted LB = 0.7·PU-est + 0.2·MAP5_OOF + 0.1·BehAP_C | computed |
| actual LB, residual | LB |

Rules:
- Residuals should be stable across submissions. A residual that jumps means a structural surprise: SCAR failure, other_coordination share, or exposure. Investigate before continuing.
- Sub001 fixes the pipeline offset; sub002 and sub003 fix the evidence offset.

---------------------------------------------------------------------------------------------------------------------

## 5. Day-by-day plan

Capacity: at most 3 concurrent agents at ≤ 3 threads and ≤ 5 GB RAM each on the local box. Kaggle T4 for the policy model (XGBoost gpu_hist on 18.6M actions) if the local run exceeds 2 h. Submission days reset at 00:00 UTC.

### D0: Tue 15 Sep (tonight). Foundations and first calibration point

| exp | hypothesis | expected gain | cost |
|---|---|---|---|
| **exp001** pair-hand engine v1 (`src/hand_engine.py`, numba). Merge the forensics code into one pass over any (phase, window, pair-set): gates, DT/SP/CI signatures, EV gift, impact dV, surprise hooks, pot_bb, hand_seq. Write pair-hand parquet per table plus a pair aggregation. | Reproduces the forensics coverage within ±1 pt on dev (acceptance test) | enabler for everything | 4–6 h dev; runtime < 20 min for both phases |
| **exp002** full dev equity replay (`src/hand_replay.py` on 1.2M dev hands), background | — | needed for U-sample and DiD features | ≈ 30–40 min at 3 threads |
| **exp003** label-free normal-play policy on ALL actions (both phases), 2-fold table cross-fit; outputs p(fold/check/call/agg) per action | pfcall_weak_max AUC stays ≥ 0.99 when the policy is not trained on the subset | +DT pair AP; generic surprise features | 1–2 h (or a T4 kernel) |
| **exp004** generic baseline: LightGBM on `pair_generic_{dev,dev_w2000,eval}` rates + multinomial behaviour (argmax for all) + rule evidence fallbacks (no chronology) → **sub001** | Pipeline and format work; first CV↔LB point (view C ≈ 0.51) | calibration | 1–2 h |

### D1: Wed 16 Sep. Detector pair model v1 and evidence A/B

| exp | hypothesis | expected gain (final score) | cost |
|---|---|---|---|
| **exp005** pair features v1 (count-aware family statistics, excess vs own baseline) + R1 LightGBM, PU training set (§3c), OOF views A/B/C | view-C AP 0.51 → ≥ 0.80 | +0.15–0.25 vs sub001 | 3 h |
| **exp006** behaviour multinomial on family features, argmax for all | family accuracy ≥ 0.93; BehAP ≈ 0.9·pairAP | +0.02–0.04 | 1 h |
| **exp007** evidence rankers v1 (gates + per-family LightGBM, in-pair target, within-pair percentiles, family mixture), **no chronology** | OOF MAP@5 ≥ 0.50 (DT .52 / SP .55 / CI .46) | +0.03–0.05 vs rule fallbacks | 3 h |
| **exp008** exp007 + within-phase chronology features | OOF MAP +0.08–0.10 | +0.015–0.02 **if the eval listing mirrors dev** | 1 h |
| **exp009** SP OOF action-LLR (top5_llr, z_llr>2) + DT pfcall_weak (via exp003) into the pair model | view C +0.02–0.05 | +0.015–0.035 | 2 h |

Submissions:
- **sub002** = exp005 + exp006 + exp007.
- **sub003** = identical except exp008 evidence (pure A/B).
- sub004 = + exp009 if ready.

### D2: Thu 17 Sep. Exposure, PU hygiene, directedness, the hunt for the 4th family

| exp | hypothesis | expected gain | cost |
|---|---|---|---|
| **exp010** exposure-matched training (2000-hand dev windows ×2) vs full-dev; binomial LLR vs rates | view C +0.01–0.03; smaller LB residual | +0.01–0.02 | 2 h |
| **exp011** PU hygiene: spy-remove flagged U; add labelled pairs' eval-phase rows as clean negatives | view C′ +0.01–0.02; fewer true positives pushed down | +0.005–0.015 | 1.5 h |
| **exp012** directedness: double-centring vs the other 28 co-players, mutual-top-lift, other-phase DiD | kills spill-over and persona false positives; view C +0.01–0.03 | +0.01–0.02 | 3 h |
| **exp013** other_coordination mining on the eval phase (phase-contrast excess of generic statistics not explained by family flags; read the top 30 pairs' hands) | a 4th mechanism with a clear signature exists | 0 to +0.10 (if φ ≈ 10–20% and it is found) | 4–6 h |
| **exp014** generic branch + LOFO validation | LOFO AP well above chance | +0 to +0.05 | 2 h |

Submissions:
- sub005 = exp010 + exp011.
- sub006 = + exp012.
- sub007 = + generic branch (other_coordination probe).
- sub008–009 in reserve for exp013 outcomes.

### D3: Fri 18 Sep. Hand-model aggregation, evidence v2, ensembles

| exp | hypothesis | expected gain | cost |
|---|---|---|---|
| **exp015** cross-fitted planted-hand detectors (per family + pooled) → top-k, top5 − median, mixture-GLR as pair features | detector separation dominates the choice of aggregator | +0.01–0.03 | 3 h |
| **exp016** evidence v2: lambdarank + binary rank-average; graded target; spy-cleaning of non-listed high-scorers (ablation); family-posterior mixture | OOF MAP +0.02–0.04 | +0.005–0.008 | 3 h |
| **exp017** other_coordination behaviour override (generic-high ∧ family-low), tuned by LOFO simulation | fewer family false positives at the top | +0.003–0.01 | 1 h |
| **exp018** seeds/bags (5×5) + R2 LR-on-LLR + XGB/CatBoost, cross-fitted LR fusion | stability; variance reduction | +0.003–0.01 | 2 h |

Submissions: sub010–014, each changing one component relative to the current best.

### D4: Sat 19 Sep. Scale and select
- **exp019** endgame: paired bootstrap (resampling tables) over the composite for all candidates (`scripts/endgame_select.py`); 3 seeds of final models; freeze the two candidate pipelines by 23:00 UTC.
- **exp020** submission audit: 5 distinct eval-phase shared hands for all 112,540 pairs, no ties in risk, valid behaviour labels, and a rerun of `validate_submission`.
- Subs 015–019: final-candidate calibration only (for example with and without DiD, with and without the generic branch).

### D5: Sun 20 Sep (deadline 22:00 UTC)
- No new features. Last-day changes only with bootstrap P > 0.95.
- At most 3 confirmation subs.
- **Tick the 2 finals manually by 21:00 UTC:**
  - slot 1 = best composite with LB consistent with CV;
  - slot 2 = the strongest file that does not share slot 1's riskiest component (chronology evidence or the generic branch/override).

### The first 3 submissions

1. **sub001 (D0)**: generic-rate LightGBM (exp004) + argmax behaviour + rule-based family evidence without chronology. Purpose: validate the pipeline and set the CV↔LB offset. Expected LB 0.50–0.70.
2. **sub002 (D1)**: detector pair model v1 (exp005) + behaviour multinomial (exp006) + learned evidence rankers without chronology (exp007). Expected LB 0.78–0.87.
3. **sub003 (D1)**: sub002 with only the evidence columns swapped for the chronology-aware rankers (exp008). A pure evidence A/B:
   - an LB delta ≥ +0.008 means the eval listing is early-biased like dev;
   - ≈ 0 means drop chronology.
   Information only until the rule call in §7 is made.

---------------------------------------------------------------------------------------------------------------------

## 6. Public notebooks vs our plan

| notebook | LB | approach | fatal limits |
|---|---|---|---|
| honghanhhh "LB 0.69" | **0.695** (best public) | pair X = mean/max/p95 of about 66 hand counts plus partner-vs-outsider response table, true-HU counts, partner-with vs partner-absent contrasts, rolling-20 maxima; XGB+LGB+CatBoost; PU sample 60 per table at weight 0.35; evidence = HGB/LGBM/XGB + XGBRanker with within-pair percentiles, all eval pairs | crude hole strength (no equity, board bug); dev 3000-hand vs eval 2000-hand scale mismatch; behaviour gated to `none` beyond the top 0.5–1.4%; iso-biased OvR argmax; early stopping on the validation fold |
| lamhuy8904 topological (fork) | ≈ 0.68 | same lineage | same |
| nomannic19 PU-aware ranker | ? (no outputs) | XGB risk + XGBRanker evidence | SCORE_FEATURES empty (bug), so the evidence fallback is net_gap for all families; behaviour gating |
| foysalemonshanto V2.2/H2 | 0.662 | exact evaluator + counterfactual residuals, family XGBRankers (OOF MAP 0.35) | ungrouped CV; wrong strength on incomplete boards; gating |
| hosen42 E23 | ≈ 0.65 lineage | PU branch + clean P/N branch rank-blend | alpha picked on labelled-only OOF; gating to top 2.5% |
| avikdas567 | ? | 29 chip-flow features, LGBM | no PU; evidence MAP measured by us at 0.091 |

Where we beat the best public approach:
1. **Phase-local, exposure-matched, count-aware features** instead of mixed-scale aggregates. Every public notebook trains on 3000-hand dev aggregates and scores 2000-hand eval aggregates.
2. **Poker-semantic signatures** (equity, normal-play policy surprise, population p_aggr, gates) with pair AUC 0.99–1.00 per family, against the public "partner response counts" with no card context.
3. **Directedness against baselines** (own-rate excess, double-centring, other-phase DiD) that removes the spill-over and persona false positives.
4. **Behaviour argmax for every pair**, a guaranteed gain over their gating.
5. **Evidence**: gates, in-pair family rankers, posterior mixture and optional chronology. Dev proxy 0.55–0.69 vs the public ≈ 0.35. Scored for every eval pair.
6. **other_coordination**: generic branch plus phase-contrast mining. The public notebooks ignore it entirely.

Public ideas worth keeping as features:
- partner-vs-outsider response table and isolation_purity_rate;
- true-HU pair counts;
- rolling-window maxima;
- EB-shrunk partner-with contrasts (rebased on "partner active in pot");
- transfer = min(loss_i, win_j).

---------------------------------------------------------------------------------------------------------------------

## 7. Risks

### Rule compliance ("infer from poker activity; no ID-format / row-order / generator-internal exploits")

| item | verdict |
|---|---|
| player_id / pair_id / hand_id strings, file row order, tie-break on pair_id | **banned**. Tie-break with a secondary model score. |
| eval-file construction artifacts: eval pairs per table, "contains a labelled-negative player", the 29 eligible-but-absent pairs, graph features from which pairs are in the labels/eval files | **banned** |
| players.parquet metadata incl. pairwise | allowed (host), useless |
| hand timestamps / hand_seq (derived from `started_at`, no ties) for episodes, windows, DiD, rolling baselines | allowed: documented gameplay field; the host notebook suggests "temporal bursts" |
| **within-phase chronology in evidence ranking** (earlier planted hands preferred because the host's list is early-biased) | **GRAY**. It uses a gameplay field and is learned from labels, but it works because of how the evidence list was built. Worth ≈ +0.018 if it transfers. **Eric decides**. The prize write-up would have to disclose it. Recommendation: A/B on the LB; if adopted, keep a no-chronology variant as a fallback. |
| CI "UTG preference" in evidence ranking | allowed (position is gameplay) |
| other-phase features for eval pairs (dev hands as baseline, DiD) | allowed (poker activity); validate on LB |
| population baselines built on all 2M hands (both phases) | allowed; label-free |

### Overfitting to 372 positives
- Signature thresholds were read off the same dev evidence, so CV is mildly optimistic. Prefer count-aware statistics against population base rates over tuned cutoffs.
- Learned LLR tables and hand models are OOF only.
- Family heads have 92–148 positives, so pool them into one risk model and put per-family statistics in as features.
- CI fold 1 has only 12 positives: never read single folds; repeat 3×5.
- Evidence listing rules (UTG, chronology, call-down under-listing) are learned from dev lists and may not transfer to eval lists. That is what the A/B tests settle.

### Public LB noise
- About 100–180 public positives; pair-AP SE ≈ 0.02. Do not chase pair-AP deltas < 0.015.
- Use LB for structural questions (chronology, DiD, generic branch) with one-component A/B tests.
- Final selection is on paired table-bootstrap OOF (parent doctrine §1).
- A 0.938 top may partly be luck on 30% of the data.

### Structural risks
1. **other_coordination share unknown.** If large and undetected, pair AP is capped at (1−φ). Mitigations: exp013/014 early (D2), not at D4.
2. **SCAR failure.** Eval positives may be subtler or at lower exposure than dev (eval shared median 76 vs dev 112).
   - Mitigations: exposure-matched view C, count-aware statistics, and the PU-est residual monitor.
   - CI AP falls to 0.88 at 38 hands.
3. **Phase-locality assumption.** If some eval positives are active in both phases, DiD features would hurt them. Settle with a sub with vs without DiD (D2).
4. **Critical path = exp001 engine.** If it slips, fall back to the per-family forensics scripts, which already have numba engines and eval-capable stages, for sub002.
5. **Compute contention.** Several agents share the box: POLARS_MAX_THREADS=3, duckdb threads=3 and memory_limit 4GB. Write per-table parquet chunks; never hold the 10M eval pair-hand rows plus features in RAM at once.
6. **Evidence validity.** Only eval-phase hands shared by both players, 5 distinct IDs, `NO_EVIDENCE` padding only if fewer than 5 shared hands exist (`common.write_submission` validates).
7. **Final-day mistakes.** Tick the finals manually (the CLI cannot select); freeze by D4 night.

### What would overturn this plan
- sub002 LB < 0.75 with view C ≥ 0.80 → exposure or SCAR shift, or a large other_coordination share. Prioritise exp010, exp013 and exp014.
- sub003 − sub002 ≈ 0 → the eval evidence list is not early-biased; drop chronology.
- The phase-contrast excess of generic statistics far exceeds the family flags → a 4th mechanism is large; move the generic branch to the centre.
- A DiD submission loses clearly → some eval positives span both phases; remove DiD and keep only same-phase baselines.

---------------------------------------------------------------------------------------------------------------------

## Critic addendum (2026-09-15, completeness critic)

Scripts: scratchpad `critic/c1_ci_phase.py`, `critic/c2_prefix.py`, `critic/c3_spill.py`, plus inline duckdb/polars queries (3 threads, each under 1 min). Every number below was **MEASURED** this session unless marked (est.).

### A. Corrections to the synthesis

| # | Where | Synthesis says | Measured / corrected |
|---|---|---|---|
| A1 | Fact 14 | "Seats are redrawn every hand" | **Wrong.** Seats persist while the lineup is unchanged: in tables 0–39, all 159,019 consecutive same-lineup hand pairs keep every seat. The lineup changes on 20.5% of hand transitions. The **button is uniformly random each hand**: step 0..5 counts are 33.0k–33.6k each. So *position* is redrawn every hand, and seats are not. The first preflop actor is button+3 in 100,000/100,000 six-handed hands, so button-derived UTG/HJ/CO are correct. The adjacency conclusion (random) stands. |
| A2 | §1 supporting facts, §2.4 | other_coordination is "reserved for evaluation… mining dev unknowns will not find it" | **Not supported.** For CI the generator activates about as many pairs in dev as in eval: 122 dev-active (83 labelled + 39 unlabelled) vs 108 eval-active in the eval file. About 32% of dev-active positives are unlabelled. "Reserved for evaluation" describes the *labels*. If OC uses the same phase assignment, a comparable number of OC pairs are active in dev and hidden in U. There they have 3000-hand exposure, and the eval phase gives a clean null. Dev-U mining is a legitimate and probably richer place to hunt OC. |
| A3 | §3f phase-contrast instrument | "each eval pair's dev phase is a clean null" | **Not clean.** 20 eval-file pairs are dev-active CI pairs: hidden dev positives that passed the eval filter, none of them active in eval. CI is about 25% of positives, so est. 70–110 dev-active hidden pairs of all families sit inside the eval file. Remove dev-flagged pairs from the null, or read the excess as a lower bound. |
| A4 | §3e(4), §7 chronology | chronology means "earlier planted hands are preferred" | **Refined** (section C). DT and SP lists are *two chronologically sorted segments*. The early bias holds within a sub-type, not overall. |
| A5 | `src/common.py` | `finalize()` docstring: "break ties by a tiny … epsilon" | The code breaks no ties, and `validate_submission` only *reports* `risk_ties`. Add `assert stats["risk_ties"] == 0` in `write_submission`. `finalize()` also defaults `predicted_behavior` to `"none"`, which §3d says never to use. |
| A6 | §6 public notebooks | — | The LB-0.69 author's public-LB ablations found that "cross-phase history" and "other_coordination as a 4th class" both **hurt**. This is independent LB evidence for phase-locality and against labelling other_coordination naively. |

### B. Phase structure, colluder identity, hidden positives (CI flag over all 174k within-table pairs)

CI flag = mutual-top-lift and excess_trash ≥ 5, computed separately per phase.

1. **Pairs are active in one phase only.** 124 pairs are dev-only, 109 eval-only and **0 are active in both**.
2. **Colluder players are disjoint across phases.**
   - 248 dev-CI-active players and 218 eval-CI-active players overlap in **0** players; random expectation ≈ 4.5.
   - The 693 labelled positive players (any family) appear in only 1 eval-CI-active pair, against ≈ 12.6 expected.
   - Implication: eval colluders look clean in dev, so a player's own dev-phase baseline is clean. A "was a colluder in the other phase" player prior would point the wrong way.
3. **No table-level carry-over.**
   - Per table, the dev CI-active count and the eval CI-active count correlate at r = 0.002.
   - Labelled positives and eval CI-active pairs per table correlate at r = 0.03.
   - Do not build a table prior from dev labels.
4. **Hidden dev CI-active pairs: 39, and all have dev shared ≥ 57** (min 58, median 115).
   - So the host's labelled subset is not selected by exposure. U(≥57) holds about 32% of dev-active positives (83 labelled vs 39 hidden among flagged pairs).
   - 20 of the 39 are in the eval file, and none is eval-active.
   - 14 miss the eval file because eval shared < 38. The other 5 touch a labelled positive player.
5. **Positive count cross-check (est.).**
   - Labelled fraction ≈ 0.68, so dev positives ≈ 372/0.68 ≈ 550.
   - The eval/dev ratio of CI-active pairs is 108/122 ≈ 0.89.
   - That gives **≈ 490 eval positives across the 3 families**, inside the 350–600 range.
6. **OPEN, decisive, LB-only.** Are the hidden dev-active pairs inside the eval file (est. 70–110; 20 for CI) positive in the private truth?
   - They have no eval-phase planted hands.
   - The host metric has an explicit `if not relevant: 0.0` branch for positives without evidence, a weak hint that such positives can exist.
   - Eval-phase-only features score them near 0.
   - **Probe (1 sub, D1–D2):** take the best sub and move the strictly dev-flagged hidden pairs to just below the top ~300 risk. About 25 of them fall in the public split, so pair AP should move by several points either way.
7. **Stake.** bb=10 tables have 47 labelled positives vs 35.1 expected (p = 0.03 one-sided; 3 stakes tested). They also have a *lower* eval CI-active rate: 0.21 vs 0.29 per table. There is no usable stake prior.

### C. Evidence listing structure (DT/SP): new, and it drives 20% of the score

**C1. Every non-chronological list has exactly one time inversion.**
- Counts: DT 91/91, CI 11/11, SP 93/96 (the other 3 have two).
- So a list is **two chronologically sorted segments concatenated**.
- Index where segment 2 starts:
  - DT: 1→11, 2→31, 3→33, 4→16
  - SP: 1→18, 2→40, 3→25, 4→10
  - CI: 4→10, 3→1 (CI is "rank 5 appended")

The two segments look different:

| | Segment 1 | Segment 2 |
|---|---|---|
| Fold share | DT: donor folds in 230/236 hands. SP: a partner folds in 195/213. | Still 42% folds (DT 89, SP 97 hands). |
| Median pot | DT 17 bb, SP 14.3 bb | DT 132.5 bb |
| Both partners at showdown | DT 2.5%, SP 8.5% | DT 58%, SP 58% |
| Other traits | DT median transfer 2.5 bb | DT: median transfer 52 bb; board_len = 5 in 144/213 hands.<br>SP: strong check 54%; preflop call of the partner's raise with eq ≥ .55 48%; postflop call of the partner with hs ≥ .8 31%. |

- **The sort key is unresolved.** It is not simply fold vs non-fold: a fold hand is ranked after a non-fold hand in 44/148 DT pairs and 67/132 SP pairs.
- Working reading: evidence_rank sorts by (visible-action sub-type, hand_seq). The sub-types are probably the generator's two variants per family: a small fold to the partner, vs a large pot with call-down, showdown or passive check.

**C2. DT/SP lists are not a strict chronological prefix overall.**

The table shows the rate of unlisted signature hands per non-evidence shared hand, as a multiple of the negative base rate. DT signatures are one-directional (evidence donor direction).

| family, signature | before first listed | between | after last listed |
|---|---|---|---|
| DT SIG_strong (donor dir) | 4.3× | 6.0× | 8.6× |
| DT SIG_gift5 (donor dir) | 4.5× | 5.4× | 7.6× |
| DT SIG_pfcall_weak (donor dir) | 19× | 26× | 38× |
| SP S_llr_gt4 (5-hand lists) | 8.9× | 11.2× | 12.8× |
| SP S_PL_and_softresp (5-hand lists) | 1.5× | 2.1× | 2.5× |

Persistent pair traits contaminate these signatures, so "planted hands before the first listed" is an upper bound. (The CI forensics found only 0.12 unlisted scenario hands per pair before the last listed, so CI *is* close to a prefix.)

**C3. A prefix within each sub-type is much closer.**

Here L = the last listed hand *carrying the same signature*.

| signature | before L | after L |
|---|---|---|
| DT call-down (SIG_call_weak15_turnriver, donor dir) | 1.9× | 8.5× |
| SP call of partner postflop, hs ≥ .8 | 1.0× | 2.3× |
| DT fold-best70-post | 3.4× | 7.2× |
| DT bluff-into-partner | 1.6× | 3.5× |

→ Evidence features should include **chronological rank within sub-type** (number of earlier same-type candidates of the pair in the phase), not only overall chronology. This is in the same rule gray zone as §7 chronology.

**C4. Short lists contain all the planted hands.**
- The 11 DT pairs that list 3–4 hands show **no** unlisted planted-like hands, all against the donor-direction negative base:

  | signature | before first listed | between | after last listed |
  |---|---|---|---|
  | SIG_strong | 0.0× | 0.64× | 0.0× |
  | SIG_gift5 | 0.47× | 0.78× | 0.61× |
  | SIG_pfcall_weak | 0.0× | 1.5× | 0.0× |

  So even the "persistent" flat-call trait is absent: **the flat-call hands are planted hands**.
- The 137 DT pairs that list 5 hands: 4.9× / 6.4× / 9.2× on SIG_strong.
- Exposure differs by family:
  - SP pairs listing fewer than 5 have lower exposure: median dev shared 76 vs 117. So SP planted counts scale with exposure.
  - DT short-list pairs do not: 116 vs 114.
- In eval (0.64× exposure) expect more short, complete SP lists. There, evidence MAP is limited by detection precision rather than by guessing the listing rule.

### D. LB arithmetic: evidence MAP is a bigger lever than §5 assumes

final = 0.7·AP + 0.2·MAP + 0.1·BehAP. The table gives the evidence MAP a team needs for each LB score, assuming AP = BehAP:

| LB | AP = 1.00 | AP = 0.97 | AP = 0.95 | AP = 0.90 |
|---|---|---|---|---|
| #1 0.938 | 0.69 | 0.81 | 0.89 | impossible (1.09) |
| #10 0.906 | 0.53 | 0.65 | 0.73 | 0.93 |
| #29 0.884 | 0.42 | 0.54 | 0.62 | 0.82 |

- **The leader must have MAP ≥ 0.69 even with perfect ranking; realistically 0.8–0.9.**
- Our dev proxies are 0.55–0.69. With AP 0.95, MAP 0.62 and BehAP 0.95 the final is ≈ 0.88, i.e. #29.
- +0.1 MAP = +0.02 final, the same as +0.029 pair AP. Give evidence (C1–C4, exp007/008/016) D1 priority equal to the pair model.
- Likely sources of the leaders' MAP:
  1. sub-type chronology (C3);
  2. short, complete eval lists (C4);
  3. high-precision planted-hand detection.

### E. Checks that passed (no change needed)

- **Players and folds.** One table per player (12,000 players, max 1 table). The fold file md5 is 322a34c8454910717d98b6a95fdeadea. Table folds leak no players and no hands.
- **Spill-over on generic statistics.**
  - Test set: colluder × non-partner dev pairs with shared ≥ 57 (CI 3,912, DT 6,113, SP 5,324).
  - Statistics: both_vpip, both_flop, sd_together, hu_final and gross flow.
  - These pairs exceed the no-positive-player p99 in 0.69–1.15% of cases, against a 1.0% reference. Colluders play normally with non-partners.
  - Partner pairs, for contrast: DT sd_together is above p99 in 40% of pairs, DT gross flow in 49%.
  - The ~28 other pairs of each eval colluder are not a generic false-positive source. Family signatures still need own-baseline directedness (§3b).
- **Metric.** `other_coordination` and `none` are valid labels. NO_EVIDENCE entries are dropped before scoring, duplicate hands raise an error, and risk must lie in [0,1].

### F. Remaining open gaps (none closable in 15 min)

1. **Hidden dev-active pairs.** Their private-truth status inside the eval file needs the LB probe in B6. It decides whether dev-phase detectors should add risk to eval pairs.
2. **other_coordination hunt in dev U (A2): not started.** Recipe:
   - take dev U pairs with shared ≥ 57 and no labelled player;
   - remove CI/DT/SP detector hits;
   - rank by family-agnostic dev statistics, using the same pairs' eval phase as the null;
   - read the hands of the top 30.
3. **Sub-type key for DT/SP lists (C1).** Fit a segment-1-vs-segment-2 hand classifier on the split pairs. Then test the "(type, time)" sort and possible per-type quotas on all 280 DT+SP pairs.
4. **Eval-phase DT and SP activity counts** (the analogue of CI's 108). This needs eval-phase DT/SP detectors on all 112,540 pairs. It would also validate the ≈ 490 estimate per family.
5. **Behaviour assignment.** A macro-AP-optimal assignment could beat plain argmax because the families differ in size. Small effect.
6. **Rule call on sub-type chronology (C3).** Same gray zone and disclosure duty as §7 chronology.
7. **Pending pipeline pieces.** The exp002 dev equity replay and the exp003 policy model are not built or verified. The DT pfcall_weak statistic depends on the policy.
8. **No host write-up search.** Nobody has searched for a Slash/host write-up on the generator or the evaluation design. Public documents would count as allowed external data.
