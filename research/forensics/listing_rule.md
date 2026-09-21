# exp036: the host's evidence listing rule

Written 2026-09-17. Code: `scripts/listing_rule.py` (stages `diag | cv | eval`). Results: `data/derived/evidence_cache/listing_rule_exp036.json`.
Scratch forensics (hand prints, per-orientation action parser, segment classifiers): session scratchpad `lr/`, not in the repo.

Scope: the 372 labelled dev positive pairs (DT 148, SP 132, CI 92) and their 45,129 dev shared hands, of which 1,817 are listed. Controls: 600 confirmed-negative pairs.

---------------------------------------------------------------------------------------------------------------------
## 0. TL;DR

1. **The list is sorted by (type, time) and cut at 5.** The host list is `sorted(EVENTS, key = (type, hand_seq))[:5]`:
   - first every type-1 event of the pair's phase, oldest first;
   - then type-2 events, oldest first;
   - cut at 5 hands.

   The "two chronological segments" in the Critic's C1 note are the two types. **Segment 1 is the complete set of type-1 events whenever a segment 2 exists.** Section 1 gives the evidence.
2. **The type is a visible property of the hand, and for CI it is exact.**

   | Family | Type 1 | Type 2 | How clean |
   |---|---|---|---|
   | **CI** | The pair's first preflop raise comes before anyone has folded (players_active = 6). | Exactly one fold before that raise. | 460/460. No listed CI hand has ≥ 2 folds before the raise. |
   | **SP** | A partner folds to the other's aggression while holding the better made hand. | A passive call of the partner's aggression. | Type 1: seg 1 182/213, seg 2 7/232. Type 2: seg 2 223/232. The 3 lists with two inversions add a **type 3 = strong check**. |
   | **DT** | The donor folds to the receiver while holding the better made hand. | Chip-dump call-downs and bets. | Type 1: seg 1 221/236, seg 2 32/213. |

   For DT and SP, **identical action strings occur in both segments**, so the type is only partly visible. A learned type model reaches OOF AUC 0.972 (DT) and 0.945 (SP).
3. **The event itself is not a crisp predicate of the public log.** An event is latent activation plus a visible action.
   - Look-alike hands with the same action string, the same position and a similar hand strength sit unlisted *between* listed hands.
   - The best deterministic predicates reproduce the listed set exactly for only **7–25% of pairs** (CI best 25%, DT best 15%, SP ≤ 3%). This is the honest null on "find the exact predicate".
4. **The structure is still worth a lot as a scorer.** The scorer has three parts:
   - an **event model** trained on censoring-derived labels;
   - a **type model**;
   - an exact **Poisson-binomial DP** for P(listed) under the rule in point 1.

   Results against exp008 (dev OOF MAP@5, true-family routing, 372 pairs):

   | Scorer | DT | SP | CI | All |
   |---|---|---|---|---|
   | exp008 (incumbent) | 0.7164 | 0.7078 | 0.7066 | 0.7109 |
   | DP alone | 0.7743 | 0.7564 | 0.7484 | 0.7615 |
   | **exp036 = 0.3·exp008 + 0.7·DP** | **0.7876** | 0.7543 | **0.7540** | **0.7675** |
   | Gain, exp036 over exp008 | **+0.071** | **+0.047** | **+0.047** | **+0.057** |

   - With real behaviour posteriors (exp016 / exp035), the gain is +0.058 / +0.056 over exp008 and +0.059 / +0.060 over exp027ev. The paired bootstrap over pairs gives **P(better) = 1.000**.
   - **Expected final-score effect: ≈ +0.011**, if eval lists follow the same rule. The chronology LB A/B (exp007 → exp008, +0.022 LB) already says they do.
5. **The eval files are written**:
   - `data/derived/evidence_scores_eval_exp036.parquet`, the blend with exp008: 2,348,893 rows;
   - `evidence_scores_eval_exp036_exp027ev.parquet`, the blend with the LB-best exp027ev: 2,300,642 rows;
   - matching dev files for the 372 dev positive pairs.

   Both eval files have all 112,540 pairs, only eval-phase hands, and at least 15 candidates per pair. A composition test with sub011's risk and exp035 posteriors passes `validate_submission`: 0 non-eval hands, 0 non-shared hands.

---------------------------------------------------------------------------------------------------------------------
## 1. Evidence for "sorted by (type, time), cut at 5"

### 1a. Lists are one or two chronological segments

This is Critic C1, re-verified:

| Family | Chronological | 1 inversion | 2 inversions |
|---|---|---|---|
| DT | 57 | 91 | 0 |
| SP | 36 | 93 | 3 |
| CI | 81 | 11 | 0 |

### 1b. Segment 1 is uniform in time, segment 2 is early

The table gives each listed hand's mean relative position in the pair's dev shared-hand timeline (0 = first shared hand, 1 = last). Uniform placement would give 0.50.

| Family | Chronological lists | Seg 1 of 1-inversion lists | Seg 2 of 1-inversion lists |
|---|---|---|---|
| DT | 0.384 | **0.498** | **0.248** |
| SP | 0.391 | **0.499** | **0.309** |
| CI | 0.316 | 0.411 | **0.172** |

This is the signature of "all type-1 events, then the *earliest* type-2 events":
- Segment 1 is not selected by time, so it is the whole type-1 set.
- Segment 2 is the first few type-2 events.

A single chronological cap over all events would push both segments early.

### 1c. Segment 1 is complete

DT, with T1 = the donor folds to the receiver with HU equity ≥ 0.5. The table gives the rate of *unlisted* T1 hands per shared hand:

| Pairs | Before the last listed hand of the reference segment | After it |
|---|---|---|
| 1-inversion (91), reference = seg 1 | 0.0148 | **0.0161** (flat) |
| Chronological 5-lists (51) | 0.0207 | **0.0485** (events beyond the cap) |
| Short lists (11), which list everything | 0.0141 | 0.0085 |

- In inversion pairs, unlisted T1-like hands sit at the natural base rate, the same as in short lists, and there is no excess after segment 1. **Segment 1 has used up the type-1 events.**
- In chronological 5-lists, the excess after the last listed hand is the capped remainder.
- Unlisted T1 per pair: 1.64 in inversion pairs vs 4.12 in chronological pairs.

### 1d. Short lists contain everything

This is Critic C4. Pairs with 3–4 listed hands have no excess planted-like hands anywhere, so the list is *all* events. In eval (0.64× exposure) short, complete lists will be more common. The DP handles this naturally: the summed P(listed) on top-risk eval pairs averages 4.5–4.7, not 5.

---------------------------------------------------------------------------------------------------------------------
## 2. The type key per family

The tables use the vectorized features from `extra_features`. "chrono" = hands of lists with no inversion.

### CI: exact

The key is the number of players who **folded** before the pair's first preflop raise (`ci_fb`).

| Group | ci_fb = 0 | ci_fb = 1 | ci_fb ≥ 2, or no pair raise |
|---|---|---|---|
| Chronological lists (405 hands) | **405** | 0 | 0 |
| Seg 1 (43) | **43** | 0 | 0 |
| Seg 2 (12) | 0 | **12** | 0 |

- Earlier forensics saw this as "UTG open" (fpa_ord = 0), but the real key is "no fold yet".
- Hands where the pair 3-bets a UTG open, or raises after a UTG limp, are type 1.
- A raise from the HJ after the UTG folds is type 2.

Example: pair P0AE7584790C9 lists [719, 720, 725, 991 | 718].

| Hand | Segment | Preflop (A, B = pair; o = other) | Folds before the pair raise |
|---|---|---|---|
| 719 | 1 | `Ac Br …` (A limps, B raises) | 0 |
| 720 | 1 | `Br …` (B raises first) | 0 |
| 725 | 1 | `Ar Bc …` | 0 |
| 991 | 1 | `or Ar … Bf` (A 3-bets the UTG open, B folds) | 0 |
| 718 | 2 | `of Ar Bf` (UTG folds, then A raises) | 1 |

Hand 718 is the *earliest* of the five, yet it is listed last.

### SP: fold-ahead first, then passive calls, then (rarely) strong checks

`fold_ahead_any` = some partner folds to the other's aggression while holding the better made hand on that street. Preflop, "better" is judged by class equity.

| Group | fold_ahead = 1 | fold_ahead = 0 |
|---|---|---|
| Seg 1 of 1-inversion lists | **182** | 31 |
| Seg 2 of 1-inversion lists | 7 | **225**, of which 223/232 of all seg-2 hands contain a passive call of partner aggression |
| Two-inversion lists (3) | seg 1: 6/6 | seg 2: 6/6 call; seg 3: 3/3 strong check (hs ≥ 0.8) |
| Chronological lists | 61 | 111 |

Example: pair P00082F54BA9A lists [2368, 2369, 2378 | 820, 1309].
- **Seg 1** holds fold-ahead hands from hands 2368–2378.
- **Seg 2** holds the two earliest passive-call hands, 820 and 1309.
- The unlisted fold-ahead look-alikes 1301, 1308, 1946 and **2374** fail the event test. Hand 2374 lies *between* two listed seg-1 hands.

### DT: donor fold-ahead first, then chip dumps

| Group | fold_ahead_don = 1 | fold_ahead_don = 0 |
|---|---|---|
| Seg 1 of 1-inversion lists | **221** | 15 |
| Seg 2 of 1-inversion lists | 32 | **181** |
| Chronological lists | 188 | 88 |

- Seg 2 is 58% showdown, with a median pot of 132 bb. Its folds are mostly low-equity folds after a dump.
- With omniscient HU equity instead of made-hand rank the split is weaker: seg-1 folds with hu ≥ 0.5 are 216/236.
- The policy model's p(fold) for the donor's fold is high in seg 2 (69/89 > 0.5), so those folds look natural.

**The DT type is not fully visible.** Pair P12DC80D8F0FD lists [2181 | 721, 787, 1481, 1785]:
- **2181 (seg 1)** and **787 (seg 2)** have the *identical* action string `P: of Rr of of of Dc | F: Dx Rr Df`.
- In 2181 the donor folds a pair of 7s (hu 0.72). In 787 the donor folds king-high with a flush draw (hu 0.78).

Pair P6D2E40BC7540 lists [103 | 77, 86, 108, 117]:
- 103 (seg 1) and 77 (seg 2) are both `of Rr Dc … | F: Rr Df`.
- In 103 the donor folds a pair of 3s against the receiver's no-pair. In 77 the donor folds a pair of Js against the receiver's 99.

A made-hand-category rule does not separate these either. The OOF type model, with engine features plus these keys, reaches:

| Family | Type-model AUC | Type-model accuracy | fold_ahead rule accuracy |
|---|---|---|---|
| DT | 0.972 | 0.927 | 0.895 |
| SP | 0.945 | 0.920 | 0.917 |

---------------------------------------------------------------------------------------------------------------------
## 3. Event eligibility: no crisp predicate

Each rule below lists the first |L| P-hands in (type, time) order. The columns are:
- **exact** = the listed set is reproduced exactly;
- **MAP** = host AP@5 of the rule list;
- **L∖P** = listed hands that fail P, per pair (should be 0);
- **P-unl<last** = unlisted P-hands before the last listed hand, 5-lists (should be 0);
- **P-unl (short)** = unlisted P-hands in 3–4-lists (should be 0).

| Family | Rule | Exact | MAP | L∖P | P-unl<last | P-unl (short) |
|---|---|---|---|---|---|---|
| CI | pair raise with ≤ 1 fold before it | 0.076 | 0.493 | **0.00** | 10.38 | — |
| CI | … & first raiser population p_aggr < 0.2 | 0.196 | **0.736** | 1.00 | 3.28 | — |
| CI | … & p_aggr < 0.3 | **0.250** | 0.734 | 0.74 | 4.09 | — |
| CI | … & p_aggr < 0.3 & partner folds or re-raises | 0.228 | 0.696 | 1.03 | 3.53 | — |
| DT | flow & donor-direction (fold-ahead, or gift > 5 bb) | 0.149 | 0.591 | 0.52 | 4.56 | 1.46 |
| DT | flow & (dt_strong or pfcall_weak), donor direction | 0.115 | 0.591 | 1.07 | 3.96 | 0.46 |
| SP | sp_soft | 0.030 | 0.435 | 0.49 | 7.07 | 4.29 |
| SP | S_PL & (fold-ahead, strong check or postflop call) | 0.015 | 0.445 | 1.13 | 3.96 | 1.81 |

Thresholding the learned event probability q (below) gives exact rates of 0.11–0.22 with MAP 0.57–0.70. That is still far from a rule.

Why no predicate exists:
- The generator activates the scenario latently.
- An activated hand counts only if the colluder's action was actually changed.
- Look-alikes are natural play, and some of them are not even rare.

Example: pair P0560CDB3D9E0, hand 1002.
- B raises J7o first to act (population p = 0.05) and A folds immediately.
- That is the same shape as listed hand 1932 (A raises J5o, B folds).
- Hand 1002 is unlisted and falls between listed hands 282 and 1932.

---------------------------------------------------------------------------------------------------------------------
## 4. The rule-guided scorer (exp036)

1. **Event model q_i.** A LightGBM per family on the family's positive pairs.
   - Features: the 321 orientation-free engine views (`evidence.hand_features`, identical to exp007's stage-1 inputs), plus `ci_fb` and the fold-ahead / fold-made keys in pair-donor and any orientation.
   - **Labels from the censoring logic of the rule:**
     - listed hand → 1;
     - unlisted hand in a 3–4-list → 0;
     - unlisted hand *before the last-ranked listed hand* → 0, because under the rule it cannot be an event;
     - later unlisted hands are excluded, since they may be events beyond the cap.
   - OOF by the frozen table folds.
   - q alone ranks poorly (MAP 0.58 / 0.61 / 0.47), as it should: it has no chronology.
2. **Type model τ_i = P(type 1 | event).**
   - DT and SP: trained on the listed hands of the 1-inversion pairs, seg 1 vs seg 2.
   - CI: τ is deterministic from `ci_fb`.
3. **Family gate, where q = 0 outside it.** Each gate covers 100% of the dev listed hands.

   | Family | Gate |
   |---|---|
   | DT | chip flow |
   | SP | G0 |
   | CI | pair raise with ≤ 1 fold before it |
4. **DP.** Take the hands in time order. For type k:

   P(listed_i) = Σ_k q_i π_ik · P( #events of type < k anywhere, other than i, + #type-k events before i < 5 )

   Each term is a Poisson-binomial CDF.
   - The first count uses p_j = q_j·P(type_j ≤ k) for j < i and q_j·P(type_j < k) for j > i.
   - It is computed exactly with prefix and suffix DPs, O(n·25) per pair and type, in numba.
   - All 9.65M eval pair-hand rows are scored in 136 s.
5. **Blend.** s_f = 0.3·s_f(partner) + 0.7·P_DP. Both terms are calibrated P(listed | family f). The weight grid was 0.3 / 0.5 / 0.7, and 0.7 was best for DT and CI and tied for SP.

### Ablation

Same OOF q throughout. Dev MAP@5, true-family routing:

| Variant | DT | SP | CI |
|---|---|---|---|
| q only (no chronology) | 0.577 | 0.612 | 0.471 |
| DP, one type (plain "first 5 events") | 0.667 | 0.718 | 0.747 |
| DP + rule type (fold_ahead / fb) | 0.741 | 0.754 | 0.748 |
| **DP + model type** | **0.774** | **0.756** | 0.748 (fb) |
| exp008 | 0.716 | 0.708 | 0.707 |

- **The (type, time) ordering is worth +0.107 on DT and +0.038 on SP** over a single chronological cap.
- On CI the gain is the censoring-label event model plus the prefix DP. Type-2 hands are rare, so the exact fb key adds only +0.001 on dev. It matters more on eval, where exposure is lower and more pairs have fewer than 5 type-1 events.
- Previous "segment quota" attempts (exp016ev, exp023ev) failed because they treated segment 2 as a quota filled by the *best* candidates. Under the true rule, segment 2 holds the *earliest* type-2 events, and only after *all* type-1 events of the whole phase. Future type-1 events count against it.

### Robustness

Composition uses `compose_evidence` with real OOF behaviour posteriors. Paired bootstrap over the 372 pairs, 2000 resamples.

| Partner | Posteriors | Partner MAP | w = 0.5 | w = 0.7 | DP only | P(w = 0.7 better) |
|---|---|---|---|---|---|---|
| exp008 | true | 0.7109 | 0.7635 | 0.7675 | 0.7615 | 1.000 |
| exp008 | exp016_full | 0.7068 | 0.7612 | 0.7651 | 0.7594 | 1.000 |
| exp008 | exp035_sf101 | 0.7103 | 0.7625 | 0.7666 | 0.7605 | 1.000 |
| exp027ev | exp035_sf101 | 0.7121 | 0.7662 | **0.7723** | 0.7605 | 1.000 |

**Caveat: uniform posteriors.** With a uniform family mixture the blend is *worse* than the partner: 0.611 vs 0.627 for exp008, and 0.623 vs 0.643 for exp027ev.
- The DP is peaked. Every family's DP puts about 5 hands near 1 in *any* pair, so wrong-family peaks compete in a flat mixture.
- Real posteriors are sharp, so this does not matter on dev.
- It does matter for eval pairs with uncertain family, including hidden other_coordination pairs (about 6.5%), where evidence is already weak (about 0.35). The expected cost is small, but it is not measured.
- If LB disappoints, a safer composition would be to route by argmax family, or to fall back to the partner score when max P < 0.8.

**Protocol note.**
- The exp036 dev numbers are single-level OOF. The event and type models use only label-free engine features, so no nested stage is involved.
- exp008's dev file is the nested-protocol file, reported earlier as about 0.014 optimistic vs its shipped model. The DP's relative gain on eval could therefore be *larger* than on dev, not smaller.
- The blend weight and the 300 / 150 rounds were not tuned beyond a 3-point weight grid.

### Eval files

| File | Rows | Pairs | Rows per pair |
|---|---|---|---|
| `evidence_scores_eval_exp036.parquet` (partner exp008) | 2,348,893 | 112,540 | min 15, median 20 |
| `evidence_scores_eval_exp036_exp027ev.parquet` | 2,300,642 | 112,540 | min 15, median 20 |

- Kept rows are the union over families of each pair's top-15 by blended score. All scored hands are eval-phase shared hands, so eval lists come from eval hands only.
- The raw DP component (rows with max P > 1e-4) is in `evidence_cache/exp036_eval_dp.parquet`.
- The composed top-5 of the 600 highest-risk pairs (exp035 posteriors) shares **4.28 / 5** hands with exp008. For comparison, exp008 vs exp027ev share 4.61 and exp036 vs exp036_exp027ev share 4.92. Much of the change is order within the top 5, which AP rewards.
- Dev files: `evidence_scores_dev_exp036.parquet` and `evidence_scores_dev_exp036_exp027ev.parquet`, with all 45,129 rows of the 372 pairs and every family head applied (OOF).

---------------------------------------------------------------------------------------------------------------------
## 5. What remains unexplained

1. **The latent event.** Only about 20% of pairs are reproduced exactly by any visible predicate.
   - The event model's OOF AUC for listed vs censored-negative hands is 0.994 (DT), 0.995 (SP) and 0.991 (CI).
   - Events are only about 1 in 12 hands, so the few high-q look-alikes before the true 5th event still displace listed hands.
   - Candidate improvements: player-specific normal-play baselines (is *this* player's action unusual?), and a q model that also trains on the capped tail with EM under the DP likelihood.
2. **The DT and SP type.** Identical action strings fall in both segments, so the generator's type is probably decided before or inside the altered decision and not from the outcome.
   - The seg-1 folds are "less natural" by our policy model (p_fold < 0.5 in 70% of cases vs 22% for seg-2 folds), but the split is not clean.
   - The 57 DT / 36 SP / 81 CI chronological lists may hide a type boundary. The DP does not need it.
3. **SP type 3.** Three lists show a strong-check type after the passive calls. It is modelled as part of type 2.
4. **Whether eval lists use the same rule.** The chronology A/B on LB (+0.022 for exp008 over exp007) says the lists are early-biased. The (type, time) refinement is untested on LB. **One submission settles it:** sub011 with `evidence_scores_eval_exp036_exp027ev.parquet` instead of exp027ev.
   - Expected LB: about +0.011 (0.2 × 0.056), minus the other_coordination and uniform-posterior risk.
   - Evidence SE on LB is about 0.006, so a real effect of this size is visible.
5. **CI p_aggr.** The CI rule with a population p_aggr < 0.2 filter already scores 0.736 without any model. The generator's trash-raise definition is probably a hand-range cut, but the exact cut was not found.

---------------------------------------------------------------------------------------------------------------------
## 6. exp037: why only 55% of the dev gain transferred (measured, 2026-09-17)

LB facts, all with identical risk and behaviour, so these are pure evidence deltas: exp027ev 0.91117 -> exp036 x exp027ev (w = 0.7) **0.91773**, pure rule (w = 1.0) 0.91549. Decomposed evidence MAP: 0.666 -> **0.699** -> 0.685.

### 6a. The whole gap is a 14% subpopulation that our 3-family routing cannot serve

1. **Exposure is not the problem.** Eval pairs share fewer hands than dev positives (median 76 vs 112). Re-weighting the dev per-pair APs to the eval shared-hand distribution *raises* the dev estimate slightly, because short lists are easier:

   | Blend weight w | Dev MAP (real posteriors) | Exposure-reweighted to eval |
   |---|---|---|
   | 0.0 (exp027ev) | 0.7115 | 0.7137 |
   | 0.5 | 0.7645 | 0.7622 |
   | **0.7 (exp036)** | 0.7710 | **0.7740** |
   | 1.0 (pure rule) | 0.7594 | 0.7678 |

2. **LOFO with the rule scorer (the honest proxy for an unseen family).** Train the event and type heads on two families, score the held-out family's pairs. Mean over the three held-out families:

   | Scorer on an unseen family | MAP |
   |---|---|
   | partner mixture over the two modelled heads (what exp027ev does for an OC pair) | **0.419** |
   | partner max over the modelled heads | 0.401 |
   | exp036 mixture at w = 0.7 | **0.258** |
   | at w = 0.5 / w = 0.3 | 0.302 / 0.353 |
   | pure DP mixture (w = 1.0) | 0.203 |
   | pooled family-agnostic rule (event + type trained on 2 families, union gate) | DT 0.336 / SP 0.385 / CI 0.059, mean 0.260 |
   | pooled agnostic q only, no DP | 0.190 |

   Per family, the exp036 mixture scores 0.358 (DT held out), 0.350 (SP), 0.067 (CI) against the partner's 0.477 / 0.530 / 0.251.

   **The DP is actively harmful when no modelled family fires.** Each family's DP is peaked on about 5 hands and confidently picks the wrong ones, while the partner's calibrated score degrades gracefully. This kills priority 2 as originally conceived: an OC-agnostic *rule* head is worse than the incumbent mixture, in line with the exp027ev LOFO result for the learned rankers.

3. **The fit.** Model each LB point as `obs_w = (1 - phi) * K_w + phi * alpha * L_w`, with K = the exposure-reweighted dev values and L = the LOFO values. Fitting the two parameters on the three matched LB points:

   **phi = 0.140, alpha = 0.90, RMSE 0.0006** (predicted 0.6666 / 0.6982 / 0.6859 vs LB 0.6660 / 0.6990 / 0.6854).

   - Known-family pairs transfer at ~100% once exposure is accounted for.
   - About **14% of true eval positives** are unexplained by the three modelled families; they score 0.38 with the partner alone but only 0.23 under exp036.
   - 0.140 * (0.377 - 0.232) = 0.020 MAP of the 0.033 lost gain, and the remaining 0.013 is the (1 - phi) discount on the known-family gain. **That accounts for all of the 45% shortfall.**
   - phi = 0.14 is above the <= 7-10% other_coordination bound in `other_coordination.md`, so the population is probably OC plus family variants that the 3-class routing mis-serves.

### 6b. exp037: gate the DP on family strength (dev-neutral, eval-positive)

`strength(pair) = max over families of the top-5 mean partner score`. Where it is below 0.6, the DP weight is set to 0 and the pair keeps the partner ranking.

- **It detects an unmodelled family.** In the LOFO setup the statistic separates held-out-family pairs from modelled-family pairs with AUC 0.899 (DT), 0.958 (SP), 0.994 (CI). The 3-class behaviour posterior does not: AUC 0.39 / 0.59 / 0.53, and 98% of the top-400 eval pairs have max posterior > 0.9, so **a confidence gate on the posterior would do nothing**.
- **Dev cost: none.** At 0.6 the gate fires on 4.8% of dev positives and dev MAP goes *up*: 0.7710 -> 0.7719 (those pairs are better served by the partner anyway).
- **Recall on unseen families:** 0.53 / 0.82 / 0.98 per held-out family, mean 0.78 at threshold 0.6 (optimistic, because the production statistic maximizes over 3 heads rather than 2).
- **Incidence on eval matches the fit:** the gate fires on 6.0% of the top-200 risk pairs, 8.5% of the top-400 and 13.3% of the top-600, the same order as phi = 0.14.
- **Projection:** +0.010 MAP at recall 0.5, +0.016 at recall 0.78, i.e. **+0.002 to +0.003 final**. The top-600 composed lists change in 8.8% of pairs, sharing 4.88/5 hands with exp036.

Files: `evidence_scores_eval_exp037_exp027ev.parquet` (2,271,506 rows) and `evidence_scores_eval_exp037.parquet` (partner exp008), plus the matching dev files. Both pass `validate_submission` composed with sub011's risk. Reproduce with `LR_TAG=exp037 LR_GATE_TH=0.6 python scripts/listing_rule.py eval`.

### 6c. Three nulls on sharpening the heads (priority 3)

Dev OOF, same protocol, paired bootstrap over the 372 pairs:

| Attempt | DT | SP | CI | Overall | Verdict |
|---|---|---|---|---|---|
| **hs_v4sf** self-trained detector scores (raw + within-pair pct/top/log) in the event and type heads | blend 0.7895 -> 0.7626 | 0.7652 -> 0.7557 | 0.7587 -> 0.7567 | **-0.0127, P(better) 0.002** | rejected |
| **EM step**: censored tail rows added with soft labels from the DP | DP 0.774 -> 0.664 | 0.756 -> 0.675 | 0.748 -> 0.547 | -0.11 DP | rejected |
| **Sequence-model margins** (`seq_cache/oof_evbag_seq.npy`) in the heads | blend -0.0125 | -0.0082 | DP +0.0204, blend **+0.0037** | +0.001 | rejected; 2-4 h of eval re-scoring for ~+0.001 final |

The pattern is the one exp027ev and exp032b already found: better *planted-hand detection* is the wrong axis for *listing*. The event head is already at AUC 0.991-0.995 against rule-certain negatives; what is missing is the latent activation bit, which no available feature block carries.

### 6d. What this implies for the rank-3 target

Rank 3 needs evidence ~0.78 overall. Known-family pairs are already at 0.774 eval-equivalent and the gate adds about 0.015, so the realistic evidence ceiling with the current routing is **~0.715**. Reaching 0.78 would require the unexplained 14% to score ~0.8, which needs the OC mechanism itself, not a better listing model. The remaining headroom for the final score is in pair AP (0.973) and behaviour AP (0.966), not in evidence.
