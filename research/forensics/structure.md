# Structure, labels, hard negatives and validation design (audit 2026-09-15)

Code: `scripts/audit_structure.py` (stages `build` ~16 s for both tables plus the w2000 window, `analyze` ~20 s, `extra` ~30 s; peak RAM < 3 GB).
Numbers: `research/forensics/structure_results.json`; per-feature AUCs are in `structure_feature_separation.csv` and `..._by_family.csv`.
Intermediates: `_audit_pair_feats_dev.parquet` (pair rate features for dev pairs with 57 or more shared hands, including pos/neg/unk group), `_audit_pu_oof.npy`, `_audit_tab_players.npy`.
Tables built: `data/derived/player_stats_{dev,eval,dev_w2000}.parquet` and `pair_generic_{dev,eval,dev_w2000}.parquet` (schema in sections 5 and 6).
My computed eval `shared_hands` matches the host's `evaluation_pairs.shared_hands` for all 112,540 pairs.

## 0. Headline findings

1. **Collusion is PHASE-LOCAL.** Labelled positive pairs show no signal in the eval phase.
   - Dev AUC vs negatives is 0.73 to 0.98. In eval the same features score 0.43 to 0.52 (281 labelled positives with 38 or more eval shared hands).
   - Median both_vpip_rate for positives: dev 0.187 (negatives 0.073). In eval: 0.070 (negatives 0.077).
   - The reverse also holds. The top-100 eval-phase suspicious pairs have a mean dev-phase percentile of 0.44, and 1% of them are in the dev top 1%. Spearman between dev and eval pair scores is 0.16.
   - So eval targets are pairs that collude in the eval phase. Their dev-phase hands are clean. Use them as a **clean per-player/per-pair baseline** for "deviation" features, not as signal.
2. **The hard negatives are not hard.** On every one of 115 pair/player/metadata features, the 1488 `confirmed_non_target` pairs look the same as random unknown pairs that share 57 or more dev hands.
   - max |AUC−0.5| = 0.035. A LightGBM OOF gives neg-vs-unknown AUC 0.494.
   - Tail rates on the tilt, weak-play, similar-strategy, opponent-selection, streak and strategy-change axes are all about 5% at the unknown p05/p95, the same as the baseline.
   - KMeans clusters of negatives absorb unknown pairs in the same proportions.
   - So negatives are, in effect, a random clean sample. Labelled-only AP measures ranking against random clean pairs at 20% prevalence. No feature separates positives from unknowns while failing to separate them from negatives, or the reverse.
3. **Eval universe rule, reproduced exactly:** within-table pair, eval shared hands ≥ 38, not a labelled pair, and neither player is a labelled positive. This gives 112,569 candidates against 112,540 actual pairs.
   - The 29 extra candidates show no pattern. Treat that as a file-construction artifact and do not use it.
   - Dev universe: unlabelled pairs with dev shared ≥ 38 number exactly 155,812, the host's "unknown" count.
   - Every labelled pair (positive and negative) has dev shared ≥ 57, i.e. 1.9% of 3000 hands, the same rate as 38/2000 in eval.
4. **Estimated number of positives** (mixture estimate; clean reference = negatives or matched-window unknowns). These cover only positives that look like the three known families:
   - Eval: about 350 to 600 among 112,540 pairs (prevalence about 0.3 to 0.5%).
   - Dev unknown (57 or more shared hands): about 140 to 220 hidden dev-active positives, about 35% of all dev positives.
   - Both 95% CIs reach 0, because 0 of 1488 negatives score above the high-precision thresholds.
5. **Seating, co-presence, sessions and metadata carry no signal.** AUCs are all between 0.48 and 0.53.
   - Partners are adjacent in 41.0% of shared hands, against 40.0% for random pairs.
   - Co-presence lift: 0.965 for positives vs 0.956 for unknowns. Co-join and co-leave AUCs are 0.53 and 0.52.
   - Metadata same-field rates and account-age gap show no lift.
6. **Strong single-feature signals exist per family** (dev, positive family vs negatives):
   - directed_transfer: gross_flow_per100 AUC 0.961, sd_together_rate 0.940.
   - soft_play: both_saw_flop_rate 0.952; min over directions of (aggression while partner active − aggression otherwise) AUC 0.118, i.e. 0.882 when inverted.
   - coordinated_isolation: the same aggression-lift feature 0.976, both_aggr_rate 0.957, reraise_partner_rate 0.925.
   - LightGBM on about 45 generic rate features: OOF labelled-only AP 0.979, AUC 0.990.

## 1. Pair universe

| quantity | value |
|---|---|
| within-table pairs (400×435) | 174,000 |
| dev pairs sharing ≥1 dev hand | 173,862 (138 share none) |
| pairs sharing ≥1 eval hand | 172,878 |
| dev shared hands, all pairs: p10/p50/p90 | 39 / 95 / 179 (max 475) |
| eval shared hands, all pairs: p10/p50/p90 | 19 / 61 / 129; eval/dev median ratio 0.642 |
| unlabelled dev pairs with shared ≥1 / ≥20 / **≥38** / ≥57 | 172,002 / 167,698 / **155,812** / 135,879 |
| labelled dev shared (min / median / max) | pos 57 / 112 / 303; neg 57 / 108.5 / 465 |
| eval pairs that are labelled pairs | 0 |
| eval pairs containing a labelled-positive player | 0 |
| **eval pairs containing a labelled-NEGATIVE player** | **42,539 (37.8%)**; none of these also contain a positive player |
| eligible (unlabelled, no positive player) | 153,450; with eval shared ≥38: 112,569 |
| eligible with eval shared ≥38 but not in eval | 29 (no structure; do not use) |
| players: positive / negative / both | 693 / 2,599 / 165 |

Per table:
- **Positives per table:** {0:155, 1:155, 2:63, 3:18, 4:8, 5:1}; 245 tables have at least one.
  - A zero-truncated Poisson fit gives λ=0.903, which predicts 162 zero tables; 155 are observed. Variance/mean = 1.01.
  - So tables without positives are ordinary Poisson zeros, not held-out tables.
  - Table-level stats barely differ between the two groups: max pair co-presence lift AUC 0.53, mean VPIP 0.53, max |flow| 0.64 (only because the positive pairs themselves are there).
  - bb=10 tables have slightly more: mean 1.21 positives vs 0.91 at bb=2.
- **Negatives per table:** 0..11, Poisson-like (var/mean 0.93). They are uncorrelated with positives (r=−0.03), and 152 tables with no positives still have negatives. Total 1488 = exactly 4×372, but only 37 tables have exactly 4× their positive count, so the 4× ratio holds globally, not per table.
- **Eval pairs per table:** falls with labelled positives: 0 positives → 316.8; 1 → 274.3; 2 → 243.5; 3 → 214.2; 4 → 194.9; 5 → 168.
- **Evidence:** 5 hands for 340 pairs, 4 for 21, 3 for 11.
  - Hands are front-loaded in dev time: 428/434/338/248/225/144 per 500-hand bin; 80% (1448/1817) have hand_seq < 2000.
  - Span within a pair: median 1370 hands, p10 277. Episodes are spread out, not bursts.

## 2. Positive graph, seating, sessions

- **Positive-player degree:** {1: 644, 2: 47, 3: 2}.
  - The 49 multi-pair players are more than random placement predicts. Placing each table's k positives among its pairs with 57 or more shared hands gives a mean of 22.2 (p95 29).
  - So colluders do reuse partners (chains) beyond chance.
- **Components:** 276 isolated pairs, 40 three-node paths, 4 four-node paths, 1 five-node path.
  - Every component is a tree; there are no triangles or complete rings.
  - 30 of the 45 multi-pair components mix behavior families.
  - 96 of 372 positive pairs share a player with another positive pair.
- **Negatives touching a positive player:** 190 (187 with one positive player, 3 with two), against 163.2 under random sampling (p95 184).
  - That is a small excess of about 27 "colluder with a non-partner" decoys.
  - Their features match the other negatives (median co-presence 0.909 vs 0.905, |flow|/100 23.2 vs 25.9).
- **Negatives sharing a player, per positive pair:** {0: 220, 1: 105, 2: 39, 3: 8}. Negative-player degree: {1: 2250, 2: 321, 3: 28}.
- **Seats:**
  - 6 seats are dealt in every hand (all 2M hands), and the seat order never changes while the lineup is constant (11,942 of 11,942 consecutive-hand checks, tables 0–2 sample).
  - Lineup changes are usually one player (2602 of 3055 changes, same sample).
  - The button seat is not sequential.
  - Player sessions average about 21 hands, with about 28 joins per player in dev.
  - Share of shared hands where partners sit adjacent (offset 1 or 5): positive pairs 0.4104 (DT 0.409, SP 0.425, CI 0.393), negatives 0.391, unknown 0.400 (the random expectation is 2/5). Opposite seat: 0.184 / 0.206 / 0.200. seat_adjacent_rate AUC is 0.517.
- **Co-presence** (shared × H / (n1 × n2)): mean positives 0.965, negatives 0.949, unknown 0.956; AUC 0.507. Co-join within ±2 hands AUC 0.530; co-leave 0.515.
  - Partners do not join or leave together and do not pick seats.
  - The "repeated opponent selection" non-target pattern does not appear as a co-presence excess in the labelled data.

## 3. Hard negatives vs positives vs unknown

All numbers are dev phase, pairs with 57 or more shared hands (372 pos / 1488 neg / 135,879 unk).

| feature | AUC pos-vs-unk | AUC pos-vs-neg | AUC neg-vs-unk | median pos / neg / unk |
|---|---|---|---|---|
| both_vpip_rate | 0.881 | 0.882 | 0.489 | 0.187 / 0.073 / 0.075 |
| sd_together_rate | 0.856 | 0.856 | 0.498 | 0.043 / 0.009 / 0.009 |
| both_saw_flop_rate | 0.841 | 0.845 | 0.486 | 0.149 / 0.043 / 0.046 |
| hu_flop_rate | 0.811 | 0.816 | 0.486 | |
| gross_flow_per100 (bb per 100 shared) | 0.806 | 0.807 | 0.494 | 209 / 60 / 62 |
| hu_final_rate | 0.798 | 0.796 | 0.496 | |
| fold_to_partner_max | 0.239 | 0.233 | 0.511 | 0.62 / 0.76 / 0.76 |
| abs_net_flow_per100 | 0.734 | 0.732 | 0.498 | 83 / 25 / 25 |
| both_aggr_rate | 0.713 | 0.718 | 0.492 | |
| player-level (best): vpip_avg_strength max | 0.325 | 0.334 | 0.491 | colluders enter pots with weaker holdings |
| weak_vpip_rate min | 0.670 | 0.666 | 0.503 | |
| metadata same_* / age_gap | 0.485–0.508 | 0.49–0.52 | 0.48–0.51 | none |
| copresence_lift / seat_adjacent / cojoin | 0.507 / 0.517 / 0.530 | | ~0.5 | none |

- **Danger lists are empty.** No feature has |AUC_pu−0.5|>0.1 with |AUC_pn−0.5|<0.05, and none has |AUC_nu−0.5|>0.1.
- **Tail-rate test** (share above unknown p95 / below p05; about 0.05 means no enrichment):
  - Negatives: tilt_vpip_delta max 0.047/0.056, weak_vpip max 0.037/0.069, vpip absdiff 0.035/0.062, copresence_lift 0.054/0.051, maxup max 0.061/0.051, drift_pfr max 0.051/0.058. All at baseline.
  - Positives are enriched only on flow and streak axes: gross_flow above p95 40.6%, |net flow| 34.9%, maxup 12.9%, maxdd 11.8%.
- **KMeans on negatives** (k=6, robust z vs unknown) finds tilt, weak-play, strategy-difference, streak and drift clusters of 185 to 359 pairs. Unknown pairs fall into the same clusters in similar shares (20k sample: 3434/2494/4492/4061/2874/2645), so these are population personas, not built negatives.
  - The tilt, weak-play and similar-strategy decoys the host describes are **player personas present across the whole population (unknown and eval too)**.
  - They are not concentrated in the labelled negatives. A model trained against the labelled negatives meets them at the same rate in eval.
- **Dev→eval stability of player stats** (Spearman): vpip 0.875, weak_vpip 0.767, pfr 0.743, aggr 0.712, 3bet 0.505, af 0.478, fold_to_bet 0.34, wtsd 0.264, bb100 0.048, hands 0.327.
  - Personas are stable, so the other phase is a valid baseline.
  - Positive-only players (528): VPIP 0.332 in dev → Δ−0.017 in eval; others +0.009. This fits collusion being switched off in eval.

## 4. Player metadata (allowed)

| rate | pos | neg | eval pairs | unknown (touching pos player) | unknown (other) |
|---|---|---|---|---|---|
| same experience bucket | 0.255 | 0.272 | 0.285 | 0.292 | 0.289 |
| same preferred stake | 0.454 | 0.450 | 0.438 | 0.438 | 0.433 |
| same region | 0.304 | 0.278 | 0.312 | 0.306 | 0.315 |
| same client | 0.403 | 0.388 | 0.395 | 0.390 | 0.393 |
| age gap median (days) | 715.5 | 735.5 | 727 | 723 | 720 |
| all 4 fields equal | 0.013 | 0.015 | 0.015 | 0.015 | 0.016 |

- Metadata carries no signal (AUC 0.48 to 0.52).
- preferred_stake is independent of table stake (bb=2 tables: 3584 micro / 2300 low / 656 mid players).
- Include it only as a cheap feature and expect about zero gain.

## 5. `player_stats_{dev,eval,dev_w2000}.parquet` (12,000 rows per phase)

- **Keys:** `player_idx, table_idx, phase`.
- **Raw counts:**
  - `hands, vpip, pfr, tb_opp, tb, limp, pre_acts, pre_aggr, fold_pre`
  - Postflop: `post_aggr, post_calls, post_checks, post_folds, post_faced, post_fold_faced`
  - Showdown and result: `saw_flop, wsd, wsd_won, net_bb, won_hands, stack_bb, allin`
  - Hand strength: `weak_dealt, weak_vpip` (Chen percentile < 0.40), `strong_dealt, strong_foldpre` (≥ 0.92), `vpip_pct_sum`
  - Halves: `h1_*`/`h2_*` (hands, vpip, pfr, net, aggr, acts), split at hand_seq 1500 in dev, 4000 in eval, 1000 in w2000
  - Tilt: `tilt_hands, tilt_vpip, tilt_pfr, tilt_net` (the 10 own hands after losing ≥30 bb), `bigloss_events`
  - Swings and sessions: `maxdd_bb, maxup_bb` (cumulative bb path), `joins` (session starts), `tot_acts, tot_aggr`
- **Rates:**
  - Preflop: `seat_share, vpip_rate, pfr_rate, threebet_rate` (re-raise when facing exactly one prior raise), `limp_rate`
  - Aggression: `af_post` (post aggr/calls), `afq_post`, `aggr_rate` (all streets)
  - Showdown and results: `wtsd` (wsd/saw_flop), `wsd_won_rate, saw_flop_rate, sd_freq, bb100, fold_to_bet_post, avg_stack_bb, allin_rate`
  - Hand strength: `weak_vpip_rate, strong_foldpre_rate, vpip_avg_strength, won_hand_rate`
  - Drift (second half − first half): `drift_vpip, drift_pfr, drift_aggr, drift_bb100`
  - Tilt (tilt window − overall): `tilt_vpip_delta, tilt_pfr_delta, tilt_bb100_delta`
  - Sessions: `mean_session_len`
- **Conventions:**
  - Aggressive = bet, raise, or all_in with amount > to_call. Call = call, or all_in with amount ≤ to_call.
  - Blinds are not in the action log.

## 6. `pair_generic_{dev,eval,dev_w2000}.parquet` (174,000 rows = every within-table pair)

- **Keys:** `table_idx, p1<p2, phase`.
- **Symmetric counts over shared hands:**
  - `shared_hands`
  - `both_vpip`, `both_saw_flop`, `sd_together`
  - `hu_final`: the last two players left, i.e. both not folded, or the winner plus the last folder
  - `hu_flop`: hu_final where both saw the flop
  - `checkdown_hu`: hu_flop with no postflop aggression
  - `both_aggr`: both made an aggressive action in the hand
  - `seat_opposite`
  - `cojoin, coleave`: sessions starting or ending within 2 hands of each other, with overlapping sessions
- **Directed (_12 = p1 toward / losing to p2):**
  - Flow:
    - `flow_*_bb`: each hand, p1's loss × p2's win / total winnings / bb. Also `flow_hu_*` (hu_final hands only) and `flow_sd_*` (both at showdown).
  - Seating:
    - `off1_12`: p2 sits directly after p1 (acts after); `off1_21` is the reverse.
    - `off2_*`: two seats away.
  - Actions with the partner still in the hand:
    - `acts_v_*`, `aggr_v_*`: actions and aggressive actions by p1 while p2 has not folded; `_post` variants cover streets ≥ flop.
    - `acts_hu_*`, `aggr_hu_*`: the same with players_active==2.
  - Facing the partner:
    - `faced_*`: p1 faces to_call>0 with p2 the last aggressor on the street.
    - `fold_faced_*`, `reraise_faced_*`, `faced_post_*`, `fold_faced_post_*`
    - `faced_pre_strong_*`, `fold_faced_pre_strong_*`: preflop, holding a Chen percentile ≥ 0.85
  - `sd_win_*`
- **Derived:**
  - Hands and co-presence: `n1_hands, n2_hands, joins_1, joins_2, copresence_lift` (shared × H / (n1 × n2), H = 3000/2000/2000), `copresence_min`
  - Net flow: `net_flow_12_bb`, `net_flow_hu_12_bb`, `net_flow_sd_12_bb` (positive = value moves p1→p2)
  - Aggression: `aggr_rate_1v2, aggr_rate_2v1`, `aggr_rate_1_not2, aggr_rate_2_not1` (aggression rate in actions while the other is not active), `aggr_rate_hu_1v2/2v1`
  - Folding and seating: `fold_rate_1_to_2/2_to_1`, `seat_adjacent_rate`
- **`dev_w2000`:** built from dev hands with hand_seq 0..1999, the same length as eval. Use it for eval-like CV.

## 7. Validation design

- **Folds.**
  - `data/folds_tables_5.csv` is already frozen (md5 322a34c8454910717d98b6a95fdeadea, 80 tables per fold). It was verified, not regenerated.
  - Per fold: positives 91/75/58/62/86, negatives 319/327/294/293/255, tables with positives 54/48/40/48/55.
  - Coordinated_isolation per fold is 24/12/15/17/24; fold 1 is thin, so average behavior AP over folds and do not read single folds.
  - Every player sits at one table, so table folds leak no players and the "no labelled-positive player" rule holds across folds automatically.
- **Train:** dev-phase features of labelled pairs, plus optionally unknown pairs (57 or more shared) as down-weighted negatives. The quick PU LGBM used weight 20 on positives against all unknowns.
- **Scoring views** (report all three; decide on C):
  - **A. Labelled-only pair AP** (positive vs negative, 20% prevalence).
    - Unbiased for ranking against random clean pairs, because negatives equal unknown in distribution.
    - It saturates: the quick model gets 0.979 (per fold 0.972 to 0.992), so it hides differences.
  - **B. Pool AP on held-out tables.**
    - Positives vs negatives plus every unknown pair with 57 or more dev shared hands (prevalence 0.27%).
    - Biased down by an estimated 140 to 220 hidden dev positives counted as FPs; a perfect detector would score about 0.63 to 0.7.
    - Quick model: 0.577 (per fold 0.53 to 0.69). Dropping unknown pairs that touch a positive player gives 0.607.
    - Keep those pairs in the primary number: eval contains colluder-with-non-partner pairs whose colluders are unlabelled.
  - **C. Matched-window pool AP (primary).**
    - Apply fold models to `pair_generic_dev_w2000` features of held-out tables, with shared ≥ 38 as in eval: 328 pos / 1319 neg / 119,349 unk.
    - Quick model: labelled-only AP 0.954, AUC 0.979, pool AP 0.510.
    - This view carries both the length shift (2000 vs 3000 hands) and the prevalence shift.
  - **Evidence MAP@5** on dev evidence; the relevant set is the 3 to 5 listed hands.
    - It is a lower bound, because unlisted planted hands count as misses.
    - For the w2000 view, restrict relevant hands to hand_seq < 2000 (1448/1817 hands; 363/372 pairs keep at least one).
  - **Behavior macro-AP:** host formula on OOF; the class score is risk if the predicted family matches.
- **Simulated exclusion rule.**
  - A random 50/50 within-table split of positive pairs (200 reps) leaves **86.6%** of held-out positives surviving "no pair containing a labelled-positive player".
  - 96/372 positive pairs are chained.
  - The eval set has had this pruning, so eval positives are slightly under-chained compared with dev. A graph prior ("a flagged player's other pairs") is weaker in eval than dev CV suggests.
- **Dev→eval shift.**
  - Eval shared hands are about 0.64× dev (eval-pair median 76, dev labelled median 110).
  - Rate features keep their medians: both_vpip 0.075→0.080, hu_final 0.060→0.059, aggr_rate_1v2 0.179→0.182.
  - Tails widen: copresence_lift p90 1.36→1.49, |net flow|/100 p90 122→153, seat_adjacent p10/p90 0.11/0.71→0.05/0.77.
  - Use rates, apply shrinkage or log-odds with pseudo-counts, never raw counts. Clean negatives in eval hit high thresholds about 3× more often: FPR at 80/90% dev-recall thresholds is 0.61%/2.1% in eval vs 0.13%/0.67% in dev.
  - Labelled positive pairs in eval are inactive (section 0.1). **Do NOT use their eval-phase hands as positives**; they are extra clean eval-length examples (281 + 1149).
- **Positive-count estimate** (mixture proportion: excess of the target pool over clean negatives above thresholds set by positive recall).
  - Dev unknown (57 or more shared): 169/174/217/191 hidden positives at 30/50/70/80% recall. Matched window: 139/210/145.
  - Eval (w2000-calibrated): 356 to 487 at 30% recall, 343 to 540 at 50%, 447 to 583 at 70%.
  - All CIs include 0 on the low side. That is an artifact of 0/1488 negative hits; the point estimates agree across thresholds.
  - An eval pool AP of 0.9 implies ranking about 450 positives above about 112k clean pairs.

## 8. Recommendations and caveats

- **Build the model on phase-local, rate-normalized interaction features.** Add player-baseline deviation computed from the *other* phase (the dev baseline for eval pairs is clean).
- **Do not spend effort on seat, co-presence, session or metadata features.** The measured lift is about 0.
- **Use pool AP on the matched 2000-hand window (view C) as the pair-AP decision metric.** Keep labelled-only AP as a sanity check.
- **Hard-negative-specific modelling is unnecessary.** Labelled negatives are a random clean sample; unknown pairs (57 or more shared) at low weight are fine, and the ~0.13% contamination is harmless.
- **Gray zone, lead to decide:** listed evidence is front-loaded in time. That may reflect how the host lists evidence, so treat any "prefer earlier hands" tie-break as a testable, rule-sensitive hypothesis.
- **Suggested LB probe:** submit an eval ranking by dev-phase score. The prediction is LB pair AP near random, which would confirm phase-locality on the private labels.
- **What would overturn these findings:**
  - Phase-locality: an LB where dev-phase-only scores beat random clearly.
  - "Negatives are random": a hand-level feature, e.g. equity-based, with neg-vs-unknown AUC far from 0.5.
  - The positive-count estimate: a model with eval pool AP well above what about 450 positives implies.
