# POLICY.md: exp003 label-free normal-play action policy

Written 2026-09-15. Code: `src/policy.py` (stages `states | train | predict | report | validate | all`).
Output: `data/derived/action_policy.parquet`, **18,609,028 rows** (every action in both phases), 452 MB.
Logs: `logs/exp003_policy_{states,train,predict,report,validate}.log`. Metrics JSON: `data/derived/policy_models/{train_meta,report,validate}.json`.

## What it is

For every action, the model gives the probability that a *normal* player takes each of 4 action classes in that exact
spot: fold, check, call, or aggressive. It also gives `surprise = -ln p(actual class)`.

The model sees only what a real player sees:
- the public state of the hand;
- the actor's own hole cards;
- the actor's perceived strength: `hs_actor` = P(beat one random hand on the current board). This does not use the omniscient `eq_*` columns.

It never sees opponents' cards, player identities, player metadata, or labels.
- **Labels are used for cleaning only.** Training excludes every action of the 69,496 hands in which both players of a labelled-positive pair are seated. Those actions are still predicted.
- **Cross-fitting.** `half = frozen table fold % 2`. The model trained on one half's tables predicts the other half, so every probability comes from out-of-table data, in both phases.

Classes are: 0 fold, 1 check, 2 call (call, or all_in with amount ≤ to_call), 3 aggressive (bet, raise, or all_in with amount > to_call).

Legal-action renormalisation:
- to_call = 0: {check, agg}.
- to_call > 0: {fold, call, agg}.
- agg is illegal when stack_before ≤ to_call.
- A fold with to_call = 0 never occurs in the data (0 of 18.6M), so it is masked.

## Decision-state features (66, numba state machine, 40 s for all 18.6M actions)

The state machine replays each hand from the blinds. It matches the logged `to_call`, `stack_before` and `players_active` on **all 18.6M actions with 0 mismatches**.

Position conventions (verified): SB = btn+1, BB = btn+2, UTG = btn+3 acts first preflop, and blinds are already deducted in `stack_before`.

| group | features |
|---|---|
| position / players | street; pos_pf (UTG0 HJ1 CO2 BTN3 SB4 BB5); pos_post (SB0 … BTN5); n_active; n_can_act_oth (others not all-in); n_allin_oth; n_left_to_act (others who still act if the actor just calls); n_after (active others later in street order); last_aggr_rel (last aggressor acts after (+1) or before (−1) the actor) |
| money | to_call_bb; pot_bb; stack_bb; pot_odds = to_call/(pot+to_call); spr; eff_stack_bb and eff_spr (vs the deepest live opponent); tocall_frac_stack; call_is_allin; bb |
| street history | n_agg_street (0 unopened, 1 bet/open, 2 3-bet, 3+); n_agg_hand; n_calls_street; n_calls_since_agg; n_checks_street; bet_level_bb; last_inc_bb (raise increment); last_agg_frac_pot (last aggressive amount / pot at that time); last_aggr_pos_pf; last_aggr_allin; n_active_street_start; n_agg_prev_street |
| actor history | act_street_inv_bb (already invested this street); act_hand_inv_bb; committed_frac; act_n_act_street; act_n_agg_street; act_n_agg_hand; act_init_prev (actor was last aggressor on the previous street); act_is_pf_aggr |
| own preflop strength | pf_eq_vsN (class equity vs min(max(n_active−1,1),5) random hands, from `preflop_equity_169`); pf_eq_vs1; pf_eq_vs5; pf_top_vsN; sklansky; chen; hi_rank; lo_rank; is_pair; is_suited |
| own postflop strength | hs_actor; hs_powN = hs^(n_active−1); rank_norm (dense rank on the current board / 7461, from `seat_strength`); own_cat; board_cat; hole_improves (own_cat > board_cat); top_pair; overpair; n_hole_match; n_overcards; flush_draw; straight_draw (4 distinct ranks in a 5-window with a hole card) |
| board texture | nboard; board_max_suit; board_mult (max rank multiplicity); board_high; board_str_win (max distinct board ranks in any 5-rank window) |

The feature states are kept in `data/derived/policy_states/part_00..07.parquet` (820 MB). Downstream detectors can reuse them, joining on (hand_idx, action_no).

Top gain features, the same in both models: pot_odds 0.28, to_call_bb 0.28, rank_norm 0.18, pf_eq_vs5 0.05, pf_top_vsN 0.04, n_agg_street 0.04, own_cat 0.03.

## Training

- **Model.** LightGBM multiclass: lr 0.1, num_leaves 127, min_data_in_leaf 200, feature_fraction 0.9, bagging 0.7, λ2 = 1, 6 threads, early stopping after 30 rounds without improvement.
- **Sample.** Street-stratified, with rates of preflop 0.25, flop 0.70, turn 0.75, river 1.0. Sampling on a covariate leaves p(y|x) unbiased.
- **Holdout.** 5% of training-table hands.
- **Unequal halves.** fold%2 gives 240 vs 160 tables, so model 0 trains on the smaller half.

| model | trained on | train rows | holdout rows | best iter | holdout mlogloss |
|---|---|---|---|---|---|
| half0 (predicts half 0) | half-1 tables | 2,704,484 | 142,434 | 99 | 0.5613 |
| half1 (predicts half 1) | half-0 tables | 4,095,759 | 214,142 | 115 | 0.5659 |

Early stopping near 100 rounds means the achievable logloss is limited by the generator's own randomness, not by model capacity.

## Held-out metrics (cross-fitted, all 18.6M actions)

The baseline is class frequencies per (street, facing, n_agg_street clipped at 3).

| street | actions | logloss | accuracy | baseline logloss |
|---|---|---|---|---|
| preflop | 13,228,162 | **0.5451** | 0.771 | 0.8439 |
| flop | 2,852,419 | **0.5744** | 0.736 | 0.7438 |
| turn | 1,614,275 | **0.6076** | 0.716 | 0.8184 |
| river | 914,172 | **0.5804** | 0.733 | 0.8145 |
| all | 18,609,028 | 0.5568 | 0.759 | — |

- **Phases.** Dev and eval score the same (preflop 0.541 dev vs 0.551 eval; postflop within 0.002), so the policy transfers across phases.
- **Positive-pair hands are more surprising.** Hands with a labelled-positive pair seated (not used in training) score 0.600 / 0.607 / 0.640 / 0.621, against 0.543 / 0.573 / 0.606 / 0.579 for clean hands.

**Calibration of p_call facing a preflop raise** (held-out). Each cell is mean predicted → observed rate (n):

| bin | [0,.05) | [.05,.1) | [.1,.15) | [.15,.2) | [.2,.3) | [.3,.4) | [.4,.5) | [.5,.6) | [.6,.7) | [.7,.8) |
|---|---|---|---|---|---|---|---|---|---|---|
| pred → obs | .023→.024 (1.42M) | .070→.072 (1.00M) | .124→.127 (571k) | .174→.177 (509k) | .250→.252 (851k) | .348→.352 (935k) | .454→.453 (653k) | .530→.524 (302k) | .645→.647 (18k) | .729→.700 (8k) |

Postflop p_agg is calibrated just as well: in every bin, predicted and observed differ by at most 0.005.

## Usefulness validation (dev, labelled pairs, `stage_validate`)

**DT pfcall_weak_max.** For each orientation (X, Y), count the dev shared hands in which X calls a preflop bet/raise whose last aggressor on the street is Y, with p_call ≤ t. Take the max over orientations, either as a count or divided by the number of dev shared hands.

| statistic | DT vs 1488 NEG AUC / AP | DT vs SP+CI AUC / AP | SP+CI vs NEG AUC |
|---|---|---|---|
| pfcall_weak_max @0.15 | **0.9930** / 0.947 | 0.9889 / 0.978 | 0.563 |
| pfcall_weak_max_rate @0.15 | **0.9933** / 0.955 | 0.9892 / 0.984 | 0.565 |
| pfcall_weak_max @0.10 | 0.9939 / 0.959 | 0.9896 / 0.982 | 0.564 |
| pfcall_weak_max_rate @0.10 | **0.9949 / 0.968** | **0.9908 / 0.987** | 0.565 |
| forensics subset policy (count @0.15) | 0.9943 / 0.950 | 0.991 | 0.556 |

- **Threshold rules** (count @0.15):
  - ≥ 3: DT 95.3%, NEG 4.6%, SP 3.0%, CI 8.7%.
  - ≥ 5: DT 89.2%, NEG 0.87%, SP 0%, CI 2.2%.
- **Responses to the partner's preflop raise**:

  | group | call rate | weak-call rate | spots |
  |---|---|---|---|
  | DT evidence hands | 0.859 | 0.482 | 786 |
  | DT non-evidence hands | 0.496 | 0.270 | 4,785 |
  | NEG | 0.194 | 0.026 | 36,444 |

  The spot counts equal the forensics counts exactly.

**SP passed-up raises.** Responses are actions by X where the last aggressor on the street is the partner Y (dev shared hands).

| group | responses | mean p_agg (all) | **mean p_agg at passive responses (fold/call)** | observed agg rate |
|---|---|---|---|---|
| SP evidence | 957 | 0.199 | **0.196** | 0.011 |
| SP non-evidence | 3,140 | 0.085 | 0.070 | 0.035 |
| DT evidence | 1,902 | 0.088 | 0.072 | 0.071 |
| CI evidence | 665 | 0.080 | 0.040 | 0.328 |
| NEG | 49,052 | 0.103 | **0.059** | 0.102 |

SP partners pass up raises that a normal player makes about 20% of the time, 3.3× the negative-pair level. They raise 1.1% of the time, against 10.2% for negative pairs.

The hand score `max_miss` is the max p_agg over passive partner responses:
- SP evidence vs NEG hands: AUC **0.959**.
- SP evidence vs SP non-evidence hands: AUC 0.962.
- The forensics count-model `max_miss` scored 0.957.

## Output schema: `data/derived/action_policy.parquet`

Rows are sorted by (hand_idx, action_no).

| column | type | meaning |
|---|---|---|
| hand_idx, action_no, player_idx | i32, i16, i32 | keys |
| street | i8 | 0..3 |
| y | i8 | actual class: 0 fold, 1 check, 2 call, 3 agg |
| n_agg_street | i8 | aggressive actions before this one on this street |
| last_aggr_player | i32 | player_idx of the last bet/raise/aggressive all-in on this street (−1 = none) |
| to_call_bb | f32 | to_call / bb |
| legal | i8 | bitmask: 1 fold, 2 check, 4 call, 8 agg |
| p_fold, p_check, p_call, p_agg | f32 | renormalised over legal actions; they sum to 1 (max deviation 1.2e-7) |
| p_act | f32 | p of the actual class |
| surprise | f32 | −ln max(p_act, 1e-6) |
| p_agg_nf | f32 | p_agg / (1 − p_fold): aggression given the actor continues |
| half | i8 | cross-fit half (table fold % 2) |
| clean | bool | False = a labelled-positive pair is seated together (hand excluded from training) |

Mean surprise by actual class: fold 0.30, check 0.46, call 1.27, agg 0.68.

## Runtime (i5-10400F, 6 threads; peak RSS not measured, estimated at about 3–4 GB from the array sizes)

| stage | time |
|---|---|
| states | 40 s |
| train (2 models) | 204 s |
| predict (18.6M × 2 boosters) | 195 s |
| report | 21 s |
| validate | 4 s |

The whole pipeline, `python src/policy.py all`, runs in about 8 minutes.

## Caveats and next levers

- **Population policy.** There are no player-style features. A tight or loose normal player's routine actions count as "surprising", so aggregate surprise must be compared with the player's own baseline, or with the pair-vs-others contrast. Adding player-level style rates, computed out-of-fold, would sharpen surprise, but it could also absorb pair-persistent collusion traits such as the DT donor's flat-calls. Build it as a separate variant rather than replacing this one.
- **Colluders in training.** Unlabelled colluders (eval-phase positives and hidden dev positives) remain in the training data. They make up only a small share of all actions, so they shift the policy only slightly toward collusion-like play.
- **hs_actor is a one-random-hand measure.** It ignores range narrowing from the betting. `rank_norm` and `own_cat` carry more gain.
- **Unequal halves.** Model 0 trains on 160 tables. Half-0 predictions score about 0.004 logloss worse preflop, which may also reflect table heterogeneity.
- **Possible cheap gains, not tried.** Equalising the halves' sample sizes, and lr 0.05 with more rounds, would each probably be worth ≤ 0.002 logloss.
- **Bet sizing is not modelled.** Only the class is predicted. A sizing head, for example amount/pot quantiles given agg, is a possible extension for detecting a min-bet into a partner.
