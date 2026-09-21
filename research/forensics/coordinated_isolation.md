# Forensics: `coordinated_isolation` (CI) evidence hands

Script: `scripts/forensics_coordinated_isolation.py` (stage-based, re-runnable; stage list in its docstring).
Intermediate outputs: `research/forensics/ci_*.parquet`, archetype print-out `research/forensics/ci_archetypes.txt`.
Data: 92 CI positive pairs, 460 evidence hands (exactly 5 per pair), 10,820 non-evidence dev shared hands of those
pairs, 400 confirmed-negative control pairs (48,969 shared dev hands), 725 DT and 632 SP evidence hands. For
pair-level work: all 1,860 labelled pairs plus 3,000 unknown dev pairs (dev_shared >= 38, no labelled-positive
player), 557,825 pair-hands.

## TL;DR

1. **What the planted action is.** A partner makes a **preflop raise with a hand the population almost never raises
   in that spot**, usually as the **first raiser from early position (UTG) with the other partner still to act
   behind**. The other partner then **folds** to it (preflop in 75% of cases) or **re-raises over it** (a
   whipsaw/squeeze: X raises, Y re-raises, X folds). Third players fold to the pressure. 97.2% of evidence hands
   start with the first voluntary pair action being a raise, and 96.5% contain "partner folds after partner's
   aggression". 69.4% contain a raise with population P(raise | class, position, prior raises, prior callers) < 0.02,
   against 0.39% of confirmed-negative shared hands (178x lift).
2. **Hands with this signature are far more common than the 5 listed.** CI members raise trash at **6.97% of hands
   when the partner is seated and 0.25% when not** (negatives: 0.196% vs 0.200%). That works out to about 15
   signature hands per pair in dev, uniform over time. The **listed evidence is roughly the chronologically first
   ~5 "scenario" hands** (UTG or early-position weak open with the partner behind). In that set, P(listed) is
   0.92, 0.85, 0.69, 0.34, 0.13 and then 0 for ranks 0 through 5+. Evidence ranks are chronological in 88% of pairs.
3. **The activation is phase-specific.** Labelled CI pairs show no signature at all in the eval phase (0.29%
   trash-raise rate per shared hand, the population baseline). 108 eval pairs show it in the eval phase only
   (14.3% per hand, uniform over hand_seq 3000-4999), and their dev-phase rate is 0 at the median. The dev-to-eval
   correlation of the rate is r = 0.08. **Eval pair features must use eval-phase hands only.**
4. **Pair level.** On dev, the trash-raise rate per shared hand separates CI from confirmed negatives and from
   unknown pairs with AUC 1.000. AP against confirmed negatives plus unknown pairs (2% prevalence) is 0.970 for the
   raw rate and **0.992 for the lift-adjusted excess count**, which removes the spill-over hard negatives: pairs
   that contain a colluder together with a non-partner.
5. **Evidence MAP@5 on dev CI pairs (GroupKFold by pair).** Random 0.032, public-notebook-style ranking ~0.04,
   trash-raise strength alone 0.12. A transparent rule, "first 5 chronological UTG weak opens", gets 0.586. The
   **LightGBM hand ranker with within-phase chronology reaches 0.686**; the same model without chronology gets
   0.464.

## 1. Population baseline (what "weak" means here)

The generator's normal players are tight. Population preflop P(aggressive action) was computed from all 2M hands,
gameplay only, per 169 class x #prior aggressive actions (0..3+) x position (BTN/SB/BB/UTG/HJ/CO) x #prior
callers (0..2+), with back-off to class x #prior aggressive actions when a cell has fewer than 150 decisions.
Tables: `ci_baseline_pre.parquet` and `ci_baseline_pre_pos.parquet`.
UTG first-in raise rates: 72o 0.08%, 62o 0.09%, J2o 0.17%, 33 1.5%, T7o 2.5%, K4s 2.8%, A7s 22.5%, Q8s 29%,
KTo 41%, ATo 51%, AKo 89%, 99 96%. The overall UTG first-in raise rate is 14.7%.
`pw` = the minimum of this probability over the pair's preflop aggressive actions in a hand.

## 2. Reading the hands (25+ evidence hands and 15+ non-evidence hands read in full)

Recurring structure in evidence hands, with roles A/B being the pair and o1..o4 third players:

- **Trash open + partner folds.** A opens 2h6d UTG, B folds 87o, everyone folds (545545). Other trash opens seen:
  24s, 68o, 62o, 43s, J2o, Q5o, T7o, Q2o, 83o.
- **Isolation 3-bet over a third-party open + partner folds.** o4 opens, B 3-bets 8s2s, A folds, o4 4-bets, B
  folds (1502345). A 3-bets K3o, B folds (1180058).
- **Whipsaw / sandwich.** o4 opens, A 3-bets KTo, B 4-bets 8h2h (22% equity 3-way), o4 folds, A 5-bets, B folds
  (1806911). B opens J6o, A 3-bets 52o (17.7% equity), o2 cold-calls, B 4-bets, A 5-bets to 70bb, o2 folds, B folds
  (1740015).
- **Squeeze over a partner's open, then fold to the partner's 4-bet.** A opens ATo, o3 calls, B squeezes 8d5d
  (26.6% equity), A 4-bets, o3 calls, B folds; A c-bets, o3 folds (1025565).
- **Postflop conflict avoidance (minority).** B opens T7o, A defends the BB, A checks a pair of 4s on Ks4hQh (69.7%
  equity vs B's actual hand), B bets, A folds (1025222).

Non-evidence shared hands of the same pairs split into two groups. Most look like normal tight play (the pair folds
or plays standard ranges). A large minority look exactly like evidence hands: trash 3-bets, whipsaws, and trash
opens with the partner folding (for example 545128, 225394, 905239, 1740002 in `printq` output). Nothing in the
action log separates those unlisted hands from listed ones except **position and time** (sections 4 and 5).

Evidence composition (460 hands):
- Partner re-raise over partner: 36.3%. Partner fold with no re-raise: 61.3%. Neither: 2.4%.
- Partner fold street, among hands with a partner fold: preflop 334, flop 78, turn 17, river 15.
- The trash raise is a first-in open (0 prior raises) in 66.5% of hands and a 3-bet in 27.9%.
- Position of the first pair aggressive action: UTG 73.7%, HJ 18.5%, CO 6.3%, BTN 1.5%, blinds 0%. The partner
  is still to act behind in 97.2%.
- Showdown 5.4%. Median pot 17.75bb. A pair member wins without showdown in 54%.

## 3. Signatures (hand-level, pair = the two labelled players)

Rates are the fraction of hands in each group. EV_CI = 460 evidence hands. same-pair non-ev = 10,820 non-evidence
dev shared hands of CI pairs. NEG = 48,969 dev shared hands of 400 confirmed-negative pairs.

| signature | exact definition | EV_CI | same-pair non-ev | NEG | EV_DT | EV_SP |
|---|---|---|---|---|---|---|
| S1 trash raise | a pair preflop aggressive action (raise, or all-in with amount > to_call) with population p_aggr < 0.02 | 0.694 | 0.097 | 0.0039 | 0.153 | 0.008 |
| S2 weak raise | same with p_aggr < 0.10 | 0.807 | 0.131 | 0.020 | 0.295 | 0.040 |
| S3 first pair vol = raise | the first voluntary preflop action (call/raise) by either pair member is a raise | 0.972 | 0.379 | 0.296 | 0.810 | 0.271 |
| S4 partner fold after aggression | a member folds later in the hand than the other member's first aggressive action | 0.965 | 0.307 | 0.204 | 0.750 | 0.633 |
| S5 partner re-raise | a member's aggressive action while the other member is the street's last aggressor | 0.363 | 0.078 | 0.020 | 0.164 | 0.017 |
| S6 sandwich | member raises, then at least one third player calls or raises, then the other member re-raises (preflop) | 0.185 | 0.031 | 0.006 | 0.019 | 0.011 |
| S7 | S1 and (S4 or S5) | 0.678 | 0.094 | 0.0026 | 0.126 | 0.006 |
| S8 | S3 and S4 | 0.937 | 0.261 | 0.167 | 0.597 | 0.185 |
| S10 | S1 and at least 3 third-party preflop folds facing pair aggression | 0.452 | 0.030 | 0.0011 | 0.086 | 0.000 |
| S11 | S2 or S5 | 0.846 | 0.154 | 0.038 | 0.419 | 0.057 |
| S13 UTG first pair aggression | the first pair preflop aggressive action is by the UTG player | 0.737 | 0.063 | 0.049 | 0.237 | 0.076 |
| S14 UTG weak open | S13 and that action's p_aggr < 0.3 | 0.626 | 0.029 | 0.011 | 0.132 | 0.010 |
| S15 | first pair aggression at UTG/HJ/CO, partner still to act, and (S4 or S5) | 0.939 | 0.202 | 0.131 | 0.495 | 0.135 |
| S16 | S15 and p_aggr < 0.3 | 0.798 | 0.090 | 0.025 | 0.241 | 0.022 |

Evidence hands not covered by S1 (141) are still weaker than normal. Median pw is 0.158, against 0.640 for
negative S3&S4 hands. They also have more third-party folds (2.87 vs 1.91) and more partner re-raises (21% vs 8%).
These are planted hands where the forced raise happened to hold a playable hand.

DT evidence also carries S1 (15%): the donor raises weak to dump chips. SP evidence does not (0.8%).

**Actor roles.** `player_1 < player_2` holds for every pair (canonical string order), so A/B carries no role.
Trash raises are split symmetrically. The mean dominant-member share is 0.584, against 0.599 expected under
Binomial(0.5), and 96.7% of pairs have both members with at least 3 trash raises. There is no fixed aggressor or
folder; either partner can raise and the other yields.

## 4. Temporal pattern

- **Dev activation is uniform and not episodic.** The CI signature rate (S1 per shared hand) by dev third is
  11.3%, 13.5%, 11.7%; for negatives it is 0.36%, 0.38%, 0.43%. Shared hands come in sessions (median gap 1 hand).
  S1 hands are not over-dispersed across sessions (observed/binomial variance 0.79). Gaps between consecutive S1
  hands have median 8, the same as a within-pair shuffle (P(gap <= 10) 0.543 vs 0.550). Session features did not
  help the evidence ranker.
- **Evidence is front-loaded in the dev phase.** Evidence per hand_seq decile: 91 67 69 58 52 29 41 25 22 6, while
  signature hands are flat (136 141 128 158 158 113 151 133 168 83). Evidence ranks are chronological in 81/92
  pairs; in 10 of the 11 exceptions only rank 5 is out of order (it looks appended). For S1 hands, P(listed) by
  chronological rank within the pair is 0.45, 0.45, 0.36, 0.39, 0.33, 0.37, 0.26, 0.23, 0.17, 0.14, 0.16, 0.08, ...,
  and 0 from rank 16. Within the scenario set K = {UTG first pair aggression with p < 0.1}, it is 0.92, 0.85, 0.69,
  0.34, 0.13, then 0. Only 0.12 non-listed K hands per pair occur before the last listed hand. **The listing is
  essentially "the first ~5 CI scenario hands of the phase".**
- **Dev to eval.** Labelled CI pairs are inactive in eval: trash-raise rate 0.29% on 71 eval shared hands (0% of
  pairs >= 5%, 1% mutual-top-lift). The same holds for DT (0.46%) and SP. Among the 112,540 eval pairs, 108 have
  mutual-top-lift and excess >= 5 computed on **eval** hands; their trash rate is 14.3%, flat across eight 250-hand
  bins (182 172 193 187 170 148 183 158). Their dev-phase rate is 0 at the median. Only 20 eval pairs look CI-like
  in dev, and their eval rate is 0 at the median. The rate correlation dev vs eval is r = 0.08. Positives are
  therefore activated for one phase only, with activity spread over that whole phase.
- On all dev pairs (shared >= 38), mutual-top-lift and excess >= 5 flags 83/92 labelled CI, 39 **unlabelled** pairs
  and 2 DT. About 30% of dev CI-like pairs are undisclosed, so remove them from the negatives in PU training (for
  example U4246_6243, U3114_7063).

## 5. Pair-level separability (dev; `pairlevel`, `pairlift`, `confound`)

Means by group: CI / DT / confirmed negatives / SP / unknown.

- rate_C02 (S1 hands per shared hand): 0.133 / 0.043 / 0.0039 / 0.0032 / 0.0038.
- rate_Cfap (S7 per shared hand): 0.130 / 0.033 / 0.0025 / 0.0017 / 0.0024.
- rate_rr: 0.096 / 0.039 / 0.020 / 0.006 / 0.020.
- rate_sandwich: 0.039 / 0.0055 / 0.0054 / 0.0015 / 0.0053.
- CI minimum rate_C02 is 0.056 (6 hands) and its 5th percentile is 0.063. Confirmed-negative maximum is 0.061,
  95th percentile 0.019.

AUC, CI vs confirmed negatives / CI vs unknown:
- rate_C02: 1.000 / 1.000. C02_logsf (binomial tail against p0 = 0.004): 1.000 / 1.000.
- excess_trash: 1.000 / 0.9998. rate_rr: 0.981 / 0.980. rate_sandwich: 0.960 / 0.959. rate_S16: 0.989.
  mutual_top_lift: 0.951 (true for 90.2% of CI pairs, 0% of confirmed negatives, 0.1% of unknown).
- n_shared alone: 0.539 / 0.593.
- AP, CI vs (confirmed negatives + unknown), prevalence 0.020: rate_C02 0.970, C02_logsf 0.980,
  **excess_trash 0.992**, loglift_sum 0.953.

Shared-hand confound:
- rate_C02 AUC within n_shared terciles: 0.9995, 1.0000, 0.9999.
- Subsampling every pair to n shared hands (20 reps), AUC / AP of the S1 rate / AP of the S7 rate:
  n=20: 0.949 / 0.746 / 0.776. n=38: 0.989 / 0.882 / 0.904. n=60: 0.998 / 0.943 / 0.956. n=100: 0.9996 / 0.989 / 0.995.
- Eval pairs have 38-419 eval shared hands. Use a count-aware statistic (binomial log-sf or LLR with p1 ~ 0.13,
  p0 ~ 0.004) instead of the raw rate.

Spill-over hard negatives: confirmed negatives that contain a positive player have rate_C02 mean 0.0075, against
0.0034 for the rest. The top-scoring negatives nearly all contain a CI or DT colluder, because the colluder
trash-raises whenever the real partner is seated, including hands shared with a non-partner. The fix is the directed
lift, X's trash rate with Y seated vs without Y (`lift PHASE`, all players, 2 s in duckdb):
- `excess_trash` = sum over both directions of t_with - n_with * rate_without.
- `loglift_min/sum`.
- `mutual_top_lift`: each member's highest-lift co-player is the other member.

CI vs DT: rate_Cfap AUC 0.986, rate_C02 0.971, rate_sandwich 0.962, mean_pwlog 0.936, rate_rr 0.914. 45 of 148 DT
pairs have rate_C02 > 0.05, so the behaviour classifier must use S5, S6 and S7 plus DT chip-flow features, not S1 alone.

## 6. Recommendations

### Evidence ranking (CI-predicted eval pairs; eval-phase shared hands only)

Dev CI MAP@5 (GroupKFold-5 by pair, `evsim`):

| recipe | MAP@5 | hand AUC |
|---|---|---|
| random | 0.032 | 0.48 |
| earliest shared hands | 0.046 | 0.64 |
| pw only | 0.122 | 0.91 |
| S1 then earliest | 0.277 | 0.86 |
| hand-crafted rule, no chronology | 0.320 | 0.94 |
| rule: first 5 chronological S14 hands (UTG open p<.3) | **0.586** | 0.94 |
| logistic, hand features | 0.469 | 0.97 |
| LightGBM, hand features | 0.464 | 0.975 |
| logistic, hand + prior_cand | 0.670 | 0.983 |
| **LightGBM, hand + prior_cand + prior_scen** | **0.686** | 0.989 |

1. **Recommended: LightGBM binary ranker.** Target = listed evidence. Train on CI dev pairs' dev shared hands.
   Features:
   - Hand features from the numba engine: pw_log, n_pair_trash, first_pair_vol_raise, pair_open, fap, rr, sandwich,
     tf_pre, tf_both_active, third_fold_facing_pair, third_aggr, third_vol, players_vol, pot_log, showdown,
     pair_win_nosd, pf_min, aggA/B_post, pair_hu_post_checks, isoA/B, **fpa_ord, fpa_nr, fpa_p,
     fpa_partner_acted_before**.
   - Chronology **within the phase**: prior_cand (# earlier shared hands with S1) and prior_scen (# earlier S14
     hands).
   - Output the top 5 by score. In the logistic fit, the strongest coefficients are fpa_ord (-7.1 SD),
     prior_cand (-1.2), fpa_p (-0.95), and pair_open / first_pair_vol_raise.
2. **Transparent fallback:** list the pair's eval shared hands where a member makes the first pair preflop
   aggressive action from UTG with population p_aggr < 0.3, in chronological order. Take the first 5; if fewer,
   append S16 hands, then S1 hands by earliest. Dev MAP@5 0.586.
3. The chronology assumption (the eval list is also "first ~5 scenario hands of the phase") cannot be checked
   locally. Settle it with one LB A/B: the same risk and behaviour, evidence with vs without chronology features
   (hand-only model 0.464 vs 0.686 on dev). CI is about 25% of dev positives (92/372), so the expected LB delta is
   about 0.2 x 0.25 x 0.22 = 0.011 if the assumption holds, which is noisy on the 30% public split.

### Pair-level features (compute on the scoring phase's hands only: dev for training, eval for eval pairs)

- `rate_C02`, `C02_logsf` / LLR (count-aware), `rate_Cfap` (S7), `n_C02`.
- `excess_trash`, `excess_trash_min`, `loglift_min/sum/max`, `mutual_top_lift`, `lift_rank_worst`: kill
  spill-over negatives. Compute over all co-players at the table in the same phase.
- Family discrimination: `rate_rr`, `rate_sandwich`, `rate_S14`, `rate_S16` (position-aware scenario rates),
  `rate_fpvr`, `rate_aggr_pre`, `mean_pwlog`, `top5_pwlog`.
- Member baseline: each member's trash rate without the partner (CI ~0.25%, population ~0.2%), to guard against
  loose-player hard negatives.
- Do NOT use dev-phase features for eval pairs: r = 0.08, and positives are phase-specific.
- Treat the ~39 unlabelled dev pairs with mutual-top-lift and excess >= 5 as likely positives (or drop them from
  the negatives).

## 7. Archetypal evidence hands (full print)

Format: seat role cards stack contrib net fold/showdown/won; actions: street role action amount-added to-street-total
pot-before to_call stack players_active (all in bb). Feature values for each hand come from `ci_feat_sub`.

### A. Trash UTG open, partner folds, table folds (pw = 0.00088, tf_pre = 4)
```
--- hand_idx 545545 seq 545 bb 2 button 0 pot 8 (4.0bb) pair P2B6AAE81BAE9 evidence_rank 3
   seat3   A 2h6d stack   83.0bb contrib 2.5 net +1.5 won=1.00
   seat4  o4 8h7d  seat5 B 8d7s  seat0 o1 Td9s  seat1 o2 8s2s  seat2 o3 5c3d
     PRE     A raise  amt 2.5 to 2.5 pot 1.5 tocall 1.0 act 6
     PRE    o4 fold   tocall 2.5 act 6
     PRE     B fold   tocall 2.5 act 5
     PRE    o1 fold / o2 fold (SB) / o3 fold (BB)
```

### B. Whipsaw: A 3-bets KTo over o4's open, B cold 4-bets 82s, table folds, A 5-bets, B folds (B pw = 0.0016; B equity 3-way 22%)
```
--- hand_idx 1806911 seq 1911 bb 4 button 2 pot 293 (73.2bb) pair P3AA5044D5AD9 evidence_rank 4
   seat0   A KsTd stack  116.2bb contrib   54.2 net   +19.0 fold=0 won=1.00
   seat1   B 8h2h stack   97.5bb contrib   15.5 net   -15.5 fold=1
   seat5  o4 JsQh stack   95.0bb contrib    2.0 net    -2.0 fold=1   (o1 9d2c, o2 4c3h, o3 3dTh)
     PRE    o4 raise  amt    2.0 to    2.0 pot    1.5 tocall   1.0 act 6
     PRE     A raise  amt    6.0 to    6.0 pot    3.5 tocall   2.0 act 6
     PRE     B raise  amt   15.5 to   15.5 pot    9.5 tocall   6.0 act 6
     PRE    o1 fold / o2 fold / o3 fold
     PRE    o4 fold   amt    0.0 to    2.0 pot   25.0 tocall  13.5 act 3
     PRE     A raise  amt   48.2 to   54.2 pot   25.0 tocall   9.5 act 2
     PRE     B fold   amt    0.0 to   15.5 pot   73.2 tocall  38.8 act 2
```

### C. Squeeze by B over A's open and o3's call, A 4-bets, B folds, A c-bets o3 out (B pw = 0.0018; B equity 26.6%)
```
--- hand_idx 1025565 seq 565 bb 4 button 2 pot 284 (71.0bb) board 6s 3h 8h pair P990D3C0B4788 evidence_rank 5
   seat5   A AdTs stack   76.5bb contrib   43.5 net   +27.5 won=1.00
   seat4   B 8d5d stack   94.5bb contrib    7.5 net    -7.5 fold=1
   seat2  o3 QdJs stack   66.8bb contrib   19.5 net   -19.5 fold=1
     PRE     A raise  amt    2.5 to    2.5 pot    1.5 tocall   1.0 act 6
     PRE    o1 fold / o2 fold
     PRE    o3 call   amt    2.5 to    2.5 pot    4.0 tocall   2.5 act 4
     PRE    o4 fold (SB)
     PRE     B raise  amt    6.5 to    7.5 pot    6.5 tocall   1.5 act 3
     PRE     A raise  amt   17.0 to   19.5 pot   13.0 tocall   5.0 act 3
     PRE    o3 call   amt   17.0 to   19.5 pot   30.0 tocall  17.0 act 3
     PRE     B fold   amt    0.0 to    7.5 pot   47.0 tocall  12.0 act 3
     FLOP    A bet    amt   24.0 to   24.0 pot   47.0 tocall   0.0 act 2
     FLOP   o3 fold
```

### D. Isolation 3-bet with 82s over o4's UTG open, partner A folds, o4 4-bets, B folds (pw = 0.0011)
```
--- hand_idx 1502345 seq 2345 bb 10 button 1 pot 195 (19.5bb) pair P4BD3F5DC8500 evidence_rank 4
   seat5   B 8s2s stack   64.4bb contrib    4.5 net    -4.5 fold=1
   seat0   A 6cJs stack   96.0bb contrib    0.0 fold=1
   seat4  o4 KsQd stack  167.2bb contrib   13.5 net    +6.0 won=1.00
     PRE    o4 raise  amt    2.0 to    2.0 pot    1.5 tocall   1.0 act 6
     PRE     B raise  amt    4.5 to    4.5 pot    3.5 tocall   2.0 act 6
     PRE     A fold   amt    0.0 to    0.0 pot    8.0 tocall   4.5 act 6
     PRE    o1 fold / o2 fold / o3 fold
     PRE    o4 raise  amt   11.5 to   13.5 pot    8.0 tocall   2.5 act 2
     PRE     B fold   amt    0.0 to    4.5 pot   19.5 tocall   9.0 act 2
```

### E. Double sandwich of o2: B opens J6o, A 3-bets 52o, o2 calls, B 4-bets, A 5-bets, o2 folds, B folds (A pw = 0.0029, B = 0.0106; A equity 17.7%)
```
--- hand_idx 1740015 seq 15 bb 2 button 0 pot 213 (106.5bb) pair PEF90594978AB evidence_rank 2
   seat5   A 2c5h stack  122.0bb contrib   70.5 net   +36.0 won=1.00
   seat4   B 6dJh stack  144.0bb contrib   23.5 net   -23.5 fold=1
   seat1  o2 8dKd stack  105.0bb contrib   10.5 net   -10.5 fold=1
     PRE    o4 call   amt    1.0 to    1.0 pot    1.5 tocall   1.0 act 6
     PRE     B raise  amt    3.0 to    3.0 pot    2.5 tocall   1.0 act 6
     PRE     A raise  amt   10.5 to   10.5 pot    5.5 tocall   3.0 act 6
     PRE    o1 fold
     PRE    o2 call   amt   10.0 to   10.5 pot   16.0 tocall  10.0 act 5
     PRE    o3 fold / o4 fold
     PRE     B raise  amt   20.5 to   23.5 pot   26.0 tocall   7.5 act 3
     PRE     A raise  amt   60.0 to   70.5 pot   46.5 tocall  13.0 act 3
     PRE    o2 fold   amt    0.0 to   10.5 pot  106.5 tocall  60.0 act 3
     PRE     B fold   amt    0.0 to   23.5 pot  106.5 tocall  47.0 act 2
```

### F. Postflop yield: B opens T7o, A defends BB, A checks pair of 4s (69.7% equity vs B), B bets, A folds
```
--- hand_idx 1025222 seq 222 bb 4 button 3 pot 46 (11.5bb) board Ks 4h Qh pair P990D3C0B4788 evidence_rank 1
   seat0   B 7hTd  contrib 8.0 net +3.5 won=1.00      seat5 A 6c4d contrib 3.0 net -3.0 fold=1
     PRE     B raise  amt 3.0 to 3.0 pot 1.5 tocall 1.0 act 6 ; o1..o4 fold
     PRE     A call   amt 2.0 to 3.0 pot 4.5 tocall 2.0 act 2
     FLOP    A check ; B bet 5.0 (pot 6.5) ; A fold
```
The unabridged prints are in `research/forensics/ci_archetypes.txt`.

## 8. What would overturn these conclusions

- **Chronology.** If an LB A/B shows the chronology-aware evidence ranker scoring no better than the hand-only
  ranker, the eval evidence list is not "first ~5 scenario hands of the eval phase". In that case, drop prior_cand
  and prior_scen and rank by the hand model; the S1, S14 and S16 detection itself stands.
- **Phase-specific activation.** This rests on unlabelled eval-phase structure: 108 eval-only CI-like pairs, and
  labelled pairs inactive in eval. If other_coordination or some eval positives turn out to span both phases,
  dev-phase features for eval pairs could add a little. It is currently r = 0.08, so they should not be added
  unless an LB probe says so.
- **Baseline scope.** The baseline table uses all hands, including eval-phase colluder hands. Contamination is
  tiny (about 1,400 trash raises out of 14M decisions), but a cleaner baseline could exclude hands from pairs with
  excess >= 5.

## Open questions

- Is `other_coordination` a variant of this mechanism, e.g. trash raises from late position or postflop pressure?
  Some of the 39 unlabelled dev CI-like pairs could belong to it.
- Why does listing favour UTG scenario hands (74% of evidence vs 14% of unlisted trash raises)? Possibly a scenario
  definition that requires at least 3-4 players left to act.
- Do DT and SP evidence lists follow the same "first ~5 scenario hands of the phase" rule? Their evidence deciles
  decline more gently: DT 95→38, SP 78→38.
