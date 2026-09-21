# EVIDENCE rankers (exp007 no chronology / exp008 with within-phase chronology)

Code: `src/evidence.py` (stages `cache | cv | eval | compose`, `python src/evidence.py all` reproduces everything).
Input: the pair-hand engine (`src/engine.py` -> `data/derived/ph_v2/phase{0,1}`; `EVID_PH` selects the directory),
one row per (hand, unordered pair).
Metric: host AP@5 per TRUE positive pair, denominator `min(|relevant|, 5)`; evidence is 20% of the final score.

## 1. Headline (OOF, frozen table folds, 372 labelled dev positive pairs)

Final run is on **ph_v2** (engine re-run with the exp003 policy columns). The policy-less `ph` run is kept as an
ablation in §4 and in `data/derived/evidence_cache/{cv_results_ph_nopolicy.json,ph_nopolicy/}`.

| ranker | DT (148) | SP (132) | CI (92) | host MAP@5, all 372 |
|---|---|---|---|---|
| exp004 rule baseline (previous) | 0.191 | 0.348 | 0.119 | 0.227 |
| forensics best recipes | 0.554 (decay) | 0.558 / 0.638 w/ chrono | 0.464 / 0.686 w/ chrono | — |
| **exp007 (no chronology), true-family routing** | **0.6000** | **0.5888** | **0.4860** | **0.56786** |
| exp007, uniform posteriors | 0.5932 | 0.5697 | 0.3525 | 0.52537 |
| **exp008 (+ chronology), true-family routing** | **0.7164** | **0.7078** | **0.7066** | **0.71091** |
| exp008, uniform posteriors | 0.6627 | 0.6304 | 0.5661 | 0.62734 |

- exp007 beats the exp004 rule baseline by +0.41 DT / +0.24 SP / +0.37 CI (+0.34 host MAP@5 = **+0.068 final**) and
  beats every no-chronology forensics recipe (DT 0.525, SP 0.558, CI 0.464).
- exp008 adds **+0.116 DT / +0.119 SP / +0.221 CI** (+0.143 host). At 0.20 weight that is **+0.029 final score**
  if the private lists follow the dev listing rule, and ~0 (not negative) if they do not — see §5.
- Uniform posteriors are the no-routing floor (family unknown). Both variants improved there once the policy columns
  landed (exp007 0.491 -> 0.525, exp008 0.544 -> 0.627). With the exp004/exp009-class behaviour model (OOF family
  accuracy 0.979-0.992) the realistic operating point is close to the true-family row.

## 2. Features (`hand_features`, 321 columns on ph_v2 after dropping constants)

Orientation-free by construction (the canonical A/B order of a pair carries no signal):

1. **Every directional engine feature in 4 views**: `max`, `min` over the two directions, plus `_don` / `_rec`
   (**hand donor** = the player with net < 0 when the partner has net > 0; else the direction with the larger EV gift),
   and 18 key features in `_pdon` / `_prec` (**pair donor** = larger summed EV gift over the pair's phase hands; 97%
   accurate on DT). CI adds first-pair-aggressor orientation (`_fpa` = aggressor's own, `_fpp` = partner's response).
2. **Family gates / signatures as flags** (validated against `research/forensics/*.md` on dev evidence, coverage in
   brackets): DT flow [1.000], DT strong [0.623], gift5, pfcall_weak, union, fold/call-down/bluff sub-types;
   SP G0 [1.000], G1, S_PL, strong-check, fold- and call-response, softresp; CI S1 trash [0.693], S14 UTG weak open
   [0.626], S15 template [0.902], S7, sandwich; and their union gate [1.000 on all three families' evidence].
3. **Within-pair percentile and top-rank** of 19 key scores (`pct_*` = rank/|pool|, `top_*` = # hands in the pair
   scoring higher). Eval candidate pools are ~0.64x dev (median 76 vs 112 shared hands), so absolute-count features
   shift but the within-pair order does not; `top_*` is the exposure-robust version.
4. **Pair context** (orderless): log shared hands, count and rate of every gate/signature in the pair's phase,
   pair-level gift asymmetry.

Chronology block (`chrono_features`, exp008 only, 57 columns):
- `q` = relative position of the hand in the pair's phase timeline;
- `prior_*` / `prel_*` = number (and share) of EARLIER hands of the pair passing each gate/sub-type (Critic §C3:
  DT fold-to-partner vs call-down/showdown, SP fold vs call-down, CI UTG weak open);
- model-based, two-stage: `s1` = stage-1 (no-chronology) probability, `s1_K` = summed stage-1 score of earlier hands
  (the forensics `p*exp(-K/2)` decay in learned form), # earlier hands above 0.1/0.25/0.5, # earlier hands inside the
  pair's stage-1 top-5/top-10, and the same counts restricted to the hand's own sub-type (`s1_*_sub`).

## 3. Models and cross-fitting

- In-pair objective: candidates = **all** dev shared hands of the family's positive pairs, target = listed evidence
  (the forensics finding that this beats "evidence vs negative-pair hands" is confirmed by the features' behaviour).
- LightGBM, `num_leaves 15, min_data_in_leaf 20, feature_fraction 0.5, bagging 0.8, lr 0.03`, rounds grid
  {100, 200, 350, 500, 750, 1000}; **binary vs lambdarank** (query = pair, truncation 20) and **per-family vs pooled
  with a family one-hot**. All OOF by the frozen table folds (`data/folds_tables_5.csv`), 4 threads.
- Chosen (OOF, ph_v2): exp007 = DT stacked per-family-binary 100 · SP stacked per-family-binary 200 · CI single
  pooled-binary 200; exp008 = DT per-family-binary 200 · SP per-family-lambdarank 750 · CI per-family-lambdarank 350.
  Differences between configs are ~0.01, i.e. inside the per-pair noise (SD of AP@5 across pairs ~0.3 => SE ~0.025-0.03 per family), so the config choice is noise and
  the *selection maximum* is optimistic by roughly that much. The exp007 -> exp008 gap is 5-10x that noise.
- exp008 is **nested**: stage-1 scores for stage-2 TRAINING rows come from inner models fitted without the outer
  validation fold, so no fold-k label ever reaches fold-k features (doctrine §1 on non-nested OOF stacking).
- Scores are **calibrated probabilities**: Platt scaling of each family model's OOF margin, fitted on that family's
  dev pairs. That makes the mixture `sum_f P(f|pair) * s_f(h)` the Bayes-correct marginal P(listed). Within-pair
  percentile normalisation was measured and is worse under uniform posteriors (0.5135 vs 0.5254 exp007,
  0.5382 vs 0.6273 exp008), because percentiles throw away *how* evidence-like the top hand is.

## 4. Ablation that matters: stacking is not the gain, chronology is

`exp007s` = the same two-stage model but with the chronology block replaced by **orderless** stage-1 summaries
(`s1`, its within-pair rank and percentile, # pair hands above thresholds):

| | DT | SP | CI |
|---|---|---|---|
| exp007 single stage (ph_v2) | 0.5983 | 0.5842 | 0.4860 |
| exp007s orderless stacking (ph_v2) | 0.6000 | 0.5888 | 0.4705 |
| exp008 + chronology (ph_v2) | 0.7164 | 0.7078 | 0.7066 |
| *the same three on the policy-less `ph`* | *0.5971 / 0.5863 / 0.7123* | *0.5808 / 0.5736 / 0.6860* | *0.4673 / 0.4695 / 0.7092* |

Orderless stacking is worth ~0 (+0.002 / +0.005 / -0.016), so the entire exp008 lift comes from *when* a hand happens
inside the pair's phase, not from the second stage. (Cross-check: the forensics DT recipe `p*exp(-K/2)` applied to our
stage-1 scores gives 0.680 vs their 0.554, purely because stage-1 is better; learned chronology adds 0.036 more.)

The exp003 policy columns (`min_polcall_pre`, `surp_face`, `min_polfold_face`, `max_polagg_resp`) are worth
+0.001 DT / +0.003 SP / +0.019 CI on exp007 and +0.004 DT / +0.022 SP / -0.003 CI on exp008 — small when routing is
perfect, much larger on the unrouted (uniform-posterior) numbers.

## 5. The chronology caveat (why exp008 is an LB A/B, not a free win)

exp008 works **because of how the dev evidence lists were built**: the host listed roughly the first ~5 planted
hands of the pair's dev phase (CI: P(listed) by chronological rank inside the scenario set 0.92/0.85/0.69/0.34/0.13;
DT/SP: two chronologically sorted sub-type segments, Critic §C1-C3). Nothing in the rules guarantees the private
eval lists were sampled the same way. What exp008 uses is only gameplay order *inside the eval phase* (hand_seq
within the pair), never row order, file order or ID structure — but it is the same gray zone flagged in
PLAN.md ("chronology in evidence") and must be disclosed in the prize write-up if used.

- If eval lists are early-biased like dev: exp008 - exp007 = +0.143 MAP = **+0.029 final**.
- If eval lists are a uniform sample of planted hands: chronology features become noise; exp008 should fall back
  toward exp007 (the chronology block would be uninformative, costing at most a little variance).
- **Decision rule:** submit the same risk + behaviour twice, evidence from exp007 vs exp008 (sub002/sub003).
  A public-LB delta near +0.03*0.3-split noise settles it. Until then keep both files.

## 6. Outputs

| file | rows | content |
|---|---|---|
| `data/derived/evidence_scores_dev_{exp007,exp008}.parquet` | 45,129 | OOF calibrated scores for ALL dev shared hands of the 372 labelled positive pairs (every family model scores every pair) |
| `data/derived/evidence_scores_eval_{exp007,exp008}.parquet` | 2.04M / 2.28M | all 112,540 eval pairs, union of each pair's top-15 hands per family |
| `data/derived/evidence_cache/cv_results.json` | — | full grids, chosen configs, Platt coefficients, compose results |

Columns: `pair_id, hand_id, hand_idx, s_directed_transfer, s_soft_play, s_coordinated_isolation`.
`compose_evidence(scores_df, posteriors_df)` (in `src/evidence.py`) returns `pair_id, evidence_hand_1..5`:
mixture `sum_f P(f|pair) s_f(h)`, top-5 distinct hands, `NO_EVIDENCE` padding only when a pair has < 5 candidates
(never happens: eval pairs share >= 38 hands). Ties are broken by a 1e-7 pot-size term, never by ID or row order.
Truncating to the top-15 union changes dev MAP@5 by 0.000, so the file loses nothing versus keeping all hands
(every eval pair keeps >= 15 candidates; median 18-20).

End-to-end check: `compose_evidence` on the eval files plus a dummy risk/behaviour passes `common.validate_submission`
for both variants — 112,540 rows, 5 distinct evidence hands each, `non_eval_phase 0`, `not_shared 0`, `risk_ties 0`.

Runtime (4 threads, < 3 GB RAM): CV 1361 s, eval scoring 324 s (9.65M eval pair-hand rows), compose 5 s;
`python src/evidence.py all` is ~28 min end to end.

## 7. Open items / next levers

1. **Routing is now the bottleneck for exp007** (uniform 0.491 vs true-family 0.560) and even more for exp008
   (0.544 vs 0.702). Feed real posteriors from the behaviour model (exp006) into `compose_evidence`; measure with
   OOF *predicted* posteriors, not one-hot, to price family errors in.
2. `src/evidence.py` reads `EVID_PH` (default `ph_v2` when it exists), so re-run `python src/evidence.py all` after
   any further engine change; `EVID_REUSE=1` reuses the cached single-stage grid and saves ~7 min.
3. `other_coordination`-routed pairs have no ranker; today they inherit the mixture of three family scores.
4. Sub-type segment model (Critic §F3: fit segment-1 vs segment-2 and test "(type, time)" sort with per-type quotas)
   is untried; it could add to exp008 on DT/SP.

---------------------------------------------------------------------------------------------------------------------

## 8. exp016ev -- evidence v2 (sub-type/segment chronology, hand-detector features, routing)

Code `src/evidence_v2.py` (stages `cv | report | rerank | final | eval`; exp007/exp008 untouched as fallbacks).
LB context that motivated it: exp007 LB 0.85923 -> eval MAP ~= 0.624, exp008 LB 0.88153 -> eval MAP ~= 0.735, so
chronology transfers to the private lists and eval MAP ~= dev OOF MAP + 0.03..0.06.

### 8.1 Dev OOF MAP@5 (frozen table folds, 372 positives)

| ranker | DT | SP | CI | all, true family | all, predicted posteriors |
|---|---|---|---|---|---|
| exp008 (LB 0.88153) | 0.7164 | 0.7078 | 0.7066 | 0.7109 | 0.7068 |
| A_base = exp008 rebuilt in v2 | 0.7164 | 0.7022-0.7078 | 0.7037-0.7066 | 0.7082-0.7108 | 0.7051-0.7084 |
| B_hs: + exp015 hand-detector scores | 0.7171 | **0.7156** | 0.6910 | 0.7101 | 0.7074 |
| C_seg: + segment (sub-type) chronology | **0.7261** | 0.7088 | 0.6990 | 0.7133 | 0.7113 |
| D_hs_seg: both | 0.7262 | 0.7115 | 0.7009 | 0.7142 | 0.7124 |
| rank-average of binary+lambdarank (D) | 0.7256 | 0.7154 | 0.7005 | 0.7158 | 0.7104 |
| **exp016ev final** (DT C_seg/binary/750, SP B_hs/binary/100, CI A_base/lambdarank/350) | **0.7261** | **0.7156** | **0.7066** | **0.7175** | **0.7150** |

Delta vs exp008: **DT +0.0097, SP +0.0078, CI 0.0000; all +0.0066 true-family, +0.0082 with real posteriors**
(= +0.0013..0.0016 final score). **Below the 0.02 "this is noise" bar** -- but the DT and SP gains are a consistent
shift across all 5 round settings and both objectives (C_seg DT 0.7196-0.7261 vs A_base DT 0.7066-0.7164; B_hs SP
0.7092-0.7156 vs A_base SP 0.7002-0.7078), not a single-cell grid maximum. Composed eval evidence shares 4.14 of 5
hands per pair with exp008, so it is a small edit, not a different answer.

The final pick prefers the hs-free configuration where it is tied (DT C_seg 0.7261 vs D_hs_seg 0.7262) because dev hs
scores are single-model OOF while eval hs is a 5-model mean -- a mild distribution shift. SP is the one place where
the hs block is the whole gain, so SP carries that shift risk; DT/CI do not.

### 8.2 Segment (sub-type) structure -- Critic C1/C3, now measured

- Recovered segment labels from the single time inversion of each list: DT 91/148 pairs, SP 93/132 (+3 with two
  inversions), CI 11/92 -> 949 labelled evidence hands (457 in segment 2). Exactly the Critic's counts.
- A segment-1-vs-segment-2 hand classifier is almost perfectly separable on listed hands:
  **OOF AUC DT 0.959 (n=449), SP 0.946 (n=445), CI 0.866 (n=55)** -- the "small fold to the partner" vs
  "big pot / call-down / showdown" split is real and learnable from the hand row alone.
- As RANKER FEATURES (per-segment earlier mass, # earlier same-segment candidates, rank within segment by time and
  by score) this is worth +0.010 on DT, +0.001..0.004 on SP, -0.005 on CI (CI lists are a single segment plus an
  appended rank-5, so there is nothing to gain).
- As an EXPLICIT TWO-SEGMENT COMPOSITION (n1 hands from segment 1 + 5-n1 from segment 2, ordered by score or
  chronologically within segment) it is **much worse** at every n1: best DT 0.618 vs flat 0.726, SP 0.647 vs 0.716,
  CI 0.683 vs 0.707. Reason: the classifier is trained only on listed hands, so on the pair's full 100+-hand
  candidate pool it labels ~90% of hands "segment 2"; hard quotas then throw away good candidates. **Use the
  structure as features, never as a quota.**

### 8.3 Routing is already solved (not a lever any more)

With the exp016 behaviour posteriors (OOF family accuracy 0.995 on dev positives):

| routing | exp008 | exp016ev final |
|---|---|---|
| true family (oracle) | 0.7109 | 0.7175 |
| predicted posteriors, soft (T=1) | 0.7068 | 0.7150 |
| sharpened T=2 / T=4 | 0.7085 / 0.7084 | 0.7153 / 0.7153 |
| argmax one-hot | 0.7078 | 0.7144 |
| exp013 rule: fam_unexplained -> SP first | 0.7059 | 0.7127 |
| uniform (no routing; floor only) | 0.6273 | 0.6418 |

Soft posteriors cost only 0.003-0.004 versus oracle routing, sharpening is inside noise (T=2 marginally best for
exp008, identical for v2), argmax is slightly worse, and the **fam_unexplained -> SP override hurts** (-0.001 to
-0.0026): on dev only 36/372 positives carry the flag (33 of them genuinely SP), and the behaviour model already
routes those correctly. Recommendation: keep soft posteriors, drop the override.

### 8.4 Where the remaining evidence headroom actually is

Diagnostics on the v2 dev OOF scores (true-family routing):

| | DT | SP | CI | all |
|---|---|---|---|---|
| listed hands inside our top-5 | 77.9% | 76.9% | 74.8% | -- |
| inside top-10 / top-20 | 96.7 / 99.6% | 95.7 / 100% | 91.7 / 98.7% | -- |
| AP@5 (the metric) | 0.7261 | 0.7156 | 0.7066 | 0.7175 |
| AP@10 / AP@20 | 0.846 / 0.857 | 0.828 / 0.844 | 0.811 / 0.836 | 0.831 / 0.847 |
| oracle ORDERING of our own top-5 set | 0.7821 | 0.7763 | 0.7478 | 0.7716 |

So ~96% of the listed hands are already in the top-10 and perfect ordering of the top-5 we already pick would be
worth +0.054. That headroom was attacked directly and **failed**: a listwise shortlist RE-RANKER (top-K = 8/10/12/16,
binary and lambdarank truncation-5, 50-400 rounds, features = flat score, within-shortlist rank/gap/percentile,
segment structure, time order inside the segment, core hand features) is *below* the flat ranker for every family
(best DT 0.7207, SP 0.7130, CI 0.7053, i.e. -0.001 to -0.006). With 372 queries x ~10 rows there is not enough
signal to learn the host's choice among near-identical planted hands. The residual is listing-rule information we
do not have, not detection or ordering skill we failed to use.

### 8.5 Closed lanes (do not re-litigate without new evidence)

1. Explicit two-segment quota composition -- large loss (8.2).
2. Shortlist re-ranking of the top-8..16 -- small consistent loss (8.4).
3. `fam_unexplained -> SP first` routing override -- small loss (8.3).
4. Posterior sharpening / argmax -- inside noise, soft is fine (8.3).
5. exp015 hand-detector scores help SP only (+0.008) and hurt CI (-0.013); as stage-1 features they are neutral to
   negative (stage-1 no-chronology MAP base DT .598/SP .584/CI .486 vs +hs .583/.591/.450).

Reopen 1-2 only with a materially better segment classifier on UNLISTED candidates (e.g. a 3-class
listed-seg1 / listed-seg2 / unlisted model), or many more labelled lists.

### 8.6 Files

`data/derived/evidence_scores_{dev,eval}_exp016ev.parquet` (eval 2.36M rows, 112,540 pairs, >= 15 candidates each,
same schema and calibrated-probability convention as exp007/exp008), `data/derived/evidence_cache/cv_results_v2.json`
(all grids, segment stats, routing/composition/re-ranker tables). Composed with the exp016 eval posteriors the file
passes `common.validate_submission` (112,540 rows, 5 distinct eval-phase shared hands per pair, no risk ties).
Runtime on 4 threads / < 4.4 GB: CV 1556 s, report 2 s, re-ranker 49 s, eval scoring 415 s.

---------------------------------------------------------------------------------------------------------------------

## 9. exp023ev -- calibrated segment composition: a second, quantified NULL (lane closed)

Code `src/evidence_v3.py` (`fit | diag | compose`). Retry of the quota lane with the calibration fix: the v2 segment
classifier was 2-class trained on listed hands only, so on the full pool it had the wrong prior (~90% "segment 2").

**Two properly calibrated formulations**
- **3-class** per family, cross-fit on the frozen table folds: 0 = not listed (in-pair: non-listed hands of positive
  pairs), 1 = listed segment-1, 2 = listed segment-2. Listed hands of zero-inversion pairs (DT 57, SP 39, CI 81) have
  no segment truth and are dropped from training, which costs this model ~48/30/88% of its positives.
- **Factored**: P(seg_s|h) = P(listed|h) x P(seg2|listed,h), with P(listed|h) = the exp016ev calibrated OOF score and
  P(seg2|listed,h) the 2-class model (OOF AUC .959 DT / .946 SP). Correct prior by construction, keeps all positives.

**Calibration is now good** (dev, one-inversion pairs, truth = split recovered from the single list inversion):

| | mean k1_hat | mean k1_true | corr(k1_hat, k1_true) | mean listed mass | mean #seg2 in our flat top-5 | true mean k2 |
|---|---|---|---|---|---|---|
| DT factored | 2.71 | 2.59 | **0.51** | 4.90 (true 4.90) | 2.10 | 2.34 |
| SP factored | 2.11 | 2.29 | 0.35 | 4.79 | 2.86 | 2.50 |
| CI factored | 3.97 | 3.91 | 0.26 | 5.00 | 0.22 | 1.09 |
| DT / SP 3-class | 1.70 / 1.57 | 2.59 / 2.29 | 0.31 / 0.31 | 3.30 / 3.61 | 2.18 / 2.88 | -- |

So the prior artefact is gone (the factored model's listed mass equals the true 4.9 hands/pair) **and the flat top-5
already has almost the right segment mix** (DT 2.10 vs 2.34 segment-2 hands; SP 2.86 vs 2.50).

**Every composition policy still loses** (dev OOF MAP@5, true-family routing; baseline = exp016ev flat):

| family | exp016ev flat | 3-class flat | best quota (est. k1) | best fixed quota | greedy E[AP@5] |
|---|---|---|---|---|---|
| DT | **0.7261** | 0.6774 | 0.6912 | 0.6481 | 0.6940 |
| SP | **0.7156** | 0.7008 | 0.7036 | 0.6721 | 0.6959 |
| CI | **0.7066** | 0.5620 | 0.6825 | 0.6834 | 0.6809 |

(The greedy policy is the decision-theoretic one: under independence, ranking by P(listed) already maximises E[AP@5]
in both set and order, so the only possible gain is the negative within-segment dependence; the greedy discounts a
segment's slot as it fills. It loses 2-3 points.)

**The oracle test that explains why, and caps the lane.** Give the quota the TRUE k1 of each pair (one-inversion
pairs only, same pairs for the baseline):

| | flat (same pairs) | oracle-k1 quota, score order | delta |
|---|---|---|---|
| DT factored | 0.7258 | **0.7407** | +0.0149 |
| SP 3-class / factored | 0.7154 / 0.7083 | 0.7242 / 0.7168 | +0.0088 / +0.0085 |
| CI factored | 0.6524 | 0.6633 | +0.0109 |

So the two-segment structure **is** worth something -- but only with a perfect split size, and only +0.009..+0.015 on
the subset of pairs that have the structure (61% of DT, 70% of SP, 12% of CI pairs). Weighted over all 372 positives
that is an **oracle ceiling of ~+0.008 MAP = +0.0016 final, below the +0.02 noise bar**, and our k1 estimate
(corr 0.26-0.51) recovers none of it: estimated-k1 quotas lose 1-3 points. Chronological ordering inside a segment
(Critic C1's own order) is consistently worse than probability order, by 2-4 points.

**Verdict: lane closed.** The remaining evidence gap (dev MAP@5 0.72 vs AP@10 0.83, oracle top-5 ordering 0.77) is
the host's choice among near-identical planted hands, and three independent attacks on it have now failed:
pool-wide quotas (v2), shortlist re-ranking (v2), calibrated segment quotas + decision-theoretic selection (v3).
No exp023ev score files were written (they would be identical to exp016ev: `factored|flat` == exp016ev flat by
construction). `src/assemble.py` needs no new composition function -- keep `evidence.compose_evidence`.
Reopen only with: a k1 predictor with corr > 0.8 on held-out pairs, or segment truth for the zero-inversion pairs.

Runtime: fit 76 s, diag+compose 2 s, oracle check 3 s (4 threads, < 3 GB).

---------------------------------------------------------------------------------------------------------------------

## 10. exp027ev -- rebuilt on hs_v2, plus an explanation of the whole dev->eval evidence gap

Code `src/evidence_v4.py` (`skew | stage1 | cv | lofo | eval`). Input change: `src/handdet.py` self-trains, and its
scores moved to `data/derived/hs_v2` (hand OOF AUC vs negative-pair hands DT .9963->.9983, SP .9931->.9957,
CI .9926->.9964; recall of listed evidence at the negative q999 threshold DT .65->.82, SP .41->.59, CI .62->.83).

### 10.1 Better detector != better ranker (per family)

Stage-1 (no chronology) dev OOF MAP@5:

| feature set | DT | SP | CI |
|---|---|---|---|
| b = engine features only | **0.5983** | 0.5836 | **0.4860** |
| h2 = engine + hs_v2 raw + rank | 0.5764 | **0.5948** | 0.4391 |
| h2r = engine + hs_v2 rank-only | 0.5894 | 0.5858 | 0.4534 |
| e = hs_v2 + pair context ONLY | 0.4911 | 0.5244 | 0.2592 |

The "sole hand-quality block" fails badly (stage-2 with chronology only reaches DT .601 / SP .587 / CI .438): the
engine features carry most of the in-pair *listing* signal, and the detector -- now also firing on the ~2x more
numerous UNLISTED planted hands by construction of the self-training -- is a weaker discriminator of *listed* vs
*planted-but-unlisted*. That is exactly why a better detector does not translate into a better ranker for DT/CI.

Nested stage-2 grid (best rounds per family, true-family routing):

| variant | DT | SP | CI |
|---|---|---|---|
| A_base (control, = exp008 features) | 0.7164 | 0.7022 | 0.7037 |
| B2_hsv2 | 0.7233 | 0.7154 | 0.6925 |
| B2r_rankonly | 0.7195 | 0.7133 | 0.6976 |
| D2_hsv2_seg (binary / lambdarank) | 0.7262 / **0.7297** | **0.7236** / 0.7249 | 0.6864 / 0.6910 |
| E_hsonly | 0.6012 | 0.5865 | 0.4379 |
| exp016ev (hs_v1) reference | 0.7261 | 0.7156 | **0.7066** |

Like-for-like hs_v1 -> hs_v2 inside the same variant (D_hs_seg|binary, mean over the rounds row): **SP +0.012,
DT +0.003, CI -0.005**. Chosen: DT `D2_hsv2_seg|lambdarank|100`, SP `D2_hsv2_seg|binary|200`, CI `A_base|lambdarank|350`
(unchanged from exp016ev -- hs_v2 hurts CI at every setting).

### 10.2 Result: a consistent but sub-bar gain

Measured under matched protocols (the only fair comparisons):

| | DT | SP | CI | all |
|---|---|---|---|---|
| nested-CV protocol: exp016ev | 0.7261 | 0.7156 | 0.7066 | 0.7175 |
| nested-CV protocol: **exp027ev** | 0.7297 | 0.7236 | 0.7066 | **0.7218** (+0.0043) |
| shipped-model protocol: exp016ev | 0.7044 | 0.7107 | 0.6907 | 0.7032 |
| shipped-model protocol: **exp027ev** | 0.7222 | 0.7191 | 0.6907 | **0.7118** (+0.0086) |

So exp027ev beats exp016ev by **+0.004 .. +0.009 MAP (= +0.001..+0.002 final), below the +0.02 bar**, consistent in
sign across both protocols, and driven by DT and SP only. Composed eval evidence shares 4.55 of 5 hands per pair with
exp016ev. Files written (they do beat exp016ev): `evidence_scores_{dev,eval}_exp027ev.parquet`, eval 2.22M rows,
112,540 pairs, >= 15 candidates each; the composed submission passes `validate_submission`.

**Protocol sensitivity (a caution on every dev number we quote).** The same configuration scores 0.703-0.722
depending on how the final stage-2 is built and measured (stage-2 trained on 80% outer-OOF stage-1 vs on noisier
3-fold stage-1, scored against sharp or noisy stage-1 rows). That +-0.014 band is *wider than the feature gain being
chased*, and it means the dev files we ship for exp008/exp016ev (written under the nested protocol) overstate their
own shipped models by about that much.

### 10.3 The dev(0.707) -> eval(0.664) gap is now fully accounted for

| component | size | source |
|---|---|---|
| hidden `other_coordination` family (~6.5% of eval positives) | **-0.024 .. -0.027** | LOFO (10.4): an unseen family scores 0.18-0.44 instead of ~0.72 |
| shipped-model vs nested-CV protocol optimism | **-0.014** | 10.2 |
| sum | -0.038 .. -0.041 | dev 0.707 - 0.040 = **0.667** vs the measured eval 0.664 |

No unexplained pathology remains in the evidence component: eval is harder for two identified, quantified reasons.

### 10.4 The OC fallback: a family-agnostic head is WORSE than what we already do

Leave-one-family-out (train the ranker on two families, score the held-out one -- the honest proxy for a family the
model has never seen):

| held out | family-agnostic head (b / h2) | uniform mixture of the two SEEN heads | max of seen heads | its own head (ceiling) |
|---|---|---|---|---|
| DT | 0.333 / 0.359 | **0.440** | 0.401 | 0.726 |
| SP | 0.372 / 0.442 | 0.425 | 0.401 | 0.716 |
| CI | 0.048 / 0.060 | **0.183** | 0.173 | 0.707 |

A pooled "planted-ness" head is worse on 2 of 3 held-out families (catastrophically so for CI) and only +0.018 on SP,
i.e. inside noise. **Recommendation: do not add an OC fallback head; the existing posterior mixture already is the
best available answer for a pair that fits no family**, and its LOFO score (0.18-0.44) is the number to use when
forecasting what OC pairs contribute.

### 10.5 hs_v2 dev/eval skew (quantified, as asked)

Comparing like-for-like populations (dev hands of confirmed-negative pairs vs eval-pair hands, tables 0-79):
mean s_dt .00213 vs .00275, p99 .0156 vs .0300, p999 .670 vs .741, frac > 0.5 .00133 vs .00184. The eval tail is
~1.2-1.4x heavier -- modest, and not the order-of-magnitude difference suggested by comparing positive-pair
candidates against all eval pairs. Raw hs_v2 features are therefore safe to ship; the rank-only variant (B2r) is
available as the skew-immune fallback at -0.004 DT / -0.010 SP.

Runtime (4 threads, < 5 GB): stage1 335 s, nested cv 1436 s, lofo 91 s, eval 281 s.
