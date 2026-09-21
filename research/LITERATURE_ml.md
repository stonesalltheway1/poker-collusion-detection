# LITERATURE_ml.md — ML methodology for "Detect Suspicious Value Transfers in Poker"

Written 2026-09-15 (D-5). Scope: the machine-learning methods for this exact problem shape (PU pair labels,
AP ranking of 112,540 pairs, evidence retrieval MAP@5, episodic coordination, an unseen 4th mechanism,
relational structure). Poker-domain feature ideas are only included where a method needs them.
Every PDF cited as `papers/ml_*.pdf` lives in `research/papers/`. Reusable aggregator code and the toy
simulation quoted below: `research/ml_sim_aggregators.py`.

---

## 0. The findings that should drive the campaign (TL;DR)

1. **The problem is multiple-instance PU learning with biased negatives.** A pair is a *bag* of shared
   hands; a positive bag has a handful of planted *instances* (≥3 listed, usually >5 planted). Pair labels
   are P (372), **biased** N (1,488 hand-picked hard negatives) and U (155,812). The three literatures
   are MIL from PU bags (Bao 2018), PU with biased negatives, "PUbN" (Hsieh, Niu & Sugiyama 2019), and PNU
   (Sakai 2017). None of them treat U as negative.
2. **Pair AP only depends on the ranking.** Under SCAR (labelled positives are a random sample of all
   positives), a classifier trained on "labelled vs unlabelled" ranks examples in the same order as the
   true P(y=1|x) (Elkan & Noto 2008). The label frequency c and the class prior only matter for fusing
   models, correcting CV, and estimating eval prevalence. They don't matter for ranking.
3. **Training only on P vs hard negatives targets the wrong population.** That model learns
   p_P/p_HN. Eval is scored on p_P / (mix of HN and easy negatives). On the ~99% of eval pairs outside
   the hard-negative support, the trees just extrapolate. Worked trap: HN include "repeated opponent
   selection", i.e. high exposure. A P-vs-HN model can learn "lower exposure → positive" and then score
   easy low-exposure eval pairs high. **Fix by chaining density ratios** (telescoping DRE, Rhodes 2020):
   `logit risk = logit f(P vs HN) + logit f((P∪HN) vs U)`. Alternatives are PUbN-style weighting or
   bagged PU. Test all three and fuse them with a cross-fitted LR.
4. **Never aggregate hand scores by their mean.** In a toy simulation with 40k pairs, 0.6% prevalence
   and eval-like exposure, mean-of-hand-score AP was 0.02–0.43. Top-k sums (k ≈ 3–5 ≈ planted count)
   got 0.53–0.99 and were the most robust to hard negatives. Mixture-GLR and HMM log-likelihood ratios
   are best when the hand model is well calibrated (0.84 vs top-5's 0.74). They fall apart when hard
   negatives carry broad weak shifts (0.27–0.34 vs top-5's 0.53). Feed several aggregators to the pair
   model.
5. **The hand-level planted-hand detector dominates everything else.** In the same simulation, raising
   hand-score separation from 2.5 sd to 3.5 sd moved top-5 AP from 0.53 to 0.98. Aggregator choice moved
   it by ≤0.3. Spend the engineering on hand-level features: equity/EV-aware decisions (all hole cards are
   visible) and residuals vs each player's local baseline. This is the S6E8 lesson again: one strong
   base model beats blending.
6. **Exposure shifts between dev and eval (measured).** Labelled pairs share a median of 108–112 hands in
   dev (q10 ≈ 67–71, q90 ≈ 180–189). The same pairs share a median of 65–66 hands in eval, and
   `evaluation_pairs.csv` has median 76 (min 38 after the host's filter). Use exposure-invariant
   statistics (top-k sums, max-window sums, n as a feature), or train on exposure-matched dev windows.
   Mean-type and count-type features will shift.
7. **The evidence target is the listed ≤5 set.** The solution file also has only 5 evidence columns.
   340/372 dev positives have exactly 5 listed (all 92 CI, 137/148 DT, 111/132 SP), so unlisted planted
   hands almost surely exist. The private truth is capped the same way, though. So: *train the ranker on
   listed=1 / other=0*. Hand-level PU cleaning is an ablation, not the default. The listing is not
   uniformly random: evidence_rank is chronological for 81/92 CI, 51/137 DT and 30/111 SP pairs, against
   ~1% expected by chance. The selection rule is structured and a ranker trained on listed hands can
   learn it.
8. **Validate with grouped CV plus PU-corrected AP.** Use `StratifiedGroupKFold` with groups = table
   (a player sits at one table, so this also groups players) and y = family. Estimate eval AP from dev
   OOF positives plus the *eval* score distribution with the Jain–White–Radivojac correction (§3.4). Also
   report HN-AP and leave-one-family-out AP (§6).
9. **other_coordination counts as a positive in pair AP (70%) and evidence (20%).** It is excluded from
   behavior AP. If a fraction φ of eval positives are other_coordination and we rank them like negatives,
   `AP ≈ (1-φ)·AP_known`, so φ = 0.2 costs about 0.14 of final score. Build a family-agnostic branch:
   collusion-table marginal impact (Mazrooei 2013) and mutual information between a player's actions and
   the partner's hole cards (Bonjour 2022). Select it on leave-one-family-out AP.
10. **The public LB is noisy.** It holds ~30% of eval positives, probably 100–250 pairs, which gives
    AP SE ≈ 0.02 as a rough binomial heuristic. LB deltas below 0.01–0.02 are noise. Decide on OOF with
    a paired bootstrap resampled by table, and use the LB for calibration only (parent doctrine §1).

---

## 1. The problem in ML terms, with measured facts

Measured from `data/derived` on 2026-09-15:

| fact | value | consequence |
|---|---|---|
| labelled positives per table | 155 tables have 1, 63 have 2, 18 have 3, 8 have 4, 1 has 5; 155 tables have 0 | Poisson-like with mean ≈0.93 → hard per-table rank normalisation of the final score is risky (§3.5) |
| hard negatives per table | 392 tables, 1–11 each (mode 3–4) | HN everywhere, so HN are not a table-level signal |
| HN pairs containing a labelled-positive player | 190/1,488 (3 contain two) | a colluder's *other* relationships are negatives → need pair-specific (double-centred) features, not player-level suspicion (§7) |
| positive-player degree | 644 players in 1 positive pair, 47 in 2, 2 in 3 | none of the 53 "wedges" (A–B, A–C positive) has B–C labelled; the two families differ in 31/53 → no evidence for closed rings, so triad features are low priority |
| evidence listed per positive | 5 for 340, 4 for 21, 3 for 11 | the 5-cap binds for 91% of pairs → unlisted planted hands exist (§4.5) |
| evidence span (dev table hands, median) | DT 1,465 / SP 1,587 / CI 960 | coordination is spread over a large part of the phase, not one burst |
| median gap between consecutive evidence hands | DT 218 / SP 220 / CI 67 table hands ≈ 8 / 8 / 2.5 *shared* hands (pair co-seat rate ≈3.7%) | CI is bursty → temporal context should matter most for CI evidence |
| evidence_rank in chronological order | CI 81/92, DT 51/137, SP 30/111 (chance ≈1/120 for 5 items) | the evidence list is ordered/selected by a structured rule |
| exposure, labelled pairs | dev median 112 (P) / 108 (HN); eval-phase median 66.5 / 65 | P and HN have the same exposure → exposure is not a label signal among labelled pairs (it may be vs U) |

Mapping to method families:

| data element | ML concept | key refs |
|---|---|---|
| pair = set of shared hands; planted hands = instances | multiple-instance learning (bag labels) | Ilse 2018 (`ml_mil_attention_ilse_2018.pdf`), Bao 2018 (`ml_mil_pu_bags_bao_2018.pdf`) |
| P / hard-N / U pair labels | PUbN, PNU | Hsieh 2019 (`ml_pu_pubn_...`), Sakai 2017 (`ml_pu_pnu_sakai_2017.pdf`) |
| listed evidence ≤5, more planted | instance-level PU / partial relevance labels | Elkan–Noto, Joachims 2017 (`ml_ltr_unbiased_joachims_2017.pdf`) |
| "episodic" coordination | sparse mixtures, scan statistics, change points, bursts | Donoho–Jin 2004, Neill 2012/2020, Kleinberg 2002, Adams–MacKay 2007 |
| other_coordination | open-set / novel-class detection | Mazrooei 2013, Bonjour 2022, Ruff 2020 (Deep SAD), Liu 2008 (iForest) |
| table (30 players) | group for CV; relational context | Mazrooei MI score, FRAUDAR, CopyCatch |
| dev phase → eval phase | covariate shift (exposure) + prior shift (prevalence) | Saerens 2002, Forman 2008, Jain 2016/2017 |

---

## 2. Positive-Unlabeled learning

### 2.1 Core results, digested

**SCAR and Elkan & Noto (2008)** (`ml_pu_elkan_noto_2008.pdf`). Let s=1 mean "labelled", which only
happens for positives. Under SCAR, P(s=1|x) = c·P(y=1|x) with constant c = P(s=1|y=1).
- A "non-traditional" classifier g(x) = P(s=1|x), trained on labelled vs unlabelled, is **rank-equivalent**
  to P(y=1|x). Pair AP needs nothing more.
- Estimator e1: ĉ = mean of g over held-out labelled positives. Then P(y=1|x) = g(x)/ĉ.
- Principled weighting: for each unlabelled x, duplicate it as a positive with weight
  w = ((1-c)/c)·g/(1-g) and as a negative with weight 1-w.
```python
# g: cross-fitted (table folds) P(labelled | x) for all dev pairs from labelled-positive vs unknown
c = g[is_pos].mean()                                   # e1 estimator, held-out folds only
w = np.clip((1 - c) / c * g / (1 - g), 0, 1)           # per unknown row: weight as positive
# final model rows: P (y=1,w=1) + HN (y=0,w=1) + U duplicated: (y=1,w=w) and (y=0,w=1-w)
```
Caveat for this comp: labelled positives are "confirmed", which can mean more blatant (SAR, below).
Eval positives are also defined as having visible evidence, so SCAR within the 3 families is a
reasonable working assumption. other_coordination violates it by construction.

**uPU / nnPU** (du Plessis 2015 `ml_pu_upu_duplessis_2015.pdf`; Kiryo 2017 `ml_pu_nnpu_kiryo_2017.pdf`).
Risk: `R(g) = π·E_P[ℓ(g,+1)] + max(0, E_U[ℓ(g,-1)] − π·E_P[ℓ(g,-1)])`. Without the max (uPU), flexible
models drive the negative-class risk below 0 and overfit. nnPU clips it and does a gradient-ascent step
when it goes negative. **For GBDTs** this needs negative sample weights, which LightGBM/XGBoost handle
badly (negative hessians). Use Elkan–Noto weights or bagging for trees. Use nnPU only for a small torch
MLP on pair features, with π taken from §2.1 prior estimation.

**Bagging PU** (Mordelet & Vert 2014, `ml_pu_bagging_mordelet_vert_2014.pdf`). Repeat T times: draw K
unlabelled at random (K ≈ |P|), train P vs that sample, and score the out-of-bag unlabelled. Average
the OOB scores. The paper shows the biggest gains when |P| is small and U is contaminated, which is our
situation. It is GBDT-friendly: T = 50–200 LightGBM models with ~300 trees each on ~2k rows is
seconds per bag. Positives are scored OOF by table fold.

**Two-step / spy technique** (Liu et al. 2002, `ml_pu_spy_sem_liu_2002.pdf`):
1. Move ~10–15% of P into U as "spies".
2. Train P vs U.
3. Set a threshold t at a low quantile of the spies' scores (5–15%).
4. Call the unlabelled rows below t "reliable negatives" (RN).
5. Train P vs RN, here pooled with HN.

This is a cheap, interpretable alternative to weighting. It is also a *diagnostic*: how many U pairs
score above most spies? That is a crude count of hidden positives.

**Class-prior / label-frequency estimation.**
- *TIcE* (Bekker & Davis 2018, `ml_pu_tice_bekker_davis_2018.pdf`): in any subdomain S, the labelled
  fraction |L∩S|/|S| is a lower bound on c. A decision tree searches for the subdomain with the largest
  bound, with a finite-sample correction. It is fast.
- *KM2* (Ramaswamy 2016, `ml_pu_km2_ramaswamy_2016.pdf`): kernel mixture-proportion estimation with the
  strongest guarantees. It is slow, so subsample to ≤2–4k points. The survey
  (`ml_pu_survey_bekker_davis_2020.pdf`) finds KM2 and TIcE most accurate on SCAR data and KM2 better
  under SAR.
- *AlphaMax* (Jain 2016, `ml_pu_alphamax_jain_2016.pdf`) tolerates noisy positives.
- **Practical score-based bound** (Blanchard/Lee/Scott-style MPE). Since U = αP + (1-α)N, for every
  threshold t: P_U(S≥t) ≥ α·P_P(S≥t). So α ≤ min_t P_U(S≥t)/P_P(S≥t). The bound is tight when the top
  of the score range is nearly pure. Verified on synthetic data: with 372 labelled positives and
  separation ≥3 sd it recovers α within ±10%. With weak separation it over-estimates, as a bound should.
```python
def mpe_alpha(s_pos_oof, s_unl, min_pos=50, z=1.0):
    """Upper bound on the positive fraction in s_unl (eval or dev-unknown scores).
    s_pos_oof = OOF scores of labelled positives, exposure-matched."""
    sp = np.sort(s_pos_oof)[::-1]; su = np.sort(s_unl); nU, nP = len(su), len(sp)
    k = np.arange(min_pos, nP + 1); t = sp[k - 1]
    q = (nU - np.searchsorted(su, t, 'left')) / nU; tpr = k / nP
    qu = q + z * np.sqrt(q * (1 - q) / nU); tl = np.maximum(tpr - z * np.sqrt(tpr * (1 - tpr) / nP), 1e-9)
    return float(np.min(qu / tl))
```

**SAR** (Bekker, Robberechts & Davis 2019, `ml_pu_sar_bekker_davis_2019.pdf`). The labelling propensity
e(x) depends on x. If the host confirmed the most blatant pairs, P-vs-U learns "blatant" and may
under-rank subtle eval positives. Cheap check: compare OOF score distributions of labelled positives
with ≥5 vs 3–4 listed evidence hands, and across families. A formal SCAR test exists
(Teisseyre 2024, `ml_pu_verify_scar_2024.pdf`) but is probably not worth the time here.

**PNU** (Sakai 2017). `R = γ·R_PN + (1−γ)·R_PU` combines supervised P/N risk with PU risk, with γ tuned
on validation. **PUbN** (Hsieh 2019) is the exact match for our negatives: they are *biased*, drawn
from a few "hard" sub-populations. PUbN introduces a latent s = "labelled or HN-like":
1. Estimate σ(x) = P(s=+1|x) with a (P∪bN)-vs-U PU classifier.
2. Take the negatives from U with weight (1−σ̂(x)) wherever σ̂ ≤ η.
3. Take P and bN with weights ∝ (1−σ̂)/σ̂ wherever σ̂ > η.

The paper beats nnPNU when N is biased. A GBDT approximation is recipe R2 in §2.3.

### 2.2 Hard-negative-only vs PU in this competition, as density ratios

The eval population density is
`p_eval(x) = π·p_P(x) + (1−π)·[β·p_HN(x) + (1−β)·p_E(x)]` (E = easy negatives).
The ideal ranking is `r*(x) = p_P(x) / [β·p_HN(x) + (1−β)·p_E(x)]`.

- **P vs HN only** estimates p_P/p_HN. It matches r* only where p_E ≪ p_HN, i.e. *inside* the hard
  region. The bulk of the 112k eval pairs sit where p_HN ≈ 0: the model never saw them and trees
  extrapolate piecewise-constantly. A split that separates P from HN in one direction goes wrong
  whenever easy negatives lie even further in that direction. Exposure is the example above; "similar
  strategies" or "weak play" on passivity features is another.
  The retrieval literature says the same thing: training on hard negatives alone makes a model collapse
  onto fine distinctions and admits false negatives. Mixing random and denoised hard negatives works
  better (RocketQA, `ml_ltr_rocketqa_hardneg_2021.pdf`). The public "E22/E23 clean-label PN" notebooks
  report that a P-vs-HN model ranks hidden positives at the top of U. That is plausible when HN
  mimic every behavioural axis, but it is not guaranteed on the axes HN do not cover.
- **P vs U** estimates p_P/p_U ∝ r*: the right target under SCAR. It wastes capacity on easy separation
  (372 P vs 155k mostly-easy U), ignores the HN information, and trees partly memorise the positives.
- **Chaining (telescoping density-ratio estimation, Rhodes 2020, `ml_dre_telescoping_rhodes_2020.pdf`):**
  `p_P/p_U = (p_P/p_L)·(p_L/p_U)`, where L = labelled set (P∪HN) is a *bridge* distribution between P and
  U. Both classifiers are well-conditioned:
  - stage A, P vs HN, is clean-label and focused on subtle cues;
  - stage B, (P∪HN) vs U, measures "hard-looking-ness" relative to the real population.

  U being contaminated by hidden positives is not a bias here, because U *is* the reference density.
  Where stage A extrapolates wildly (easy pairs), stage B's large negative logit dominates the sum and
  pushes the pair down.
  `risk_logit = clip(logit f_A, ±15) + clip(logit f_B, ±15)`. The ±15 clip keeps a single extreme
  logit from dominating the sum. A learned variant a·logit f_A + b·logit f_B is fit by cross-fitted LR.

### 2.3 Recipes (GBDT-friendly), in priority order
| id | recipe | notes |
|---|---|---|
| R1 | **Chain**: A = P vs HN, B = (P∪HN) vs U-sample (30–50k U rows stratified by table); risk = logit A + logit B | OOF by table for both stages; the most principled use of all three label types |
| R2 | **Weighted PNU/PUbN-lite**: one model on P (y=1), HN (y=0, w=1), U-sample (y=0, w = 1−ŵ(x)); ŵ = Elkan–Noto positive-weight from a first-pass P-vs-U model | down-weights U rows that look positive, so hidden positives are not trained as negatives |
| R3 | **Bagged PU**: 100 bags of P vs random U (K = 2·|P|), OOB-averaged | robust baseline, cheap |
| R4 | **Spy RN + HN → PN** | diagnostic plus alternative |
| R5 | **nnPU torch MLP** on pair features (π from `mpe_alpha`) | only if R1–R3 plateau |

**Cross-fitting protocol (non-negotiable).** Every first-stage model whose output becomes a feature or a
weight for dev rows must be OOF under the *same* table folds. Eval predictions are the fold-model
average. Players sit at one table, so table grouping also stops player-identity leakage. IEEE-CIS's
winners dropped UID features for the same reason: they wanted models that generalise to unseen entities.

---

## 3. Optimising pair AP (70% of the score)

### 3.1 Metric mechanics, from the host's reference code
- `_average_precision`: `order = np.argsort(-scores, kind="mergesort")` on a frame **sorted by pair_id**.
  Ties are therefore broken by pair_id ascending, which is random with respect to labels. Tied blocks
  at the top of the list cost AP. Break ties with a secondary *legitimate* score, e.g. `+1e-9·rank` of a
  second model. Never use pair_id; that exploits the file format and is prohibited.
- risk_score must lie in [0,1]. Use `rank/(N+1)` or a sigmoid; calibration is irrelevant to AP.
- Behavior AP for family f uses `where(pred==f, risk, 0)`. The ordering inside family f is the *global*
  risk; a separate per-family score is not possible.
- Evidence MAP counts only true positives. Always submit 5 distinct hands for every pair; it costs
  nothing.

### 3.2 What to optimise
- **Probability ranking principle** (Robertson 1977): sorting by P(y=1|x) maximises expected precision@k
  for every k and is near-optimal for expected AP when labels are conditionally independent. So the
  goal is the best posterior *ranking*, not a ranking-loss trick.
- AP is global across tables, so per-table LambdaMART queries optimise the wrong thing. Use a binary
  objective with strong regularisation, many seeds and bags. With 372 positives, variance dominates;
  `scale_pos_weight` barely changes the ranking. Early stopping on OOF AP should use nested folds or a
  fixed round count chosen on the CV mean.
- Blending model classes with **rank averaging** is a robust baseline. Doctrine §1 bans nonlinear
  stackers on non-nested OOF.

### 3.3 Score fusion across families and model branches
- Families are mutually exclusive, so the ideal is `P(pos|x) = Σ_f P(pos_f|x)`. For small p,
  noisy-OR `1−Π(1−p_f)` ≈ sum. **Sum and noisy-OR are only meaningful if every head is calibrated
  against the same negative population**, the eval-like one: fit Platt/isotonic on P + HN + U-sample
  with Elkan–Noto weights. **max** under-scores family-ambiguous pairs but survives one badly calibrated
  head.
- Recommended structure: **one pooled binary risk model** trained on all 372 positives (more data per
  split than per-family heads with 92–148 each) and **one multinomial family model** trained on
  positives only. Per-family heads go in as *features*/stack members. Fuse the branches (R1/R2/R3,
  family heads, generic branch §6) with a **cross-fitted LR on clipped logits** plus log n_shared.
  Fit its weights on the PU-proxy target (labelled positives vs HN ∪ U, which is ranking-consistent
  under SCAR) and check with the corrected AP in §3.4.

### 3.4 Eval prevalence and a label-free AP estimate (the CV↔LB bridge)
- **Prevalence**: apply `mpe_alpha` to (a) dev U scores for the dev-U prior and (b) eval scores for π_eval.
  Both use OOF scores of dev positives computed on **exposure-matched** features. Cross-check with
  Saerens–Latinne–Decaestecker EM prior adjustment (Neural Computation 2002, DOI
  10.1162/089976602753284446) and with Forman's adjusted classify-and-count (DMKD 2008; see
  `ml_quant_weiss_kdd09.pdf`).
- **PU-corrected AP** (Jain, White & Radivojac 2017, `ml_pu_eval_jain_white_radivojac_2017.pdf`). Under
  SCAR, held-out labelled positives give TPR(t), and the U score distribution gives
  Q(t) = α·TPR(t) + (1−α)·FPR(t). Precision(t) = α·TPR(t)/Q(t), and AP is the mean precision at the
  positives' thresholds. Verified on synthetic mixtures (112k U, 372 labelled, 5 reps): mean absolute
  error ≈ 0.01–0.03, and the error is shared across models, so **paired** comparisons are sharper than
  that.
```python
def pu_ap_estimate(s_pos_oof, s_unl, alpha):
    """Expected AP of this score on population s_unl (e.g. EVAL pairs) with positive fraction alpha."""
    sp = np.sort(s_pos_oof)[::-1]; su = np.sort(s_unl); nU, nP = len(su), len(sp)
    tpr = (np.arange(1, nP + 1) - 0.5) / nP
    Q = nU - np.searchsorted(su, sp, 'left')          # count of population at or above each positive
    TP = alpha * nU * tpr; FP = np.maximum(Q - TP, 0)
    return float(np.mean(TP / np.maximum(TP + FP, 1e-12)))
```
  Compute it on the **eval** score distribution (the real population) with dev OOF positives. That makes
  every submission a CV↔LB calibration point. It also flags a candidate whose eval score distribution
  shifts: when the fitted α jumps between models, suspect exposure or covariate shift.
  **What would break it:** SCAR failing (subtler eval positives), other_coordination (those positives
  are missing from s_pos), and unmatched exposure.

### 3.5 Per-table normalisation
Labelled positives per table look Poisson (0–5), so forcing each table to contribute equally to the top
of the list (within-table rank as the final score) throws away real between-table variation. Normalise
*features* (in bb units, and z-scored within table for stake- or table-dependent quantities). Give the
fusion model *context* features such as the pair's rank in its table, the table's max/mean score and the
gap to the table's 2nd-best pair, and let CV decide. Score-level per-table normalisation is an ablation,
never the default.

### 3.6 Behavior assignment (10%)
Assigning family f to a pair adds it to f's list at its global risk. That helps AP_f iff
P(f|x, pos)·P(pos|x) exceeds the running precision of f's list at that rank. Low-risk pairs sit where
precision ≈ prevalence, so assigning them is always ≥ neutral. **Default: argmax_f of the family model
for every pair.** Override to `other_coordination` (which removes the pair from all three lists) only
for high-risk pairs whose best family posterior is low and whose generic-branch score is high (§6).
Tune the override threshold on OOF behavior AP.

### 3.7 LB noise and final selection
Public = 30% of eval, stratified by family. With N+ eval positives, public AP has roughly
SE ≈ sqrt(AP(1−AP)/(0.3·N+)), a rough binomial heuristic: ≈0.02 at AP = 0.9 and N+ = 600. Final
selection uses a paired bootstrap on dev OOF, resampling **tables**, over the full proxy
`0.7·PU-AP + 0.2·MAP@5 + 0.1·behaviorAP` (parent doctrine: `_toolkit/templates/endgame_select.py`).

---

## 4. Evidence retrieval, MAP@5 (20% of the score)

### 4.1 Target definition
- Relevant set = the listed evidence hands, ≤5 in both public dev and private truth. AP@5 has
  denominator min(|rel|, 5), and order matters: a hit at rank 1 is worth more than one at rank 5.
- **Train on listed = 1, all other shared hands = 0.** For the 91% of pairs with exactly 5 listed hands,
  the private relevant set is also a 5-subset of the planted hands, chosen by the host's rule. That rule
  looks structured (chronological ordering is far above chance), so a ranker trained on *listed* labels
  learns planted-ness plus whatever the selection prefers. That is the metric's target.
- The host guarantees each listed hand "contains a behavior-specific action visible in the public
  action log". **Use that guarantee as a candidate filter**: e.g. hands where both partners took a
  voluntary action in the same hand, or one acted facing the other's bet or raise, or one folded with
  the other still in. Measure listed-evidence recall under the filter on dev (it must be ≈100%) and the
  resulting reduction in candidates. Fewer candidates means higher precision@5.

### 4.2 Models
- **(a) Pointwise binary LightGBM** hand model, i.e. the "planted-hand detector". Positives = listed
  evidence. Negatives = non-listed hands of positive pairs, plus hands from HN and U-sample pairs
  (instance-level negatives that teach "planted vs normal interaction"). The same model feeds pair-level
  aggregation (§5).
- **(b) LambdaMART / XE-NDCG** with query = positive pair and doc = candidate hand:
```python
params = dict(objective='lambdarank',            # or 'rank_xendcg' (Bruch 2021; faster, similar quality)
              metric='map', eval_at=[5], label_gain=[0, 1],
              lambdarank_truncation_level=20,   # docs: set slightly above k
              lambdarank_norm=True, learning_rate=0.03, num_leaves=31, min_data_in_leaf=40,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=3, verbose=-1)
# rows sorted by (pair, hand); group = candidates per pair in that order
dtr = lgb.Dataset(Xtr, ytr, group=gtr)
```
  Refs: Burges 2010 (`ml_ltr_lambdamart_burges_2010.pdf`), Bruch 2021 (`ml_ltr_xendcg_bruch_2021.pdf`).
  The H&M 6th-place solution blended LGBM lambdarank, LGBM xendcg, XGB lambdarank and a CatBoost ranker.
  Rank-average objectives within each query.
- **Query-normalised features matter for LTR**: the hand's percentile within the pair, z-score within the
  pair, gap to the pair's top hand. Use percentiles rather than raw ranks because eval candidate pools
  are ~2/3 the size of dev pools.
- **Family conditioning.** Train per-family pointwise models, which give calibrated P(evidence | hand, f).
  At inference mix them: `p(h) = Σ_f P(f | pair)·p_f(h)`. Validate on dev OOF with *predicted* family
  posteriors so the cost of misclassifying the family is included. other_coordination has no training
  evidence, so its hands get the family-agnostic detector (§6).
- **Temporal context features**: the per-hand posterior from the burst HMM (§5.3), the count of pair
  hands in the top decile within ±5/±15 shared hands, and the distance to the nearest other top-5 hand.
  In the toy simulation the HMM posterior improved evidence P@5 by +0.01–0.03 when planted hands were
  clustered. CI (median evidence gap ≈2.5 shared hands) is where this should pay.

### 4.3 Unlisted planted hands (instance-level PU)
- **Default (option 0):** non-listed = 0. This is unbiased for the metric, since the target is "listed".
- **Option 1 (spy-style cleaning):** in 5-listed pairs, drop from training the non-listed hands whose OOF
  score beats the pair's weakest listed hand; keep all hands of 3–4-listed pairs, whose listing is
  probably complete. This can only help by learning planted-ness better. Accept it only if metric-exact
  OOF MAP@5 improves.
- **Option 2 (propensity / unbiased LTR, Joachims 2017):** needs a model of the listing propensity. It is
  not identifiable here, so skip it.

### 4.4 Validation
Use the same `StratifiedGroupKFold` table folds as the pair model, and the metric's own MAP@5 code on
OOF, reported per family. 372 queries give SE ≈ 0.02–0.03, so repeat CV 3×5 before accepting changes
below +0.01.

---

## 5. Episodic / temporal detection, and aggregating hands into pairs

### 5.1 Framing
Pair score = pooling of instance scores (MIL). Standard pools:
- max
- mean
- log-sum-exp `(1/r)·log mean exp(r·s)`
- noisy-OR `1−Π(1−p_i)`
- top-k mean
- learned attention (Ilse 2018)

Deep MIL is unnecessary here. Handcrafted pools feeding a GBDT are the Kaggle-practical version.

### 5.2 Likelihood-ratio view
A calibrated hand probability p_i at training base rate π_h gives `llr_i = logit(p_i) − logit(π_h)`.
- **Top-k sum** Σ of the k largest llr. By Neill's *linear-time subset scanning* property
  (`ml_ts_subset_scan_neill_2020.pdf`, McFowland 2013 `ml_ts_fast_generalized_subset_scan_mcfowland_2013.pdf`),
  the best unconstrained subset for many scan scores is the top-j records by a priority function. So
  "max over k of a penalised top-k sum" *is* the exact unconstrained subset scan.
- **Sparse-mixture GLR**: `max_ε Σ_i log(1 − ε + ε·LR_i)`, with ε on a grid of 0.01…0.4. Under the null it
  self-penalises: correlation with exposure ≈ 0.
- **CUSUM** (Page 1954): `S_t = max(0, S_{t−1} + llr_t − κ)`, statistic max_t S_t = the max over contiguous
  windows of Σ(llr − κ), i.e. a temporally constrained scan.
- **Burst HMM** (adapted from Kleinberg 2002's two-state burst automaton, `ml_ts_bursty_kleinberg_2002.pdf`;
  Kleinberg models arrival gaps, here the model runs over per-hand emissions). Two states: normal (emission 1) and
  active (emission `1−ρ+ρ·LR_i`). Transitions normal→active 1/200 and active→normal 1/40 per shared
  hand, ρ ≈ 0.15. Pair statistic = forward log-likelihood ratio vs all-normal. The per-hand posterior
  `P(planted_i | all)` feeds the evidence ranker.
- **BOCPD** (Adams & MacKay 2007, `ml_ts_bocpd_adams_mackay_2007.pdf`) is online. We see the whole
  sequence offline, so the HMM forward-backward is the better tool. BOCPD is only useful for per-player
  change points ("strategy change" hard negatives).
- **Higher Criticism** (Donoho & Jin 2004, `ml_ts_higher_criticism_donoho_jin_2004.pdf`): per-hand null
  p-values p_(i), `HC = max_{i≤n/2} √n·(i/n − p_(i))/√(p_(i)(1−p_(i)))`. It is self-normalised for n.
  The null p-values come from the hand-score distribution of U pairs in the same context bucket.
  Distribution-free rank scans: Arias-Castro 2016 (`ml_ts_rank_scans_arias_castro_2015.pdf`).

### 5.3 Toy simulation (`research/ml_sim_aggregators.py`)
Setup:
- 40k pairs, prevalence 0.6%, exposure lognormal clipped to 38–419 (median 76);
- positives get 5–8 planted hands inside an episode of 25–70 shared hands, with hand-score shift MU sd;
- a fraction HARD of negatives get a broad +0.6 sd shift over a 30–90-hand streak (tilt/streak-like).

AP results:

| MU / HARD | mean | max | top3 | top5 | top8 | mixGLR | CUSUM | HMM-LLR | HC |
|---|---|---|---|---|---|---|---|---|---|
| 2.5 / 0 | .21 | .44 | – | .74 | – | .83 | .48 | **.84** | .52 |
| 2.5 / .08 | .02 | .34 | **.56** | .53 | .39 | .27 | .28 | .34 | .38 |
| 3.0 / .08 | .03 | .66 | **.87** | .85 | .76 | .78 | .63 | .81 | .71 |
| 3.0 / .15 | .02 | .59 | .85 | .82 | .71 | .72 | .61 | .76 | .64 |
| 3.5 / .08 | .04 | .88 | – | **.98** | – | .97 | .88 | .97 | .91 |

Lessons (toy, but the direction is robust):
1. Mean pooling is catastrophic.
2. Top-k with k at or slightly below the planted count is the most robust single pool; k too large hurts.
3. Likelihood accumulators (GLR/HMM) win only when the hand model is calibrated against the *real*
   negatives, hard ones included.
4. Subtracting the pair's median llr before top-k (`top5-median`) helps when hard negatives carry
   broad shifts (.82→.87 at HARD = .15).
5. Hand-detector quality (MU) matters far more than the pool.

### 5.4 Multiple testing and exposure debiasing
- Scans inflate the null maximum with n. Correlation of the statistic with n among negatives was
  +0.3 (max), +0.5 (top5) and +0.27 (CUSUM), against ≈0 for GLR and HC and −0.1 to −0.5 for HMM. Options:
  (i) give the pair GBDT `log n` alongside raw statistics, the simplest and usually best;
  (ii) exposure-conditional null quantiles from U pairs, `cond_null_norm` in the module, which helped
  weak detectors (+0.06 AP at MU = 2.5) and cost strong ones slightly;
  (iii) Bonferroni-style penalties of log(#windows).
- **Exposure-matched training**: build dev pair features on contiguous 2,000-hand dev windows (2–3
  windows per pair, keeping positive windows that contain ≥4 of 5 listed hands) as well as on the full
  dev phase. Ablate the two on LB with one submission each.
- **Player-local baselines against the listed hard negatives** (tilt, weak play, streaks, strategy
  changes): score every action against the *acting player's* rolling baseline (e.g. ±150 table hands,
  excluding hands with the partner) and against the table context. Player-level change points then
  cancel before the scan, so only *partner-specific* deviations accumulate.
- "Repeated opponent selection" means high exposure without coordination. Never let exposure count
  alone raise the score: use it only as a denominator or conditioning variable.

---

## 6. Open-set detection of other_coordination

### 6.1 Is it worth anything?
Yes: it is a positive in 90% of the metric's weight (pair AP plus evidence). If a fraction φ of eval
positives are other_coordination and get negative-like scores, AP ≈ (1−φ)·AP_known and their evidence
AP is ≈0. With φ = 0.15 that is ≈ −0.10 AP (−0.07 final) plus −0.03 from evidence. The LB top (0.938)
suggests leaders catch most positives, so either φ is small or generic signals find them.

### 6.2 Family-agnostic signals, from the collusion literature
- **Collusion tables** (Mazrooei, Archibald & Bowling 2013, `ml_domain_collusion_tables_mazrooei_2013.pdf`).
  With a value function V (here: exact equity × pot, computable because all hole cards are visible), the
  impact of agent k's action on agent j's value is `C(j,k) = Σ_{k's actions} [V_j(after) − V_j(before)]`.
  - *Total Impact*: `TI(a,b) = Σ_{i,j∈{a,b}} C(i,j)`.
  - *Marginal Impact*: `MI(a,b) = [C(b,a) − mean_{i≠a,b} C(i,a)] + [C(a,b) − mean_{j≠a,b} C(j,b)]`, i.e.
    "treats the partner differently from everyone else".

  This captures chip dumping, soft play and squeeze pressure under one definition, *without* modelling
  any family, which is exactly what an unseen mechanism needs.
- **Information-theoretic** (Bonjour, Aggarwal & Bhargava 2022, `ml_domain_infotheoretic_collusion_bonjour_2022.pdf`):
  mutual information between agents' actions is elevated under collusion. Smed et al.'s taxonomy has
  spectator (information-sharing), assistant (dumping) and association (mutual benefit) collusion.
  Information sharing is the obvious candidate for an undisclosed 4th mechanism. Test with
  `I(action_A ; strength(hole_B) | strength(hole_A), public state)`, estimated as the lift in predicting
  A's fold/raise from B's hole-card equity compared with A's other opponents.
- **Lockstep / dense-block** (CopyCatch, Beutel 2013 `ml_graph_copycatch_beutel_2013.pdf`; FRAUDAR, Hooi
  2016 `ml_graph_fraudar_hooi_2016.pdf`): synchronised joint actions in the same hands (joint entries,
  back-to-back raises around a third player). FRAUDAR's camouflage-resistant column weighting,
  log(1/(deg+5)), carries over as "weight interactions by rarity".

### 6.3 Anomaly scoring
- **Isolation Forest** (Liu 2008, `ml_anomaly_iforest_liu_2008.pdf`) or robust Mahalanobis (MCD) on the
  vector of **double-centred** pair residuals (§7). EA's team-game collusion system (Greige 2022,
  `ml_domain_collusion_team_games_greige_2022.pdf`) does exactly this: pair features from graph and
  behaviour, then iForest.
- Caveat: hard negatives *are* anomalies. Use them as labelled normals, as in **Deep SAD** (Ruff 2020,
  `ml_anomaly_deepsad_ruff_2020.pdf`), which uses labelled normals and labelled anomalies plus unlabelled
  data. GBDT form: a PU classifier on **family-agnostic features only**, with all 372 positives pooled
  and HN as negatives.

### 6.4 Validation: leave-one-family-out (LOFO)
Train the generic branch on 2 families (+HN+U) and score the held-out family's positives against HN ∪ U
at held-out tables. Mean LOFO AP over the three folds is the best available proxy for detecting an
unseen mechanism. The generic branch enters fusion only if LOFO AP clearly beats the family-specific
model's LOFO AP. Behavior label for generic-only detections: `other_coordination` (§3.6).

---

## 7. Relational modelling (no GNN needed)
The data is 400 disconnected 30-node graphs, which is ideal for matrix features plus a GBDT.

- **Directed statistic matrices per table**, S[j,k]. Examples: EV impact C(j,k) per 100 shared hands,
  fold-to-k rate, aggression toward k vs others, top-5 hand-llr sum. Shrink rates by empirical Bayes
  first: `(x + m·p0)/(n + m)`.
- **Double centring** = the interaction residual: `R = S − row_mean − col_mean + grand_mean`, computed
  with n-weighted means. It removes player-level traits (weak play, tilt, loose/aggressive style) that
  make a player look "suspicious with everyone". It matters here: 190 HN pairs involve a real colluder
  with a non-partner.
- **Exclusivity / reciprocity**:
  - partner share `S[j,k]/Σ_k S[j,k]`;
  - rank of k among j's partners;
  - `mutual_top1 = argmax_k R[j,:]==k & argmax_j R[:,k]==j`;
  - gap between the best and second-best partner.

  Colluders have one special partner. Tilted or weak players have flat profiles.
- **Triads**: `max_C min(R[A,C], R[B,C])`. Low priority, since labels show no closed rings (§1).
- **Player-level suspicion** (max over partners) is a feature *of the pair* only through
  "is this the player's argmax partner". Raw player suspicion leaks into all of a colluder's pairs,
  which are negatives.
- **Guardrail**: compute relational features only from gameplay (co-played hands). Never compute them
  from which pairs appear in `evaluation_pairs.csv` / `development_labels.csv`. Pair-list structure is a
  construction artifact, the same class as Quora Question Pairs' "magic" graph features, and the rules
  prohibit it.

---

## 8. Kaggle precedents and what won

Items marked (h) are recalled from memory and could not be re-verified online today; Kaggle writeup pages
do not render for the fetch tool.

| competition | similarity | what won | lesson for us |
|---|---|---|---|
| IEEE-CIS Fraud Detection (2019), 1st Deotte & Yakovlev | entity-level labels, groups | constructed client UIDs (card1+addr1+D1), aggregated features per UID, removed UID-identifying columns so the model generalises to unseen clients, averaged predictions per UID (h) | the pair is our entity. Aggregate hands into pairs; group CV by table; no identity features |
| H&M Fashion Recs (2022) / OTTO (2023) | top-k retrieval metric (MAP@12 / recall@20) | candidate generation + GBDT ranker; 6th place H&M blended LGBM lambdarank + xendcg + XGB lambdarank + CatBoost ranker | evidence = pure reranking of ~40–400 candidates. Ensembling ranker objectives helps |
| Eedi Mining Misconceptions (2024–25), 1st "MTH 101" | MAP@25 retrieve-and-rerank, **unseen classes in test** | retrieve (embeddings) → rerank (LLM); kept external misconceptions for generalisation | open-set: build detectors that generalise beyond the labelled families (§6) |
| Coleridge "Show US the Data" (2021), 1st ZALO FTW "metric learning and GPT" | hidden-test labels not in train | context models rather than string-matching the training labels; big public→private shake-up (h) | don't overfit the 3 disclosed families on the public LB |
| LLM – Detect AI Generated Text (2024), 1st place (code: github rbiswasfc/llm-detect-ai) | hidden generators | large, *diverse* fine-tuned-LLM generated training data; heavy shake-up (a 21st-place title: 0.986 public → 0.932 selected private) | diversity of positives for an unseen mechanism; distrust public LB |
| Elo Merchant (2019) | ~1% extreme outliers | dedicated outlier classifier blended with a regressor trained without outliers (h) | a dedicated rare-positive head (pooled binary risk) rather than a regression-ish score |
| HPA Single-Cell (2021), 43rd | image-level labels applied to cells = MIL | PU-learning-based solution | bag→instance label transfer is standard; our hand detector is the same idea |
| Quora Question Pairs (2017) | pair classification | "magic" graph features from the pair list itself | here this class of feature is prohibited (construction artifact) |

---

## 9. Recipe for this comp (D-5 → deadline 2026-09-20 22:00 UTC)

**Principles:**
- A single strong base model first; blending only on D-1.
- Every decision on grouped OOF.
- LB for calibration only.
- Every experiment gets a ledger row with a mechanism and a verdict.

### Day 0 (15 Sep): foundations
1. **Folds**: `StratifiedGroupKFold(5, shuffle=True, random_state=42)`, groups = table_idx,
   y = family (0 for HN/U). Write `data/folds_table5.csv` (table → fold) and freeze it. Every pair and
   hand model uses it; a second fold seed (43) is only for the accept/reject stability check.
2. **Candidate filter** from the host's evidence guarantee (§4.1). Report recall of listed evidence
   (target ≥99.5%) and candidate reduction.
3. **Hand-level feature table** for (a) all labelled dev pairs, (b) a U sample (all pairs of ~40 tables
   plus ~30k random pairs), (c) all eval pairs in the eval phase:
   - equity-aware decision residuals (fold/call/raise vs pot odds and exact equity, using all hole cards);
   - EV impact between the two partners (collusion-table cells);
   - partner-directed vs other-directed aggression;
   - action residuals vs the acting player's rolling baseline;
   - table context in bb.

### Day 1 (16 Sep): the base model
4. **Planted-hand detector** (E2): LightGBM binary; positives = listed evidence; negatives = hands of HN
   and U-sample pairs. Non-listed hands of positive pairs go in at weight 0.3, with 0 as the ablation.
   Train pooled plus per-family heads, OOF by table.
5. **Pair aggregation features** from E2 OOF, per head:
   - top-k sums (k = 1, 2, 3, 5), top-5 minus pair median;
   - mixture-GLR, HMM-LLR (ρ = .15, episode length 40), CUSUM (κ = .5, 1), HC;
   - `log n`, and exposure-conditional versions;
   - double-centred / exclusivity / mutual-top1 versions of the top-5 sum and of the EV-impact matrix.
6. **Pair models** R1 chain, R2 weighted PNU and R3 bagged PU, each with 3 seeds. Report, all on the
   same folds:
   - PU-corrected AP on **eval** scores (`mpe_alpha` + `pu_ap_estimate`);
   - HN-AP (P vs HN only);
   - AP of P vs (HN ∪ dev-U), the uncorrected proxy;
   - LOFO AP.
7. **First submission**: risk = best of R1/R2 (rank-averaged with ties broken); family = argmax of a
   multinomial on positives; evidence = top-5 pooled E2 score among filtered candidates. Log
   CV-proxy vs LB.

### Day 2 (17 Sep): evidence + exposure
8. **Evidence ranker**: pointwise per-family mixture + lambdarank + xendcg, rank-averaged within query.
   Add HMM-posterior and local-density features. Metric-exact OOF MAP@5 by family. Accept at ≥ +0.01
   (3×5 CV).
9. **Exposure-matched windows** (§5.4) vs full-dev features: OOF plus one LB submission each.

### Day 3 (18 Sep): generic branch and fusion
10. **Generic branch**: collusion-table TI/MI scores, action↔partner-hole-card MI lift, iForest/MCD on
    double-centred residuals, family-agnostic PU model. Keep only what wins on LOFO.
11. **Fusion**: cross-fitted LR on clipped logits {R1, R2, R3, family heads, generic} plus log n. Tune the
    behavior assignment and the other_coordination override on OOF behavior AP. Submit.

### Day 4 (19 Sep): scale and select
12. Seed-bag the winners (5–10 seeds). Run a paired bootstrap resampling tables over the full OOF proxy
    `0.7·PU-AP + 0.2·MAP@5 + 0.1·behaviorAP`.
    - **Slot 1**: highest P(beat).
    - **Slot 2**: best candidate that does *not* share slot 1's risk branch (chain vs weighted-PNU), or its
      exposure choice.
13. Freeze by 20 Sep 12:00 UTC. Tick finals manually.

### Accept rule and what would overturn it
- **Accept** a change when the OOF proxy improves ≥ +0.003 with the same sign on fold seeds 42 and 43
  (372 positives make smaller steps noise).
- **If the LB disagrees with the PU-corrected AP across ≥3 submissions**, the proxy's assumptions are
  failing: SCAR, exposure matching, or a large other_coordination share. Switch selection weight toward
  HN-AP plus the LB trend, and revisit exposure matching first.
- **If LOFO AP of the generic branch beats the family model on all 3 folds**, promote the generic branch
  from "fusion member" to base model.

---

## 10. References (local copies in `research/papers/`)

PU learning:
- `ml_pu_elkan_noto_2008.pdf`: Elkan & Noto, *Learning classifiers from only positive and unlabeled data*, KDD 2008. https://cseweb.ucsd.edu/~elkan/posonly.pdf
- `ml_pu_upu_duplessis_2015.pdf`: du Plessis, Niu, Sugiyama, *Convex formulation for learning from PU data*, ICML 2015.
- `ml_pu_nnpu_kiryo_2017.pdf`: Kiryo et al., *PU learning with non-negative risk estimator*, NeurIPS 2017. arXiv:1703.00593
- `ml_pu_bagging_mordelet_vert_2014.pdf`: Mordelet & Vert, *A bagging SVM to learn from positive and unlabeled examples*, PRL 2014. arXiv:1010.0772
- `ml_pu_spy_sem_liu_2002.pdf`: Liu, Lee, Yu, Li, *Partially supervised classification of text documents* (spy / S-EM), ICML 2002.
- `ml_pu_tice_bekker_davis_2018.pdf`: Bekker & Davis, *Estimating the class prior in PU data through decision tree induction* (TIcE), AAAI 2018.
- `ml_pu_km2_ramaswamy_2016.pdf`: Ramaswamy, Scott, Tewari, *Mixture proportion estimation via kernel embedding* (KM1/KM2), ICML 2016. arXiv:1603.02501
- `ml_pu_alphamax_jain_2016.pdf`: Jain, White, Radivojac, *Estimating the class prior and posterior from noisy positives and unlabeled data*, NeurIPS 2016. arXiv:1606.08561
- `ml_pu_eval_jain_white_radivojac_2017.pdf`: *Recovering true classifier performance in PU learning*, AAAI 2017. arXiv:1702.00518
- `ml_pu_sar_bekker_davis_2019.pdf`: Bekker, Robberechts, Davis, *Beyond the SCAR assumption* (SAR-EM). arXiv:1808.08755
- `ml_pu_survey_bekker_davis_2020.pdf`: Bekker & Davis, *Learning from positive and unlabeled data: a survey*, MLJ 2020. arXiv:1811.04820
- `ml_pu_pnu_sakai_2017.pdf`: Sakai et al., *Semi-supervised classification based on classification from PU data* (PNU), ICML 2017. arXiv:1605.06955
- `ml_pu_pubn_hsieh_niu_sugiyama_2019.pdf`: Hsieh, Niu, Sugiyama, *Classification from positive, unlabeled and biased negative data*, ICML 2019. arXiv:1810.00846
- `ml_pu_positive_rather_negative_2019.pdf`: *Revisiting sample selection approach to PU learning: turning unlabeled data into positive rather than negative*. arXiv:1901.10155
- `ml_pu_verify_scar_2024.pdf`: Teisseyre et al., *Verifying the SCAR assumption in PU learning*. arXiv:2404.00145
- `ml_dre_telescoping_rhodes_2020.pdf`: Rhodes, Xu, Gutmann, *Telescoping density-ratio estimation*, NeurIPS 2020. arXiv:2006.12204
- `ml_quant_weiss_kdd09.pdf`: Xue & Weiss, *Quantification and semi-supervised classification methods for handling changes in class distribution*, KDD 2009. Covers Forman's ACC; Forman DMKD 2008 DOI 10.1007/s10618-008-0097-y. Saerens et al. 2002, DOI 10.1162/089976602753284446, is not downloadable.

MIL, ranking and retrieval:
- `ml_mil_attention_ilse_2018.pdf`: Ilse, Tomczak, Welling, *Attention-based deep MIL*, ICML 2018. arXiv:1802.04712
- `ml_mil_pu_bags_bao_2018.pdf`: Bao et al., *Convex formulation of MIL from positive and unlabeled bags*. arXiv:1704.06767
- `ml_ltr_lambdamart_burges_2010.pdf`: Burges, *From RankNet to LambdaRank to LambdaMART*, MSR-TR-2010-82.
- `ml_ltr_xendcg_bruch_2021.pdf`: Bruch, *An alternative cross entropy loss for learning-to-rank* (LightGBM `rank_xendcg`). arXiv:1911.09798
- `ml_ltr_unbiased_joachims_2017.pdf`: Joachims, Swaminathan, Schnabel, *Unbiased learning-to-rank with biased feedback*, WSDM 2017. arXiv:1608.04468
- `ml_ltr_rocketqa_hardneg_2021.pdf`: Qu et al., *RocketQA* (denoised hard negatives, false negatives among hard negatives). arXiv:2010.08191
- LightGBM ranking parameters: https://lightgbm.readthedocs.io/en/latest/Parameters.html (`lambdarank_truncation_level` default 30, "set slightly higher than k"; `rank_xendcg` "faster than and achieves the similar performance as lambdarank"; `lambdarank_position_bias_regularization`).

Temporal and scan:
- `ml_ts_bursty_kleinberg_2002.pdf`: Kleinberg, *Bursty and hierarchical structure in streams*, DMKD 2003.
- `ml_ts_bocpd_adams_mackay_2007.pdf`: Adams & MacKay, *Bayesian online changepoint detection*. arXiv:0710.3742
- `ml_ts_higher_criticism_donoho_jin_2004.pdf`: Donoho & Jin, *Higher criticism for detecting sparse heterogeneous mixtures*, AoS 2004.
- `ml_ts_subset_scan_neill_2020.pdf`: Neill, *Subset scanning for event and pattern detection* (LTSS overview); original: *Fast subset scan for spatial pattern detection*, JRSS-B 2012.
- `ml_ts_fast_generalized_subset_scan_mcfowland_2013.pdf`: McFowland, Speakman, Neill, *Fast generalized subset scan for anomalous pattern detection*, JMLR 2013.
- `ml_ts_rank_scans_arias_castro_2015.pdf`: Arias-Castro et al., *Distribution-free detection of structured anomalies: permutation and rank-based scans*, JASA.

Open-set, anomaly and graph:
- `ml_anomaly_iforest_liu_2008.pdf`: Liu, Ting, Zhou, *Isolation forest*, ICDM 2008.
- `ml_anomaly_deepsad_ruff_2020.pdf`: Ruff et al., *Deep semi-supervised anomaly detection*, ICLR 2020. arXiv:1906.02694
- `ml_graph_copycatch_beutel_2013.pdf`: Beutel et al., *CopyCatch: lockstep behavior*, WWW 2013.
- `ml_graph_fraudar_hooi_2016.pdf`: Hooi et al., *FRAUDAR: bounding graph fraud in the face of camouflage*, KDD 2016.

Collusion detection (domain ML):
- `ml_domain_collusion_tables_mazrooei_2013.pdf`: Mazrooei, Archibald, Bowling, *Automating collusion detection in sequential games* (collusion tables, TI/MI scores), AAAI 2013.
- `ml_domain_infotheoretic_collusion_bonjour_2022.pdf`: Bonjour, Aggarwal, Bhargava, *Information-theoretic approach to detect collusion in multi-agent games*, UAI 2022.
- `ml_domain_collusion_team_games_greige_2022.pdf`: Greige et al. (EA), *Collusion detection in team-based multiplayer games* (graph features + iForest), AAAI-MAKE 2022. arXiv:2203.05121

Kaggle write-ups (URLs; pages need a browser):
- IEEE-CIS 1st place part 2: https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2
- H&M 6th place: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/writeups/hard2rec-6th-place-solution ; 2nd place: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324197
- Eedi 1st place: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution
- Coleridge 1st place: https://www.kaggle.com/competitions/coleridgeinitiative-show-us-the-data/writeups/zalo-ftw-1st-place-solution-metric-learning-and-gp
- LLM-Detect 1st place code: https://github.com/rbiswasfc/llm-detect-ai ; 21st place (shake-up numbers): https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470148
- HPA single-cell 43rd (PU): https://www.kaggle.com/competitions/hpa-single-cell-image-classification/writeups/lv97-5-43rd-positive-unlabeled-learning-based-solu
