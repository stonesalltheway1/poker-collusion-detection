# PLAN — Slash poker collusion · state and endgame (updated 2026-09-20 02:40 UTC)

Deadline **2026-09-20 22:00 UTC**. Reading order: this file → `research/CLOSED_LANES.md` (measured dead ends) →
`research/forensics/listing_rule.md` (the evidence breakthrough) → `experiments/LEDGER.md` (append-only run log).
Prize deliverables: `research/WRITEUP.md`, `README.md` + `src/run_all.py`, `research/CASE_REVIEWS.md`.

## 1. Where we stand

Best public **0.91954** (sub024) / 0.91942 (sub025) — **11th of 357**, not 9th; the board moved on 19 Sep.

| # | team | public |
|---|---|---|
| 1 | Pardheev Krishna | .93876 |
| 2 | blastyy | .93823 |
| **3** | **Tejasv Bhatia** | **.93783** ← prize line, gap **+0.01829** |
| 4 | John Tyler | .93672 |
| 5 | [Deleted] | .93048 |
| 6–9 | Leo / Marc Donovici / thisray / seantangth | .92902 … .92486 |
| 10 | Eduardo Trabattoni | .91973 |
| **11** | **Exposed** | **.91954** |
| 12 | Amin Mohamed | .91898 |

**Corrected component decomposition** (exp048_fuse_bags risk + exp042_ramp evidence = sub025 0.91942):

| component | weight | ours | loss of final score |
|---|---|---|---|
| pair AP | 0.70 | **0.9745** | 0.0179 |
| evidence MAP@5 | 0.20 | **0.7048** | 0.0590 |
| behaviour macro-AP | 0.10 | **0.9635** | 0.0037 |

`0.7(.97445) + 0.2(.70480) + 0.1(.96345) = 0.91942`. **Measured directly**, not inferred: sub031 (2026-09-20,
sub025 with every behaviour set to `other_coordination`) = **0.82325**, which isolates
`0.7·pairAP + 0.2·evMAP + 0.1·b0` for *this* risk model. Only b0 is assumed, and it moves pair AP by ±0.0001.

> **CORRECTION (2026-09-20, twice).** The original table (pair AP .9750 / behaviour .9664) was wrong. The sub014 probe
> sets every behaviour to `other_coordination`, which makes all three behaviour score vectors **all-zero** — and
> the host's `_average_precision` uses a *stable* argsort, so an all-zero vector is ranked in `pair_id` order and
> returns the AP of a **random** ranking, not 0.
> The first fix of this put that AP at 0.00466, which was ALSO wrong: `metric.py:131` scores each family
> against *its own* positives, so the relevant count is ~147/3 per family, not 147. The correct value is
> **0.00175 ± 0.00061** at public size (0.00121 at private size; `scripts/phi_behap_bound.py`). Adding it
> back gives behaviour AP **0.9682** and pair AP **0.9738** — robust to the unknown eval-positive count
> (pair AP 0.97375–0.97381 for 420–560 positives). A tell we missed: under the wrong b0 the
> behaviour-minus-pair gap was −0.0023, *tighter than the −0.0070 that is physically possible at zero
> other_coordination*. The corrected −0.0056 sits right on that baseline.

## 2. The two structural findings that produced the score

1. **Phase-locality.** Coordination exists only in the phase being scored. Same labelled pairs, same statistics:
   dev AUC 0.73–0.88 → eval AUC 0.47–0.49; colluder player sets disjoint; dev↔eval Spearman of our pair scores
   0.16 (**not** 0.08 — that number was the CI *rate* correlation, `research/SYNTHESIS.md` line 48).
2. **The evidence listing rule**: `sorted(EVENTS, key=(type, hand_seq))[:5]`. CI's type is exact (folds before the
   pair's first preflop raise, 0 vs 1, on all 460 listed CI hands); DT/SP types are latent (learned, AUC .972/.945).
   A Poisson-binomial DP blended **0.7** with the learned in-pair ranker: dev MAP .711 → .775, LB evidence
   .666 → .705.

## 3. What actually limits the score — the corrected diagnosis

The old story ("φ = 0.14 of eval positives belong to an undisclosed 4th family; evidence caps at ≈0.715") is
**wrong in its interpretation and unidentified in its arithmetic**. Four independent results, all new on 2026-09-20:

1. **The fit is not identified.** `obs = (1−φ)K + φαL` is linear in `((1−φ)c, φαc)`, so any assumed global transfer
   factor `c` gives the same RMSE: c = 0.92 → φ = 0.065; c = 1.00 → φ = 0.139; c = 1.10 → φ = 0.218. Even with
   c = 1 forced, φ = 0.14 ± 0.04. A different 2-parameter model (`obs = c·K − g·w`, c = .933, g = .032) fits the
   same three points essentially as well (RMSE .00074 vs .00057) and posits **no** dead subpopulation.
   The only identified quantity is the **transfer factor: a dev gain arrives on eval at 0.86–0.93.**
2. **The behaviour probe rules out `other_coordination` outright.** An OC truth pair is a negative for all three
   disclosed families but a positive for pair AP, so high-ranked OC truth costs ≈0.95 points of behaviour macro-AP
   per unit. Simulation at the real ranking (`scripts/phi_behap_bound.py`, output in
   `research/forensics/phi_behap_bound_output.txt`): φ_OC = 0.14 ⇒ behAP 0.84–0.89; φ_OC = 0.05 ⇒ 0.93–0.94;
   **observed 0.9635 against pair AP 0.9745 ⇒ φ_OC ≲ 0.02** (a gap of −0.011 is consistent with either behaviour
   accuracy ≈ 0.995 at φ = 0, or perfect routing at φ ≈ 0.02).** That leg only binds if other_coordination
   pairs rank *high*; the second leg closes the other case — exp039's leave-one-family-out prices an unseen
   mechanism's pairs at AP 0.070–0.079, so `pairAP ≤ (1−φ) + 0.079φ` with pairAP 0.9738 forces
   **φ_undetected ≤ 0.0277**. High or low in the ranking, the share is ≤ ~3%, not 14%. After the direct probe the
   second leg carries most of the weight, which is why it was worth adding. The pairs we detect are essentially all
   disclosed-family, and we route their family essentially correctly.
3. **The mis-served cohort was identified positively**: low-exposure **soft play**. Its 16-statistic event profile
   matches labelled dev SP positives in an exposure-matched 2000-hand window, and a directed statistic calibrated
   to fire on 0/1488 clean pairs fires on 183/174,000 dev and 193/174,000 eval pairs — ratio 1.05, i.e. **no
   eval-only population exists**.
4. **The strength gate is an exposure gate, not a family detector**: 0/25 of the gate-firing top-200 eval pairs
   carry `fam_unexplained`; their median shared-hand count is 52 vs 76. The ledger's "eval incidence matches φ"
   was a coincidence.

**Where the evidence loss actually is.** Oracle decomposition on the same folds: true *type* is worth +0.018 dev
MAP, true *event* (was this hand's action altered?) is worth **+0.157** (.775 → .932). And recall@5 is .820 while
**recall@10 is .970** — candidate retrieval is solved; the whole remaining loss is discriminating 5 of ~10
candidates. Perfect selection from our own top-10 would be worth **+0.039 final**. That is the lane, and it is
locked behind a latent bit that six separate attempts failed to expose.

**Arithmetic on the leaders.** With our evidence 0.70485, #1's 0.93876 would need pair AP ≥ 0.9968 *and*
behaviour AP = 1.0 — impossible. So #1 beats us on **evidence**, in roughly [0.73, 0.86], most plausibly 0.76–0.80.
Equally, nobody has cracked the exact planting predicate: perfect known-family listing plus our other components
scores 0.942–0.950, well above the actual best of 0.93876.

## 4. Conventions that must not drift

- Frozen, never regenerate: `data/folds_tables_5.csv`, `data/derived/spies_exp005.parquet`, `spies2_exp022.parquet`
  (all three now committed with `git add -f`; `.gitignore` carries explicit `!` exceptions).
- **Same-fold eval hand scoring** (`SAMEFOLD_EVAL=1`).
- Decide on view C′/C2 + paired table-bootstrap P(better) ≥ 0.8, never on a single LB delta (noise ≈ ±0.006).
- One component per LB test.
- Relabelling experiments must carry `y_orig` through every scorer.

## 5. Endgame — CLOSED. All 5 submissions spent; only the manual tick remains.

**2026-09-20 UTC submissions (5/5 used):** sub026 01:33 · sub027 02:29 · sub028 04:44 · sub030 05:0x · sub031 05:1x.

| file | public | Δ vs sub025 | what it isolated | verdict |
|---|---|---|---|---|
| sub026 = exp048 + exp027ev evidence | 0.91169 | −.00773 | evidence anchor | pinned evidence(exp042_ramp) = .70480 |
| sub027 = exp048 + exp036 FLAT evidence | 0.91918 | −.00024 | the ramp gate | ramp slightly better; kept |
| sub028 = **exp068 famspec-mix w=.15** + ramp | 0.91935 | **−.00007** | risk only | **NULL on eval** — do not ship |
| sub030 = exp048 + **exp060_sp09** evidence | 0.91905 | **−.00037** | evidence only | **FALSIFIED** by its own rule |
| sub031 PROBE = sub025, behaviour all `other_coordination` | 0.82325 | — | pair AP, directly | see §1 — overturned the inferred triple |

**FINAL SELECTION — must be ticked by hand at
<https://www.kaggle.com/competitions/detect-suspicious-value-transfers-in-poker/submissions> before 22:00 UTC.**

| slot | file | public | why |
|---|---|---|---|
| **1** | `sub025_fusebags_exp042ramp.csv` | 0.91942 | Best measured file. 3-seed LightGBM bag ⊕ 3-seed XGBoost bag, rank-averaged 50/50. Tied with sub024 (Δ +0.00012, paired bootstrap P 0.543) but the wider XGBoost bag (3 fits vs 1), so lower variance on the private split. |
| **2** | `sub028_famspecw15_exp042ramp.csv` | 0.91935 | **The best hedge available**: a genuinely different risk model (Spearman 0.9955, 5/500 of the top-500 differ) at an essentially **zero** measured cost (−0.00007). Evidence and behaviour columns are bit-identical to sub025, so the hedge is isolated to the pair ranking. E[max] gain ≈ +0.0004, the largest of any pairing. |

E[max(private)] over the candidate pairings, using the measured Δ as the mean and the decorrelation as the spread:
sub028 **+0.00037** · sub024 +0.00023 · sub027 +0.00018 · sub023 +0.00014 · sub030 +0.00007.

**Do NOT tick sub024 + sub025** (Kaggle's default if nothing is selected): byte-identical evidence and behaviour
columns on all 112,540 rows, risk Spearman 0.99930, and an identical top-500 set — one bet in two slots.
**Ticking is therefore required, not optional.** Do not use sub011 (0.91117): it gives up 0.008 to insure the
listing-rule DP, which a matched A/B confirms at +0.0066. Note the ramp's +0.00025 (sub016→sub023) and +0.00024
(sub025→sub027) are **not** independent replications — evidence(sub016) ≡ evidence(sub027) and
evidence(sub023) ≡ evidence(sub025), so both are the same quantity on the same 147 public positives.

**Methodological note worth keeping.** A *paired* public-LB delta between two one-component variants is not
subject to the ±0.006 absolute noise: both files are scored on the same public subset, so their difference is a
measurement of that subset. That is why −0.00007 and −0.00037 are readable at all, and it is what made the last
three slots worth spending.

## 6. Re-audited on 2026-09-20 — all null, do not reopen

**Second push, 7 lanes, 2026-09-20 03:00–06:00 UTC — two candidates built, both dead on eval:** the family-specialist
risk mixture (C′ +0.0028 at P 1.000 but LB −0.00007 — views C′/C2 delete the very population where its cost
lives, and every eval positive is an *unlabelled* positive) and the soft-play event-count rescale (dev MAP +0.0027
at P 0.887 but LB −0.00037, transfer factor −0.68). Five further lanes were clean nulls: counterfactual-value
features in the event head, a per-player×spot baseline fit on partner-absent hands, exact expected-AP@5 ordering
under the DP’s joint law, partner seed-bagging, and a 5+5 seed bag.

Pair fusion re-weighting (CatBoost at every weight 0.05–0.33; L:X ratio; rank vs logit vs prob; cross-fitted
hill-climb — the last is *worse* out-of-fold than fixed weights); adding the sequence model (CV-positive at
P 0.990 but LB-falsified at −0.00063); per-family evidence blend weights (+0.00098, P 0.766); ramp-bound grid
(dev has no pairs below the low bound, so dev carries no information); partner averaging (best standalone ranker,
**worst** blend member — the blend needs a calibrated P(listed), not a better ranking); the listing type key for
DT/SP (best mechanical rule 0.911/0.924, below the incumbent learned model); player-specific normal-play baselines
in the event head (−0.0096, P 0.004); per-pair q recalibration (already calibrated: 4.89/4.76/4.93 vs true
4.90/4.79/5.00); a within-pair ranking objective (−0.007…−0.032); forum/kernel/web intel (no new thread since
2026-09-15, no public generator).

## 7. Honest odds

Top-3 needs +0.01829. The 2026-09-20 push built the two best candidates the architecture admits and the
leaderboard killed both, so the realistic remaining headroom is **0.000**: sub025 is our best measured file and
nothing in the repo beats it. Private
is 70% of the eval set drawn from the same pool, and the score difference between two teams is largely
common-mode, so a 0.018 flip is a ≫3σ event. **P(top 3) ≈ 0.1–0.3%.** P(holding a top-15 finish) is high; ranks 8–13 sit
within 0.006 of each other, so a ±0.005 shakeup reorders that block and our realistic range is roughly 8th–14th.

One thing that is *not* negligible: rule 2.8a says the sponsor reviews teams **in private-LB order until the prize
positions are filled**, so a team above us that cannot produce a write-up, reproducible code and case reviews is
skipped. That makes the deliverables worth more than another 0.0002 of score — they are complete and committed.
