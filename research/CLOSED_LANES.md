# Closed lanes — measured dead ends (do not re-litigate)

Every row was measured, not guessed. Each states what would reopen it. Ledger rows in `experiments/LEDGER.md`.

## Evidence (20% of the score) — transfer factor 0.86–0.93 measured; **no ceiling established**

> The old header said "capped at ≈ 0.715". That was our own routing's value restated through an
> unidentified model, not a property of the task (see the phi row below). What *is* measured is that a
> development gain arrives on evaluation at 0.86–0.93, and that an oracle event bit would be worth
> +0.157 development MAP.

| lane | result | reopen if |
|---|---|---|
| Evidence **selection/ordering** policies: pool-wide segment quotas, shortlist re-ranking, greedy expected-AP@5 | 3 nulls. Ranking by P(listed) is provably E[AP@5]-optimal under independence; the flat top-5 already picks the right segment mix (DT 2.10 seg-2 hands vs a true 2.34) | a k1 predictor with corr > 0.8, or segment truth for the zero-inversion pairs |
| **Exact E[AP@5] under the DP's JOINT law** (the "only optimal under independence" objection, settled exactly) | NULL. `scripts/lr_joint.py` computes u_i = E[rel_i/L] and v_ij = E[rel_i rel_j/L] exactly (leave-one-out / leave-two-out Poisson-binomial, validated to 1e-16 against brute force) and searches all ordered 5-subsets of the top-10. The model's own E[AP] gain over marginal ranking is **+0.00004** (pure DP) / +0.00008 (blend); measured dev MAP moves -.0005..+.0006, every P(better) < 0.8; invariant to candidate set (10/15/20) and to conditioning on L >= 3. The dependence is strong (implied corr +0.45) but **nested** - listed sets are prefixes of the (type, time) order - and marginal ranking is AP-optimal under a nested law. The 1/min(L,5) channel is calibrated, not broken (model E[L] vs true n_rel corr +0.730; Q1 exposure true 4.714 / model 4.711), and is a per-pair scale factor that barely correlates with WHICH hands are relevant | a listing law that is genuinely MULTIMODAL. Measured on the family mixture: +.0003 at max-P .9, +.0006 at .8, **+.0065 at .5 (P .927), +.0131 at 1/3 (P .959)** - so reopen if evidence is ever routed through a flat family posterior, or if the event head is ever given a latent pair-level mixture. Today the eval top-200/400 risk pairs have max-P .998/.995 with 0% below .8, so there is no population to collect it from |
| **Oracle** segment quotas (true k1 handed to the rule) | ceiling +.008 MAP = +.0016 final, on the 61%/70%/12% of DT/SP/CI pairs with the structure | — |
| **Family-agnostic** evidence head (pooled event+type, union gate) for unmodelled families | .260 vs .419 for the plain mixture — *worse*. A family-specific DP picks confidently wrong hands when the family is wrong | — (this is why exp037 switches the DP **off** instead) |
| "**Earliest detected planted hands**" rules | .42–.49 vs .711 for the learned ranker; the better self-trained detectors are *worse* at it (they fire on unlisted planted hands too) | — |
| "**Type-1 planted then type-2**, each chronological" (hand-crafted type proxies) | .49–.51 | superseded by the learned DP (exp036), which is the same idea done properly |
| **Detection-only ranking for short lists** (lists of 3–4 contain all planted hands) | .816 vs .795 for the ranker on the 32 true short-list pairs, but short lists are not identifiable on eval; by estimated planted count the ranker wins (.734 vs .607) | a reliable estimator of "this pair has ≤5 planted hands" |
| Better **planted-hand detectors** (self-trained hs_v2/v3/v4, sequence-model margins) as ranker features | −.0127 overall (P(better) .002); seq margins DT −.013, SP −.008, CI +.004 | — better *detection* is the wrong axis for *listing* |
| **EM step** on the censored tail | −0.11 DP | — |
| **Pure rule** (blend w=1.0) | LB .91549 vs .91820 at w=0.7 | — |
| **Per-family blend weights** (8^3 grid on dev, real posteriors) | best cell (.7,.6,.7) = +0.00098, P(better) .766, per-cell SE .00139 — inside selection noise; the DT and CI optima are already exactly 0.7 | a per-family effect that survives a bootstrap at P >= 0.8 |
| **Ramp-bound tuning on dev** | impossible: the minimum dev-pair strength is 0.4847, so no dev pair is below the ramp's low bound and only 11.6% sit in its active region. The correct validation set is the LOFO simulation, never dev OOF | — |
| **Averaging the two in-pair partner rankers** (raw / rank / max / geometric mean of exp008 + exp027ev) | the rank-average is the BEST standalone ranker (+0.0047) and the WORST blend member (−0.0105). `s=(1−w)·partner + w·DP` needs a calibrated P(listed | family) on the DP's scale, not a better ordering | a partner that is re-calibrated after averaging |
| **Seed-bagging the exp027ev partner** (3 full reseeds; probmean / margmean / re-Platt-after-averaging) — this tests the row above's own reopen condition | **standalone it works** (+0.00625 over the mean seed, P .977–.997 vs each seed) but **in the blend it is −0.00081, P .21–.47**, and re-calibrating after averaging changes the blend by 0.00003. Not a scale problem: the partner's within-pair top-5 sd is unchanged by averaging. The whole partner is worth only +0.0117 of blend MAP over the pure DP at w = 0.7, so a 0.9% better partner cannot clear the ±0.0018 bootstrap sd. Null holds across the whole flat-w grid | the partner's weight (1 − w) rises above ~0.5 |
| **Player-specific normal-play baselines in the event head** (shrunk per-(player, phase, bucket) policy recalibration) | −0.0096 with 32 features (P .004), −0.0036 with 6 (P .104). The premise is real — per-player fold rates span .42–.79 against a .61–.67 population band — but every baseline enters the within-pair ranking as a pair-constant transform | a player x situation baseline fine-grained enough to reorder hands WITHIN a pair |
| **Per-pair q rate calibration** | already calibrated: summed P(listed) = 4.89/4.76/4.93 vs true list lengths 4.90/4.79/5.00 on dev, and 4.82/4.77 on the eval top-200/400. Rescaling would move it away from the truth | — |
| **Within-pair ranking objective** for the event head (lambdarank, level-matched) | −0.007 to −0.032, P(better) .000 at every level. The DP consumes the absolute level, not just the order | — |
| **Temporal / spacing priors on planted events** | the process is an exact memoryless Bernoulli process: on 162 complete lists the first four listed hands sit at scaled positions [.195, .400, .587, .799] against a Poisson null of [.2, .4, .6, .8], spacing-variance ratio .97 | — |
| **The DT / SP listing type key, mechanically** | independent 231-column battery of action-log facts: best DT rule .9109, best SP .9239, LGBM on all of them .9198/.9022 — strictly less type information than the incumbent 321-feature model (.927/.920). Falsified: fold-count unification, street index, showdown, action index, orientation bit, position. The CI key reproduced exactly (43/43, 12/12, 405/405), validating the harness | a predicate beating 0.95 accuracy on segment 1 vs 2 |
| **A hole-card range cut behind the CI "trash raise"** | none: 144 of 169 preflop classes appear among the 460 listed CI hands; the best threshold (pf_eq <= .50) covers 49.6% of listed vs 23.9% of unlisted gate hands — a 2.1x enrichment, not a predicate | — |
| **Exposure-law correction of the DP (lane M3)** | the *law* is real — censored-Poisson `n_listed = min(N,5)`, `log lam = a + b log n_shared + family`, gives **b = 0.517, 95% CI [0.28, 0.78]**, b=1 rejected (LR 11.9, p .0006) — but the DP does not need it. (a) The event model is already sub-linear in support (slope +.08/+.47/+.30) AND out of it: refitting the heads on `n_shared >= 86` (and `>= 112`) and applying them to held-out low-exposure pairs gives the *same* crossing slope (+.20/+.57/+.16 and +.20/+.51/+.25), so **there is no kink at the support boundary**. (b) Forcing `sum q := lam_hat(n)` costs −.0088 overall (P .000), −.0153 on the bottom quartile, −.0080 held-out (P .067); every de-kink arm loses on the simulated out-of-support pairs. (c) The "implied list length 2.9–3.5 below support" was a routing artefact: the posterior-routed `sum_f P_f·L_f` on the top-300 eval pairs is **4.42–4.99 in every exposure band, including 38–49 shared hands**. (d) S is an effective parameter at a sharp optimum: x0.85 −.0138, x1.15 −.0302, fold-honest optimum exactly 1.0 in 5/5 folds | a *measured* eval-side calibration (not a fitted law) showing routed implied list length < 4 on high-risk low-exposure pairs |
| **ORACLE PRICING of the lane** (not a dead end — the map) | oracle TYPE +0.018 dev MAP; oracle EVENT **+0.157** (.775 -> .932; CI .759 -> .976). recall@5 .820, **recall@10 .970**: candidate retrieval is solved and the entire residual is discriminating 5 of ~10 candidates — worth +0.039 final if solved perfectly | this IS the lane; it needs a feature block that exposes the latent activation bit |

## The undisclosed 4th mechanism

| lane | result | reopen if |
|---|---|---|
| **Mechanism hunt**: 65 label-free directed statistics (ghosting/information sharing, best-hand-plays, blind-steal agreements, mutual pot building, chip-dump variants, squeeze/whipsaw, synchronised sit-down, seat geometry, Mazrooei impact/transfer/gift) against three nulls (binomial IPF, scaled labelled negatives, the same pairs' other phase) | **documented null** — no unexplained excess. Instrument validated: it recovers DT .68–.74 / SP .30–.61 / CI .40 recall at 0/1488 negative FPR | a statistic outside the bank shows a large, phase-replicated excess |
| **Bet sizing signalling channel** | Tested bet size distributions and partner response conditional on size across 1.1M dev actions: response shifts match known soft play / flat-call traits; no separate sizing signalling channel | a sizing-specific anomaly shows significant excess over table/player baselines |
| **Three-player ring/triad coordination** | Audited all 53 open 2-hop paths (Y, Z) connected via multi-pair colluders X. 51/52 have dev OOF risk < 0.05 (mean 0.0012, 0 spies). Generator plants strictly pairwise trees, 0 triads. | — |
| Share bound | ≤ 7–10% by LB arithmetic and by 96% of the eval top-450 carrying a known-family signature. **Tightened on 2026-09-20 to ≤ 2%** by the behaviour probe: an `other_coordination` truth pair is a negative for all three disclosed families but a positive for pair AP, so high-ranked OC truth costs ~0.95 of behaviour macro-AP per unit; observed behAP 0.9635 vs pair AP 0.9745 (measured directly by probe sub031) (simulation at the real ranking: φ_OC = .14 ⇒ behAP .84–.89, φ_OC = .05 ⇒ .93–.94). SECOND, INDEPENDENT LEG for other_coordination that hides LOW in the ranking: exp039's leave-one-family-out prices an unseen mechanism's pairs at AP .070–.079, so `pairAP ≤ (1−φ) + .079φ`; pairAP .9745 ⇒ **φ_undetected ≤ .0277**. `scripts/phi_behap_bound.py` | a behaviour macro-AP well below pair AP on a future probe |
| **φ = 0.14 "unserved by 3-family routing"** — the INTERPRETATION | **REFUTED 2026-09-20.** (a) The fit is unidentified: `obs=(1−φ)K+φαL` is linear in `((1−φ)c, φαc)`, so c = .92/1.00/1.10 give φ = .065/.139/.218 at identical RMSE; with c = 1 forced, φ = .14 ± .04. A competing 2-parameter model `obs=cK−gw` fits as well with no dead subpopulation. (b) The behaviour probe caps high-ranked other_coordination at ≤ 2%, and pair AP itself caps any undetected mechanism at ≤ 2.85% via the LOFO AP of .079. (c) The mis-served cohort profiles positively as **low-exposure soft play**, and a 0/1488-calibrated statistic fires 183/174,000 dev vs 193/174,000 eval (ratio 1.05) — no eval-only population. (d) The strength gate is an **exposure** gate: 0/25 of gate-firing top-200 eval pairs carry `fam_unexplained` | a probe that separates the two competing transfer models |
| Predicting `other_coordination` as a behaviour class via strength gate | **MEASURED NULL & REJECTED**: LOFO simulation dropped macro-AP by -0.012 (DT), -0.001 (SP), -0.018 (CI), and -0.025 on dev positives. Low strength is often due to low exposure on true known-family pairs. Setting behaviour to other_coordination turns them into false negatives with score 0. Keep argmax family for all pairs. | — |
| Censoring threshold extension (`max_seq` in listing rule) | Tested max_seq vs last_by_rank (added 8,157 training negatives). Dev MAP dropped .7615 -> .7358 (-0.0257): hands after segment-2 cutoff contain truncated type-2 events; treating them as negative labels poisons the event model with false negatives. Claude's cutoff was correct. | — |

## Pair model (70% of the score)

| lane | result | reopen if |
|---|---|---|
| **Transductive pseudo-labelling** (top-350 target-window pairs as positives) | C′ .9438 vs .9581, P(better) .000 — 26% wrong pseudo-labels poison the fit | — |
| **Exposure-stratified calibration** of pair scores | .929 vs .948 — low-exposure pairs genuinely carry less evidence; their lower scores are correct | — |
| **Eval-window rows of labelled pairs as extra clean negatives** | −.0023, P(better) .013 | — |
| **Within-table z-scores** of the top-25 features | +.0001 (noise) | — |
| **Self-training round 4** | +.0001, P .525 — converged at round 3 | — |
| **Sequence model as a replacement** for engine features | worse on both tasks (hand detector AUC .9936 vs .9960 SP; ranker worse on DT/SP) | — |
| Sequence-model features **inside the hand detector** | SP C′ +.002, P(better) .755 — below bar alone; **but the model fuses well at pair level** (+.0027, P .973 — accepted as exp038_f1) | — |
| Model diversity: LightGBM cfg B, rank fusion A+B+C | fusion ties cfg C alone (P .675); seed-bagging ties the single model | — |
| **Metadata, seat adjacency, co-presence, co-join/leave** | AUC .48–.53 | — |
| Hard-negative-specific machinery | labelled negatives are indistinguishable from random unlabelled pairs (neg-vs-unknown AUC .494) | — |
| **CatBoost in the pair fusion, at every weight** (not just equal weights) | C' .9585 (w .05) -> .9577 (w .333); paired bootstrap w=.10 P(better) .410, w=.20 P .293 | — |
| **Fitted second-level weights** (cross-fitted Caruana hill-climb over 4 pair models) | worse out-of-fold than fixed equal weights (.9635 vs .9639); per-fold weights swing .33–.87. With ~354 validation positives the weight surface is flat and fitting chases noise | nested OOF at 5x5 scale |
| **Aggregation function** for the fusion (rank vs logit vs probability mean) | identical: .9586 / .9586 / .9587 | — |
| **The sequence model inside the pair fusion** | CV-positive (C' +.0016 at w .25, P .990) but LB-falsified: sub020 vs sub016 is a clean same-evidence A/B and gives −0.00063 pair AP against a +.0017…+.0024 prediction | — |
| **Listing-rule DP aggregates as PAIR features** | not shippable with OOF hygiene in the time available (dev evidence scores exist only for the 372 positives, so a dev-wide rescore of ~157k pairs is required); low prior because the DP is normalised within-pair, i.e. conditional on the pair being positive | a dev-wide OOF rescore |

## Data / external

| lane | result |
|---|---|
| **UB/AP real-world hand histories & "master spreadsheet"** | declined: generator-specific planted rules don't transfer; the UB superuser mechanism (hole-card knowledge) is exactly the "information sharing" family already null-tested; external data must be publicly and equally accessible to all participants |
| Original/source dataset | none exists — the data is fully synthetic |

## Intel

| lane | result |
|---|---|
| Forum / public kernels / web search for the generator | repeated at D-1 (2026-09-20): `kaggle competitions topics list` returns the same 5 threads, newest 2026-09-15, none naming a component score; best public kernel LB 0.824; no public generator code, paper or write-up exists. Documented null |
| What the leaders have | arithmetic bound: with our evidence 0.70485, #1's 0.93876 would need pair AP >= 0.9968 AND behaviour AP = 1.0, so they beat us on **evidence** ([.73, .86], most plausibly .76–.80). Conversely a team with perfect known-family listing would score .942–.950 > the actual best .93876, so **nobody cracked the exact planting predicate** |

## Lane M2 — the per-player baseline at action granularity (added 2026-09-20, exp062_pllr)

| lane | result | reopen if |
|---|---|---|
| **Per-player residual policy in the event head** (logistic offset over the population policy on the SAME state features, fitted on the player's own decisions; cross-phase baseline so the fit is collusion-free by phase-locality) | **MEASURED NEGATIVE, -.0072 (9 feats, P(better) .008) / -.0101 (26 feats, P .000)** on dev OOF MAP@5 vs .7748. The premise is confirmed three ways — per-player style transfers across phases at **+0.044 nats per decision** held out, the per-hand surprise separates listed hands at **AUC .902**, and **98.4% of its variance is within-pair** (the exact fix exp050's diagnosis called for) — but the event head's AUC moves by **+0.0002** (it is redundant with the 321 engine features, spearman .32-.40 with `surp_face_*`) and the DP's implied list length is **unchanged** (4.90/4.77/4.92 vs true 4.90/4.79/5.00), so it is not a recoverable calibration shift either | a per-player signal that raises the EVENT head's OOF AUC by more than ~0.001, not merely its standalone AUC |
