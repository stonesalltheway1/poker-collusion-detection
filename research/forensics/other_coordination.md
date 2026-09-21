# exp013 — hunt for the hidden 4th mechanism (`other_coordination`)

Written 2026-09-15/16 (D-5/D-4). Code: `scripts/oc_hunt.py` (stages `events | stats | excess | phase | mine | profile | print | export`).
Artifacts: `data/derived/oc/*` (event bank, per-pair stats), `research/forensics/oc_excess_all.csv`,
`oc_excess_by_event.csv`, `oc_phase_mcnemar.csv`, `oc_family_feats.json`,
`data/derived/oc_pair_scores_{dev_full,eval}.parquet`, `data/derived/oc_hand_candidates_eval.parquet`.

## 0. Verdict: NULL (with one bounded caveat)

**No hidden 4th coordination mechanism was found.** 62 label-free directed interaction statistics, each turned into an
own-baseline (row × column, leave-pair-out) Poisson LLR with mutual-top flags, reproduce every disclosed family at
high precision (0 / 1488 labelled negatives fire while 25–95 % of the family's positives do), and show **no tail
population** in either phase that (a) exceeds the negative-calibrated and binomial-null expectations, (b) replicates
in the other phase for a disjoint pair set, or (c) survives reading the hands.

Three independent lines say the residual `other_coordination` share is small:

1. **Eval top ranks are already explained.** Of exp005's top 450 eval pairs (≈ the estimated number of eval
   positives), **96 % carry a high-precision known-family signature** (25 unexplained), and those 25 have a
   soft-play profile (limp/joint-entry statistics), i.e. they are SP positives that the crude SP flag misses,
   not a new mechanism. The unexplained fraction only rises past rank ~450, where false positives take over.
2. **LB arithmetic.** sub002 (LB 0.78984) implies pair AP ≈ 0.93. If a share φ of eval positives were invisible to
   exp005, AP ≤ (1−φ) ⟹ **φ ≲ 0.07** (≲ 0.10 with public-LB noise). Whatever OC is, it is either rare or already
   ranked high by the current model.
3. **Symmetric phase test.** On 37,098 pairs selected symmetrically (dev-2000 shared ≥ 50 **and** eval shared ≥ 50,
   dev exp005 OOF below the positives' p05, not in the exp005 eval top-800), a McNemar contrast of dev-only vs
   eval-only flags is flat for every discrete event (|z| ≤ 1.5). If OC were planted only in the eval phase (host
   notebook: "reserved for evaluation"), it would appear here as an eval-only excess. It does not.

Caveat (one residual, weak): `third_folds_pair` (a third player folds to X's aggression while Y is still active)
shows a ~2× tail excess in dev candidates (285 flagged vs 140–150 expected from both nulls, NEG 3/1488). Reading the
top pairs' hands shows ordinary poker (button steals with the partner in the blinds, one cooler); it is best
explained by position-pairing structure the row×column model does not absorb, plus a residue of weak DT/CI-like
hidden positives (the flag has DT recall 0.43, CI 0.40). It does not replicate as an eval-side excess (+8).

## 1. Instrument

**Event bank** (`scripts/oc_hunt.py`, numba, one pass over all 2 M hands, ~90 s for both phases).
For every ordered within-table pair (X acts, Y seated) and every window (dev full 3000, dev first 2000, eval 2000)
it accumulates a **count** and an **opportunity count** for 62 directed events plus 3 continuous sums
(Mazrooei impact X→Y, transferred chips X→Y, EV gift X→Y), into `[window, 400 tables, 30, 30, event]` arrays.
Events cover, by hypothesis class:

| hypothesis | events |
|---|---|
| information sharing / ghosting (action depends on the partner's hidden cards) | `inf_fold_Ystrong/Yweak`, `inf_raise_Ystrong/Yweak` (X's first preflop decision *before Y acts*, split by Y's class strength), `fold_Yahead/Ybehind`, `pump_Yahead`, `aggr_Ybehind` (postflop, split by the partner's HU equity) |
| best-hand-plays (weaker partner steps aside) | `bhp_fold_Ybetter/Yworse`, `foldpost_Ysd`, `both_sd`, `mw_fold_3rd`, `mw_fold_3rd_eq60`, `foldeq50_Yactive` |
| blind-steal agreements / walks | `bvb_walk`, `bvb_raise`, `blind_fold_steal`, `steal_open_Yblind` |
| chip dumping variants | `flow`, `flow_big`, `gift_to_Y`, `imp5`, `call_Yallin`, `allin_Yactive`, `betfold_Y`, `pot3_Ywins`, `sum_{imp,transfer,gift}` |
| joint pot building / mutual limping | `pf_enter_afterY`, `pf_limpbehind`, `pf_iso_overY`, `pf_overcall_3rd_Yin`, `limp_first`, `pre_both_vol`, `mw_call_3rd`, `mw_bigbet` |
| squeeze / whipsaw / isolation | `pf_squeeze_3rd_Yin`, `pf_3bet_Yraise`, `pf_fold_3rd_Yin/Ybehind`, `mw_raise_3rd`, `third_folds_pair` |
| soft play / conflict avoidance | `hu_fold/raise/bet/smallbet`, `hu_strongcheck`, `mw_strongcheck`, `hu_checkdown`, `mw_checkthrough`, `pf_call_Yraise`, `pf_fold_Yraise` |
| synchronised presence / seating | `cojoin`, `coleave`, `seat_next` |

**Pair statistic.** Per event and ordered pair: expected count from a shrunk double-centred (row × column,
leave-pair-out) model `E_xy = o_xy · p_x^(−y) · q_y^(−x) / p0` with a 20-opportunity prior toward the table base
rate; signed Poisson LLR `G`; shrunk log-lift; the rank of Y among X's 29 co-players (`rt`) and of X among the
players acting toward Y (`ct`). Flags: `dirtop_G*` (mutual directed top + G ≥ t), `symtop_G*` (both directions
top), `gmax_*`, `deficit_*`. Nulls: (i) a **binomial IPF null** (counts resampled at the fitted rate, same
exposure), (ii) the **1,488 labelled negatives** scaled to the candidate pool, (iii) the **same pairs' other
phase** (exposure-matched windows).

**Validation of the instrument** (dev, flag recall at 0/1488 negatives): DT — `gift_to_Y` 0.74, `flow_big` 0.70,
`imp5` 0.74, `pf_call_Yraise` 0.68; SP — `limp_first` 0.61, `pre_both_vol` 0.61, `pf_overcall_3rd_Yin` 0.40,
`hu_checkdown` 0.30; CI — `third_folds_pair` 0.40, `pf_3bet_Yraise` 0.18 (+ CI is caught at recall 1.00 by the
existing trash-raise pair features). A planted mechanism of comparable strength inside the bank would therefore
surface 20–60 % of its pairs with a negative-clean flag.

## 2. Candidate pools and tests

- **Dev candidates (CANDX, 73,651 pairs):** unlabelled, dev shared ≥ 57, neither player in any labelled pair,
  exp005 OOF below the positives' 5th percentile, and not in the exp005 eval top-800.
- **Eval candidates (EVCX, 82,521 pairs):** in the eval file, eval shared ≥ 57, exp005 eval rank > 800, dev OOF low.
- **Reference groups:** 372 labelled positives by family, 1,488 labelled negatives, 199 "hidden" dev pairs
  (unlabelled with high exp005 OOF), and both nulls above.

Per-event excess (max over flags of observed − max(binomial null, negative-scaled expectation)) is in
`research/forensics/oc_excess_by_event.csv`. Summary of what was tested and found (discrete flags only):

| mechanism / event | dev excess (of 73.6 k) | eval excess (of 82.5 k) | note |
|---|---|---|---|
| `third_folds_pair` | +137 | +8 | only residual; read hands = normal poker (see §0 caveat) |
| `mw_checkthrough` (multiway check-through with a third player) | +27 | +9 | 27 pairs, NEG 0/1488 (bound ≤ 148); SP-flavoured |
| `both_sd`, `hu_checkdown`, `pre_both_vol`, `pf_3bet_Yraise`, `pf_overcall_3rd_Yin` | +11 … +24 | +0 … +22 | all inside the negative-based Poisson bound; SP/DT-flavoured |
| `cojoin` / `coleave` (synchronised sit-down / leave) | +5 / +11 | +2 / 0 | no synchronised-presence mechanism |
| information sharing (`inf_fold_Ystrong` vs `Yweak`, `inf_raise_*`, `bhp_fold_*`, `fold_Yahead` vs `Ybehind`, `pump_Yahead`) | 0–1 | 0–11 | **hard null**: 0 pairs anywhere with a strong preflop card-conditioned flag; the *asymmetry* test (pairs with "ahead" flag and no "behind" flag vs the reverse) matches the null exactly in both phases |
| blind-steal agreements (`bvb_walk`, `blind_fold_steal`, `steal_open_Yblind`) | 0 | 0–1 | null |
| all remaining 40 events | 0 | 0–2 | null |

Additional tests:

- **Multi-event anomaly** (`oc_score` = Σ over events of the mutual-top G, clipped at 30): dev medians NEG 4.5,
  DT 209, SP 54, CI 42; CANDX p99 77.7 vs NEG p99 84.7, CANDX max 186 vs NEG max 154 — no excess tail.
- **Metadata / multi-accounting:** `players.parquet` similarity (region, client, stake, experience, |age gap| < 30 d)
  is distributed identically for positives and negatives; the 34 eval pairs matching on all five fields have a
  *lower* anomaly score and none is in the exp005 top 600. Null.
- **Phase-rate control:** eval-phase per-opportunity rates are uniformly ~5 % higher than dev (`sum_transfer` 1.062,
  `pre_both_vol` 1.057). This global shift, not a pair mechanism, explains the only large McNemar z-values
  (`sum_*` deficit flags, z ≈ +6…+8): they are mass effects on tens of thousands of pairs.
- **Hands read:** the 2 top residual dev pairs and the 2 top unexplained high-risk eval pairs, plus their event
  profiles (`scripts/oc_hunt.py profile|print`). Eval pair 4101:7572 (rank 79) = mutual limping + joint entry +
  multiway check-throughs (soft-play-like); eval pair 663:8311 (rank 114) = one-directional EV gift over 39 hands
  (DT-like, small sample); dev pairs 7783:8659 and 1053:4399 = ordinary button-steal + cooler poker.

## 3. What would overturn this null

- A public post or a top solution naming a 4th mechanism with a concrete hand signature.
- An LB probe: a submission that re-ranks the ~500 eval pairs with the highest *family-agnostic* `oc_score` but no
  family signature. If pair AP rises materially, an uncovered population exists (cheap: one submission).
- A rise of the exp005 implied LB pair AP well above 0.93 with the same evidence/behaviour columns would shrink the
  bound further; a *fall* of implied AP on a better single model would reopen the question.
- The instrument is blind to a mechanism that needs **more than 2 players** (a ring of 3), or that is expressed only
  in bet *sizing* sequences, or in time gaps between actions (not modelled: `started_at` deltas within a hand are
  not in the data). A 3-player ring would still show up pairwise, so this is a weak gap.

## 4. Deliverables and how to use them

| file | content |
|---|---|
| `data/derived/oc_pair_scores_{dev_full,dev_w2000a,dev_w2000b,eval}.parquet` | 174,000 within-table pairs per window (the four pair-model windows): `oc_score` (mutual-top ΣG), `oc_gsum`, `oc_k10`, `oc_gmax`, `G_*`/`mtop_*` for 26 selected events, plus the high-precision family flags `f_DT/f_SP/f_CI/fam_unexplained` (+ `pair_id` in eval). Identical columns/dtypes across windows; base rates, shrinkage, own-baseline ranks and family thresholds are recomputed **within** each window, so a model trained on the 2000-hand dev windows transfers to eval (median shared hands 62/61/61 vs eval 61; `oc_score` p50/p99 3.5/86 in every 2000-hand window). Labelled-only AUC of `oc_score`: 0.951 dev_full, 0.907 w2000a, 0.920 w2000b (the drop is pure exposure). Rebuild: `python scripts/oc_hunt.py events` then `OC_WINDOWS=… OC_NO_NULL=1 … stats` and `OC_EXPORT_WINDOWS=0,1,2,3 … export` |
| `data/derived/oc_hand_candidates_eval.parquet` | top-5 generic-anomaly hands (\|imp\|+2·gift+3·both-voluntary) for the 496 eval pairs in the exp005 top-1200 with no family signature — a fallback evidence list for pairs the family routers cannot classify |
| `research/forensics/oc_family_feats.json` | the 12 best pf statistics per family (threshold = labelled-negative maximum in the same window) used for the family flags |

**Recommendations**

1. **Do not spend more of the sprint on OC.** The share is bounded at ≲ 7–10 % of eval positives and the most likely
   reality is that OC pairs, if present, look like the disclosed families to a generic detector.
2. **Feed `oc_pair_scores_*` into exp014 as the generic branch.** `oc_score` alone gets labelled-only AP 0.867 /
   AUC 0.951 (DT 0.977, SP 0.951, CI 0.910) and pool AP 0.306, and its rank correlation with exp005 on unknown dev
   pairs is only 0.21 — genuinely different information, label-free, and computed identically in both phases.
   Use the event-level `G_*`/`mtop_*` columns as features and validate with LOFO as planned.
3. **Behaviour:** keep argmax over the three families for every pair. The measured "unexplained" rate inside the eval
   top 450 (≈ 5 %) is smaller than the dev positives' own unexplained rate (9.7 %), so an `other_coordination`
   override would remove more true family pairs than it fixes. If an override is wanted at all, gate it on
   `fam_unexplained & oc_score high`, which selects ~25 pairs in the top 450.
4. **Evidence:** for pairs with `fam_unexplained`, route the SP ranker first (their profile is soft-play-like:
   `limp_first`, `pre_both_vol`, `pf_overcall_3rd_Yin`), and use `oc_hand_candidates_eval` only as a padding source.
5. **Rules note:** every statistic here is computed from gameplay only (actions, cards, positions, `hand_seq`,
   lineup changes). No ID formats, no file/row order, no eval-file construction features.
