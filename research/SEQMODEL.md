# exp032 — learned sequence model over hands (src/seqmodel.py)

*Written 2026-09-16, D-4, inside a hard 4 h time box. Verdict in §5.*

## 1. Why

Everything the pipeline consumes today is a **hand-crafted aggregate**: `src/engine.py` reduces a hand to ~90
numbers per (hand, pair) — gift/impact EV, policy surprises, fold/call/raise counts, family gate flags — and both
downstream consumers are at their ceiling with those numbers:

| consumer | incumbent | metric |
|---|---|---|
| hand detector (`src/handdet.py`, LGBM 4-class) | hs_v4, round-4 self-trained | OOF AUC vs clean-pair hands DT .9985 / SP .9960 / CI .9970 / any .9951 |
| evidence ranker (`src/evidence.py` exp008, `evidence_v4.py` exp027ev) | two-stage LGBM + chronology | dev OOF MAP@5 .7164/.7078/.7066 (exp008), .7261/.7156/.7066 (exp016ev), .7297/.7236/.7066 (exp027ev) |

Hypothesis: a model that reads the **ordered action stream** of a hand (who acted, what, how much, in what spot,
with what equity) has a different inductive bias and may see the planted pattern that the aggregates smear out.
The likely payoff was never "replace the aggregates" but "add an orthogonal member".

## 2. Representation

One example = (hand, unordered pair {A,B}), A = lower `player_idx` (the engine's canonical orientation).

**Tokens** (one per action; mean 9.3, max 33, padded to 34):
`emb(action type 6) + emb(street 4) + emb(role A/B/other) + emb(position in hand) + Linear(28 continuous)`

* 15 seat-independent: log amount / to_call / pot / stack in bb, bet fraction `amount/(pot+to_call)`,
  players_active, the actor's perceived strength `hs_actor` + a missing flag, the label-free normal-play policy
  `p_fold/p_check/p_call/p_agg` and `surprise` (`research/POLICY.md`), `n_agg_street`, all-in, to_call>0.
* 8 A/B-paired: omniscient `eq_pre`, `eq_post`, `Δeq` for A and for B; "the actor is facing A" / "facing B"
  (last aggressor on this street is A/B and to_call > 0).
* 5 pair-derived: the actor's own `eq_pre` and `Δeq`, the A-vs-B heads-up equity on this street, the best
  `eq_pre` among the other seats, to_call>0.

**Static (46)**: 10 hand-level (n_actions, board_n, final pot in bb, players dealt / at showdown, blind level,
pre/post action counts, aggressive-action count), 6 pair-symmetric (A-vs-B heads-up equity per street, both
players active post-flop, both at showdown), and 2×15 A/B blocks (position vs the button, folded, went to
showdown, won_share, net bb, contribution, stack, preflop equity vs 1 / vs 5, own action and aggression counts,
fold street, final hand category, #opponents better, best-hand-at-showdown).

**Orientation carries no signal by construction.** The layout is built so that exchanging A and B is a fixed
column permutation (`SWAP_C`, `SWAP_S`) plus a role remap 1↔2; a unit check confirmed that `build_examples` with
the seats exchanged equals the permuted canonical tensor exactly. Training swaps a random half of the batches;
inference averages both orientations.

**Trunk**: 3-layer pre-norm transformer encoder, d_model 96, 4 heads, ff 192, dropout .1, padding-masked;
masked mean+max pooling concatenated with LayerNorm(static) → 128-unit GELU MLP → task head.
(`--arch gru` swaps in a 3-layer bi-GRU of the same width; unused in the reported runs.)

**Training**: AdamW, OneCycle to lr 2e-3 (1.5e-3 for evidence), weight decay 1e-2, grad-clip 1.0, fp16 autocast,
batch 512 (256 for evidence). Cross-fit on the frozen table folds `data/folds_tables_5.csv` — model k trains on
tables of folds ≠ k and scores fold k — so every OOF number below is comparable with the incumbents' on the same
rows and the same folds. The whole population lives GPU-resident (390 MB for the detector, 96 MB for evidence);
peak allocation 942 MB, well inside the ~1.2 GB budget.

## 3. Data cache (`python src/seqmodel.py cache`, 37 s)

`data/derived/seq_cache/` — 223,749 (hand, labelled-pair) rows over 204,325 dev hands / 1.90 M action tokens
(353 MB npz). Built from `actions`, `action_equity_dev`, `action_policy`, `seats`, `seat_strength`,
`hu_equity_dev`, `hands` and the `ph_v2` pair index. Rows carry label / family / fold / is_ev / n_rel and the
self-training flag (`is_extra`: unlisted hands of positive pairs with hs_v3 family score > .5, i.e. exactly
`handdet.py`'s `SELFTRAIN=1` population). Ragged token storage + on-the-fly padded-batch assembly keeps the
memory flat; nothing uses ID formats, file/row order or eval-file membership.

## 4. Results

All numbers are OOF on the frozen table folds, on the identical rows the incumbent uses. Sanity anchors: the
harness reads `evidence_scores_dev_exp008.parquet` and reproduces its published MAP@5 **exactly**
(.7164/.7078/.7066), and the LightGBM baseline re-fit here on the seq population reproduces the shipped hand
detector's AUCs to within .0001. Both sides of every comparison below are therefore genuinely paired.

### 4.1 Experiment 1 — hand detector (`train_hd`)

Population = `handdet.py`'s `SELFTRAIN=1` population: 182,801 rows = 178,620 hands of the 1,488
confirmed-negative pairs (class 0) + 1,817 listed evidence hands + 2,364 self-train extras, weights 1 / 25 / 8.

**OOF AUC vs clean-pair hands**

| model | DT | SP | CI | any | DT vs other fams | SP | CI |
|---|---|---|---|---|---|---|---|
| shipped LGBM (hs_v4 `oof_metrics.json`) | .9985 | .9960 | .9970 | .9951 | .9910 | .9888 | .9948 |
| LGBM re-fit here (same rows/folds) | .99854 | .99596 | .99712 | .99516 | .99082 | .98887 | .99512 |
| **sequence model** | .99773 | .99356 | .99644 | .99378 | .99211 | .98933 | .99689 |
| rank-average (75 % LGBM / 25 % seq) | **.99865** | **.99617** | **.99731** | **.99555** | .99354 | .99100 | .99678 |
| LGBM + seq class margins as 4 extra features | .99851 | .99599 | .99710 | .99522 | .99147 | .98831 | .99507 |

The sequence model is **below** the GBDT on every family head (−.0008 DT / −.0024 SP / −.0007 CI / −.0014 any).
The blends move AUC by +.0001…+.0004 — inside noise at this level. Spearman rank correlation between the two
models' per-hand scores is only .72–.76, so they really do disagree; the GBDT is simply better.

**But AUC is not what the pair model consumes.** Its features are top-k aggregates and counts above the
negative-hand q999 threshold, so the decision-relevant statistic is recall of *listed* evidence hands at that
threshold — the statistic exp024's self-training moved when it bought +.0045 C':

| model | DT | SP | CI |
|---|---|---|---|
| LGBM re-fit here | .6579 | .4098 | .6457 |
| sequence model alone | .6069 | .4193 | .6196 |
| rank-average (.25) | .6690 | .4747 | .6413 |
| **LGBM + seq margins as features** | **.6731** | **.4810** | .6478 |

Paired McNemar over the listed hands (gain / loss / z): **DT 27/16 z 1.68 · SP 69/24 z 4.67 · CI 5/4 z 0.33.**
So the seq score, used as four extra columns inside the incumbent GBDT, buys a **significant +.071 listed-evidence
recall on soft play** (+17 % relative) and a marginal +.015 on DT, at flat CI and flat AUC. Soft play is the
campaign's weakest family at both levels (pair C2 SP .8845 vs DT .9508 / CI .9765; listed-hand recall .41 vs
.66/.65). (Published `recall_ev_*` numbers are higher because they average over the self-labelled extras, which
are much easier — .83–.98; the table above is listed-only and therefore non-circular.)

Runtime: seq 2,250 s for 5 folds × 8 epochs, LGBM 79 s, LGBM+seq 81 s.

### 4.2 Experiment 2 — evidence ranker (`train_ev`, `confirm`)

Population = the 45,129 shared dev hands of the 372 positive pairs (1,817 listed), aligned 1:1 with
`evidence_cache/dev_pos.parquet`; 3 family heads on a shared trunk, each trained only on its own family's pairs,
20 epochs, pos_weight 5. The model sees **no chronology** (no `hand_seq`, no within-pair position), so its
natural comparison is the engine's no-chronology stage-1.

| dev OOF MAP@5 (true-family routing) | DT | SP | CI |
|---|---|---|---|
| sequence model alone (no chronology) | .5304 | .5406 | **.4886** |
| engine stage-1, same protocol/rows/folds | .5913 | .5748 | .4661 |
| engine stage-1, as published (exp007 / exp027ev) | .598–.600 | .576–.584 | .439–.486 |

Below the engine on DT (−.061) and SP (−.034), **above it on CI** (+.023). Spearman between the seq score and the
engine stage-1 on CI rows is .767.

**Nested two-stage (exp008 protocol: stage-2 trained on inner 3-fold stage-1 scores, served the sharper 4-fold
ones), paired with and without the 3 seq head margins appended to the feature matrix.** Row mean over stage-2
rounds 100/200/350:

| variant | DT | SP | CI |
|---|---|---|---|
| base (engine features only) | .7187 | .7065 | .6837 |
| base + seq | .7122 | .7011 | **.7187** |
| Δ | −.0065 | −.0054 | **+.0350** |
| published incumbents (exp008 / exp016ev / exp027ev) | .7164 / .7261 / .7297 | .7078 / .7156 / .7236 | .7066 / .7066 / .7066 |

CI-only confirmation with both stage-2 objectives the shipped heads use (`confirm` stage — all six cells):

| stage-2 objective | base 100 / 200 / 350 | base+seq 100 / 200 / 350 | Δ row-mean |
|---|---|---|---|
| binary | .6824 / .6845 / .6841 | .7188 / .7201 / .7172 | **+.0350** |
| lambdarank | .6810 / .6940 / .6940 | .7210 / .7203 / .7228 | **+.0317** |

Every cell improves, so this is not a grid maximum. A pure post-hoc rank average of the **shipped** exp008 dev
score file with the seq score (w = .15) reproduces the same shape without rebuilding anything:
DT .7177 (+.0013) · SP .6973 (−.0105) · CI **.7179 (+.0113)**.

Runtime: 1,783 s for 5 folds × 20 epochs; each nested two-stage grid 82 s.

### 4.3 Seed stability, seed bag, and a pretrained trunk (the data-poverty test)

Seq-alone (chronology-free) listing MAP@5 is stable across runs, so its level is a property of the model, not of
the seed:

| seq score | DT | SP | CI |
|---|---|---|---|
| from scratch, seed 0 (20 ep) | .5304 | .5406 | .4886 |
| from scratch, seed 1 (20 ep) | .5194 | .5445 | .4860 |
| **trunk pretrained on the 183 k-row detector task, fine-tuned 12 ep at lr 7e-4** | .5302 | .5453 | .4824 |
| engine stage-1 (reference) | .5913 | .5748 | .4661 |

**Pretraining does not lift the seq-alone listing score at all** — so the data-poverty hypothesis for lane (2) is
falsified: a planted-ness trunk is the wrong prior for *which* planted hands the host listed, the same mechanism
exp027ev found for the GBDT hand scores ("better detection is the wrong axis for in-pair listing").

CI nested two-stage, paired with vs without each seq score (row mean over rounds 100/200/350; paired bootstrap over
the 92 CI pairs, 4,000 reps, P(base+seq > base) per rounds cell):

| seq score added | stage-2 obj | base | base+seq | Δ | P(better) per cell | Spearman(seq, stage-1) |
|---|---|---|---|---|---|---|
| seed 0 | binary | .6837 | .7187 | +.0350 | .996 / .993 / .986 | .767 |
| seed 0 | lambdarank | .6897 | .7214 | +.0317 | .999 / .978 / .979 | .767 |
| seed 1 | binary | .6837 | .7296 | +.0459 | .998 / .996 / .998 | .790 |
| seed 1 | lambdarank | .6897 | .7346 | +.0449 | 1.00 / .996 / .993 | .790 |
| bag (seed 0+1) | binary | .6837 | .7262 | +.0425 | .997 / .997 / .991 | .810 |
| bag (seed 0+1) | lambdarank | .6897 | .7256 | +.0359 | .996 / .994 / .987 | .810 |
| pretrained + fine-tuned | binary | .6837 | **.7343** | **+.0506** | 1.00 / 1.00 / 1.00 | .784 |

**21 of 21 cells improve, every P(better) ≥ .978.** Against the frozen published CI number (.7066) the gain is
+.012 (seed 0) … +.028 (fine-tuned). The pretrained trunk, useless on its own, gives the *most complementary*
CI score.

All three families with the seed bag (binary, row mean): DT .7187 → .7143 (−.0044) · SP .7065 → .7085 (+.0020) ·
**CI .6837 → .7262 (+.0425)** — so the seq column belongs in the CI head only.

## 5. Verdict

**As a replacement: a clean null. As an added member: one seed-stable, statistically solid lead on CI evidence (the
one number that had not moved all campaign), plus one unverified lead on soft-play hand recall.**

1. **Replacement — no.** The sequence model loses to the GBDT on the engine features at both tasks (hand AUC
   −.0014 any; listing MAP −.061 DT / −.034 SP), and pretraining the trunk on 100× more rows does not change that.
   The hand-crafted aggregates in `engine.py` beat a 3-layer transformer trained on 1,817 listed hands. That is now
   measured, not assumed.
2. **Lead A — CI evidence (dev-validated, not shipped).** Appending the seq head margins to the CI stage-1/stage-2
   feature matrix gives +.032…+.051 paired MAP@5 (21/21 cells, P ≥ .978, two seeds, a bag and a pretrained
   variant) and +.012…+.028 over the shipped .7066. If only the CI head changes:
   92/372 × ~+.02 ≈ **+.005 overall evidence MAP ⇒ ≈ +.001 final.** The public LB cannot resolve that
   (evMAP SE ≈ .03 ⇒ ≈ .006 on the total), so this is a private-LB expectation play, not an LB-testable one.
3. **Lead B — SP hand-level recall (unverified downstream).** The seq margins as four extra GBDT columns raise
   listed-evidence recall at the q999 operating point on soft play by +.071 (McNemar z 4.67) at flat AUC.
   It only pays through a pair-model refit (exp024's analogous move was worth +.0045 C'); not attempted.
4. **Caveats that still stand on Lead A.** (a) My `base` CI reproduction (.684–.690) sits .017–.023 *below* the
   published .7066, so the paired Δ overstates the gain over the shipped file; the vs-published range above is the
   honest one. (b) The seq scores fed to the GBDT are cross-fit but **not nested** (a training row's seq score comes
   from a model that saw the outer validation fold) — the same protocol the shipped hs features use; a nested
   check needs 20 seq fits per seed (≈ 2–4 h on this GPU) and was not run. (c) exp027ev documented ±.014 of
   protocol sensitivity in exactly this comparison; the fine-tuned/seed-1 gains over the published number
   (+.023/+.028) clear it, the seed-0 gain (+.012) does not.
5. **Why nothing was shipped from this box, and what shipping Lead A would take** (≈ 1.5–2 h of focused work):
   (i) persist fold models for the CI head's seq score (the pretrained trunks are saved in `seq_cache/trunk8/`;
   the fine-tuned heads are not); (ii) a seq cache over the eval phase — ~800 k hands, 6.9 M evidence-candidate
   rows, built in chunks (the dev cache took 37 s for 204 k hands; `build_examples` runs ≈ 37 k rows/s);
   (iii) chunked inference, 5 folds × 2 orientations (≈ 30–60 min on the 1660S at batch 4096);
   (iv) add the three seq columns to the CI head in `evidence_v4.py` (A_base | lambdarank | 350), re-fit on dev,
   re-Platt, and rerun `eval`; (v) submit as a **single-component A/B** (CI evidence head only) against whichever
   of sub010/sub011 is the incumbent. Expected +.001 final; do it only if a sub slot and ~2 h are free before the
   D-1 freeze.
6. **What would overturn this verdict.**
   * Lead A dies if a nested-seq-score rerun loses most of the +.03 paired gain, or if the eval-side CI swap
     scores below its sibling on the LB by more than noise (−.006).
   * Lead B becomes actionable if a pair-model refit shows ≥ +.002 C' at P(better) ≥ .8.
   * The replacement verdict would reopen only with a far larger label set (e.g. hidden-positive pseudo-labels at
     hand level — already partly exploited by the GBDT self-training) or a fundamentally different target.

## 6. Files

* `src/seqmodel.py` — stages `cache | train_hd | train_ev | confirm` (`--save` persists fold trunks from
  `train_hd`; `--init` fine-tunes `train_ev` from them). There is deliberately **no `score`/eval stage**: it would
  only exist to write eval score files; §5.5 lists what it would take.
* `data/derived/seq_cache/` — `rows.parquet`, `seq.npz` (353 MB), `trunk8/fold{0..4}.pt` (pretrained trunks),
  `oof_{hd,ev,ev1,evbag,evft}_*.npy`, `res_hd.json`, `res_ev.json`, `res_ev1.json`, `res_evft.json`,
  `res_confirm_{ev,ev1,evbag,evft}.json`, `res_bag_allfams.json`.
* `logs/seq_{hd,ev,ev1,evft,pretrain,confirm}.log` — the raw runs behind every table above.
* Runtime of the whole study on the GTX 1660S: cache 37 s · detector 2,250 s (+ LGBM 160 s) · listing 1,783 /
  1,584 / 518 s · pretrain 1,991 s · each two-stage confirm ≈ 100 s. Peak VRAM 942 MB.

## 7. exp033 — the SP recall lead, followed end to end into the pair model

**Question.** §4.1 found that the seq margins, as 4 extra columns in the hand detector, raise listed soft-play
recall at the q999 operating point (z 4.67). Does that move **pair ranking** (70 % of the score), where soft play
is the weakest family (C′ SP .879 vs DT .955 / CI .973 for exp028_selfv3)?

**Pipeline** (`src/seqmodel.py hd_lgb → hs_score → hs_agg`, then the incumbent pair-model command with
`PFH_TAG=seq`; nothing incumbent was written to):

1. `hd_lgb` — round-3 detector exactly as hs_v3 was made (`handdet.py` SELFTRAIN population, extras = unlisted
   positive-pair hands with **hs_v2** family score > .5 → 2,097 extras, matching the exp028_train log; weights
   1/25/8; same LightGBM params, 400 rounds) **plus the 4 seq margins** from the saved `trunk8` fold models
   (dev rows scored by their own table fold). Fold models + thresholds + diagnostics →
   `data/derived/handdet_models_seq/`.
2. `hs_score` — every `ph_v2` pair-hand row that passes the **soft-play gate G0** (neither partner's first
   preflop action is a non-facing fold) is re-scored by that detector; every other row keeps its hs_v3 score
   verbatim. G0 passes 22.5 % of rows (dev 4.0 M / eval 2.7 M), 21.9 % of negative-pair hands, **100 % of
   listed DT and SP hands** and 93.5 % of listed CI hands. Dev rows use their table's fold model; eval rows use
   the 5-model mean (handdet convention). Output `data/derived/hs_seq/phase{0,1}/`, hs_v3 schema, 18 M + 12 M
   rows. Check: re-scored dev rows reproduce the training OOF to 4e-8 and ungated rows equal hs_v3 exactly.
   Peak GPU footprint ≈ 0.6 GB (4,096-row inference chunks); ~60 s per dev part, ~50–250 s per eval part
   depending on CPU contention.
3. `hs_agg` — `handdet.agg()` unchanged, with its threshold directory pointed at `handdet_models_seq/`
   (thresholds recomputed on the **mixed** negative-hand OOF scores) → `pfhseq_<window>.parquet`.
4. Pair model: `CFG=C PF_TAG=v2 PFH_TAG=seq USE_OC=1 USE_HANDDET=1 USE_MIX=1 ADD_SPIES_POS=1 W_SPY=0.5
   python src/exp005_pairmodel.py exp033_seq 0.1 600 all3`.

**Controls.** Re-running the same command with `PFH_TAG=v3` reproduces exp028_selfv3 **bit for bit** (C′ .9581,
identical eval risk), and with `SEED=101` reproduces exp029_C101 bit for bit. So every comparison below differs
**only** in the hand-score block, seed for seed. (`scripts/fuse.py exp028_selfv3 exp029_C101 exp029_C202`
reproduces exp029_bag exactly, so the bag comparison uses the same recipe.)

**Hand-level pre-check on the round-3 population** (listed-only recall at the negative q999 / q99 thresholds):

| detector | DT q999 | SP q999 | CI q999 | SP q99 | SP AUC |
|---|---|---|---|---|---|
| hs_v3 (shipped round 3) | .6634 | .4209 | .6435 | .8244 | .99596 |
| LGBM re-fit, no seq | .6648 | .4272 | .6478 | .8259 | .99600 |
| **mixed (G0 rows: LGBM + seq)** | .6593 | **.4873** | .6435 | **.7975** | .99578 |

So the seq columns sharpen the extreme SP tail (+.066 at q999) but *lose* SP recall at q99 (−.027) and a hair of
AUC. Univariate pair AP of the SP aggregates on labelled w2000a pairs rises ≈ +.03 (e.g. `h_s_sp_top5`
.872 → .903, `h_s_sp_mean` .888 → .912).

**Pair model, view C′ (original labels, paired table bootstrap, 400 reps)**

| comparison | C′ base → seq | Δ | P(better) | 90 % CI | C2 | SP C′ | DT C′ | CI C′ |
|---|---|---|---|---|---|---|---|---|
| **exp033_seq vs exp028_selfv3** (seed 42, the specified test) | .9581 → **.9604** | +.0024 | **.755** | [−.0032, +.0074] | .9657 → .9676 | .8789 → **.8857** | .9551 → .9550 | .9727 → .9726 |
| exp033_seq_C101 vs exp029_C101 | .9565 → .9596 | +.0033 | .853 | [−.0017, +.0083] | .9639 → .9663 | .8783 → .8845 | .9532 → .9552 | .9717 → .9746 |
| exp033_seq_C202 vs exp029_C202 | .9574 → .9580 | +.0007 | .583 | [−.0047, +.0056] | .9647 → .9649 | .8814 → .8813 | .9544 → .9532 | .9728 → .9729 |
| **exp033_seqbag vs exp029_bag** (3-seed rank fusion both sides) | .9578 → .9599 | +.0022 | .738 | [−.0032, +.0070] | .9652 → .9667 | .8802 → .8851 | .9548 → .9549 | .9724 → .9743 |
| exp033_seqbag vs exp028_selfv3 | .9581 → .9599 | +.0019 | .715 | [−.0032, +.0066] | | | | |

**Verdict: FAILS the bar (P(better) ≥ .8 AND SP C′ up).** The specified comparison has SP up (+.0068) but
P = .755. The effect is positive in sign for all three seeds and for the bag (+.0007 … +.0033 C′; SP up in 2 of 3
seeds and in the bag), but only one seed clears .8 and the bag sits at .738. This is a consistent,
**sub-threshold** gain of about +.002 C′ — the same size as exp026's marginal accept, one notch less certain.
Not recommended for staging on this evidence.

**Eval-side risk that dev C′ cannot see — and its fix.** Averaging 5 fold models *twice* (seq features, then the
LightGBM) shrinks the eval score tail relative to the single-model dev OOF the pair model was trained on. Rate of
rows above the q999 thresholds, eval ÷ dev (dt / sp / ci / any):

| hand scores | eval/dev exceedance ratio |
|---|---|
| hs_v3 (incumbent, 5-model mean) | .952 / **.834** / 1.127 / .955 |
| hs_seq (5-model mean) | .865 / **.651** / 1.071 / .845 |
| hs_seqsf (each eval table scored by its own frozen-fold model, like dev) | 1.107 / **.990** / 1.190 / 1.133 |

`SEQ_EVAL_SAMEFOLD=1` (→ `hs_seqsf/`, `pfhseqsf_*`, `oof/exp033_seqsf_*`) removes the shrinkage; its dev views
are the same model (C′ .9605; the ±.0001 against exp033_seq is `handdet.agg` nondeterminism — its
double-centring sums differ at 1e-8 between runs, enough to move LightGBM bins). Eval top-500 family mix
(DT/SP/CI): exp028_selfv3 190/155/155 · exp033_seq 189/146/165 · exp033_seqsf 194/145/161. **Side-finding for
the incumbent:** hs_v3's own SP eval tail is already shrunk to .83 of dev by the same 5-model averaging, so SP
pairs may be slightly under-ranked in eval relative to DT/CI today. A same-fold rescoring of hs_v3's eval phase
(8 parts × LightGBM only, ~10 min) + a pair refit would test that; it is not LB-resolvable either.

**What would overturn the null.** More seeds pushing the bag to P ≥ .8 (the three seeds say the mean is ≈ +.002;
the table bootstrap width ≈ ±.005 will not shrink with seeds, so this is unlikely), or a variant that keeps the
q999 sharpening without the q99 loss (e.g. apply the seq detector only to SP-gated rows whose hs_v3 s_sp is
already above the q99 threshold).

**Artifacts.** `data/derived/handdet_models_seq/` (fold models, thresholds, `oof_metrics.json`,
`oof_population.npz`), `data/derived/hs_seq/`, `data/derived/hs_seqsf/`, `data/derived/pfhseq_*`,
`data/derived/pfhseqsf_*`, `oof/exp033_seq*`, `logs/exp033_*`. Runtime: detector 8 min; scoring 12 min dev +
45 min eval (5-mean, shared CPU) / 7 min eval (same-fold); aggregation 40 s; pair model 4.5–15 min per seed
depending on CPU contention (an unrelated 8-worker job held the CPU from 07:20 to ~08:35).
