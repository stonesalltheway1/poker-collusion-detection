# TOOLS.md: tooling, repos, hand evaluator, derived strength tables

Written 2026-09-15. Machine: i5-10400F (12 threads), 32 GB RAM, GTX 1660S. Python 3.13 at `C:\Python313`.

## 1. Python packages

### Installed this session (each `pip install --dry-run` checked first: no protected package changed)
| package | version | smoke test |
|---|---|---|
| pulearn | 0.2.0 | `BaggingPuClassifier(RF)` on a synthetic PU set (70% of positives hidden) gave AUC 0.956 against the true labels; `ElkanotoPuClassifier` gave 0.926 |
| ruptures | 1.1.10 | `Pelt(model='rbf')` found breakpoints [120,250,370] where the truth was [123,249,369] |
| leidenalg | 0.12.0 | Leiden modularity on Zachary's karate club gave 4 communities |
| statsmodels (+patsy 1.0.3) | 0.15.0 | imports OK |

### Already present (checked)
networkx 3.6.1, optuna 4.8.0, igraph 1.0.0 (python-igraph), hdbscan 0.8.42, shap 0.51.0, phevaluator 0.6.0,
treys 0.1.8, pokerkit 0.6.3, tqdm 4.67.1.

### Skipped
* **pyod 3.6.5**: its dry-run would upgrade joblib 1.4.2 to 1.6.0, which sklearn and lightgbm use while other agents are running. Not installed.
* **eval7**: does not build on py3.13/Windows. Not needed, because `src/handeval.py` does the same job.

### Protected stack (unchanged, re-checked after installs)
numpy 2.2.6, pandas 2.3.3, polars 1.39.3, duckdb 1.5.0, pyarrow 23.0.0, scipy 1.17.1, scikit-learn 1.6.1,
lightgbm 4.6.0, xgboost 3.0.0, catboost 1.2.10, torch 2.6.0+cu124, numba 0.61.2 (llvmlite 0.44.0).

## 2. Repos: `research/repos/` (git clone --depth 1)

| dir | source @ commit | size |
|---|---|---|
| pokerkit | uoftcprg/pokerkit @54571dd (2026-08-22) | 8 MB |
| PokerHandEvaluator | HenryRLee/PokerHandEvaluator @10be452 | 718 MB (397 MB is `test_data/`, can be deleted) |
| treys | ihendley/treys @70fbaad | 0.2 MB |
| poker-suspicious-detection | phucthaiv02/poker-suspicious-detection, branch `feat/evidence-first-baseline` @0f48bfc (main has only a README) | 0.1 MB |
| pulearn | pulearn/pulearn @51e62fb (2026-06-13) | 4 MB |
| nnPUlearning | kiryor/nnPUlearning @3d9769e | 0.3 MB |
| PUExtraTrees | jonathanwilton/PUExtraTrees @5cb15d7 | 0.1 MB |
| DEDPUL | dimonenka/DEDPUL @04c0281 | 58 MB |
| OMPEval | zekyll/OMPEval @4aec210 | 0.5 MB |
| multiagent-poker-collusion | m1chae11u/multiagent-poker-collusion @3b98a5b | 3 MB |

**pokerkit.** A pure-Python poker engine (University of Toronto CPRG) with complete No-Limit Hold'em state machines: blinds, legal actions, minimum raises, side pots, showdown, multi-runout. It also has PHH hand-history I/O and a slow built-in evaluator. Use it to replay a suspicious hand action by action, for example to check what a minimum raise or call amount should have been, or to validate our action-log decoding in edge cases such as all-in-for-less or side pots. It is far too slow for 12M-row work.

**PokerHandEvaluator (phevaluator).** A perfect-hash C evaluator with a Python binding that covers 5-7 card hands (plus Omaha). It returns ranks 1..7462 where lower is better. Its card id is identical to our code (`rank*4+suit`, `cdhs`), so `phevaluator.evaluate_cards(*codes)` works on our int8 columns directly. This is the ground truth `handeval.py` is verified against. From Python it runs at only about 0.65-0.75 M hands/s, because of per-call overhead.

**treys.** A pure-Python Cactus-Kev evaluator (a deuces fork). It uses its own bit-packed card integers, which are **not** our encoding, and returns ranks 1..7462 with lower better. It also has PLO support. It is slow, so keep it for reference or cross-checks only.

**poker-suspicious-detection.** A public "evidence-first" baseline for **this competition**, about 900 lines of polars. It frames the task as positive-unlabelled pair ranking plus multiple-instance evidence retrieval:
* The evidence-hand model uses planted hands as positives and hands from confirmed non-target pairs as the only negatives. Unknown pairs are never used as negatives.
* OOF evidence scores from GroupKFold by table are stacked into a pair model, together with top-k evidence statistics and exposure features.
* Three one-vs-rest behaviour heads plus an open-set threshold route high-risk pairs that fit no known family to `other_coordination`.
* The top-5 shared hands by evidence probability are submitted as evidence.
* Its features are generic numeric aggregates of seats and actions with no poker semantics such as equity or hand strength.

It is mainly useful as a design reference for what public competitors do, and as a leakage-control checklist.

**pulearn.** Scikit-learn-compatible PU learning (installed 0.2.0). It provides `ElkanotoPuClassifier` and its weighted version, `BaggingPuClassifier` (PU bagging, which accepts any estimator including LightGBM), `NNPUClassifier`, and Bayesian PU classifiers (naive Bayes and TAN). It also has class-prior estimators (label frequency, histogram match, SCAR-EM), propensity estimators for SAR settings, and `PUStratifiedKFold`/`PUCrossValidator`. This is the quickest way to try PU bagging on pair features where the 1,488 confirmed negatives plus about 156k unknown pairs form the unlabelled set.

**nnPUlearning.** The reference Chainer code for the non-negative and unbiased PU risk estimators (Kiryo et al., NeurIPS 2017). `pu_loss.py` is about 50 lines and ports easily to torch, or to a custom LightGBM/XGBoost objective (non-negative risk with a class prior π). It is the canonical formula to copy.

**PUExtraTrees.** Random forest and extra-trees in numpy that split by directly minimising the uPU/nnPU risk (quadratic or logistic loss, which are equivalent to gini or entropy), from arXiv 2210.08461. It supports PU and PN modes and needs a class prior. It is pure Python, so use it on subsamples, for example the labelled pairs plus sampled unknown pairs.

**DEDPUL.** Class-prior (α) estimation and posterior correction for PU data. It includes DEDPUL, Elkan-Noto, KM mixture-proportion estimation, TIcE and nnPU (in `algorithms.py`, `KMPE.py`, `TIcE.py`). Note that for legacy reasons the code returns the prior of the NEGATIVE class. Use it to estimate how many of the ~156k unknown development pairs are latent positives. That estimate is the π needed by nnPU and for calibrating pair scores.

**OMPEval.** A C++ evaluator and multithreaded range-vs-range equity calculator with 200 kB tables. Its rank is 16-bit with category = rank/4096, and it runs at about 270 M random 7-card evaluations/s on one thread with SSE. There is no Python 3.13 binding and it was not built. It is the algorithmic reference if we ever need 10x faster equity (its incremental Hand objects cache rank sums and suit counters).

**multiagent-poker-collusion.** A 6-max NLHE simulator (texasholdem package plus a Rust postflop solver via maturin) in which two LLM agents are prompted to collude against CFR/LLM opponents. It logs JSON hand histories, and `analyze_winnings.py` sums the colluders' combined chip gain. It has no detector. It is only a qualitative reference for how prompted collusion shows up (soft-play and squeeze patterns).

Found but not cloned (low value): mijanur132/Leduc_Poker_Collusion_Detection (Leduc toy game), AndrewChang-cpu/Poker-Collusion (a CFR blueprint bot, not detection), the Bittensor "Poker44" miners (bot detection, not collusion), and Bonjour et al. 2022 "Information-theoretic approach to detect collusion" (mutual information of actions on Leduc; the idea is useful: pairwise MI of action streams). GitHub has no dedicated chip-dumping or soft-play detector with code.

## 3. `src/handeval.py`: numba evaluator and equity

Import it with `sys.path.insert(0, 'src'); import handeval as he`. The import takes about 4 s, which covers the numba cache load and building the dense table. Numba threads default to `HANDEVAL_THREADS` (default **3**).

### Conventions
* Card code = `rank*4+suit`, ranks `23456789TJQKA` (0..12), suits `cdhs` (0..3). 2c=0, As=51. This is identical to phevaluator.
* **rank** = dense hand-class index **0..7461, higher = better**. 0 = 7-5-4-3-2 unsuited, 7461 = royal flush. Ranks are exact for 5, 6 or 7 cards (best five). `-1` means not available.
* `rank == 7462 - phevaluator.evaluate_cards(...)` exactly.
* Category start ranks: high_card 0, pair 1277, two_pair 4137, trips 4995, straight 5853, flush 5863, full_house 7140, quads 7296, straight_flush 7452.

### API
| function | purpose |
|---|---|
| `rank_cards(cards)` | one hand of 5-7 codes (ints, or strings such as `'As'`) returns an int rank |
| `rank_batch(cards2d, counts=None)` | (n, ≤7) int8 in, int16[n] out, parallel. Rows with count <5 give -1 |
| `table_ranks(holes, board5, nboard, upto)` | holes (n,P,2), board (n,5) padded with -1, nboard (n,) in; int16 (n,P) rank of hole + first `upto` board cards out. -1 if the board is shorter or the seat is empty |
| `rank_hole_board(c1, c2, board5, nboard, upto)` | single-seat version of `table_ranks` |
| `hand_category(rank)` | int8 0..8 (`CATEGORY_NAMES`), -1 passes through; works on scalars or arrays |
| `describe_rank(rank)` | e.g. `'two_pair 987'`, `'flush A9742'` |
| `preflop_class(c1, c2)` | 0..168 13×13 grid (row/col 0 = A). Pairs on the diagonal, suited above it (row = high card), offsuit below. AA=0, AKs=1, AKo=13, 22=168. `preflop_class_name(i)`, `PREFLOP_NAMES`, `PREFLOP_COMBOS` (6/4/12) |
| `equity_hu(h1, h2, board=())` | exact enumeration; returns `(equity_h1, p_win_h1, p_tie)` |
| `equity_multi(holes, board=(), dead=())` | exact equity (win plus split share) for 2-6 known hands; extra dead cards allowed |
| `equity_batch(holes, active, board5, nboard, max_exact=20000, n_mc=2000, seed=0)` | returns float32 (n,P) all-in equity for **active** seats (NaN elsewhere), parallel over rows. **Every** hole card in the row, folded seats included, is a dead card. Exact when the runout count ≤ `max_exact`, otherwise deterministic Monte-Carlo with `n_mc` runouts |
| `preflop_equity_table(n_samples, seed)` | MC equity of the 169 classes against 1 and against 5 random hands |
| `card_code / card_str / parse_cards('AhKd2c')` | conversions |

### Verification (`python src/handeval.py 300000`)
* **Ranks: 300,000 random hands each of 5, 6 and 7 cards gave 0 rank mismatches and 0 ordering mismatches** against phevaluator. The ranks are identical after the 7462-x transform.
* **Real data:** 18,174 street ranks on 3,000 random real hands matched phevaluator with 0 mismatches.
* **Showdowns:** in all 311,742 real 5-card-board showdowns, the best-ranked showdown player received a pot share (100%).
* **Equity:** exact equity matched brute force (itertools + phevaluator) with diff 0.0 for:
  * AsKs vs QdQc preflop: 0.462145, which matches published values;
  * AhAd vs 7c2d on Ac7d7h;
  * 9s8s vs AcKd on Ts7s2c2h;
  * a 4-way flop;
  * a 6-way flop.
* **Preflop table** (400k MC samples per class, SE ≈ 0.0008): AA 0.8517 against 1 hand and 0.4913 against 5 hands; 72o 0.3454; 32o 0.3235. These match published figures (85.2%, 34.6%, 32.3%).

### Benchmark (i5-10400F)
| operation | throughput |
|---|---|
| `rank_batch` 7-card random hands, 1 thread | **12.6 M hands/s** |
| `rank_batch` 7-card random hands, 3 threads | **33.6 M hands/s** |
| all 12M seats × 3 streets (`table_ranks`) + joins + parquet write | 10 s wall |
| `equity_hu` preflop, exact (1,712,304 runouts) | 170-230 ms |
| `equity_hu` flop/turn | <0.1 ms |
| `equity_multi` 6-way preflop exact (658,008 runouts) | 150 ms |
| `equity_multi` 3-way preflop exact | 300 ms |
| `equity_batch`, 6-handed random spots (~3.5 active), 3 threads, turn exact | 437,000 spots/s |
| same, flop exact (666 runouts) | 32,600 spots/s |
| same, preflop MC 2000 | 5,100 spots/s |
| phevaluator python loop (for comparison) | 0.65-0.75 M hands/s |

**Estimate for street-start equity against the non-folded opponents on every street** (not built yet):
* flop: 424k hands at ~13 s;
* turn: 220k hands at <1 s;
* river: trivial;
* preflop: 2M hands at ~6.5 min with MC 2000, or much less if restricted to hands with ≥2 voluntary players.

Total ≈ 7 min, so this is cheap enough to build whenever it is needed.

## 4. Derived tables

### `data/derived/seat_strength.parquet`: 12,000,000 rows, 20 cols, 108 MB
Built by `python src/build_seat_strength.py`: 10 s, plus 19 s the first time to create `preflop169.parquet`; peak RSS 1.19 GB. Rows are sorted by (hand_idx, seat_no), one row per dealt seat (every hand has 6), with row-group statistics so `filter(hand_idx range)` prunes.

| column | type | meaning |
|---|---|---|
| hand_idx, player_idx, seat_no | i32, i32, i8 | keys (join to seats/hands) |
| preflop_class | i16 | 0..168 grid index (see above) |
| pf_eq_vs1, pf_eq_vs5 | f32 | class-level MC all-in equity against 1 or 5 random hands |
| rank_flop, rank_turn, final_rank_7 | i16 | dense rank of hole + board prefix of 3/4/5 cards. **null when the hand's board never reached that street** |
| cat_flop, cat_turn, cat_river | i8 | made-hand category 0..8 of the rank above (null likewise) |
| n_opp_better_river | i8 | how many of the other 5 dealt players (**folded included**) hold a strictly better 7-card hand. null without a river |
| n_opp_tied_river | i8 | other dealt players with an identical rank |
| best_hand_at_table | bool | `n_opp_better_river == 0`: would win or chop at showdown had nobody folded. null without a river |
| last_board_n | i8 | number of board cards the hand revealed (0/3/4/5) |
| rank_last | i16 | rank on the hand's final revealed board. null if last_board_n = 0 |
| n_opp_better_last, best_at_last_board | i8, bool | the same comparison against all 6 hands on the final revealed board, e.g. "folded the best made hand on the flop" |
| fold_street | i8 | street 0..3 where this player folded (from actions). null = never folded. Consistent with `seats.folded` (100%) |

Coverage and data notes:
* **The board is revealed only up to the street reached.** 934,586 hands (46.7%) have no flop, 423,691 end on the flop, 219,865 on the turn, and 421,858 (21.1%) have a river. So:
  * `rank_flop` is non-null on 6.39M rows, `rank_turn` on 3.85M, `final_rank_7` on 2.53M;
  * `rank_last` is null on 5.61M rows.
* River category distribution (non-null seats): high 395k, pair 1.08M, two pair 637k, trips 140k, straight 112k, flush 76k, full house 82k, quads 6.0k, straight flush 857.
* River hands average 1.092 `best_hand_at_table` seats. 7.49% of river hands have a tie for best among all six players.
* Fold streets: preflop 8.58M, flop 646k, turn 293k, river 132k; never folded 2.35M. `fold_street` agrees with `seats.folded` on all 12M rows.
* 86,954 seats folded on the flop or later while holding the best made hand on the final revealed board. Many are legitimate folds against a draw or bet, but these are raw material for soft-play and dumping features.

### `data/derived/preflop169.parquet`: 169 rows
`preflop_class` i16, `name` str (`AA`, `AKs`, `AKo`...), `combos` i16, `eq_vs1` f32, `eq_vs5` f32, `order_vs1`/`order_vs5` i16
(1 = strongest). MC with 400k samples per class, seed 12345.

### Usage example
```python
import os, sys; os.environ["POLARS_MAX_THREADS"] = "3"; sys.path.insert(0, r"F:\kaggle competitions\suspicious poker\src")
import polars as pl, handeval as he
ss = pl.scan_parquet(r"F:\kaggle competitions\suspicious poker\data\derived\seat_strength.parquet")
# e.g. seats that folded the best made hand on the final board after the flop
x = ss.filter((pl.col("fold_street") >= 1) & pl.col("best_at_last_board")).select("hand_idx", "player_idx", "fold_street").collect()
he.equity_hu("AsKs", "QdQc")                 # (0.46214, 0.46018, 0.00393)
he.equity_multi(["AhKh", "7s7c", "QdJd"], "Th6c3h")
```

## 5. Files
* `src/handeval.py`: evaluator, equity, preflop classes, self-test and benchmark (`python src/handeval.py [n]`)
* `src/build_seat_strength.py`: builds both derived tables (chunked by 250k hands, <1.2 GB)
* `data/derived/seat_strength.parquet`, `data/derived/preflop169.parquet`
* `research/repos/*`: 10 clones (section 2)
