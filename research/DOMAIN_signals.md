# DOMAIN_signals.md: poker domain signals for hand-level collusion detectors (all hole cards visible)

Written 2026-09-15 (D-5). Every predicate below can be computed from `data/derived/{actions,seats,hands}.parquet`.
The numbers in section 0 were **measured on the dev labels** (372 positive and 1,488 hard-negative pairs, dev-phase shared hands).
They are not taken from the literature.

**Artifacts produced by this note**

| file | what |
|---|---|
| `data/derived/preflop_equity_169.parquet` | 169 classes: `class_id, grid_idx, hand_class, hi_rank, lo_rank, pair, suited, combos, equity_vs1..5, win_vs1..5, chen, sklansky_group, top_pct_vs1..5, nsim`. MC with 1,000,000 deals per class per opponent count (SE <= 0.0005). |
| `data/derived/preflop_class_map.parquet` | all 2,652 ordered `(c1,c2)` -> `class_id, grid_idx, hand_class`. `grid_idx` matches `src/handeval.py::preflop_class` (AA=0, AKs=1, AKo=13, 22=168; verified on all combos). |
| `src/poker_equity.py` | numba evaluator + `equity_exact`, `equity_mc`, `hand_strength_vs_random`, `class_id`, `chen_score`, `sklansky_group`. Validated below. |
| `src/hand_replay.py` | per-action equity replay: `eq_pre_s0..5`, `eq_post_s0..5`, `hs_actor` per action row, plus the heads-up matrix `hu_s{street}_{i}{j}` per hand. |
| `data/derived/action_equity_labeled_dev.parquet`, `hu_equity_labeled_dev.parquet` | replay of the 204,325 dev hands shared by labelled pairs (1,898,654 actions, 4 min at 4 threads) |
| `data/derived/action_equity_eval.parquet`, `hu_equity_eval.parquet` | replay of all 800,000 eval-phase hands (7,493,544 actions) |
| `data/derived/pair_hand_detectors_labeled_dev.parquet` | 223,749 (labelled pair x dev shared hand) rows with every detector in section 2, plus `is_ev`, `label`, `fam` |
| `src/domain_detectors_check.py`, `src/domain_detectors_ci_check.py` | reproduce the measurement tables in section 0 |

---

## 0. What matters: measured summary

The table covers single predicates. The `% ev` columns give the fire rate on listed evidence hands; `% non-ev` gives the fire rate on the same positive pairs' other dev shared hands; `% hard-neg` gives the fire rate on hard-negative pair-hands. MAP@5 means ranking each positive pair's dev shared hands by the predicate, scored with the host metric's evidence AP; the random baseline is about 0.02. Pair AUC is the per-pair fire rate, positives of the family vs the 1,488 hard negatives. The thresholds were fixed a priori after reading about 15 evidence hands and were **not tuned**; there are no folds, so treat the numbers as mildly optimistic.

| detector (section) | DT % ev / non-ev | SP % ev / non-ev | CI % ev / non-ev | % hard-neg | MAP@5 DT / SP / CI | pair AUC DT / SP / CI |
|---|---|---|---|---|---|---|
| `fold_best_vs_partner` (2.1b) | **56.8** / 2.44 | **32.9** / 1.12 | 7.8 / 3.12 | 1.39 | **.33 (.35 pot-weighted)** / .26 / .05 | .88 / .69 / .81 |
| `aggr_dead_into_partner` (2.1c) | 35.9 / 2.90 | 29.4 / 1.39 | 1.1 / 1.38 | 0.97 | .20 / .24 / .02 | .91 / .80 / .58 |
| `call_dead_vs_partner` (2.1a) | 21.1 / 3.38 | 7.9 / 0.64 | 0.4 / 0.94 | 0.76 | .13 / .09 / .02 | **.92** / .58 / .56 |
| `passive_strong_hu_vs_partner` (2.2a) | 21.5 / 2.40 | 33.7 / 2.38 | 2.2 / 1.56 | 0.98 | .15 / .25 / .02 | .85 / .89 / .65 |
| `river_check_nuts_hu_vs_partner` (2.2b) | 3.7 / 0.40 | 20.4 / 1.73 | 0.9 / 0.39 | 0.19 | .05 / .17 / .02 | .65 / **.93** / .59 |
| `raise_over_3rd_after_partner_aggr_weak` (2.3d) | 5.9 / 1.54 | 1.9 / 0.18 | 13.3 / 3.31 | 0.09 | .06 / .04 / .06 | .92 / .58 / **.997** |
| `sandwich` (both partners aggress, same street, 3+ live) (2.3a) | 7.9 / 1.78 | 1.9 / 0.58 | 39.3 / 8.01 | 1.79 | .07 / .03 / .09 | .56 / .24 / **.988** |
| `rr_fold_vs_partner` (raise, partner re-raises, fold) (2.3b) | - | - | 29.8 / 5.77 | 1.15 | CI .11 | .44 / .27 / **.987** |
| `o_fold_facing_pair_after_both_aggr` (2.3c) | - | - | 34.1 / 6.42 | 1.28 | CI .17 | .55 / .26 / **.985** |
| `third_party_fold_to_pair_aggr` (2.3c) | 90.3 / 20.3 | 65.3 / 10.0 | 80.2 / 22.4 | 13.6 | .19 / .17 / **.23** | .88 / .45 / .90 |
| `raise_vs_partner` (conflict) | 16.4 / 3.22 | **1.7 / 0.55** | 36.3 / 7.84 | 2.01 | - | .75 / **.21** (SP raises partner LESS) / .98 |

The luck-free value-impact score (Mazrooei et al. 2013, section 1.9) needs no hand-crafted rule. Summed per 100 dev shared hands, max direction, in bb, the medians are DT 481, CI 176, SP 120 and hard negatives 79. Pair AUC vs hard negatives is **DT .968**, SP .68 and CI .80. As hand-level evidence MAP@5 it scores DT .24 and SP .28.

Family-routed hand scores reach evidence MAP@5 of **DT .35** (`s_fold_best`), **SP .36** (`generic` = count of all DT/SP predicates + 0.01 x impact) and **CI .23** (`third_party_fold_to_pair_aggr`).

Structural findings:
1. **Unlisted planted hands exist.** On every positive family, the non-evidence shared hands fire 2-4x more often than hard-negative pair-hands (for example, CI `raise_vs_partner` fires on 7.8% vs 2.0%). Pair-level rates therefore carry signal beyond the 3-5 listed hands.
2. **The CI evidence list is chronological.** `evidence_rank` equals the chronological order of the listed hands in 88% of CI pairs (Spearman .91). The figure is 39% for DT and 27% for SP (Spearman .43 and .39). This probably means "the first 5 planted CI hands". Among equally suspicious CI candidates, earlier eval hands may be the better picks. That is gameplay time, but it concerns how the label list was built, so decide consciously whether to use it.
3. **Evidence pots are big.** Median final pot is 34.5 bb for DT, 17.0 for SP and 17.8 for CI, against 9.25 bb for all dev hands. The 90th percentiles are 214, 180 and 155 against 97. Pot size (in bb) is a strong prior for evidence ranking.
4. **Seat geometry is null.** Seats are redrawn every hand: every player uses all 6 seats (median distinct seats = 6). The seat-distance distribution {1: .40, 2: .40, 3: .20} is identical for positives, negatives and evidence hands. Do not build "partner sits to the left" features.
5. **Episodes are only mildly clustered.** Evidence hands cover 61% (DT) and 68% (SP) of the pair's shared timeline, close to the 67% expected for 5 uniform draws. CI covers 41%, so it is clustered. Over dev time, the evidence-hand quantiles are 10%/50%/90% = 0.07/0.36/0.82 of the dev phase (CI 0.04/0.31/0.71).

---

## 1. Hand-strength machinery

### 1.1 Evaluator (`src/poker_equity.py`)
- The card code is `rank*4+suit`, rank `23456789TJQKA` (0..12), suit `cdhs`. This is identical to phevaluator ids and to `src/handeval.py`.
- `eval_cards(cards, n)` returns an int score (higher is better) with `category = score >> 20`: 0 high card, 1 pair, 2 two pair, 3 trips, 4 straight, 5 flush, 6 full house, 7 quads, 8 straight flush. The low 20 bits hold 5 x 4-bit rank nibbles (kickers).
- Design:
  - Rank-multiplicity bitmasks `m1..m4` (seen >= 1..4 times) and 4 suit bitmasks are built in one pass.
  - Three 8192-entry lookup tables: `STRAIGHT_HIGH[mask]`, `TOP5[mask]` and `HIGHBIT[mask]`.
  - The wheel is handled via `mask & 0x100F`. There is no allocation.
  - A flush and a full house/quads cannot both exist in 7 cards, so the branch order is safe.
- Validation:
  - 7,462 distinct 5-card classes.
  - Category counts over all C(52,5) hands match the textbook values exactly.
  - 0 ordering mismatches vs phevaluator on 400,000 random 7-card pairs.
  - AsAh vs KdKc full enumeration = 0.8125549, identical to phevaluator enumeration.
- Speed (1 thread): 9.3M 7-card evals/s and 43M 5-card evals/s.

### 1.2 Preflop tables
- `equity_vsK` is the pot-share equity (win + split share) against K random hands that all go to showdown. `win_vsK` is the outright win probability.
- `top_pct_vsK` is the combo-weighted cumulative share of starting hands at least as strong (AA = .0045, 32o = 1.0 vs 1).
- Checkpoints against published charts:

| hand | vs1 | vs2 | vs5 |
|---|---|---|---|
| AA | .852 | .734 | .491 |
| KK | .824 | | |
| AKs | .670 | | |
| AKo | .653 | | |
| 22 | .503 | | |
| 72o | .346 | | |
| 32o | .324 (worst) | | |

- Use `equity_vs{n_opp}` with `n_opp = players_active - 1` as the *perceived* preflop strength of the decider. The honest player does not know the other cards. This slightly overstates strength against players who already continued (range narrowing). For "fold to raise" spots, prefer `top_pct_vs1` deciles as strata.
- **Chen formula** (`chen_score`):
  1. Score the high card: A 10, K 8, Q 7, J 6, T..2 = face/2.
  2. Pairs: x2, minimum 5.
  3. Suited: +2.
  4. Gap penalty: 0 -> 0, 1 -> -1, 2 -> -2, 3 -> -4, 4+ -> -5.
  5. +1 if the gap is 0 or 1 and both cards are below Q.
  6. Round up.
  Examples: AA 20, AKs 12, 72o -1.
- **Sklansky-Malmuth groups** (`sklansky_group`, 9 = unlisted):

| group | hands |
|---|---|
| 1 | AA KK QQ JJ AKs |
| 2 | TT AQs AJs KQs AKo |
| 3 | 99 JTs QJs KJs ATs AQo |
| 4 | T9s KQo 88 QTs 98s J9s AJo KTs |
| 5 | 77 87s Q9s T8s KJo QJo JTo 76s 97s A9s-A2s 65s |
| 6 | 66 ATo 55 86s KTo QTo 54s K9s J8s 75s |
| 7 | 44 J9o 64s T9o 53s 33 98o 43s 22 K8s-K2s T7s Q8s |
| 8 | 87o A9o Q9o 76o 42s 32s 96s 85s J8o J7s 65o 54o 74s K9o T8o |

### 1.3 Made-hand classes per street (computable)
Notation: `b1 >= b2 >= ...` are the board ranks, `h_hi >= h_lo` are the hole ranks, `cat = eval(hole+board) >> 20`.

- **Plays a hole card:**
  - River: `eval(hole+board) > eval(board5)`.
  - Flop/turn: compare against the best 5 of the board plus any 2 unseen cards. In practice, use `cat(hole+board) > cat_of_board_multiplicity`. The board alone has pair/trips/quads by multiplicity; a straight or flush needs 5 board cards.
- **One pair (cat 1):**
  - Pocket pair: overpair if `h > b1`, underpair if `h < min(board)`, otherwise middle pocket pair.
  - Board-matched pair: top pair if the matched rank is `b1` (the kicker is the other hole card; a kicker >= J is a "good kicker"), second pair if `b2`, otherwise weak pair.
  - Board pair with no hole card involved: nothing.
- **Two pair:** both hole cards paired is strong; one hole card plus a board pair is weak.
- **Trips:** a set (pocket pair plus one board card) is strong and disguised; trips (one hole card plus a board pair) is weaker because of kicker issues.
- **Relative strength (the robust choice):** `hand_strength_vs_random` = P(win)+P(tie)/2 of the current made hand against every unseen 2-card holding (at most 1,081 evals).
  - The river value is exact.
  - Nut indicator: HS >= 0.99, i.e. no holding beats you.
  - Stored as `hs_actor` in the replay.

### 1.4 Draws and outs (flop/turn, perceived potential)
- **Flush draw:** some suit has exactly 4 cards among hole+board with at least 1 hole card of that suit.
  - Nut flush draw: the hole card of that suit is the highest rank of the suit not on the board.
  - Backdoor draw (flop): 3 cards of a suit with a hole card involved.
- **Straight draws:** let `M` be the rank bitmask of hole+board. The completing ranks are `R = {r : STRAIGHT_HIGH[M | 1<<r] >= 0 and STRAIGHT_HIGH[M] < 0}`, with the condition that the straight is not fully on board (`STRAIGHT_HIGH[boardmask | 1<<r] < 0`).
  - `|R| = 2` means an OESD or double gutter (8 outs); `|R| = 1` means a gutshot (4 outs).
  - Subtract outs already visible. With omniscience, all dealt cards are removable.
- **Outs to equity:** rule of 4 and 2 (flop two cards to come ~ 4% per out, turn ~ 2%).
- **Hand potential (Billings et al.):** `EHS = HS + (1-HS)*PPOT`. Estimate it with MC over random opponent holdings and runouts (about 200 x 45).
- Use potential to **exclude honest semi-bluffs** from "aggression with a dead hand" detectors. A flop bet with 8% omniscient equity is normal when perceived HS+potential is at least about 0.35.

### 1.5 Exact equity when every hole card is known
All 6 seats are dealt in every hand, so 12 hole cards are known (folded hands become dead cards).

| street | remaining deck | runouts (all 12 known) | runouts if other hands unknown, HU |
|---|---|---|---|
| preflop | 40 | C(40,5) = 658,008 | C(48,5) = 1,712,304 |
| flop | 37 | C(37,2) = 666 | C(45,2) = 990 |
| turn | 36 | 36 | 44 |
| river | 35 | 1 | 1 |

- **Cost:** preflop exact is about 0.8 s per HU spot. Use `equity_mc` instead: SE is about 0.5/sqrt(n), so 400 samples give ~2.5% and 2,000 give ~1.1%. Alternatively, cache HU equities by suit-isomorphic matchup. Postflop exact costs under 0.1 ms per spot.
- **Fast numba approach** (`hand_replay.py`):
  - `prange` over hands.
  - Per hand, walk the actions maintaining the live-seat bitmask.
  - Cache the equity vector per `(street, live_mask)`: the vector only changes on folds and board cards.
  - Nested runout loops over the remaining deck; `_settle` evaluates all live hands and splits ties.
  - HU matrix: 15 seat pairs x streets reached, with all other 4 hands dead.
- **Measured replay speed:** about 850 hands/s on 4 threads (including the HU matrix and `hs_actor`). The 800k eval hands take about 20 min on 3 threads; all 2M hands would take about 40 min.

### 1.6 Two notions of strength (central to collusion)
- **Omniscient equity** (`eq_pre_s*`, `hu_*`) is what the money actually does. Use it to measure value transferred and EV lost.
- **Perceived strength** (`hs_actor` postflop, `equity_vsN`/`top_pct` preflop) is what an honest decider could know.
- Honest mistakes are explained by perceived strength: folding a hand that looks weak but is actually best. Planted collusion is explained by omniscient or partner information: folding trips (perceived HS .98) to a 1/2-pot partner bet.
- **Always condition on perceived strength**, and measure the damage with omniscient equity.

### 1.7 EV of a decision with all cards known
The schema's `pot_before` already includes the bet being faced; `to_call = c`.
- **Facing a bet, heads-up, ignoring future betting:**
  - `EV_call = eq*(P+c) - c`, `EV_fold = 0`, break-even `eq* = c/(P+c)`.
  - `EVloss_fold = max(0, eq*(P+c) - c)`, `EVloss_call = max(0, c - eq*(P+c))`.
- **River** (no cards to come, eq in {0, .5, 1}) is exact:
  - Folding the winner loses `P`.
  - Calling with the loser loses `c`.
  - Betting `a` with the loser and being called loses `a`.
  - A foregone value bet (checking the nuts) is a counterfactual. Proxy it by the event "check with eq >= .99 heads-up".
- **Multiway:** use `eq_pre` vs all live seats for the chips at stake, and `hu_s{st}_{ij}` for "who is ahead between these two".
- **Normalization:** always divide by `bb` (tables run at different stakes, e.g. bb 2, 4, 10).
- **Thresholds used above:**
  - "fold best": eq >= .5 facing the partner's bet (a pot-odds-aware variant is `eq >= eq* + .3`).
  - "dead": eq <= .15 while the partner has eq >= .7.
  - "strong perceived": preflop `top_pct_vs1 <= .15`, postflop `hs_actor >= .80`.
- **All-in adjusted EV per hand:** `EV_j = eq_j(at the moment the money went in)*pot - contrib_j`, and `luck_j = net_j - EV_j`. This removes coolers and streaks.
- **Caveat:** an all-in for less than `to_call` creates side pots, so equity x total pot is only approximate there.

### 1.8 Schema gotchas (verified)
- Blinds are **not** action rows. The first action has `pot_before = sb+bb`.
  - A player's posted blind is `amount_to - amount` on his first preflop row.
  - Fold rows carry `amount_to` = the blind already posted.
  - The BB's option shows as `check` with `amount_to = bb`.
- `players_active` is the number of non-folded players **before** the action, including the actor.
- `all_in` (5) is a call if `amount <= to_call`, otherwise a bet/raise. `amount` is the additional chips; `amount_to` is the street total.
- Last aggressor on a street: `pl.when(is_aggr).then(seat_no).shift(1).forward_fill().over([hand_idx, street])`. Preflop starts null (blinds).
- In `seats.parquet`, `seat_no` indexes the per-hand seat; `hand_replay` columns `eq_pre_s{seat_no}` use the same index.

### 1.9 Luck-free value accounting: the Mazrooei "collusion table"
- **Value function:** `V_j = eq_j * pot`.
- **Impact of action t** (actor k, chips `a`, pot before `P`) on seat j:
  `dV_j = (eq_post_j - eq_pre_j)*P + eq_post_j*a - [j==k]*a`
  - For a fold: `dV_k = -eq_k*P`, and the other seats gain the redistributed equity.
  - For a bet/call: `dV_k = -(1-eq_k)*a`, and `dV_j = eq_j*a`.
- **Properties:** `sum_j dV_j = 0` for every action. Board-card (chance) impacts are excluded, so the accounting is luck-free.
- **Collusion table:** `C(j,k) = sum over hands of sum_{t: actor k} dV_j(t)`, i.e. what k's decisions did to j.
- **Total impact:** `TI(a,b) = C(a,a)+C(a,b)+C(b,a)+C(b,b)`.
- **Marginal impact:** `MI(a,b) = [C(b,a) - mean_{i not in ab} C(i,a)] + [C(a,b) - mean_{j not in ab} C(j,b)]`. Compute the means only over hands where i was live at k's action, and normalize by co-live exposure.
- In AAAI-13 (synthetic 3-player limit poker), the two real colluding pairs ranked 1st and 2nd by TI (100.6 and 38.6). MI ranked an "accidental" colluder pair first and the strong colluders second (109.2). TI is better for real teams, and MI alone gets fooled by style-correlated players. The mechanism-agnostic design is exactly what the undisclosed `other_coordination` family calls for.
- Measured here: pair AUC DT .968 from the raw max-direction impact alone. Implement the marginal version (subtract impact on others) next.

---

## 2. Collusion archetypes as action-log predicates
X is the actor, Y the partner, O any third party. "Facing partner" means `to_call > 0` and `last_aggr == seat(Y)`. The real evidence hands cited are in the `hand_idx` space.

### 2.1 Directed transfer / chip dumping (one player feeds the other)
a) **Pay-off with a dead hand** (`call_dead_vs_partner`): X calls Y's postflop bet/raise with eq <= .15 while Y has eq >= .7, often on several streets.
   - Hand 290305: donor Q4o calls turn and bets river with 0% equity into the partner's straight.
   - Hand 290483: T9o calls a raise and a river all-in vs AA, with 0%.
   - Hand 1770651: T9o calls flop, turn and a pot-sized river bet vs AQ (two pair), with 9% then 0% equity.
b) **Fold the winner to a partner bet** (`fold_best_vs_partner`): X folds facing Y's bet with eq >= .5.
   - Hand 1770847: donor folds 100%-equity river vs a 20-chip bet into 50.
   - Hand 1770650: 67% equity turn fold.
   - Also preflop: AKo folded to the partner's raise (hand 1771305).
c) **Bet or lead into the partner's monster** (`aggr_dead_into_partner`): X bets/raises with eq <= .15 while Y is live with eq >= .7. Y raises and X calls off ("pump": bet, get raised, call). Also min-raise-then-fold and overbet shoves into the partner's nuts.
d) **Direction and asymmetry:** the same donor shows up across hands. `|C(B,A) - C(A,B)|` is large (median asymmetry 288 bb/100 for DT vs 21 for negatives). Net chip flow X->Y in contested pots, minus luck (1.7), follows the same pattern.
e) **Small-pot variants:** partner raises, X calls with junk, then X folds to the partner's c-bet a street or two later (hands 290005, 290008). These transfer only a few bb per hand. Do not require big pots.

### 2.2 Soft play (partners avoid normal aggression against each other)
a) **Passive with a strong hand heads-up vs the partner** (`passive_strong_hu_vs_partner`): postflop, 2 live, X eq >= .8, X checks or just calls. Hands 581139 and 581272: calls with .83-.97 and checks back the river with 100%.
b) **River nuts check HU vs the partner** (`river_check_nuts_hu_vs_partner`): the single best SP pair-rate feature (AUC .93).
c) **Fold strong to the partner's aggression.**
   - Hand 1895716: QJ with trip jacks (.98) folds to the partner's flop bet.
   - Hand 1897045: KK folds preflop to the partner's raise.
   - Hand 1897361: A4s (.93) folds to the partner's bet.
   - Hand 1897371: .86 equity folds to the partner's all-in in a 3-way pot, leaving the partner vs the third player.
d) **Absence of conflict.** SP pairs re-raise each other **less** than hard negatives (pair AUC .21 for `raise_vs_partner`, i.e. inverted). Build "3-bet rate facing Y's open", "bet-into-Y rate" and "HU showdown rate" as **deficits** vs the actor's baseline.
e) SP evidence also fires `aggr_dead_into_partner` (29.4%). This is the mirror image of (c): one partner bets air, the other folds the best hand. **The best evidence hands contain both sides.**

### 2.3 Coordinated isolation / whipsaw / squeeze (pressure a third party, no conflict between partners)
Typical CI evidence action strings, preflop (`r` raise to N bb, `c` call, `x` fold, `k` check):
- `A r2 | o r8 | B r22 | A r50 | o x | B x`: a 4-bet/5-bet war that ends with the partner folding (hand 1675924).
- `A r2 | o c | B r6 | A r20 | o x | B x` (hand 1512461).
- `o r2 | B c | A r6 | o x ... B x`: A squeezes after B's call, the opener folds, then B folds (hand 541740).
- `o r2 | B r5 | A x | o r16 | B x` (hand 1771006) and `B r2 | A x | o x`: the partner steps aside for the partner's raise.

Predicates:
a) `sandwich`: both partners are aggressive on the same street with 3+ live (CI ev 39.3% vs hard-neg 1.8%).
b) `rr_fold_vs_partner` ("fake conflict"): X raised, Y re-raised, X folds. The whole raise war resolves without the partners fighting (AUC .987).
c) `o_fold_facing_pair_after_both_aggr`: a third party folds after both partners were aggressive on the street (AUC .985). The weaker `third_party_fold_to_pair_aggr` gives the best CI evidence MAP@5 (.23).
d) `raise_over_3rd_after_partner_aggr_weak`: X re-raises a third party's raise after Y was aggressive on the street, with a weak perceived hand (preflop `top_pct_vs1 >= .4`, postflop `hs <= .5`). Pair AUC .997, and only 0.09% of hard-negative pair-hands fire it.
e) **Partner steps aside:** fold to the partner's raise preflop with a playable hand (`top_pct_vs1 <= .35`) (CI ev 17.6% vs 3.7% hard-neg). The raw fold-to-partner-raise fires on 81% of CI evidence hands but also on 27% of non-evidence and 18% of hard-negative pair-hands. Use it only stratified (section 4).
f) **Victim loss:** Mazrooei impact of the pair's actions on third parties in hands with a sandwich.

**CI evidence retrieval is inherently hard.** Planted CI hands are frequent, and the listed ones are the chronologically first (section 0). Expect MAP@5 around .2 unless a learned ranker plus a time prior is used.

### 2.4 Information sharing / ghosting (candidate mechanism for `other_coordination`)
With full visibility this is testable as **dependence of X's action on Y's hidden cards**. This is the IET patent's Bayesian-network idea: a causal link from the partner's hand rank to the player's action.
- **Ghost folds:** X folds hands with good perceived strength more often when Y holds a strong hand (removing himself so the partner gets heads-up). Or X folds when X's cards block Y's outs.
- **Pump or whipsaw on the partner's strength:** X raises weak holdings more often when Y is strong ("colluder raises regardless of cards after partner's raise"; Mazrooei et al.).
- **Card removal:** X continues with draws whose outs are not in Y's (folded) hand. Compare continue rates vs outs-remaining with vs without Y at the table.
- **Signalling via sizing:** Y's bet-size distribution correlates with Y's HS more than baseline, and X's responses follow the sizing.

### 2.5 Seat and position
Classic whipsaw needs the victim seated between the partners in action order. **In this generator seats are redrawn every hand and evidence hands show a uniform seat-distance distribution**, so drop seat features. Per-hand action order (who acts between the partners) still matters inside the sandwich predicates.

### 2.6 Other generic mechanisms to cover `other_coordination` without knowing it
- Mazrooei MI score (value-agnostic).
- Conditional-dependence score (2.4).
- **Pair-surprise residual:** fit a per-player action model (public state + own cards, e.g. GBDT on `street, n_live, to_call/pot, position, hs/top_pct`) on hands without Y. Then compute the mean log-likelihood ratio of X's actions in hands with Y vs hands with the other Z.
- **Blind-steal agreement:** never defend the BB vs the partner's button/SB open.
- **Mutual pot-building vs a third party:** both call down vs O with medium hands.

---

## 3. Benign look-alikes and how to separate them

| look-alike | what it resembles | separator (compute this) |
|---|---|---|
| **Tilt** (loose-aggressive after losses) | DT (EV loss), CI (raise wars) | Loss is spread over all opponents. Define tilt windows (the player's last-k net <= -X bb) and compare the directed share: the partner's share of X's EV loss inside the window vs co-live exposure share. Tilt hits whoever is present; DT concentrates on Y. Use the double-centred score (4.1). |
| **Weak / fishy play** | DT (calls with dead hands, folds winners) | Same predicates fire vs everyone. Use the player-specific baseline `p_X(k)` per stratum; the directed excess O/E ~ 1 for fish. Weak players also lose to hard negatives in showdowns at similar rates. |
| **Similar strategies** (two nits, two passive players) | SP (rarely raise each other) | Deficit measured against each player's own raise rate vs others in the same strata. Correlated marginal stats (both low PFR) are symmetric and not directed. |
| **Repeated opponent selection** (sitting with the same players) | any co-occurrence feature | `shared_hands` is uninformative (pair AUC of n = .50-.54). Normalize every count by exposure (co-live spots), never by calendar. |
| **Streaks / variance / coolers** | DT net-chip flow | Use omniscient-equity impact (chance excluded) or all-in EV, not net chips. In coolers both players' actions are perceived-strength justified (hs >= .8 on both sides). |
| **Strategy changes** (style shift mid-timeline) | episodic collusion | The change affects play vs all opponents. Estimate baselines in the same time window (rolling or change-point per player) and compare the directed deviation within the window. |
| **Hero folds / bluff-catch folds** | SP/DT fold-best | Honest folds of the best hand happen facing large bets with medium perceived strength. Stratify by bet size (`c/P` buckets) and perceived HS. Folding trips (HS .98) to a 1/3-pot bet has no honest explanation. |
| **Semi-bluffs** | DT "aggression with dead hand" | Exclude or stratify by perceived potential (draws, 1.4) and fold equity (bet into multiple opponents). |
| **Hard negatives in the labels** | everything | Hard-negative pair-hands fire at 1-2%, comparable to positive non-evidence hands, so single-hand triggers are weak. **Rates plus directedness plus episodes** separate them (pair AUCs .9+). |

---

## 4. Directedness statistics, per family

### 4.1 General recipe (apply to every event)
For ordered (X -> Y) define a spot set with event `E_t` (binary) or loss `L_t` (bb):
- **Spots:** X decisions where the counterparty is Y (facing Y's aggression, or HU with Y).
- **Reference spots:** the same decision types with counterparty Z != Y, for X at the same table.
- **Strata** `k(t) = (street, n_live in {2, 3+}, perceived-strength decile of X, bet-size bucket c/P in {<=.35, <=.7, <=1.1, >1.1}, [omniscient eq bucket {<.15, .15-.5, .5-.85, >=.85} where the event is defined on it])`.

1. **Expected count from X's own behaviour vs others**, with shrinkage to the population rate in the stratum:
   `p_X(k) = (e_Xk^others + m*p_pop(k)) / (n_Xk^others + m)` (m about 5-20, method of moments).
   `E0 = sum_{t in partner spots} p_X(k(t))`, `O = sum E_t`.
2. **Test statistics:**
   - Poisson signed LLR `G = sign(O-E0) * 2[O ln(O/E0) - (O-E0)]`.
   - Normal `z = (O-E0)/sqrt(sum p(1-p))`.
   - Mantel-Haenszel common OR over strata: `OR_MH = sum_k(a_k d_k/N_k) / sum_k(b_k c_k/N_k)`, where a and b are events and non-events vs Y and c and d vs others. Use Robins-Breslow-Greenland SE, and add 0.5 to empty cells.
   - Continuous losses: `z_L = sum_t (L_t - mu_X(k)) / sqrt(sum_t sigma^2_X(k))`.
3. **Empirical Bayes across pairs** (shared hands run 38..419): `O ~ Poisson(lambda*E0)`, `lambda ~ Gamma(a, a)`, so the posterior mean is `(a+O)/(a+E0)`. Fit `a` on all eval pairs. Use the posterior mean or its 10% quantile as the model feature. It is automatically conservative for small-exposure pairs, which matters for pair AP.
4. **Placebo / double-centring** (kills tilt, fish, style, table effects): compute `S(X,Z)` for all 29 other Z at the table. The feature is `S*(X,Y) = S(X,Y) - median_Z S(X,Z)` (and symmetrically for Y), or the within-player percentile rank of Y among X's opponents. Tables have 30 players and about 281 eval pairs, so this is cheap.
5. **Episodes:** add a scan statistic, `max over windows of consecutive shared hands (25, 50, 100, 200) of G_window`. Calibrate by permuting `E_t` within X's partner spots, or penalize with `log(#windows)`. Also add the top-m sum: the sum of the 5 largest hand-level scores. It mirrors evidence extraction and is robust to on/off activity.
6. **Symmetry:** SP and CI are mutual, so sum the two directions. DT is directional, so keep `max(dir)` and `|A->B - B->A|`.

### 4.2 Directed transfer
- `VT(X->Y)` = Mazrooei `C(Y,X)` minus the exposure-matched mean `C(Z,X)`, in bb per 100 co-live hands (the raw version already gives AUC .968).
- Fold-best-to-Y: event = fold facing Y with omniscient eq >= .5. Strata: perceived strength x bet size.
- Pay-off-Y: loss `L = c - eq(P+c)` on calls facing Y postflop. Also the event (eq <= .15 and eq_Y >= .7).
- Bet-into-Y's-monster: event = aggression with eq <= .15 while Y is live and eq_Y >= .7. Strata: perceived HS + potential (the bluff baseline).
- Luck-adjusted net flow X->Y in hands both played to the end.

### 4.3 Soft play
- **Aggression deficit vs Y:** event = bet/raise at opportunities HU vs Y (to_call = 0, or facing Y's bet) with X eq >= .7 or hs >= .8. Expect `O/E0 << 1`. Build it both ways and sum `-G`.
- **River nuts check:** event = check with eq >= .99 HU vs Y (AUC .93 even without baseline).
- Fold strong-perceived to Y's aggression (hs >= .8 or top-15% preflop).
- **3-bet deficit:** 3-bet rate facing Y's opens vs facing others' opens, per (position, class decile).
- **Checkdown:** rate of HU postflop hands vs Y with zero bets while someone holds eq >= .8.

### 4.4 Coordinated isolation
- Raise-war-without-conflict: `rr_fold_vs_partner`, plus `reraise_partner` followed by a partner fold in the same hand, vs the baseline re-raise-then-fold rate for X vs others.
- Weak re-raise over a third party after the partner's aggression (strata: street, X class decile). Hard-negative base rate 0.09%.
- Third-party squeeze-out: O's fold rate facing the pair when both were aggressive, vs O's fold rate facing a single aggressor in the same strata (to_call/P bucket, O perceived strength). Aggregate over all O.
- Step-aside: fold to Y's raise with `top_pct_vs1 <= .35` vs X's fold-to-raise rate vs others (strata: position, raise size).
- Victim damage: the pair's summed impact on third parties in sandwich hands.

### 4.5 Information sharing / other_coordination
- **Score test for Y's hidden strength in X's policy:** `logit P(continue_t) = f(X perceived strength, strata) + beta * s_Y(t)`, where `s_Y` is Y's HU equity vs X or Y's preflop `equity_vs1`.
  - Honest play gives `beta = 0`. Report `z_beta` per ordered pair and double-centre against Z.
  - Do the same for aggression (`beta > 0` = pump/whipsaw).
  - Do the same for spots where Y has already folded (card removal, `beta` on "Y's folded cards contain X's outs").
- Pair-surprise residual (2.6) and the Mazrooei MI score.

### 4.6 Evidence hand ranking (20% of the score)
- **Candidate hands:** eval shared hands of the pair where at least one family predicate fires.
- **Hand score:** a family-routed, pot-weighted sum.
  - DT: `eq*P/bb` for fold-best, `+ (1-eq)*a/bb` for dead calls, `+ eq_Y*a/bb` for dead aggression.
  - SP: fold-best + passive-strong + dead aggression + 5 x river-nuts-check.
  - CI: 3 x sandwich + 3 x weak re-raise over O + 2 x partner re-raise + third-party fold + fold-to-partner-raise.
  - Tie-break by final pot in bb (evidence pots are about 2-4x typical).
- **Better:** a hand-level classifier (LightGBM) on `pair_hand_detectors_labeled_dev.parquet` features (plus pot bb, n_live at key spots, and the Mazrooei impacts of both directions), with `GroupKFold` by pair. The label is `is_ev`, and the model learns per family.
- Choose the family from the pair model, then pick the top 5 distinct hands.
- For CI, consider a mild chronological prior (section 0, finding 2).

### 4.7 Pitfalls
- **PU labels:** dev "unknown" pairs are not negatives, and positive pairs contain unlisted planted hands. Hand-level `is_ev = 0` inside positive pairs is noisy-negative.
- **Look-ahead in baselines:** player baselines may use both phases, but never use label-derived hands (evidence) to build them.
- **Equity MC noise preflop** (400 samples, SE about 2.5%): avoid hard thresholds within ±.05 of preflop eq cutoffs. Prefer class tables for perceived strength.
- **Side pots** make `eq*pot` approximate when stacks differ. It is exact for single-pot hands and the error is small in practice.
- **Multiple testing:** many detectors across 112k pairs. Let the pair model combine them (shrunk features) rather than thresholding individual z-scores.

---

## 5. Reproduce
```
set NUMBA_NUM_THREADS=4 & set POLARS_MAX_THREADS=3
python src/poker_equity.py build_preflop 1000000              # ~2 min, writes preflop_equity_169 + class map
python src/hand_replay.py labeled_dev                          # 204k dev hands shared by labelled pairs, ~4 min
python src/hand_replay.py eval <parquet with hand_idx list>    # e.g. all phase==1 hands, ~20 min @3 threads
python src/domain_detectors_check.py data/derived/pair_hand_detectors_labeled_dev.parquet   # section-0 tables
python src/domain_detectors_ci_check.py                        # CI-specific rows
```

## Sources
- Mazrooei, Archibald, Bowling, *Automating Collusion Detection in Sequential Games*, AAAI-13: https://poker.cs.ualberta.ca/publications/AAAI13.pdf (local copy in `research/papers/`). Source of the collusion table and the TI/MI scores.
- US9652931B2 (CFPH), *Collusion detection*: https://patents.google.com/patent/US9652931B2/en. Chip dumping = a transfer inconsistent with the player's style; soft play = a passive action where the profile predicts aggression; profile stats AF (<1 passive, >1.5 aggressive), VPIP bands, "collusions / total actions > threshold".
- US7604541B2 (IET), *Detecting collusion via conditional behavior*: https://patents.google.com/patent/US7604541B2/en. A Bayesian-network causal link from the partner's hidden hand rank to the player's action; the odds ratio of collusional vs non-collusional graphs; about 500-800 games to discriminate.
- US7618321B2 (PokerTek), *Detecting collusion between poker players*: https://patents.google.com/patent/US7618321B2/en
- Smed, Knuutila, Hakonen, *Towards Swift and Accurate Collusion Detection* (GAMEON 2007), and the SCAI 2006 paper: https://www.semanticscholar.org/paper/Can-We-Prevent-Collusion-in-Multiplayer-Online-Smed-Knuutila/ae644637f1785c3c0fccf0db6ac5002999fcec37. Mutual information between colluders' actions.
- Bonjour et al., *Information theoretic approach to detect collusion in multi-agent games* (UAI 2022): https://proceedings.mlr.press/v180/bonjour22a.html
- Chip dumping / soft play practitioner descriptions: https://www.pokercode.com/blog/chip-dumping, https://pokercoaching.com/blog/chip-dumping/, https://pokercoaching.com/blog/cheating-in-poker/, https://sumsub.com/blog/chip-dumping-guide/
- Preflop equity checkpoints: https://www.thepokerbank.com/strategy/mathematics/equity/, https://www.888poker.com/magazine/strategy/poker-equity-charts, https://cryptopokerdb.com/tools/preflop-odds-chart
