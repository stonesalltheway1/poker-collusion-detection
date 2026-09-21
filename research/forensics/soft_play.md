# Evidence-hand forensics: `soft_play`

Written 2026-09-15. Script: `scripts/forensics_soft_play.py` (you can re-run it: `python scripts/forensics_soft_play.py all`, which takes about 70 s,
or run a single stage: `load|actions|features|llr|analyze|archetypes|print <family> <n> <seed>`).
Working data goes to `data/derived/forensics_sp/`. Outputs go to `research/forensics/`: `sp_signatures.csv`, `sp_pair_auc.csv`,
`sp_evidence_ranking.csv`, `sp_action_shares.csv`, `sp_action_llr_table.parquet`, `sp_analysis_output.txt` (the full printed log) and `sp_archetypes.txt`.

Study set (dev phase only): 132 SP positive pairs (15,699 shared hands, 632 evidence hands); 148 DT and 92 CI positives;
1,488 confirmed negatives (178,620 shared hands); 3,000 random unknown dev pairs (311,091 shared hands; 2,677 of them
contain no labelled-positive player, called "UNKclean"). All rates below are fractions of pair-hands (a hand where both
players of the pair are seated).

Definitions used everywhere:
- **A/B** = player_1/player_2. The IDs are string-sorted, so the order carries no meaning.
- **partner aggression** = the last bet, raise or aggressive all-in (amount > to_call) on the current street was made by the
  partner.
- **hs** = the decider's made-hand strength postflop: the chance of beating one random hand on the current board, with no
  runout (the decider can compute this). **pre_eq** = preflop class equity against one random hand (MC, 169 classes).
- **pact** = partner not folded. **HU** = only A and B are still active.

---------------------------------------------------------------------------------------------------------------------
## 0. TL;DR (the mechanism)

A planted soft_play hand has two parts.

1. **Forced joint participation.** Neither partner folds as their first preflop action. This holds in **100%** of the
   632 SP evidence hands; in confirmed-negative shared hands it holds only 21.9%. In **97.6%** of evidence hands *both*
   partners enter voluntarily (limp, call, raise, or check the big blind), against **10.0%** in negatives. In 93% of
   evidence hands both see the flop (negatives: 5.7%).
2. **A direct confrontation, answered passively.** In 98.7% of evidence hands one partner bets or raises and the other
   has to respond. The response is almost never a raise: 1.8% of evidence hands contain one, against 10.1% of
   negative hands that contain a confrontation. The response is a call or a fold *regardless of hand strength*.
   Separately, when a partner is still active and nobody has bet, strong hands are checked instead of bet.

The visible "behavior-specific action" is therefore one of these:
- **call instead of raise** against partner aggression (preflop flat of the partner's raise; postflop call with
  two pair or better),
- **fold to partner aggression** (postflop fold even with top pair or two pair; preflop fold of QQ, KQ or 77 after
  limping),
- **check a strong hand** (hs ≥ 0.8) while the partner is active. This often becomes a heads-up check-down to showdown
  with an overpair, trips or a flush.

The roles are **symmetric**. There is no fixed soft or donor player: the passive responder is player_1 in 51.6% of
cases, and within-pair role concentration matches a 50/50 binomial (0.668 observed vs 0.668 expected). Chips flow
both ways (mean net per evidence hand: player_1 +0.1 chips, player_2 +5.0 chips).

Softness is a **property of the pair across its whole shared history**, not only of the listed hands:
- SP pairs have about 16.6 "planted-like" hands per ~119 dev shared hands. The negative base rate would give 7.1, so
  there are about 9.4 excess hands, of which 4.5 are listed.
- The rate is flat across the timeline, with no local episodes.
- Outside the listed hands, partners almost never raise each other.

**Pair-level separability is very high.** The pair statistic `top5_llr` (mean of the pair's 5 highest hand log-LR
scores) gives these AUCs: **0.991** against confirmed negatives, **0.993** against unknown pairs, **0.923** against DT
pairs and **0.971** against CI pairs.

**Evidence retrieval is the hard part.** There are about 2× more soft-looking hands than listed ones.
- Proxy MAP@5 on dev, ranking all shared hands of a positive pair: random 0.026, best hand-crafted rule 0.436, LightGBM
  hand ranker **0.55–0.56**.
- Adding the hand's position in the pair's timeline raises this to **0.638**, because listed evidence is skewed early.
  Whether that carries over to eval is unknown; see §6.

---------------------------------------------------------------------------------------------------------------------
## 1. What reading the hands showed

I read 46 evidence hands in full (14 + 16 random, plus 16 hands not covered by the first rule set), 15 random
non-evidence shared hands and 9 top-scored non-evidence hands (`sp_print` output, reproducible with `python scripts/forensics_soft_play.py print soft_play 16 1`).
The anomalies repeat.

- **Heads-up check-down with a strong hand after a preflop raise and call between the partners.** Examples:
  - KK checks flop, turn and river against the partner's AJ (H067FA4AFCA9CEF);
  - 77 overpair checks down (H0FC60CC12E0876);
  - QQ makes a full house and checks the river (H1B260EB0BEC266);
  - a flush checks the river (H95435CE4D77C25).

  The non-evidence hands of the same pairs also include limped heads-up check-downs with weak hands. Those are *not*
  listed.
- **Fold to the partner's bet with a real hand.**
  - Postflop: two pair K2 folds to the partner's 106-chip turn bet made with J9-high (H13B5DD9B5BB68A); top pair K
    folds to the partner's flop bet (HADB53959522807); top pair J folds (H6299889B8B6E74, H2A9E7F2EC13182); Q9 folds a
    pair of nines on the river to the partner's pair of fives (H49C36A1B5E4160).
  - Preflop: QQ folds to the partner's open raise (H5BFF709DC36AFA); a limped KQo or 77 folds to the partner's raise
    (H3D455754DF3D23, H343F2FA8DA7649).

  Many folds involve junk (T9 high, A6 high). The fold itself, not the strength, is what is planted.
- **Call instead of raise against the partner.**
  - Trips 7s just call the partner's turn raise, then the river is checked heads-up (HC87612371A8222).
  - Top pair calls flop and turn, then checks trips on the river (HA532C1A81DF5BC).
  - Two pair calls bets on the turn and river (H81CE4DE4D41F8F).
  - A straight just calls on the river (H273472AA2E43A3).
- **Both partners enter every planted hand.** Limps with junk such as T6o and cold-calls of other players'
  raises are the preflop part of nearly every evidence hand.
- **Multiway hands where both partners check and call a third player's bets with strong hands and never raise** (QQ
  and JJ against AA, H42B38D83B0FB5F) score very high but are *unlisted*. These are probably planted without a listed
  "visible action" or simply not chosen for the list.
- **Nothing unusual in the chip flow.** Soft play is about avoiding pots against each other, not dumping chips.

---------------------------------------------------------------------------------------------------------------------
## 2. Action-level conditional behaviour (SP evidence hands vs confirmed-negative shared hands)

Hand strength is binned into fifths. Numbers are action shares within each context and strength bin (full table:
`sp_action_shares.csv`; preflop bins are (pre_eq − 0.3)/0.1 clipped to 0..4, so the top bin is pre_eq ≥ 0.7; the LLR version is in
`sp_action_llr_table.parquet`).

| context | strength bin | NEG | SP evidence | SP non-evidence |
|---|---|---|---|---|
| postflop, partner active, nobody has bet: **bet/raise rate** | hs 0.8–1.0 | 75.7% | **15.2%** | 22.7% |
| same | hs 0.6–0.8 | 57.6% | 28.9% | 28.9% |
| postflop, facing **partner** bet: **raise** | hs 0.8–1.0 | 27.5% | **1.4%** | 8.0% |
| same: **fold** | hs 0.8–1.0 | 18.6% | **47.6%** | 18.5% |
| same: **fold** | hs 0.6–0.8 | 35.9% | 52.7% | 47.0% |
| postflop, facing **other** bet (partner active): fold | hs 0.8–1.0 | 19.9% | 10.3% | 19.5% |
| preflop, facing **partner** raise: **3-bet** | pre_eq ≥ 0.7 | 73.0% | **4.7%** | 40.4% |
| preflop, facing partner raise: call | pre_eq 0.6–0.7 | 34% | **87%** | 62% |
| preflop, no raise yet: **fold** | pre_eq 0.4–0.5 | 78% | **0%** | 67% |
| preflop, no raise yet: fold | pre_eq 0.5–0.6 | 52% | **0%** | 32% |
| preflop, facing **other** raise: fold | pre_eq 0.4–0.5 | 89% | **2%** | 78% |

The most anomalous partner action in each evidence hand (highest out-of-fold log-LR) is:
- preflop limp or call with partner active: 247 hands;
- turn or river check with partner active: 137 hands, 135 of them with hs ≥ 0.8 (plus 7 flop checks, all with hs ≥ 0.8);
- preflop call of the partner's raise: 99 hands;
- preflop cold-call of another player's raise: 58 hands.

---------------------------------------------------------------------------------------------------------------------
## 3. Candidate signatures (measured; `sp_signatures.csv`)

Columns: SP_ev = coverage of the 632 SP evidence hands; SP_non = non-evidence shared hands of the SP pairs;
NEG = confirmed-negative shared hands; UNK = unknown-pair shared hands; DT_ev and CI_ev = evidence hands of the other
families.

| signature | exact definition | SP_ev | SP_non | NEG | UNK | DT_ev | CI_ev |
|---|---|---|---|---|---|---|---|
| S_G0_no_first_fold | no partner's first preflop action is a fold, unless that fold faces partner aggression | **1.000** | 0.263 | 0.219 | 0.222 | 1.000 | 0.935 |
| S_G1_both_enter | both partners' first preflop action is call, raise or check | **0.976** | 0.204 | 0.100 | 0.102 | 0.949 | 0.496 |
| S_both_see_flop | a postflop street starts with both partners active | 0.930 | 0.150 | 0.057 | 0.059 | 0.891 | 0.109 |
| S_faced_partner_aggr | at least one partner action taken while facing the partner's bet or raise | 0.987 | 0.170 | 0.199 | 0.201 | 1.000 | 0.920 |
| **S_PL** ("planted-like") | G1 AND faced_partner_aggr AND no raise against partner anywhere in the hand | **0.946** | 0.105 | **0.060** | 0.062 | 0.785 | 0.117 |
| S_strong_check_pact | postflop check (to_call = 0) with partner active and hs ≥ 0.8 | 0.459 | 0.045 | **0.0079** | 0.0081 | 0.172 | 0.013 |
| S_fold_to_partner_post | postflop fold facing partner aggression | 0.505 | 0.056 | 0.034 | 0.035 | 0.646 | 0.078 |
| S_fold_to_partner_post_hs60 | same, with hs ≥ 0.6 | 0.316 | 0.010 | 0.0081 | 0.0083 | 0.484 | 0.017 |
| S_call_partner_post_hs80 | postflop call of partner aggression with hs ≥ 0.8 | 0.206 | 0.013 | 0.0073 | 0.0073 | 0.092 | 0.007 |
| S_call_partner_pre_eq55 | preflop call of partner raise with pre_eq ≥ 0.55 | 0.381 | 0.020 | 0.024 | 0.025 | 0.193 | 0.078 |
| S_fold_to_partner_pre | preflop fold facing partner raise | 0.049 | 0.078 | 0.145 | 0.145 | 0.094 | **0.735** |
| S_hu_checked_street | a postflop street starts heads-up between the partners and nobody bets | 0.331 | 0.025 | 0.0089 | 0.0094 | 0.108 | 0.024 |
| S_missed_raise30 | passive response (call or fold) to partner aggression where NEG would raise with P ≥ 0.3 in the same context | 0.411 | 0.021 | 0.025 | 0.025 | 0.225 | 0.072 |
| S_U1 | strong_check_pact OR fold_to_partner_post OR call_partner_post_hs80 | 0.820 | 0.096 | 0.041 | 0.043 | 0.724 | 0.087 |
| S_PL_and_U1 | S_PL AND S_U1 | 0.794 | 0.076 | 0.033 | 0.034 | 0.619 | 0.043 |
| **S_PL_and_softresp** | S_PL AND (U1 OR missed_raise30 OR call_partner_pre_eq55) | **0.881** | 0.081 | **0.039** | 0.041 | 0.663 | 0.059 |
| S_llr_gt2 | out-of-fold hand log-LR (sum over partner actions) > 2 | 0.756 | 0.120 | 0.021 | 0.021 | 0.517 | 0.139 |
| S_llr_gt4 | same, > 4 | 0.495 | 0.054 | **0.0050** | 0.0047 | 0.190 | 0.033 |

Hand log-LR (`llr_sum`) is a smoothed count model. For each partner action it computes
log P(action | street, facing {none, partner, other}, partner active, HU, strength bin, already-invested preflop)
under SP evidence, minus the same under NEG. It backs off to the context without HU and strength. It is fitted
out-of-fold on the frozen table folds.
- Hand AUC for SP_ev vs NEG: **0.980**. For SP_ev vs SP_non: 0.922.
- `max_miss` (the largest NEG probability of raising that the responder passed up) has AUC 0.957 and 0.959 on the same
  two comparisons.

Family contrast at hand level:
- **DT** evidence shares the forced-participation gate (G0 100%, G1 94.9%) and the confrontation (100%). The
  difference is that DT raises the partner in 16.4% of hands (SP: 1.7%) and folds or calls more. It checks strong
  hands far less (17% vs 46%).
- **CI** evidence is a different shape: a preflop fold to the partner's raise in 73.5% of hands, and a raise against
  the partner in 36%.
- The best single SP-vs-DT discriminators are `strong_check_pact` and "raise against partner when both are in the pot".

---------------------------------------------------------------------------------------------------------------------
## 4. Temporal pattern

- **Where evidence sits in each pair's dev shared-hand timeline (deciles):** 95, 102, 86, 69, 69, 54, 43, 42, 41, 31.
  The mean position is 0.393, so listed evidence is skewed **early** (linear decline). By dev hand_seq deciles:
  78, 98, 87, 73, 66, 61, 37, 52, 42, 38.
- **Listing order:** evidence_rank is only partly chronological (mean Spearman 0.37). DT is similar (0.44). CI is
  almost chronological (0.91) and even more front-loaded.
- **The planted-like rate is flat over time.** For S_PL across all SP shared hands (evidence included), the quintile
  rates are 0.142, 0.143, 0.145, 0.133 and 0.132; NEG sits at 0.059–0.061. So planted hands are spread evenly and the
  early skew comes from how the list is chosen.
  - In 5-listed pairs, high-scoring *unlisted* hands mostly lie between or after the listed ones: 25 before, 220
    between and 184 after, where uniform placement would give about 130 after.
  - So the list is not strictly "the first 5 planted"; it is a sample biased toward earlier hands.
- **No local episodes.** Among non-evidence hands, S_PL is flat against hand_seq distance to the nearest evidence hand:
  0.100 at 1–10, 0.113 at 11–30, 0.106 at 31–100, 0.101 at 101–300, 0.117 at 301–1000 and 0.101 at 1001–3000.
  Visible "clusters" in hand_seq are just the sessions in which the two players sit together.
- **Consistency across halves:** the split-half S_PL rate (evidence excluded) correlates at 0.28 across SP pairs.
  Rates are 0.093 in the first half and 0.112 in the second (the second is higher because the removed evidence is
  early). Over the last 40% of the timeline, 21% of SP pairs look inactive (rate ≤ NEG + 1 point); for NEG pairs this is 67%.
  Episodic activation happens, but most pairs stay active.
- **Planted density:** about 9.4 excess planted-like hands per SP pair per 119 dev shared hands (~8%). For eval pairs
  with 38–419 shared hands, expect roughly 3–35 planted hands.

---------------------------------------------------------------------------------------------------------------------
## 5. Pair-level separability (`sp_pair_auc.csv`)

Rates are per shared hand. The z-score is (count − NEG rate × n) / sqrt(NEG rate × (1 − NEG rate) × n).

| pair statistic | SP vs NEG | SP vs UNK | SP vs UNKclean | SP vs DT | SP vs CI | DT vs NEG | CI vs NEG |
|---|---|---|---|---|---|---|---|
| n shared hands (confounder) | 0.499 | 0.596 | 0.601 | 0.467 | 0.455 | 0.531 | 0.540 |
| **top5_llr** (mean of the 5 largest hand llr_sum) | **0.991** | **0.993** | **0.993** | 0.923 | 0.971 | 0.924 | 0.777 |
| z_S_llr_gt2 | 0.984 | 0.987 | 0.986 | 0.878 | 0.965 | 0.907 | 0.758 |
| r_S_strong_check_pact | 0.983 | 0.978 | 0.978 | **0.943** | 0.978 | 0.738 | 0.534 |
| z_S_both_see_flop | 0.952 | 0.958 | 0.956 | 0.660 | 0.924 | 0.910 | 0.576 |
| mean_llr | 0.952 | 0.950 | 0.949 | 0.832 | 0.889 | 0.816 | 0.746 |
| z_S_U1 | 0.943 | 0.947 | 0.946 | 0.712 | 0.886 | 0.868 | 0.580 |
| −(raises vs partner)/(both-in-pot hands) | 0.936 | 0.927 | 0.926 | **0.952** | **1.000** | 0.553 | 0.043 |
| z_S_G1_both_enter | 0.925 | 0.936 | 0.934 | 0.664 | 0.728 | 0.866 | 0.822 |
| z_S_PL | 0.903 | 0.912 | 0.909 | 0.487 | 0.844 | 0.899 | 0.591 |
| −raise-response rate to partner aggression | 0.832 | 0.831 | 0.828 | 0.837 | 0.994 | 0.558 | 0.067 |

- **Shared-hand confound is small.** AUC vs NEG within shared-hand tertiles:
  - ≤91 hands: top5_llr 0.988, z_S_PL 0.905;
  - 91–128 hands: top5_llr 0.998, z_S_PL 0.877;
  - >128 hands: top5_llr 0.998, z_S_PL 0.924.

  Unknown pairs share fewer hands (103 vs 119), so raw counts would be biased; use rates or z-scores.
- **Average precision.** top5_llr has AP 0.932 for SP vs UNKclean (prevalence 4.7%). At a simulated 1% prevalence it
  is 0.833. For all three positive families against UNKclean, AUC is 0.927 and AP 0.730.
  - UNKclean holds hidden positives, so these are lower bounds.
  - 28 of 2,677 UNKclean pairs (1.0%) and 21 of 1,488 NEG pairs (1.4%) score above the 10th percentile of SP pairs.

---------------------------------------------------------------------------------------------------------------------
## 6. Recommendations

### Evidence ranking (evidence MAP@5)

Proxy measurement: rank *all* dev shared hands of each SP pair and score against the listed evidence
(`sp_evidence_ranking.csv`). This proxy understates true performance if the private set counts every planted hand.

| recipe | proxy MAP@5 | with −3·timeline position |
|---|---|---|
| random | 0.026 | 0.031 |
| S_PL binary | 0.213 | 0.282 |
| llr_sum | 0.261 | 0.271 |
| RULE = 10·S_PL + 4·max_miss + 1.5·strong_check + n_call_responses_to_partner + 0.3·llr_post | 0.436 | 0.479 |
| LightGBM hand ranker trained in-pair (listed vs unlisted hands of SP pairs), OOF table folds | 0.551 | 0.561 |
| LightGBM trained on all 3 families' positive pairs | **0.558** | 0.566 |
| LightGBM within SP + timeline position q as a feature | **0.638** | – |
| LightGBM "SP evidence vs NEG hands" detector | 0.409 | 0.384 |

Recommended recipe for the eval pairs you predict:
1. **Gate.** Drop hands where either partner's first preflop action is a fold (G0 = false) unless it is a fold to the
   partner's raise. No listed evidence hand is lost; 78% of ordinary hands are removed.
2. **Score** the remaining eval shared hands with a LightGBM hand ranker trained on dev positive pairs' shared hands,
   with target = listed evidence. Train it across all families, and optionally add a family one-hot for the predicted
   family. Use the per-hand features in `phf.parquet` / `sig.parquet`:
   - counts of fold, call and raise responses to partner aggression, pre- and postflop;
   - max_miss and max_fsur;
   - check_pact_hs, fold_vs_p_post_hs/eq, call_vs_p_post_hs, call_vs_p_pre_str;
   - n_both_streets, n_hu_streets, n_hu_aggr_streets;
   - llr_pre, llr_post, llr_sum;
   - G0, G1, preA, preB.

   **Train the ranker in-pair** (listed vs unlisted hands of positive pairs), not on planted vs negative hands. SP pairs
   are soft in many unlisted hands, and the in-pair objective learns what makes a hand *listable*: a direct
   confrontation answered passively with a strong hand.
3. **Fallback** if there is no model: the RULE score above (0.44).
4. **Timeline-position prior.** Listed dev evidence is skewed early (+0.08 MAP on the dev proxy). For eval this is a
   *hypothesis*: the private evidence may or may not be sampled the same way within the eval period. Check it on the
   public LB with two otherwise identical submissions, one ranker with q and one without. The evidence component is
   20% of the score, and the difference should be clearly measurable when pair AP is high.
5. Submit 5 hands, always. Order matters, so put the highest score first.

### Pair-level features (for the risk model)

- **top5_llr** is the single strongest statistic. Refit the action-LLR table on all dev labels and apply it to eval
  actions (`fit_llr` in the script; contexts are phase-independent).
- Also useful: `z_S_llr_gt2`, rate or z of `S_strong_check_pact`, z of `S_both_see_flop`, z of `S_G1_both_enter`, z of
  `S_U1`, z of `S_PL`, and the mean of the top-5 `max_miss`.
- **Softness contrasts** (these also separate families): raises against the partner per both-in-pot hand (SP ≈ 0;
  DT and CI high), raise-response rate to partner aggression, and call-response rate.
- Family classification: SP vs DT is best separated by strong_check rate (0.94) and raises against partner per
  both-in-pot hand (0.95). SP vs CI is separated by raises against partner (1.00) and preflop folds to the partner's
  raise (CI 73%).
- **Use z-scores or rates, not counts.** Shared-hand counts differ between labelled and unknown pairs.

---------------------------------------------------------------------------------------------------------------------
## 7. Archetypal evidence hands (full prints; A = player_1, B = player_2, oN = others)

Column key: stack = starting stack; contrib = total chips put in; net = net chips; won = pot share; final = made hand on
the full board. Action lines: amt = chips added; to = street total after the action; tocall = amount facing the player.

**(a) Canonical "Pc-Ff": the partner raises preflop, the other flats, then folds to the partner's c-bet.** This pattern
(preflop call of the partner's raise, then flop fold to the partner) is the single most frequent response sequence.
The counts are: Pc alone 104, Ff alone 103, Pc-Ff 75, Pc-Tf 40, Tc 36, Pf 30, Tf 28.
```
--- hand HCDF38FB6CD488F seq=1166 [EVIDENCE SP pair=PE60E87200AA1 rank=5] blinds 5/10 board=6s 7c Qd final_pot=80
  seat4   B 9hTc stack=925  contrib=20 net=-20 fold=1
  seat5   A KcJs stack=1421 contrib=50 net=+30 won=1.00
  [PRE]  o2 fold, o3 fold, o4 fold, B call 10, A raise to 20, o1 fold, B call 10
  [FLOP] 6s 7c Qd   A bet 30 (pot 50)   B fold
```

**(b) Fold of two pair to the partner's bet with junk.**
```
--- hand H13B5DD9B5BB68A seq=2522 [EVIDENCE SP pair=PA4A4744909BA rank=2] blinds 5/10 board=7c 2h Ks 5c final_pot=406 sd=0
  seat0  o1 QcKc stack=  789 contrib=   30 net=   -30 fold=1
  seat2  o3 KhAh stack= 1096 contrib=   90 net=   -90 fold=1
  seat4   A 2dKd stack= 1288 contrib=   90 net=   -90 fold=1   (two pair Ks+2s on the turn)
  seat5   B 9cJd stack=  979 contrib=  196 net=   210 won=1.00 (no pair)
  [PRE]  o2 fold | o3 call 10 | o4 fold | A call 10 | B call 5 (SB) | o1 raise to 30 | o3 raise to 90 | A call 80 | B call 80 | o1 fold
  [FLOP] 7c 2h Ks   B check | o3 check | A check          <- A checks two pair with partner active
  [TURN] 7c 2h Ks 5c  B bet 106 (pot 300) | o3 fold | A fold (two pair, facing partner)   max_miss=0.32 llr_sum=6.45
```

**(c) Heads-up check-down with an overpair after the partner flats the raise.**
```
--- hand H067FA4AFCA9CEF seq=1303 [EVIDENCE SP pair=PA357524DE19A rank=4] blinds 1/2 board=7d 6s 5c 9s 9c final_pot=11 sd=2
  seat2   B KdKh stack=182 contrib=4 net=+7 won=1.00 final=2pair
  seat4   A JhAd stack=147 contrib=4 net=-4 final=pair
  [PRE]  o2 fold | B raise to 4 | o3 fold | A call 4 | o4 fold | o1 fold
  [FLOP] 7d 6s 5c      B check | A check      (KK overpair checks, HU vs partner)
  [TURN] 7d 6s 5c 9s   B check | A check
  [RIVER] ... 9c       B check | A check      llr_sum=3.44
```

**(d) Trips flat-call the partner's raise, then the river is checked heads-up (call instead of raise, and no river
value bet).**
```
--- hand HC87612371A8222 seq=2345 [EVIDENCE SP pair=P049F7CE27361 rank=3] blinds 2/4 board=Ts Jd 7c 7s 6h final_pot=298 sd=2
  seat2  o3 Qc4c contrib=34 net=-34 fold=1
  seat3   B Td8h stack=634 contrib=130 net=-130 final=2pair
  seat4   A 7h8c stack=475 contrib=130 net=+168 won=1.00 final=trips
  [PRE]  o1 fold | o2 fold | o3 raise to 10 | B call 10 | A call 8 | o4 fold
  [FLOP] Ts Jd 7c      A check | o3 check | B check
  [TURN] Ts Jd 7c 7s   A check | o3 bet 24 | B raise to 120 | A call 120 (trips vs partner raise) | o3 fold
  [RIVER] ... 6h       A check (trips) | B check          max_miss=0.38 llr_sum=11.10
```

**(e) Preflop: QQ folds to the partner's open raise.** This is rare (4.9% of evidence hands contain a preflop fold to
the partner) but highly anomalous.
```
--- hand H5BFF709DC36AFA seq=1103 [EVIDENCE SP pair=P375F7CE38E9E rank=3] blinds 2/4 final_pot=20
  seat0   B TdJc stack=447 contrib=10 net=+10 won=1.00
  seat1   A QhQc stack=527 contrib=0  net=0  fold=1
  [PRE]  o3 call 4 | o4 fold | B raise to 10 | A fold (QQ!) | o1 fold | o2 fold | o3 fold     max_miss=0.88
```

The full machine prints of all six archetypes (the five above plus HA532C1A81DF5BC, which is top pair calling and then
trips checking the river) are in `research/forensics/sp_archetypes.txt`.

---------------------------------------------------------------------------------------------------------------------
## 8. Open questions / what would overturn these conclusions

- **Is the private eval evidence set "up to 5 listed" (sampled early) or "all planted hands"?** This decides whether
  the timeline prior helps (§6.4). Test it on the LB.
- **The hs thresholds are my proxy for the generator's strength variable.** `max_miss` (a NEG-calibrated raise
  probability) works better than any fixed hs threshold. The generator may use equity against a random range; the
  eq_omni columns are available to test this.
- **About 5% of SP evidence hands fall outside S_PL.** Examples: preflop-only sequences where one partner limps and
  folds to the partner's raise, or both fold to a third player's 3-bet after a raise and flat between the partners.
  They are still covered by G0 (100%).
- **SP-like hard negatives.** 1.4% of confirmed negatives look SP-like at pair level. These are probably the "similar
  strategies" or "weak play" hard negatives. Confirm by reading their top hands before trusting a pure top5_llr risk
  score.
- **Other_coordination (hidden) is not examined here.** A risk model built on "forced participation + passive
  confrontation" may also catch it, which would be good for pair AP.
