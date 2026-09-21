# LITERATURE_poker.md: collusion detection in poker and multiplayer games

Written 2026-09-15 for the Kaggle competition "Detect Suspicious Value Transfers in Poker".
Primary sources are in `research/papers/`. Plain-text extractions are in `research/papers/txt/`
(patents are scanned images, so they have no text; their summaries below come from the Google Patents full text).
Papers named `ml_*` in the same folder were downloaded by another agent (PU learning, scan statistics,
CopyCatch, FRAUDAR v2, LTR). They are only cross-referenced here.

---------------------------------------------------------------------------------------------------

## 0. TL;DR: what the literature says that matters for this competition

1. **Score what a pair of actions does to utility, not how much money was won.** Every serious academic
   method removes chance from the outcome first: Mazrooei/Archibald/Bowling (AAAI 2013) with
   value-function "collusion tables", Billings & Kan with DIVAT, Zinkevich 2006 with advantage-sum estimators,
   and Burch 2018 with AIVAT. What is left is each player's action-caused change in every other player's value.
   Money screens fail. The best agents usually do not collude, weak colluders still lose money overall,
   and cheats' payoff distributions overlap honest players' (Mazrooei Table 3; Kocsis & György Fig. 1).
2. **Four pair scores from collusion tables are directly usable.** Total Impact (the pair's joint self-benefit),
   Marginal Impact (a player helps the partner more than they help the average third party), Minimum Impact
   (both partners must contribute; this suppresses one-sided "accidental" pairs) and Differential TI.
   The thesis recommends combining them. The "always-call" value function failed and a CFR-quality value
   function worked. Our analogue is all-in equity (AIE) with known hole cards, or a rollout policy.
3. **Collusion = the action depends on the partner's hidden state.** IET patent US7604541 learns a Bayes-net
   edge "opponent hand rank -> player action". Bonjour et al. (UAI 2022) use conditional mutual information
   I(a_i ; a_j | s_i, s_j) minus the maximum over third parties ("net influence"). Yan (AAAI 2010) uses
   "action is justified by the pair's combined hand but not by own hand". We see every hole card, including
   folded ones, so all of these can be computed exactly. Johansson et al. (2003) called passive collusion
   "next to impossible" to tell from cautious play. That was without hole-card visibility, and we have it.
4. **Episodic collusion dilutes whole-history averages.** Bonjour: with collusion probability CP=0.1, MI
   detection stays near chance even with 10k games. CP=0.2 needs about 10k games for 82% detection
   (1000 games gives 24%). IET patent: about 500-1000 games are needed at intensities of 0.25-0.75.
   Laasonen & Smed (2015): soft-play detection collapses when colluders still attack each other half the time.
   Our pairs share 38-419 eval hands, with about 3-5+ planted hands. **So use top-k or scan statistics
   over per-hand suspicion, not means.** HoloScope uses bursts, CopyCatch lockstep windows, and Smed's
   taxonomy calls these "opportunistic" agreements. The same per-hand scores become the evidence ranking
   that feeds the 20% MAP@5 part of the metric.
5. **The operators' own definitions match the host's families.** CFPH patent US9652931 (spec shared by
   US10699523, US11610453 and US12100265):
   - Chip dumping: "transfer of chips ... when the player should not have lost the game (based on the cards
     in his or her hands)".
   - Soft play: "plays in a non-aggressive manner when he should play in an aggressive manner (e.g., based on
     having good or the best cards)".
   - Other listed patterns: "lack of raising against a particular player" and "combining betting patterns with
     a teammate to squeeze out another player".

   Both chip dumping and soft play are judged against the *player's own style profile* in the same context
   (hand strength, position, street, opponent type). PokerStars reviews with all cards visible and names
   "best hand playing", soft play and "chip dumping/stack balancing". Wikipedia's taxonomy adds whipsawing
   (raise/re-raise sandwich) and signalling.
6. **Hard negatives need a player-specific, partner-vs-others baseline.** Mazrooei's "accidental colluders"
   (positional strategies that always help the left or right neighbour) rank at the top of a naive table.
   The CFPH patent warns that an inexperienced player's misfold "could look like collusion". IGT US8360838
   uses each player's "characteristic range of deviation" plus "other players consistently win during a
   player's lapse times". Waterleaf US7883412 flags "an inordinate percentage" of a player's winnings or losses
   against one opponent, so normalize by shared hands. The discriminating contrast is always
   (behaviour vs partner) minus (the same player's behaviour vs third parties in the same equity bucket).
7. **Supervised training can miss the 4th undisclosed mechanism.** The generic, model-free scores (collusion-table
   TI/MaI/MiI and MI net influence) exist precisely to catch "new forms of collusion [that] go undetected" by
   pattern-specific detectors (Mazrooei §1; Greige 2022 switches to unsupervised Isolation Forest for the same
   reason). Kocsis & György (2010) is the PU recipe: per-state deviation-from-optimal frequency plus excess
   reward as features, synthetic positives vs unlabeled. A classifier trained on one cheat type still
   caught an unseen type.
8. No 2024-2026 paper on LLM or agent collusion is poker-specific (checked arXiv). COLOSSEUM (2026) measures collusion
   as *regret vs the cooperative optimum*, and NARCBench (2026) as *distributed anomaly types needing
   different detectors*. Both are conceptually consistent with the above and not directly usable.

---------------------------------------------------------------------------------------------------

## 1. Source index (files in `research/papers/`)

| File | Type | Relevance |
|---|---|---|
| Mazrooei_Archibald_Bowling_2013_Automating_Collusion_Detection_Sequential_Games_AAAI.pdf | AAAI paper | ***** core method (collusion tables) |
| Mazrooei_2012_MSc_Thesis_Collusion_Detection_Sequential_Games_UAlberta.pdf | MSc thesis | ***** 5 scores, value functions, PIVAT, what failed |
| Bonjour_etal_2022_Information_Theoretic_Collusion_Detection_MultiAgent_Games_UAI.pdf | UAI paper | ***** MI / net-influence; detection floor vs collusion probability |
| Patent_US9652931_CFPH_2013_..._chipdump_softplay.pdf (+ US10699523, US11610453, US12100265 same family) | patent | ***** operator-style definitions, style-deviation scoring, severity ranking |
| Patent_US7604541_IET_2006_..._BayesNet.pdf | patent | **** action conditional on opponent hole cards |
| Patent_US7883412_Waterleaf_2003_Collusion_Detection_and_Control.pdf | patent | *** win/loss concentration vs one opponent, change flags |
| Patent_US7618321_PokerTek_2005_Detecting_Collusion_Between_Poker_Players.pdf | patent | ** triggers ("unusual action given hand"), co-seating |
| Patent_US8360838_IGT_2006_Detecting_Bots_and_Cheating_Online_Gaming.pdf | patent | *** lapse-time/beneficiary coincidence |
| Patent_US11717757_EA_2021_... / Patent_US12115458_EA_... | patents | *** pairwise features + Isolation Forest |
| Greige_etal_2022_Collusion_Detection_Team_Based_Multiplayer_Games.pdf | CEUR/arXiv 2203.05121 | *** unsupervised pairwise anomaly scoring (EA) |
| Billings_Kan_2006_DIVAT_Direct_Assessment_Poker_Decisions.pdf | ICGA J. | ***** luck removal; AIE/ROE; why hindsight EVAT fails |
| Zinkevich_etal_2006_Optimal_Unbiased_Estimators_Agent_Performance_AAAI.pdf | AAAI | **** advantage-sum estimators (theory behind DIVAT) |
| Burch_etal_2018_AIVAT_Variance_Reduction_Agent_Evaluation.pdf | AAAI | **** control variates for chance *and* actions |
| Davidson_Archibald_Bowling_2013_Baseline_Control_Variates_Agent_Evaluation_AAMAS.pdf | AAMAS | *** baseline-agent control variate (3-player poker) |
| Kocsis_Gyorgy_2010_Fraud_Detection_Generating_Positive_Samples_Unlabeled_poker.pdf | ICML-W | **** PU fraud detection in games; deviation-frequency features |
| Smed_Knuutila_Hakonen_2007_Towards_Swift_Accurate_Collusion_Detection_GAMEON.pdf | GAMEON | *** taxonomy (express/tacit, total/partial, enduring/opportunistic), accuracy vs swiftness |
| Laasonen_Smed_2015_Soft_Play_Detection_Shooter_Games_Hit_Matrix_INTETAIN.pdf | INTETAIN | **** soft play as a directed "hit matrix"; 6 pair scores; failure modes |
| Yan_2010_Collusion_Detection_Online_Bridge_AAAI.pdf | AAAI | *** "action justified by combined hands, not own hand"; critical decision points |
| Zehnder_2012_ETH_BA_Thesis_Collusion_in_Online_Poker_Pays_Off.pdf | BA thesis | ** real colluding 6-max NLHE bots (passive info sharing); undetected |
| Johansson_Sonstrod_2009_Fish_or_Shark_Data_Mining_Online_Poker.pdf | IEEE CIG | ** player-profile stats (VPIP/PFR/AF) as style baseline |
| Teofilo_Reis_2011_Identifying_Player_Strategies_NLHE_arXiv1301.5943.pdf | EPIA | ** clustering moves into player types (style baseline) |
| Farina_Celli_Gatti_Sandholm_2018_Ex_Ante_Coordination_Collusion_Multiplayer_EFG_NeurIPS.pdf | NeurIPS | ** team-maxmin w/ coordination (theory of optimal collusion) |
| Farina_etal_2020_Faster_Algorithms_Ex_Ante_Coordinated_Collusive_Strategies.pdf | arXiv | ** same line |
| Carminati_Cacciamani_Gatti_2022_Adversarial_Team_Games_Kuhn_Leduc_ICML.pdf | ICML | ** team-public-information representation |
| Xu_Feng_Fang_2024_Deviate_or_Not_Coalition_Structure_Learning_arXiv2412.10636.pdf | arXiv | * active coalition learning (not applicable: we cannot design games) |
| Hooi_etal_2016_FRAUDAR_Bounding_Graph_Fraud_Camouflage_KDD.pdf (also ml_graph_fraudar_hooi_2016.pdf) | KDD | ** camouflage-resistant dense subgraph |
| Liu_Hooi_Faloutsos_2017_HoloScope_Topology_Spike_Fraud_Detection_CIKM.pdf | CIKM | *** bursts/drops; contrast suspiciousness |
| Jiang_etal_2014_CatchSync_KDD_SLIDES.pdf | KDD slides | ** synchronicity + normality |
| Nakamura_etal_2026_COLOSSEUM_Auditing_LLM_Agent_Collusion_arXiv2602.15198.pdf | arXiv 2026 | * LLM agent collusion (regret-based) |
| Rose_etal_2026_NARCBench_Detecting_MultiAgent_Collusion_Interpretability_arXiv2604.01151.pdf | arXiv 2026 | * LLM collusion probes |

Sources that could not be downloaded (paywalled or ResearchGate-only), summarized from abstracts and citing papers:
- Yampolskiy, "Online poker security: problems and solutions" (GAME-ON-NA 2007).
- Yampolskiy, "Detecting and controlling cheating in online poker" (CCNC 2008).
- Yampolskiy, "Mimicry attack on strategy-based behavioral biometric" (ITNG 2008).
- Smed/Knuutila/Hakonen, "Can we prevent collusion in multiplayer online games?" (SCAI 2006).
- Laasonen/Knuutila/Smed, "Eliciting collusion features" (SIMUTools 2011).
- Johansson/Sönströd/König, "Cheating by sharing information: the doom of online poker?" (ADCOG 2003).
- Vallvè-Guionnet, "Finding colluders in card games" (ITCC 2005).
- Palshikar & Apte, "Collusion set detection using graph clustering" (DMKD 2008).
- White & Bowling, MIVAT (IJCAI 2009).
- Fiedler, "Online gambling as a game changer to money laundering?" (SSRN 2013).

---------------------------------------------------------------------------------------------------

## 2. Per-source summaries

### 2.1 Mazrooei, Archibald & Bowling (AAAI 2013) and Mazrooei MSc thesis (U. Alberta 2012)
**Setting.** Synthetic population in 3-player limit [2-4] Hold'em: 14 CFR agents (strong and weak abstractions).
- Colluders: CFR trained on a modified utility u_i + λ·u_j with λ=0.9, giving about +53 mbb/g joint advantage.
- Defenders and paranoid agents.
- "Accidental colluders": always collude with their left or right seat.

Each triple played 1M hands. The paper also reran with 100k hands.

**Method: the collusion table.** For hand g, C_g(j,k) = Σ over k's actions h→ha of [V_j(ha) − V_j(h)].
This is the change in j's expected value caused by k's actions. Chance gets its own column, so luck is excluded
by construction. The zero-sum property makes each column sum to 0. Row sum plus chance plus start value equals
realized utility.

Value functions: V is defined implicitly by a (purified, i.e. argmax) CFR strategy playing out the rest of the
hand, or by "always call". The thesis also proposes PIVAT, a linear value function learned to minimise
the variance of C, subject to a zero-luck constraint (value before chance = expectation after chance).

Tables are averaged over hands. Scores per pair (a,b):
- **Total Impact** TI = Σ_{i,j∈{a,b}} C(i,j): the pair's actions benefit the pair.
- **Marginal Impact** MaI = [C(b,a) − mean_{i∉{a,b}} C(i,a)] + [C(a,b) − mean_{j∉{a,b}} C(j,b)]: a treats b better than the average opponent.
- **Mutual Impact** MuI = C(a,b) + C(b,a).
- **Minimum Impact** MiI = min over i∈{a,b} of Σ_{j∈{a,b}} C(j,i): both must contribute, which suppresses one-sided accidental benefit.
- **Differential TI** DI = TI(a,b) − max over d∉{a,b} of max(TI(a,d), TI(b,d)): isolates the single most suspicious pair.

**Results.**
- TI and MaI put both true colluding pairs in the top 1-4 out of 91 pairs; the top 4 were all intentional or accidental colluders.
- Weak colluders who *lose money overall* were still detected.
- 95% CI ≈ ±15 mbb/g at 1M hands. With 100k hands per configuration colluders still stand out.
- The results were robust to which CFR value function was used (the strong-agent one or a coarser abstraction).

**Failures.**
- The always-call value function did not work.
- Mutual Impact failed to detect strong colluders with the CFR value function.
- Accidental (positional) colluders are indistinguishable by these scores and need human review.
- Money-won screens miss weak colluders, and the best winners are honest.
- Needs a lot of data (thousands of hands per pair) for whole-history averaging.

**Relevance here.**
- We can compute V exactly or approximately from **known hole cards**: AIE per street, or a rollout with a simple policy.
- Per-hand C_g gives per-hand evidence scores. Table aggregates give pair features.
- MaI and MiI are the right tools against the "weak play/tilt" and "repeated opponent selection" hard negatives.

### 2.2 Bonjour, Aggarwal & Bhargava (UAI 2022): information-theoretic collusion detection
Individual influence ι(i;j) = conditional mutual information Σ p(s_ij) Σ p(a_i,a_j|s_ij) log[p(a_i,a_j|s_ij)/(p(a_i|s_i)p(a_j|s_j))].
For sequential games only pairs where j acts after i are counted. The joint state includes both
private states, so in poker it includes both hole cards.

**Net influence** ν(i;j) = ι(i;j) − max_{k≠i,j} ι(k;j). Collusion is declared if both ν(i;j) and ν(j;i) exceed a threshold (0.05).

Experiments:
- 3-player iterated RPS: assistant collusion with collusion probability CP, and DDQN auto-colluders.
- 3-player Leduc Hold'em "association collusion": both raise if either partner holds an ace or pair. Also DDQN
  "high-payoff" colluders and "low-payoff" colluders trained to keep payoff near 0.

Findings:
- The approach detects collusion even when payoff is about 0, where payoff-based methods are blind.
- The detection rate (CDA) grows with the number of games. At CP=0.3: 37% after 100 games, 100% after 1000.
  At CP=0.2: 24% after 1000 games and 82% after 10k. At CP=0.1: about 8%, effectively undetectable.
- There is a **minimum collusion intensity** below which whole-history MI fails.

Relevance: this is a direct test for "other_coordination" and for isolation. Does player B's aggression depend on A's hole-card strength,
beyond B's own strength and the public state, more than it depends on any third player's cards?
Because collusion here is episodic, compute it on the top suspicious windows or hands, or weight by per-hand suspicion.

### 2.3 Kocsis & György (ICML 2010 Workshop on ML and Games): PU fraud detection
The unlabeled pool is almost all honest, and synthetic cheaters can be generated. Label U as negative and P as positive, then train a
classifier (RProp neural net) with reweighting. Player features: for each MDP state s, [1 − π̂(s, π*(s))], the
frequency of deviating from the (approximately) optimal action, and R̂(s) − E R*(s), the excess reward after the deviations.

For multi-player collusion the pair is described by deviation from the optimal *product* policy plus the resulting extra reward.

In blackjack (next-card cheats vs hole-card cheats, with disguise), a model trained only on next-card cheats
**also detected the unseen hole-card cheats**. Payoff alone could not separate cheats from honest players.

Relevance:
- Our dev labels are PU: 372 positives, 1488 hard negatives and about 156k unknown pairs.
- The features to build are per-context deviation rates plus excess value accruing to the partner.
- Other_coordination generalization suggests keeping the model's features mechanism-agnostic (deviation plus beneficiary)
  rather than purely family-specific triggers.

### 2.4 CFPH patent family "Collusion detection" (US9652931 B2, priority 2013; continuations US10699523, US11610453, US12100265)
**Style profile.** Vector per player of HUD-like dimensions: small-blind completion, aggression factor
(bet+raise)/call, VPIP, PFR, 3-bet, c-bet, showdown %, stack vs start. Some dimensions are situation-generic and some
are *context-specific* (hole-card strength bucket × hand strength × street × position × opponent type).
There is a 9-category taxonomy, for example "lion": PFR > 8% and AF > 2.

**Chip dumping.** "Transfer of chips from one player to another when the player should not have lost the game
(based on the cards in his or her hands)", or when caused by an action "not in line with the player's determined style".

**Soft play.** "Passive play in a situation when the player would take a non-passive play according to his determined play style."

**Other patterns listed.** "Deliberately losing or causing another player to win", "passive play rather than aggressive
play against a particular player", "lack of raising against a particular player", "combining betting patterns with a
teammate to squeeze out another player".

**Scoring.**
- The deviation threshold depends on severity: a higher deviation is needed for low severity, a lower one for high severity.
- Severity grows with the money involved and with how far the action departs from the style.
- There is an ongoing collusion rating = the % of actions that are possibly collusive, with example thresholds of 0.001%, 1% and 10%.

**False-positive note.** A player folding a made straight or flush they did not notice "could look like collusion"; history
decides.

Relevance: this is almost literally directed_transfer and soft_play and coordinated_isolation. Their "style vs this particular
opponent" contrast is the hard-negative control. Severity-weighted ranking of actions maps to evidence-hand ranking.

### 2.5 IET patent US7604541 (Aikin, Goldfedder, Ostheimer; filed 2006): collusion via conditional behaviour
Variables: player hand rank, opponent hand rank, player action and opponent action. Bayesian-network structure learning is
done per pair, with a search over candidate graph structures. A direct edge from *opponent's hidden hand rank* to *player action* is
illegal conditional behaviour. The score is a Bayes-score odds ratio of the "colluding" graph vs the "non-colluding" graph.

Bot tests at collusion intensities X=0.25/0.5/0.75: colluders were "uniquely identified most of the time in on the order of 500 to 800 games" for
looser strategies, and about 1000 for tighter ones. Example of the pattern: "Player 2's action of raising with a weak hand is conditional upon
Player 1 having a strong hand."

Relevance: this is exactly coordinated isolation (partner raises with air when the partner is strong) and
information-sharing coordination. It is easy to implement as a logistic model of P(aggressive | own equity, public state,
partner equity) per pair, with the partner-equity coefficient compared against third-party coefficients.

### 2.6 Waterleaf patent US7883412 (filed 2003)
- Primary statistic: cumulative outcome / hands, per stake level, computed after at least 300 hands.
- Flags: a change in the primary statistic beyond a threshold; an "inordinate percentage" of a player's winnings *or losses*
  coming from one opponent; "raises without a raising hand" (R/RH).
- Also suspends pairs sharing a geographic location.

This is a crude, unadjusted money screen. It is useful only as a feature after luck adjustment and shared-hand normalization.

### 2.7 PokerTek patent US7618321 (filed 2005)
Rank players by winnings/losses and take the top as suspects. Partners are those who routinely play at the same time and table.
Then scan hands for "collusion triggers": "betting, folding, calling, and/or checking in uncommon situations", for example raising with a normally
losing hand to build a pot. Alerts only mean "possibility" and are followed by human replay. There are no quantitative thresholds.

It shows the operator workflow of **trigger hands, then human review**, which matches the host's "evidence hands contain a
behavior-specific action visible in the action log".

### 2.8 IGT patent US8360838 (filed 2006)
Per player: a "characteristic percentage of optimal decision-making" and a "characteristic range of deviation".
Collusion flags are:
- "lapse times during which a player's game play is worse than normal" coinciding with "one or more other players consistently win[ning] during a player's lapse times";
- repeated opponents;
- shared physical location.

Relevance: an **episodic** formulation. Find windows where A's decision quality drops, and test whether B is the beneficiary within those windows.

### 2.9 EA patents US11717757 / US12115458 and Greige et al. 2022 (arXiv 2203.05121)
Battle royale games, not poker. Pairwise feature sets for opposing pairs: social links, co-participation counts,
consecutive matches, landing proximity, final rank difference, damage dealt between the pair, and engaging the same third parties
"more often ... and closer in time than average". Unsupervised Isolation Forest, because there were fewer than 100 confirmed colluders.
Human review follows. False positives are costlier than misses.

Relevance: the isolation analogue is the pair attacking the same third player in the same hand, close in action order.
Unsupervised anomaly scores are one family-agnostic input.

### 2.10 Billings & Kan (ICGA J. 2006): DIVAT, and why hindsight analysis fails
**EVAT** penalizes every decision that differs from the perfect-information decision (Sklansky's "Fundamental Theorem
of Poker"). It is "highly unrealistic" because it expects omniscience. For example, a second-best hand on the river "would be
expected to fold whenever the opponent happens to hold the best possible hand". It is unreliable as a predictor.

**LFAT** measures equity change at chance events. EVAT and LFAT are not independent.

**DIVAT** compares the actual betting to a *baseline betting sequence* from simple quasi-equilibrium policies applied to both players:
- Fold if 7-card hand rank < bet/(pot+bet) on the river.
- Bet or raise thresholds use EHR = max(IHR, 7cHR).

The DIVAT difference is the equity difference between the actual and baseline pot contributions, using AIE (all-in equity with
all cards known) or ROE (roll-out equity with a betting policy). It is provably unbiased (Zinkevich 2006) and needs far fewer hands.

Metric definitions:
- IHR = (ahead + tied/2)/total vs uniform opponent holdings.
- 7cHR averages over future boards.
- AIE = fraction of runouts won given all hands. Net = AIE·pot − invested.
- AIE is exact for all-in players and overstates draws and weak made hands otherwise. Use ROE when future betting matters.

Relevance: two complementary views are needed.
- **Own-information view** (IHR/EHR vs uniform or range): was the action sane given what the player could see? This is the dumping and soft-play signature.
- **Omniscient view** (AIE vs actual hands): did the action benefit a specific other player? This is the beneficiary, and the information-sharing signature.

EVAT's failure warns *not* to penalize every omniscient mismatch. Weight by own-information irrationality.

### 2.11 Zinkevich et al. (AAAI 2006), Burch et al. AIVAT (AAAI 2018), Davidson et al. Baseline (AAMAS 2013)
- **Advantage-sum / MIVAT:** add correction terms E_o[u(o)] − u(o) at each chance event. The correction has zero mean, which removes card luck without bias.
- **AIVAT** also adds control variates on actions of players whose strategy is known, plus "imaginary observations" (other private hands consistent with public actions). It cut standard deviation by 85% (44× fewer hands) in the DeepStack evaluation.
- **Baseline:** replay the same cards with a baseline agent in self-play and subtract its outcome as a control variate. It beat other methods in 3-player poker.

Relevance: for any per-pair "chips transferred" feature, subtract card luck. Use either (a) AIE at every all-in or street
(the standard "all-in adjusted" winnings), or (b) the realized outcome minus a baseline-policy outcome on the same cards.
Both are computable because all hole cards are known.

### 2.12 Smed, Knuutila & Hakonen (GAMEON 2007; SCAI 2006) and Laasonen/Knuutila/Smed (2011)
Taxonomy of collusion agreements:
- **Consent:** express vs tacit.
- **Scope:** total vs partial.
- **Duration:** *enduring* vs *opportunistic* ("formed, disbanded, altered continuously").
- **Content:** knowledge (in-game info, or "stance: playing softly against one another") vs resources (donations).

Roles: spectator, assistant (sacrificial play to help the partner win), association (mutual benefit), self-collusion (one person, many seats),
and player-controller collusion (ghosting).

Detection: a measure m(Q,D) compared to expected results. Suspicion arises when results become too good (above r_g) **or too bad**
(below r_b, which indicates an assistant or dumper). Detection methods are judged on accuracy vs swiftness.

The Pakuhaku testbench shows colluder advantages but gives no detection results. Laasonen 2011 tried decision trees on synthetic
features (not downloadable).

Relevance:
- directed_transfer ≈ assistant collusion (one side "too bad").
- soft_play ≈ stance-sharing.
- coordinated_isolation ≈ association collusion.
- "Opportunistic duration" = the host's "episodic".

### 2.13 Laasonen & Smed (INTETAIN 2015): soft-play detection via hit matrices
A directed matrix H[a,b] records hits from a to b. It is converted to a collusion table (negate off-diagonal entries, put total hits on
the diagonal), and TI/MaI/MuI/MiI/DI are computed. Clustering is done on the inverted-nearness graph (a player's nearest neighbour is
the one they hit least), with shared-NN, mutual-NN, Palshikar-Apte collusion clustering and modularity community detection.
The collusion set is chosen by collusion index I(C)/E(C).

Results:
- 2 colluders with no mutual aggression were found in most cases.
- With 3 colluders, collusion clustering failed and mutual-NN and community detection worked.
- **When colluders still shot each other with probability 0.5, all methods failed.**
- PCA on hit-matrix rows (van der Knyff 2009) is significant only when collusion is the dominant source of variation, so it has "little practical use".

Relevance: our analogue is a directed *aggression matrix* per table (bets and raises made into a pot where the target is
the main opponent), normalized by opportunities and conditioned on equity. Pure frequency matrices will fail on episodic
soft play. The contrast must be equity-conditioned and per-hand.

### 2.14 Yan (AAAI 2010): collusion detection in online bridge
Detect "decisions deemed too good to be drawn from partial information". For each key action, build A_h (candidate
honest actions from own hand plus public auction) and A_c (candidates given the partnership's *combined* hands).
The action is suspicious if it is in A_c but not in A_h.

Focus on **critical cash-out points**: contract bid, penalty double and opening lead, "where online cheaters often cash
their collusive advantages". A single suspicious play proves nothing (luck, misclick, genius). The evidence must accumulate over
time and account for skill level. Publicly known detectors invite evasion.

Relevance: this is the poker analogue for other_coordination and isolation. Take actions that are irrational given own cards
but correct given the partner's cards. Prioritize evidence at cash-out decisions: river actions, all-in calls or folds, and
folds facing a bet with sufficient equity.

### 2.15 Johansson, Sönströd & König (2003); Johansson & Sönströd "Fish or Shark" (2009)
2003 (via citations): showed there are no pre-emptive or real-time countermeasures to collusion. "Active collusion can be detected
afterward by analysing the game data, but it is next to impossible to discern passive collusion from cautious normal
play". Built information-sharing strategies in 3-player Kuhn poker.

2009: decision trees and NN ensembles on player profiles (VPIP, PFR, aggression, WTSD and similar) separate winners from losers. It gives the HUD
stat set that defines a player's *style baseline* (tight or loose × passive or aggressive: calling station, rock, maniac, solid).

Relevance: style clustering helps control the "similar strategies" hard negative. Two rocks show low mutual aggression without colluding.

### 2.16 Yampolskiy (2007, 2008; not downloadable)
- GAME-ON-NA 2007: taxonomy of online poker cheating (collusion, bots, account compromise and similar).
- CCNC 2008: a non-obtrusive, CAPTCHA-like continuous verification against bot assistance.
- ITNG 2008: strategy-based behavioural biometrics, where poker action profiles identify a player. A mimicry attack shows
  profiles can be spoofed by training a generator on observed actions.

Relevance: marginal. Profile similarity is a *hard negative* (similar strategies) rather than evidence of coordination.
Profile change over time is a ghosting or strategy-change signal.

### 2.17 Zehnder (ETH Zürich BA thesis, 2012): building colluding 6-max NLHE bots
Up to 4 robots shared hole cards through a server. They removed partners' cards from equity enumeration and used a win/loss-ratio
decision rule. Deliberately *passive* collusion (information use only, no whipsawing), because "active collusion ... would be much more detectable".
Collusion helped more with each extra bot, but the robots still lost at real-money micro-stakes. The platform detected nothing.

Relevance: information-sharing collusion changes fold, call and raise thresholds (card removal) but produces no obvious betting
pattern. Only equity-vs-action analysis conditional on partner cards reveals it. That is a plausible "other_coordination".

### 2.18 Team games theory: Farina et al. (NeurIPS 2018; arXiv 2020), Carminati et al. (ICML 2022), Xu/Feng/Fang (2024)
- Collusion in poker is an adversarial team game.
- Team-maxmin equilibrium with ex-ante coordination (TMECor): the team is one player with *imperfect recall*.
  An optimal collusive strategy is a correlated distribution over joint pure plans.
- Coordinated strategies can gain substantially over uncoordinated behavioural strategies (3-player Kuhn and Leduc testbeds).
- Carminati: a team-public-information representation where a coordinator prescribes actions for each member's private state.
- Xu et al.: learning coalition structure by *designing* games, O(log n) rounds (not applicable: observational data only).

Relevance: optimal colluders' actions are correlated through a shared plan. That produces dependence between A's action and
B's private state, the MI/Bayes-net test, not necessarily different marginal frequencies. Detectors based only on marginals
(VPIP, AF) can miss sophisticated collusion.

### 2.19 Graph and fraud-ring mining: FRAUDAR (KDD 2016), HoloScope (CIKM 2017), CatchSync (KDD 2014), Palshikar-Apte (DMKD 2008)
- **FRAUDAR:** greedy peeling for dense bipartite blocks with column weights 1/log(degree + c). It resists camouflage (fraudsters adding honest edges).
- **HoloScope:** "contrast suspiciousness" plus **temporal bursts and drops** of fraud activity plus rating deviation. Sub-quadratic.
- **CatchSync:** synchronized (similar neighbour behaviour) and abnormal (rare) nodes.
- **Palshikar-Apte:** collusion sets = clusters with heavy internal trading vs external. Dempster-Shafer combines candidates from several clusterers.

Relevance here is limited because every pool is 30 players at one table and pairs are the unit. Two ideas transfer:
(a) HoloScope/CopyCatch-style *time-window burst* scoring of a pair's per-hand suspicion series;
(b) group structure: some players appear in more than one positive pair (693 players across 372 pairs), so a player-level
"is part of a ring" score propagates.

### 2.20 Industry sources (web; not archived as PDF)
- **PokerStars:** reviews hands "with all cards showing"; checks "best hand playing" (only the better of the two colluders' hands plays on), soft play ("refuse to play aggressively against each other"), and "chip dumping/stack balancing" (the big stack folds to the small stack). A third-party security blog (guardianstack.ai) claims PokerStars removed 3,000+ accounts since Jan 2025 using ML detection. This is unverified.
- **ACR/WPN:** hand-history plus video review; GTO Wizard partnership for "GTO analysis protocols"; PLO reshuffle (folded cards go back into the deck) to reduce information collusion; hundreds of accounts suspended; funds redistributed.
- **GGPoker:** security team, RTA and ghosting bans (GGMillion$ 2025), GTO Wizard Fair Play Check (was the board solved at the hand's time). Forum case descriptions of rings: accounts "super close to each other, but never compete", with synchronized session starts.
- **GTO Wizard:** screens "GTO profiles" for superhuman play and withholds specifics.
- **Wikipedia "Cheating in poker":**
  - Soft play: "fail to bet or raise in situations that normally warrant it".
  - Whipsawing: "partners raise and re-raise each other to trap players between them".
  - Dumping: "deliberately loses to a partner".
  - Signalling.
  - Ghosting and multi-accounting.
  - Detection: sites flag patterns such as "folding good hands to a small bet".

  A common heuristic in player guides and forums for whipsaw: raise wars with a third player in the pot "but rarely have a showdown".
- Common operator language in public materials: "repeated large losses to the same player, illogical bet sizing, correlated account activity", "VPIP versus win rate outliers", "consistent strategic folds between partners". Device/IP signals are not available here.

---------------------------------------------------------------------------------------------------

## 3. Consolidated table: family -> signatures in a full-visibility action log -> luck-adjusted scoring

Notation:
- eq_i = all-in equity of i vs the *actual* active hands at a decision (known hole cards, remaining runouts enumerated).
- eq_ij = heads-up equity of i vs j.
- EHR_i = i's own-information strength (hand rank vs uniform or range: IHR/7cHR).
- potodds = to_call/(pot_before + to_call).
- A decision is "facing j" if j made the last aggressive action.
- Every rate is compared to **the same player's rate vs non-partners in the same (street, EHR bucket, potodds bucket, players_active, position) cell**.
  This is the CFPH style baseline and Mazrooei's marginal impact.

| Family (host) | Literature names | Signatures visible in the action log (with all hole cards) | Luck-adjusted per-hand score (evidence ranking) | Pair aggregation (risk features) |
|---|---|---|---|---|
| **directed_transfer** (one player deliberately loses value to another) | chip dumping, stack balancing, assistant collusion (Smed "too bad" r_b), "should not have lost based on cards" (CFPH), "folding good hands to a small bet" (Wikipedia), lapse times with a consistent beneficiary (IGT) | (1) **Fold facing partner's bet with equity well above pot odds**: eq_i − potodds >> 0 and EHR_i high, especially on turn and river, and after the dumper invested. (2) **Calling off or overbetting into a partner who is far ahead**: i puts chips in with eq_ij << potodds, own EHR low; all-in by dumper with trash vs receiver. (3) Dumper bets or raises (builds the pot) then folds to receiver's raise ("bet-fold"). (4) Pot goes to receiver without showdown, often heads-up after others fold. (5) **Asymmetry**: dumper's EV loss accrues to one opponent, not to the field (weak and tilted players leak to everyone). (6) Dumper's decision quality drops only in hands vs receiver. | Directed EV transfer T_g(k→j) = Σ over k's actions of max(0, EV_best − EV_taken) weighted by the share that goes to j. For a fold: (eq_k·(pot+to_call) − to_call) forfeited, split among the remaining players by their equity (j's share = eq_j/Σeq). For a call or shove: to_call·(potodds − eq_k)⁺ paid to the leader. Equivalent to the per-hand collusion value C_g(j,k) − C_g(j,k \| baseline policy). Multiply by an own-info irrationality weight (EHR-based) to avoid the EVAT trap. | Top-k (k=3..5) sum of T_g(k→j); count of hands with T > x bb; Marginal Impact: mean T(k→j) − mean over third parties of T(k→i); Minimum Impact is low for one-way pairs, so keep a *directed* score as well as MiI; all-in-EV-adjusted net chips k→j per shared hand; burst score (max over windows). |
| **soft_play** (partners avoid normal aggression vs each other) | soft play, "stance" sharing (Smed), passive collusion (Johansson), "lack of raising against a particular player" (CFPH), hit-matrix soft play (Laasonen & Smed) | (1) **Missing value bet or raise when strong vs partner**: i has eq_ij ≥ ~0.7 or nut advantage, heads-up with j (or j is the only one left to act), and i checks or just calls where i's own baseline bets or raises. (2) **Check-down** of the whole hand heads-up between partners with a large equity gap (river check-back with the nuts). (3) Lower 3-bet or isolation frequency vs partner's open than vs others' with the same EHR. (4) Folding marginal-plus hands when partner bets instead of raising (avoiding conflict). (5) Smaller bet sizing vs partner. (6) Lower rate of heads-up confrontations than expected from co-dealt hands and seat order (avoidance). | Forgone value F_g(i,j) = Σ over passive decisions of i where the baseline (own style or DIVAT policy) would bet: P(baseline bets \| own cell) × est. value of bet ≈ b·(2·eq_ij − 1)·P(call), with b = baseline bet size (e.g. 0.66 pot). The score is large when eq_ij is high and the action is passive. The mirror case, bluff avoidance, is weaker evidence. Condition on equity so that luck does not enter: only the choice is scored. | Equity-conditioned aggression deficit: logit P(aggr \| cell, opponent=partner) − logit P(aggr \| cell, opponent≠partner), estimated with shrinkage; directed 2×2 "aggression matrix" between the pair vs row/column means (Laasonen hit-matrix scores MaI/MiI); top-k F_g; symmetric version (both soft), since soft play is usually mutual → MiI. |
| **coordinated_isolation** (pressure others while limiting conflict between themselves) | whipsawing/raise sandwich (Wikipedia, 2+2), squeeze-out (CFPH "combining betting patterns with a teammate"), association collusion (Smed; Bonjour Leduc colluders "both raise if either has an ace"), best-hand playing (PokerStars), conditional behaviour on partner's hidden cards (IET patent), "engage same opponents closer in time" (EA) | (1) **Sandwich**: A bets or raises → third player C calls or acts → B raises or re-raises (C trapped between); C folds with non-trivial equity. (2) **B's aggression conditional on A's strength**: B raises with low EHR_B when eq_A is high (and B folds later to A or does not contest A). (3) **Best-hand playing**: in pots both enter with a third player, the partner with the lower eq drops out early while the higher one continues; rarely both to showdown (the 2+2 "rarely have a showdown" heuristic). (4) Multiway pots where the pair's joint aggression makes third parties fold more than their equity warrants. (5) After isolation, the pair checks or folds to each other (soft play inside the isolated pot). (6) Seat geometry: the partners sit on both sides of the victim in action order. | Third-party harm with pair benefit: TI_g(A,B) = Σ C_g over the pair's own actions on the pair, plus H_g = −Σ_{C∉pair} [C_g(C,A) + C_g(C,B)] (value the pair's actions took from third parties, computed with AIE or rollout). Sandwich indicator × third-party forfeited equity (C's eq·pot at fold). Conditional-dependence term: B's aggression residual (after own EHR, position and pot) × A's equity. | Mazrooei TI and DI; Bonjour net influence ν(A;B), ν(B;A) on action bins with the joint state including partner equity; IET-style logistic coefficient of partner equity in B's aggression model minus the max coefficient for any third party; sandwich rate per multiway co-involved hand vs table baseline; best-hand-playing rate; top-k per-hand scores. |
| **other_coordination** (undisclosed 4th; never in positives) | information sharing / "cheating" in the Mazrooei sense (Zehnder bots; Johansson 2003), Yan's "too good for partial information", self-collusion or multi-accounting (Smed), stack balancing, signalling, synchronized presence (GGPoker rings), opportunistic agreements | Candidates: (a) **card-removal-informed thresholds**: calls or folds that are wrong on own info but right given partner's hole cards (A folds a marginal hand exactly when B holds its outs; A calls light when B's hand blocks the opponent's range); (b) **synchronized participation** (both enter or sit out the same hands, correlated VPIP beyond seat effects, correlated session changes over hand_seq/ts); (c) protecting partner (betting a third player out when partner is all-in, or checking down a side pot); (d) walks and blind steals uncontested to partner; (e) mutual chip passing through alternating direction (net ~0, low-payoff collusion as in Bonjour's LP agents). | Yan-style indicator: action ∈ A_c (optimal given the pair's combined information) and ∉ A_h (own info) → score = EV_c(action) − EV_h(action); the MI contribution of the hand (pointwise MI log p(a_i \| s_i, s_j)/p(a_i \| s_i)); generic per-hand collusion value \|C_g(i,j)\| minus third-party baseline. | Family-agnostic: collusion-table TI/MaI/MiI/DI; Bonjour net influence; PU/anomaly score on pair features (Isolation Forest à la Greige/EA; PU à la Kocsis-György); co-participation correlation adjusted for seat and players_dealt; HoloScope-style burst score. Keep these in the final risk_score blend so the 4th mechanism is not scored 0. |
| **Hard non-targets** (host list) | accidental colluders (Mazrooei CR/CL), misfolds by weak players (CFPH), profile similarity (Yampolskiy), money-concentration screens (Waterleaf), strategy change (Waterleaf change flag), repeated opponents (IGT/PokerTek) | tilt / weak play: EV leaks to *all* opponents; similar strategies: correlated marginals, but no dependence on partner's *hidden* cards given public state; repeated opponent selection or high shared_hands: large raw counts; streaks: realized chips without an equity edge; strategy change: behaviour shift vs everyone at once | Use a luck-adjusted per-hand score only (AIE or baseline control variate); never use raw net_chips | Always use **partner minus third-party contrasts** (MaI, net influence, logit deficits); **normalize per shared hand with Bayesian shrinkage** (the evidence-count problem: 38-419 hands); condition on own-info strength (EVAT trap); difference-in-differences over time (behaviour vs partner changes while behaviour vs others does not); require both partners' contribution (MiI) except for directed_transfer. |

---------------------------------------------------------------------------------------------------

## 4. How the literature handles luck and variance, and a recipe for this data

1. **Remove chance, keep choices.** Advantage-sum, MIVAT and AIVAT subtract E[u] − u at each chance event.
   Collusion tables attribute value changes at chance nodes to a "chance" column. With all hole cards known:
   - At each street start, compute per active player AIE vs the actual active hands by exact enumeration of the remaining board:
     flop 2 cards to come ≈ C(52−3−2n, 2) runouts, turn ≤ 44. Preflop needs Monte Carlo (about 500-2000 boards) or a cache keyed by (hands, players).
   - Recompute when the active set changes, after a fold. This is cheap: equity changes only on folds and chance, not on bets.
   - Value of a history for player i ≈ eq_i·pot_total − invested_i (AIE; exact for all-ins).
   - C_g(i,k) = Σ over k's actions of ΔV_i. A fold by k changes everyone's eq, so it directly measures how k's fold redistributed equity.
2. **All-in EV adjustment:** when players are all-in before the river, replace realized won_share with equity share.
   This is the standard "all-in adjusted" metric in tracking software.
3. **Own-information rationality** (DIVAT baseline): IHR/7cHR vs uniform or range and pot-odds invariants
   (fold if strength < bet/(pot+bet) on the river). An action is only "irrational" relative to this view. The omniscient view
   identifies the *beneficiary*. The EVAT failure mode is penalizing correct plays that lose to a hidden better hand.
4. **Player-specific baselines** (CFPH style vector, IGT characteristic deviation, Johansson/Teófilo profiles): fit
   P(action | street, EHR bucket, potodds bucket, position, players_active, aggressor identity class) per player on
   hands *without* the partner as main opponent, then score deviations in hands vs the partner.
5. **Minimum intensity and sample size** (Bonjour, IET, Mazrooei): whole-history averages need hundreds to thousands of hands
   at high intensity. For episodic manipulation use **top-k sums, max over sliding windows or a scan statistic**
   (see the other agent's `ml_ts_*` scan-statistics papers) on per-hand scores. Add a shrinkage prior per pair based on
   shared hands, so that 38-hand pairs do not dominate by noise.
6. **Critical decision points** (Yan): colluders cash in at a few decisions. In NLHE these are river bets and folds, all-in
   calls and shoves, folds facing a bet with high equity, and re-raises in multiway pots. Weight per-hand evidence by chips at
   stake (CFPH severity = money involved × deviation).

---------------------------------------------------------------------------------------------------

## 5. Ranking suspicious hands (evidence MAP@5), per the literature

- Per-hand score = severity (chips of equity moved, in bb) × deviation (how far from the player's own baseline or
  own-info rationality) × beneficiary alignment (the share of the value that went to the partner). This is the CFPH severity rule
  combined with the Mazrooei per-hand C_g and the Yan A_c/A_h test.
- Separate scorers per family (directed EV transfer, forgone value vs partner, sandwich or third-party harm), then choose
  the family-appropriate list for the predicted behaviour and fall back to the generic |C_g| score.
- Evidence hands must contain "a behavior-specific action visible in the public action log", so prefer hands with an explicit
  trigger action by one of the pair: fold with equity, passive line with nuts, re-raise sandwich. Latent card
  configurations alone do not qualify.
- Planted evidence is in the dev phase only, while eval evidence is scored on eval hands. Tune the per-hand scorer on dev positives
  (1817 listed hands) and check that it ranks the listed hands above other shared hands of the same pair.
  Unlisted planted hands exist ("up to five"), so this is a PU ranking check.

---------------------------------------------------------------------------------------------------

## 6. What failed or does not work (consolidated)

| Approach | Failure | Source |
|---|---|---|
| Rank by money won or win-rate outliers | The best winners are honest; weak colluders lose overall; cheat payoffs overlap honest ones | Mazrooei 2013 Table 3; Kocsis & György Fig. 1; PokerTek and Waterleaf are only screens |
| Hindsight "perfect information" misplay counting (EVAT, Sklansky FToP) | Unrealistic omniscient baseline; penalizes correct plays; poor predictor | Billings & Kan 2006 |
| Always-call value function | Did not detect colluders | Mazrooei thesis ch. 7 |
| Mutual Impact score alone | Missed strong colluders with the CFR value function | Mazrooei thesis Table 7.11 |
| Whole-history MI at low collusion probability | CP=0.1 undetectable even at 10k games | Bonjour 2022 |
| Frequency-only soft-play matrices when colluders still attack each other ~50% | All clustering methods failed | Laasonen & Smed 2015 |
| PCA of interaction-matrix rows | Works only if collusion is the main source of variance | van der Knyff 2009 via Laasonen & Smed 2015 |
| Shared-NN clustering; collusion clustering with 3 colluders | Poor, or failed | Laasonen & Smed 2015 |
| Passive collusion without hole-card visibility | "Next to impossible" to separate from cautious play | Johansson et al. 2003 |
| Positional "accidental colluders" | Ranked as colluders by impact scores; need human review | Mazrooei 2013 |
| Rule triggers without a baseline | Only "possibility" alerts; misfolds by weak players look like dumping | PokerTek; CFPH |
| Operator detection of real colluding bots (2012, micro-stakes) | Nothing detected | Zehnder 2012 |
| Publicly known detectors | Colluders adapt (co-evolution) | Yan 2010 |

---------------------------------------------------------------------------------------------------

## 7. Concrete feature list to build (derived from the above; for the modelling agents)

Per hand, per ordered pair (i,j) with both dealt in:
1. `aie_street[i]` at each street start and after each fold (exact enumeration postflop, MC preflop); `eq_hu[i,j]`.
2. `C[i,k]`: per-hand collusion value (ΔV_i summed over k's actions, V = AIE·pot − invested).
   Hand totals give TI, MaI, MuI, MiI and DI per pair. Also a directed `C[j,k] − mean_{third} C[t,k]`.
3. `dump_ev[k→j]`: equity forfeited by k's folds (share to j) plus overpayment by k's calls or shoves into j when behind,
   gated by k's own-info strength (EHR) and pot odds.
4. `soft_forgone[i vs j]`: passive actions with eq_hu ≥ τ where i's own baseline aggression in that cell is ≥ ρ;
   forgone ≈ bet·(2·eq − 1)·P(call) using i's field baseline.
5. `sandwich[A,B]`: A aggressive → third player acts → B raises, in the same street; third player's forfeited equity if it folds.
6. `best_hand_play[A,B]`: both voluntarily in with a third player, the weaker by eq folds first without being bet at by the partner.
7. `cond_dep[A←B]`: residual of A's aggression (after own EHR, position, potodds, players_active) regressed on B's hidden
   equity, minus the max over third parties (IET patent, Bonjour net influence). Also pointwise MI per hand.
8. `yan_flag`: action optimal given combined pair information but not given own information (EV gap).
9. Pair aggregation: top-k (k=1,3,5) and mean with a shrinkage prior over shared hands; sliding-window max (e.g. 50-hand
   windows over hand_seq); counts above thresholds; partner-minus-field contrasts; burstiness (HoloScope-style).
10. Player-level "ring" propagation: a player's max pair score across partners (693 positive players across 372 pairs).

Pitfalls to remember:
- Never use raw net_chips or won_share without equity adjustment.
- Always contrast against the same player's behaviour vs non-partners.
- Condition on own-information strength.
- Keep a generic, mechanism-agnostic score in the final blend for other_coordination.
- The evaluation excludes pairs containing labelled positive players. Do not rely on player identity carry-over from dev;
  player-level style baselines must be learned from each player's own hands (dev and eval phases both available).

---------------------------------------------------------------------------------------------------

## 8. Web references (for traceability)
- Mazrooei et al. AAAI13: https://poker.cs.ualberta.ca/publications/AAAI13.pdf ; thesis: https://poker.cs.ualberta.ca/publications/mazrooei.msc.pdf
- Bonjour et al. UAI22: https://proceedings.mlr.press/v180/bonjour22a/bonjour22a.pdf
- Kocsis & György 2010: https://www.szit.bme.hu/~gya/publications/KocsisGyorgy.pdf
- DIVAT: https://poker.cs.ualberta.ca/publications/divat-icgaj.pdf ; AIVAT: https://poker.cs.ualberta.ca/publications/aaai18-burch-aivat.pdf ; Zinkevich06: https://poker.cs.ualberta.ca/publications/AAAI06.pdf ; Baseline: https://poker.cs.ualberta.ca/publications/AAMAS13-baseline.pdf
- Smed et al. 2007: https://people.rennes.inria.fr/Sophie.Pinchinat/PAKUdir/inpSmKnHa07a.pdf
- Laasonen & Smed 2015: https://eudl.eu/pdf/10.4108/icst.intetain.2015.259565
- Yan 2010: https://prof-jeffyan.github.io/aaai10.pdf
- Greige et al. 2022: https://arxiv.org/abs/2203.05121
- Zehnder 2012 (ETH): https://pub.tik.ee.ethz.ch/students/2012-FS/BA-2012-03.pdf
- Johansson & Sönströd 2009: https://www.diva-portal.org/smash/get/diva2:886961/FULLTEXT01.pdf
- Farina et al. 2018: https://proceedings.neurips.cc/paper_files/paper/2018/file/c17028c9b6e0c5deaad29665d582284a-Paper.pdf ; Carminati 2022: https://proceedings.mlr.press/v162/carminati22a/carminati22a.pdf
- FRAUDAR: https://bhooi.github.io/papers/fraudar_kdd16.pdf ; HoloScope: https://arxiv.org/abs/1705.02505 ; CatchSync code: https://github.com/mjiang89/CatchSync
- Patents (full text): https://patents.google.com/patent/US9652931B2/en , https://patents.google.com/patent/US7604541B2/en , https://patents.google.com/patent/US7883412B2/en , https://patents.google.com/patent/US7618321B2/en , https://patents.google.com/patent/US8360838B2/en , https://patents.google.com/patent/US11717757B2/en , https://patents.google.com/patent/US12100265B2/en
- Industry: https://en.wikipedia.org/wiki/Cheating_in_poker ; https://www.acrpoker.eu/security/ongoing-security-efforts/ ; https://blog.gtowizard.com/towards-a-safer-poker-ecosystem/ ; https://ggpoker.com/network/security-ecology-policy/ ; https://www.pokerstarsmtairycasino.com/embedded/help/articles/unfair-play-master/ ; https://blog.guardianstack.ai/online-poker-cheating-detection/
- LLM-agent collusion 2026: https://arxiv.org/abs/2602.15198 (COLOSSEUM), https://arxiv.org/abs/2604.01151 (NARCBench)
