# Post-mortem — Slash poker collusion, finished **8th of ~370** (target: top 3)

Written 2026-09-20 before the private LB; §0 added 2026-09-21 with the final result. Companion to `experiments/LEDGER.md` (what we ran) and
`research/CLOSED_LANES.md` (what was measured dead). This file is about **why the campaign lost**, which is a
different question, and it is written so the parent playbook can be amended.

---

## 0. FINAL RESULT (private LB, 2026-09-21)

**8th of ~370.** Private **0.92817** (`sub028_famspecw15_exp042ramp.csv`). Public was 11th/0.91954, so the
private split moved us up three places. Prizes went to Tejasv Bhatia (0.94320), blastyy (0.94263) and
John Tyler (0.94196). **Gap to 3rd: +0.01379.**

| rank | team | private |
|---|---|---|
| 1 | Tejasv Bhatia | 0.94320 |
| 2 | blastyy | 0.94263 |
| 3 | John Tyler | 0.94196 |
| 4 | Pardheev Krishna | 0.93514 |
| 5 | Leo | 0.93416 |
| 6 | thisray | 0.92888 |
| 7 | Marc Donovici | 0.92850 |
| **8** | **Exposed** | **0.92817** |
| 9 | seantangth | 0.92677 |

### The endgame decisions were all correct

- **The hedge won.** Of the two ticked finals, `sub028` (0.92817) beat `sub025` (0.92777) by **+0.00040**.
  The E[max] argument that put it in slot 2 — a genuinely different risk ranking at a measured cost of
  −0.00007 — paid exactly as designed. Kaggle's auto-pick (sub024 + sub025) would have returned 0.92777.
- **Selection cost us essentially nothing.** The best private score across all 29 submissions was `sub020`
  at 0.92822, only **+0.00005** above what we ticked.
- **Every D-1 rejection held up on private.** `sub030` (the falsified evidence change) → 0.92754, below
  sub028. `sub027` (flat gate) → 0.92777, below sub028. The public readings pointed the right way.
- One caveat worth recording: `sub020` (the sequence-model risk fusion we rejected on a −0.00044 public
  delta) scored the highest private of anything we built. The margin is 5e-5, i.e. nothing — but it
  confirms that a −0.0004 public reading is **not** falsification, only an absence of evidence.

### Private component decomposition (from the private probes sub011/sub012/sub014/sub026/sub031)

| component | public | private |
|---|---|---|
| pair AP | 0.9745 | **0.9807** |
| evidence MAP@5 | 0.7048 | **0.7218** |
| behaviour macro-AP | 0.9635 | **0.9692** |

The private half was kinder on all three (mean public→private shift across our 29 submissions: **+0.0126**,
sd 0.0071). That is why the board reshuffled and why public rank was a poor guide.

### What 3rd place actually had — the diagnosis is slightly different on private

3rd (0.94196) minus our 0.92817 is **+0.01379**. Solving `0.7P + 0.2E + 0.1B = 0.94196`:

| their pair AP | their behaviour | implied evidence |
|---|---|---|
| 0.9807 (ours) | 0.9692 (ours) | 0.793 |
| 0.985 | 0.985 | 0.770 |
| 0.988 | 0.985 | 0.759 |
| 0.990 | 1.000 | 0.745 |
| **1.000 (perfect)** | **1.000 (perfect)** | **0.710** |

So even a *perfect* pair ranking and *perfect* behaviour routing would still need evidence ≈ 0.710 — just
below our 0.722. The realistic reading is that 3rd beat us on **both**: evidence ≈ 0.75–0.77 (+0.03–0.05 over
ours, worth 0.006–0.010 of score) and pair AP ≈ 0.985–0.990 (+0.005–0.010, worth 0.0035–0.007).

**This refines the public-LB conclusion.** On public the entire gap looked like evidence. On private, pair AP
carries roughly a third of it. Both were lanes we had identified; neither was closable in five days.

---

## 1. The gap, as seen from the public board (refined by §0 above)

Measured components of our shipped file (probe sub031, direct):

| component | weight | ours | a leader at 0.94068 needs |
|---|---|---|---|
| pair AP | 0.70 | 0.9745 | 0.974–0.985 |
| **evidence MAP@5** | 0.20 | **0.7048** | **0.77–0.81** |
| behaviour macro-AP | 0.10 | 0.9635 | 0.96–0.98 |

Pair AP and behaviour are within ~0.01 of anything achievable. **The entire 0.021 gap is evidence retrieval**,
and inside evidence it is one latent variable: *which hands did the generator actually plant?*

The oracle decomposition (computed at D-1, which is the problem — see §2):

- knowing the true **type** → +0.018 dev MAP
- knowing the true **event** → **+0.157 dev MAP ≈ +0.027 final**
- our recall@5 = 0.820, **recall@10 = 0.970**, AP@5 = 0.775

So the candidates were always in our top-10. Our *ordering* of a retrieved set was already near-optimal
(AP/recall = 0.945). What we never got was the ability to tell a planted hand from an identical-looking natural
one. The leaders did. That is the whole competition.

---

## 2. Four causes — reordered 2026-09-21 after the final data refuted the first one

### (a) ~~We started late and were submission-limited~~ — **REFUTED by the final data, 2026-09-21**

This was my original ranked-first cause. The final leaderboard kills it.

| team | private | submissions |
|---|---|---|
| Tejasv Bhatia (1st) | 0.94320 | **30** |
| blastyy (2nd) | 0.94263 | **20** |
| John Tyler (3rd) | 0.94196 | **25** |
| **Exposed (8th)** | **0.92817** | **29** |
| Pardheev Krishna (4th) | 0.93514 | 52 |
| seantangth (9th) | 0.92677 | 65 |
| Brothers (11th) | 0.92237 | 60 |
| Amin Mohamed (16th) | 0.92056 | 57 |

**The three prize winners used 20–30 submissions. We used 29.** The teams with the *most* submissions
(65, 60, 57, 52) finished 9th, 11th, 16th and 4th. Across the top of the board the correlation between
submission count and rank is nil, if not negative.

Worse for the excuse: on 2026-09-15 19:56, the moment we joined, **John Tyler sat 15th at 0.89612 with five
submissions** — already ahead of where we were a day later (0.88153) — and finished **3rd at 0.94196 with 25
submissions total**. He worked the same five-day window we did, submitted fewer times, and beat us by 0.0138.
Both of us gained ≈0.046 over that window; he was ~0.014 ahead at the start of it and ~0.014 ahead at the end.

**The gap was never about time or submissions. It was about method, and it was already there on day one.**
A day-0 start would have helped — more time to think is never worse — but it is not the explanation, and
listing it first was self-serving. The real causes are (b), (c) and (d) below, all of which are process, and
all of which are fixable without a single extra day.

### (b) Our validation view was structurally blind, and nobody checked it for five days

View C′ — the view every model decision was made on — **deletes** the 137 suspected hidden positives from
scoring. But **every evaluation positive is, by construction, an unlabelled positive** — i.e. exactly a spy-like
pair. C′ was therefore blind to the one failure mode that matters on eval.

The cost was proven on the last night: the family-specialist risk mixture scored **C′ +0.0028 at P(better) 1.000**,
positive in 5/5 folds and all three families and replicated at a second seed — and delivered **−0.00007** on the
leaderboard. When the complementary "view D" was built (the 137 counted as *positives*), the same candidate
read +0.0006 at P 0.728, and direct retrieval of those 137 pairs *dropped* from 0.9254 to 0.9229.

How many earlier decisions were priced on the same blind view? Self-training, spy promotion, the whole
model-selection ladder. We cannot now tell.

**Fix: build the adverse view at the same time as the optimistic one, and require a candidate to win on BOTH.
State explicitly which population the test positives resemble, and validate on that population.**

### (c) A refuted narrative steered the last three days into the wrong lane

`φ = 0.14 of eval positives belong to an undisclosed 4th mechanism` ⇒ `evidence caps at ≈0.715` ⇒ *stop pushing
evidence, go push pair AP*. That inference was written into the planning docs as measured
fact and inherited as a premise by everything that came after.

It was wrong. `obs = (1−φ)K + φαL` is linear in `((1−φ)c, φαc)`, so φ trades 1:1 against an assumed global
transfer factor: c = 0.92 → φ = 0.065, c = 1.10 → φ = 0.218, **identical RMSE**. Two free parameters, three data
points. A completely different 2-parameter model with *no* dead subpopulation fits as well. And two independent
bounds — the behaviour probe, and pair AP against the LOFO value of an unseen family — cap any unmodelled
mechanism at **≤3%**, not 14%. The mis-served cohort is low-exposure **soft play**, a disclosed family.

The consequence: we redirected effort from the component carrying 0.059 of loss to the one carrying 0.018, on the
strength of a fit that was never identified. The last three days went into CatBoost fusions, seed bags, rank
averages and hill-climbs worth ≤0.001 combined, while the event bit sat there worth +0.027.

**Fix: no inference that CLOSES a lane may enter the docs until a dedicated adversarial review has tried to break it.
Any fit gets an identifiability line: free parameters, data points, and what else fits equally well.**

### (d) We measured where the value was on the last night instead of the first

The oracle decomposition in §1 takes about two hours to compute and tells you, before you optimise anything,
exactly how much each latent variable is worth. We ran it at D-1. Had we run it on day 1 we would have seen
+0.157 sitting behind the event bit and +0.018 behind everything else in evidence, and the campaign would have
pointed at the event bit from the start instead of for about two days at the end.

Related: the score decomposition we priced every decision against was **wrong twice** (the all-tied behaviour
probe does not return AP = 0; and that floor is per-family; and even then the triple was inferred across risk
models). Pair AP was off by +0.001, behaviour by −0.005.

**Fix: day-1 oracle decomposition per component. And audit the measuring instrument before using it to steer —
probe the model you actually ship.**

---

## 3. The process failure underneath all three

Causes (b), (c) and (d) are the same mistake wearing three hats: **a plausible story was allowed to become a
premise without ever being attacked.**

φ = 0.14 fitted three leaderboard points at RMSE 0.0006 and explained everything. It was written into the
planning docs as a measured fact. From that moment every later decision inherited it, and nobody re-derived it,
because it was no longer a hypothesis, it was background. The same thing happened to view C′: it was built early
for a good reason (pooled AP penalises a model for finding hidden positives) and then used for five days without
anyone asking whether its *positive population* matched the one being scored. It did not.

The general shape is worth naming, because it is not specific to this competition:

1. **A coherent story that fits the data is not the same as an identified one.** Always write down free
   parameters against data points, and what else fits equally well, before letting a fit close a lane.
2. **Written project documentation is an echo chamber.** Once an inference lands in the plan, it reads as
   measurement to everyone downstream, including your future self. Inferences and measurements need different
   typography and different burdens of proof.
3. **Breadth is cheap and closes lanes; depth is expensive and opens them.** This campaign ran 65 statistics,
   8 hypothesis families and 40 documented lanes. The plausible winning approach was one idea pursued properly.

The uncomfortable part is that the doctrine file written *before* this campaign, from the previous one's
post-mortem, already said "never infer a ceiling from saturation", "the centre of gravity is single-model
iteration, not blending", and "start the iteration engine on day 1". It named this failure mode in advance. The
campaign violated all three, and the φ story is what made the violation feel justified.

---

## 4. The seven changes that would actually have mattered

1. **Join on day 0.** Submission-days are the binding resource.
2. **Day-1 oracle decomposition.** For every metric component, measure the value of perfect knowledge of each
   latent variable. Rank the campaign's lanes by that, and re-rank weekly.
3. **Two validation views, always.** An optimistic one and an adverse one, chosen so that the *population* of
   the adverse one matches the test population. Ship only what wins on both.
4. **Audit the instrument before steering by it.** Probe the exact model you ship; re-probe when it changes.
5. **Identifiability line on every fit.** Free parameters, data points, competing models with equal RMSE.
   No closing a lane without it.
6. **A standing red team.** A daily review whose only job is to break the campaign's current central belief.
   The D-1 audit found three real errors in a few hours, one of them three days old and load-bearing.
   Run that on day 2, not D-1.
7. **50% of the campaign on the single hardest unsolved bit.** Everything else — fusion, bagging, calibration,
   blending — was worth ≤0.002 combined here, against +0.027 for the one thing we never properly attacked.

---

## 5. The honest counterfactual

**Rewritten 2026-09-21.** The original counterfactual leaned on the day-0 start, which the final data refutes
(§2a): the winners used 20–30 submissions against our 29, and 3rd place worked the same five-day window.

So the honest counterfactual is harder and less comfortable. We finished 0.92817 against 3rd's 0.94196 — a gap
of 0.01379 that splits roughly 60/40 between evidence (theirs ≈0.75–0.77 vs our 0.7218) and pair AP (theirs
≈0.985–0.990 vs our 0.9807). Neither is a resource gap. Both are a *model* gap that existed on the day we
joined and never closed, because the campaign spent its five days on breadth (65 mechanism statistics, 231
type-key features, 40 closed lanes, a dozen fusions worth ≤0.002 combined) instead of on the one latent
variable that an oracle decomposition would have priced at +0.027 on day one.

**The version of this campaign that wins is not a longer one. It is one that measures where the value is on
day one, validates on the population it will be scored on, refuses to let an unidentified fit close a lane, and
then spends three of its five days attacking a single bit with hardware chosen to fit that bit's shape.**
