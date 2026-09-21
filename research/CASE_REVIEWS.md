# Case reviews — `sub028_famspecw15_exp042ramp.csv`

Five pairs from the submitted ranking, chosen to cover **all three disclosed behaviour families** (the top of the ranking is dominated by directed transfer, so a plain top-5 would show only one pattern). For every submitted evidence hand this lists both partners' hole cards, every other seat's cards, the board, every action with pot and price, the omniscient equity at each partner decision, the signature that fired, and a benign alternative for that specific hand.

The data is synthetic and the labels are a benchmark, not an adjudication. A high score is a prompt for human review, never a finding of guilt.

## Case 1 — pair `PE4D799DCE834` · predicted **directed transfer**

- Rank **1 of 112,540** by submitted risk score (0.99999); the score is a rank within the evaluation set, so the top of the list is where review effort should go.
- Players: **A = `U1503AA4813BD`**, **B = `U3C54D3A546E2`**, sharing 222 hands in the scored period.
- Net over those shared hands: A +307 bb, B -399 bb.
- Submitted evidence: `H9F6E6EAB15A72D`, `HEADF53E86AF271`, `HF745F4A26D300F`, `HEA5A04F5B8A9B8`, `H328A8672EA1EE3`

### Evidence 1: `H9F6E6EAB15A72D` — table 138, hand #3778, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | · | Qd 2d | -2.5 | no |
| 1 | · | 3s 4d | +0.0 | no |
| 2 | · | 6d 4c | +0.0 | no |
| 3 | A | Kh 8h | +37.0 | yes |
| 4 | B | 9h 3d | -34.5 | yes |
| 5 | · | 7d Js | +0.0 | no |

Board: **5h 2c 8c As Ks** · final pot 71.5 bb

*preflop*
- other: fold (pot 1.5bb, to call 1.0bb)
- other: call 1.0bb (pot 1.5bb, to call 1.0bb)
- other: fold (pot 2.5bb, to call 1.0bb)
- other: fold (pot 2.5bb, to call 1.0bb)
- **A**: raise 2.0bb (pot 2.5bb, to call 0.5bb)  _[equity now: A 47%, B 25%]_
- **B**: call 1.5bb (pot 4.5bb, to call 1.5bb)  _[equity now: A 47%, B 25%]_
- other: call 1.5bb (pot 6.0bb, to call 1.5bb)
*flop*
- **A**: check (pot 7.5bb, to call 0.0bb)  _[equity now: A 65%, B 14%]_
- **B**: check (pot 7.5bb, to call 0.0bb)  _[equity now: A 65%, B 14%]_
- other: check (pot 7.5bb, to call 0.0bb)
*turn*
- **A**: check (pot 7.5bb, to call 0.0bb)  _[equity now: A 72%, B 14%]_
- **B**: bet 4.0bb (pot 7.5bb, to call 0.0bb)  _[equity now: A 72%, B 14%]_
- other: fold (pot 11.5bb, to call 4.0bb)
- **A**: raise 14.0bb (pot 11.5bb, to call 4.0bb)  _[equity now: A 86%, B 14%]_
- **B**: call 10.0bb (pot 25.5bb, to call 10.0bb)  _[equity now: A 86%, B 14%]_
*river*
- **A**: bet 18.0bb (pot 35.5bb, to call 0.0bb)  _[equity now: A 100%, B 0%]_
- **B**: call 18.0bb (pot 53.5bb, to call 18.0bb)  _[equity now: A 100%, B 0%]_

**What the model saw.** No fold-ahead in this hand: the chips move by **B paying A off** (B -69 chips, A +74). Under the listing rule this is a type-2 event -- a call-down or bet that transfers value without the fold signature.

**Benign alternative.** Paying off a better hand is the most ordinary loss in poker. One hand of this shape is pure variance; only the direction being consistently one-way across the pair's shared hands, and absent against everyone else at the table, makes it notable.

### Evidence 2: `HEADF53E86AF271` — table 138, hand #4181, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | A | 5h Qc | +97.5 | yes |
| 1 | · | 4c 6s | +0.0 | no |
| 2 | · | 4d Js | +0.0 | no |
| 3 | B | 5c 6c | -94.0 | yes |
| 4 | · | Td 7h | -1.0 | no |
| 5 | · | 8c Ac | -2.5 | no |

Board: **Th Qs 6d 4h 9d** · final pot 191.5 bb

*preflop*
- other: raise 2.5bb (pot 1.5bb, to call 1.0bb)
- **A**: raise 5.5bb (pot 4.0bb, to call 2.5bb)  _[equity now: A 20%, B 6%]_
- other: fold (pot 9.5bb, to call 5.5bb)
- other: fold (pot 9.5bb, to call 5.5bb)
- **B**: call 5.0bb (pot 9.5bb, to call 5.0bb)  _[equity now: A 22%, B 13%]_
- other: fold (pot 14.5bb, to call 4.5bb)
- other: fold (pot 14.5bb, to call 3.0bb)
*flop*
- **B**: bet 11.0bb (pot 14.5bb, to call 0.0bb)  _[equity now: A 95%, B 5%]_
- **A**: raise 47.5bb (pot 25.5bb, to call 11.0bb)  _[equity now: A 95%, B 5%]_
- **B**: call 36.5bb (pot 73.0bb, to call 36.5bb)  _[equity now: A 95%, B 5%]_
*turn*
- **B**: check (pot 109.5bb, to call 0.0bb)  _[equity now: A 97%, B 3%]_
- **A**: bet 27.5bb (pot 109.5bb, to call 0.0bb)  _[equity now: A 97%, B 3%]_
- **B**: call 27.5bb (pot 137.0bb, to call 27.5bb)  _[equity now: A 97%, B 3%]_
*river*
- **B**: bet 9.5bb (pot 164.5bb, to call 0.0bb)  _[equity now: A 100%, B 0%]_
- **A**: all-in 13.5bb (pot 174.0bb, to call 9.5bb)  _[equity now: A 100%, B 0%]_
- **B**: call 4.0bb (pot 187.5bb, to call 4.0bb)  _[equity now: A 100%, B 0%]_

**What the model saw.** No fold-ahead in this hand: the chips move by **B paying A off** (B -188 chips, A +195). Under the listing rule this is a type-2 event -- a call-down or bet that transfers value without the fold signature.

**Benign alternative.** Paying off a better hand is the most ordinary loss in poker. One hand of this shape is pure variance; only the direction being consistently one-way across the pair's shared hands, and absent against everyone else at the table, makes it notable.

### Evidence 3: `HF745F4A26D300F` — table 138, hand #3751, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | B | 8s 9h | -19.0 | no |
| 1 | · | 7h 2d | +0.0 | no |
| 2 | · | Kd 5s | -0.5 | no |
| 3 | · | Qs 7s | -1.0 | no |
| 4 | A | 6d 6c | +23.0 | no |
| 5 | · | Js 2c | -2.5 | no |

Board: **Qh Ac 4s 8d Ts** · final pot 63.5 bb

*preflop*
- **A**: raise 2.5bb (pot 1.5bb, to call 1.0bb)  _[equity now: A 22%, B 20%]_
- other: call 2.5bb (pot 4.0bb, to call 2.5bb)
- **B**: call 2.5bb (pot 6.5bb, to call 2.5bb)  _[equity now: A 22%, B 20%]_
- other: fold (pot 9.0bb, to call 2.5bb)
- other: fold (pot 9.0bb, to call 2.0bb)
- other: fold (pot 9.0bb, to call 1.5bb)
*flop*
- **A**: check (pot 9.0bb, to call 0.0bb)  _[equity now: A 55%, B 27%]_
- other: check (pot 9.0bb, to call 0.0bb)
- **B**: bet 9.0bb (pot 9.0bb, to call 0.0bb)  _[equity now: A 55%, B 27%]_
- **A**: call 9.0bb (pot 18.0bb, to call 9.0bb)  _[equity now: A 55%, B 27%]_
- other: fold (pot 27.0bb, to call 9.0bb)
*turn*
- **A**: bet 7.5bb (pot 27.0bb, to call 0.0bb)  _[equity now: A 6%, B 94%]_
- **B**: call 7.5bb (pot 34.5bb, to call 7.5bb)  _[equity now: A 6%, B 94%]_
*river*
- **A**: bet 21.5bb (pot 42.0bb, to call 0.0bb)  _[equity now: A 0%, B 100%]_
- **B**: fold (pot 63.5bb, to call 21.5bb)  _[equity now: A 0%, B 100%]_

**What the model saw.** **B folds on the river facing A's aggression while holding the better hand** (B 100% equity vs A 0% at that moment, all cards visible). That is the type-1 signature for directed transfer: the value goes to the partner without a showdown.

**Benign alternative.** B cannot see A's cards. Folding 100% equity to a large bet is a normal, if tight, laydown -- the equity edge is only visible to us. A single such fold is unremarkable; what the model actually scores is how often it recurs between these two and not with their other 28 table-mates.

### Evidence 4: `HEA5A04F5B8A9B8` — table 138, hand #4171, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | A | 6h 6s | +208.0 | yes |
| 1 | · | 4c Jc | +0.0 | no |
| 2 | · | 4s 7h | +0.0 | no |
| 3 | B | 8h Js | -105.0 | yes |
| 4 | · | Ac As | -102.0 | yes |
| 5 | · | 3c 5c | -1.0 | no |

Board: **6d 5s Kh 3s 4d** · final pot 436.5 bb

*preflop*
- **A**: raise 3.0bb (pot 1.5bb, to call 1.0bb)  _[equity now: A 19%, B 11%]_
- other: fold (pot 4.5bb, to call 3.0bb)
- other: fold (pot 4.5bb, to call 3.0bb)
- **B**: raise 7.0bb (pot 4.5bb, to call 3.0bb)  _[equity now: A 15%, B 11%]_
- other: call 6.5bb (pot 11.5bb, to call 6.5bb)
- other: fold (pot 18.0bb, to call 6.0bb)
- **A**: raise 18.0bb (pot 18.0bb, to call 4.0bb)  _[equity now: A 19%, B 14%]_
- **B**: call 14.0bb (pot 36.0bb, to call 14.0bb)  _[equity now: A 19%, B 14%]_
- other: raise 47.5bb (pot 50.0bb, to call 14.0bb)
- **A**: all-in 82.5bb (pot 97.5bb, to call 33.5bb)  _[equity now: A 19%, B 14%]_
- **B**: call 82.5bb (pot 180.0bb, to call 82.5bb)  _[equity now: A 19%, B 14%]_
- other: all-in 172.5bb (pot 262.5bb, to call 49.0bb)
- **B**: all-in 1.5bb (pot 435.0bb, to call 1.5bb)  _[equity now: A 19%, B 14%]_

**What the model saw.** No fold-ahead in this hand: the chips move by **B paying A off** (B -210 chips, A +416). Under the listing rule this is a type-2 event -- a call-down or bet that transfers value without the fold signature.

**Benign alternative.** Paying off a better hand is the most ordinary loss in poker. One hand of this shape is pure variance; only the direction being consistently one-way across the pair's shared hands, and absent against everyone else at the table, makes it notable.

### Evidence 5: `H328A8672EA1EE3` — table 138, hand #4186, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | A | Kd 4c | +57.5 | yes |
| 1 | · | As Jd | -1.0 | no |
| 2 | · | 9c Td | +0.0 | no |
| 3 | B | 4d 8c | -56.5 | yes |
| 4 | · | 3h 2c | +0.0 | no |
| 5 | · | 4h 3s | +0.0 | no |

Board: **Ad 2h Ac 2s 5h** · final pot 114.0 bb

*preflop*
- other: fold (pot 1.5bb, to call 1.0bb)
- **B**: call 1.0bb (pot 1.5bb, to call 1.0bb)  _[equity now: A 20%, B 18%]_
- other: fold (pot 2.5bb, to call 1.0bb)
- other: fold (pot 2.5bb, to call 1.0bb)
- **A**: raise 2.0bb (pot 2.5bb, to call 0.5bb)  _[equity now: A 23%, B 19%]_
- other: fold (pot 4.5bb, to call 1.5bb)
- **B**: call 1.5bb (pot 4.5bb, to call 1.5bb)  _[equity now: A 71%, B 29%]_
*flop*
- **A**: bet 5.0bb (pot 6.0bb, to call 0.0bb)  _[equity now: A 85%, B 15%]_
- **B**: call 5.0bb (pot 11.0bb, to call 5.0bb)  _[equity now: A 85%, B 15%]_
*turn*
- **A**: bet 11.0bb (pot 16.0bb, to call 0.0bb)  _[equity now: A 89%, B 11%]_
- **B**: call 11.0bb (pot 27.0bb, to call 11.0bb)  _[equity now: A 89%, B 11%]_
*river*
- **A**: bet 38.0bb (pot 38.0bb, to call 0.0bb)  _[equity now: A 100%, B 0%]_
- **B**: call 38.0bb (pot 76.0bb, to call 38.0bb)  _[equity now: A 100%, B 0%]_

**What the model saw.** No fold-ahead in this hand: the chips move by **B paying A off** (B -113 chips, A +115). Under the listing rule this is a type-2 event -- a call-down or bet that transfers value without the fold signature.

**Benign alternative.** Paying off a better hand is the most ordinary loss in poker. One hand of this shape is pure variance; only the direction being consistently one-way across the pair's shared hands, and absent against everyone else at the table, makes it notable.

## Case 2 — pair `PC178304F6F93` · predicted **soft play**

- Rank **26 of 112,540** by submitted risk score (0.99977); the score is a rank within the evaluation set, so the top of the list is where review effort should go.
- Players: **A = `U2B73D38F0F64`**, **B = `U54EBF17596CB`**, sharing 195 hands in the scored period.
- Net over those shared hands: A +46 bb, B +200 bb.
- Submitted evidence: `HF745A6904D873E`, `H22311A667D6951`, `H3965E8228C0661`, `H8EEE93EC74FFB5`, `H90FC271477B452`

### Evidence 1: `HF745A6904D873E` — table 212, hand #4659, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | · | 7h Kd | -1.0 | no |
| 1 | · | 9s 5c | +0.0 | no |
| 2 | A | 9d Ad | +5.0 | yes |
| 3 | · | 5d 4h | +0.0 | no |
| 4 | · | Ks 5h | +0.0 | no |
| 5 | B | Kh 8h | -4.0 | yes |

Board: **3d 4d 8d Ac Kc** · final pot 9.0 bb

*preflop*
- other: fold (pot 1.5bb, to call 1.0bb)
- **A**: call 1.0bb (pot 1.5bb, to call 1.0bb)  _[equity now: A 40%, B 23%]_
- other: fold (pot 2.5bb, to call 1.0bb)
- other: fold (pot 2.5bb, to call 1.0bb)
- **B**: call 0.5bb (pot 2.5bb, to call 0.5bb)  _[equity now: A 61%, B 24%]_
- other: check (pot 3.0bb, to call 0.0bb)
*flop*
- **B**: check (pot 3.0bb, to call 0.0bb)  _[equity now: A 97%, B 3%]_
- other: check (pot 3.0bb, to call 0.0bb)
- **A**: check (pot 3.0bb, to call 0.0bb)  _[equity now: A 97%, B 3%]_
*turn*
- **B**: bet 3.0bb (pot 3.0bb, to call 0.0bb)  _[equity now: A 100%, B 0%]_
- other: fold (pot 6.0bb, to call 3.0bb)
- **A**: call 3.0bb (pot 6.0bb, to call 3.0bb)  _[equity now: A 100%, B 0%]_
*river*
- **B**: check (pot 9.0bb, to call 0.0bb)  _[equity now: A 100%, B 0%]_
- **A**: check (pot 9.0bb, to call 0.0bb)  _[equity now: A 100%, B 0%]_

**What the model saw.** **A only calls B's bet** (3.0bb into 6.0bb) and never raises, and no raise between the partners appears anywhere in the hand; A's equity at that point was 100%.

**Benign alternative.** Loose-passive is the single most common recreational style: calling stations under-raise everyone, not just one player. This only becomes evidence if the same player raises other opponents in the same spots.

### Evidence 2: `H22311A667D6951` — table 212, hand #3107, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | · | 5h Th | -1.0 | no |
| 1 | · | 8s 4s | +0.0 | no |
| 2 | B | 9s Ks | +4.0 | no |
| 3 | A | Qc Kc | -2.5 | no |
| 4 | · | 3s Jc | +0.0 | no |
| 5 | · | 2h Td | -0.5 | no |

Board: **2s 7c As 9h 8h** · final pot 9.5 bb

*preflop*
- other: fold (pot 1.5bb, to call 1.0bb)
- **B**: raise 2.5bb (pot 1.5bb, to call 1.0bb)  _[equity now: A 36%, B 19%]_
- **A**: call 2.5bb (pot 4.0bb, to call 2.5bb)  _[equity now: A 36%, B 19%]_
- other: fold (pot 6.5bb, to call 2.5bb)
- other: fold (pot 6.5bb, to call 2.0bb)
- other: fold (pot 6.5bb, to call 1.5bb)
*flop*
- **B**: check (pot 6.5bb, to call 0.0bb)  _[equity now: A 57%, B 43%]_
- **A**: check (pot 6.5bb, to call 0.0bb)  _[equity now: A 57%, B 43%]_
*turn*
- **B**: check (pot 6.5bb, to call 0.0bb)  _[equity now: A 6%, B 94%]_
- **A**: check (pot 6.5bb, to call 0.0bb)  _[equity now: A 6%, B 94%]_
*river*
- **B**: bet 3.0bb (pot 6.5bb, to call 0.0bb)  _[equity now: A 0%, B 100%]_
- **A**: fold (pot 9.5bb, to call 3.0bb)  _[equity now: A 0%, B 100%]_

**What the model saw.** **A only calls B's bet** (2.5bb into 4.0bb) and never raises, and no raise between the partners appears anywhere in the hand; A's equity at that point was 36%.

**Benign alternative.** Loose-passive is the single most common recreational style: calling stations under-raise everyone, not just one player. This only becomes evidence if the same player raises other opponents in the same spots.

### Evidence 3: `H3965E8228C0661` — table 212, hand #3031, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | A | 9d Kh | +6.5 | no |
| 1 | · | 8d 2s | -0.5 | no |
| 2 | · | 6h Kd | -3.0 | no |
| 3 | B | Jc Qs | -3.0 | no |
| 4 | · | 7c Qd | +0.0 | no |
| 5 | · | 2c 7h | +0.0 | no |

Board: **Ad Jd 2h 7s 9s** · final pot 12.5 bb

*preflop*
- **B**: call 1.0bb (pot 1.5bb, to call 1.0bb)  _[equity now: A 19%, B 39%]_
- other: fold (pot 2.5bb, to call 1.0bb)
- other: fold (pot 2.5bb, to call 1.0bb)
- **A**: raise 3.0bb (pot 2.5bb, to call 1.0bb)  _[equity now: A 32%, B 36%]_
- other: fold (pot 5.5bb, to call 2.5bb)
- other: call 2.0bb (pot 5.5bb, to call 2.0bb)
- **B**: call 2.0bb (pot 7.5bb, to call 2.0bb)  _[equity now: A 41%, B 39%]_
*flop*
- other: check (pot 9.5bb, to call 0.0bb)
- **B**: check (pot 9.5bb, to call 0.0bb)  _[equity now: A 8%, B 86%]_
- **A**: check (pot 9.5bb, to call 0.0bb)  _[equity now: A 8%, B 86%]_
*turn*
- other: check (pot 9.5bb, to call 0.0bb)
- **B**: check (pot 9.5bb, to call 0.0bb)  _[equity now: A 6%, B 94%]_
- **A**: check (pot 9.5bb, to call 0.0bb)  _[equity now: A 6%, B 94%]_
*river*
- other: check (pot 9.5bb, to call 0.0bb)
- **B**: check (pot 9.5bb, to call 0.0bb)  _[equity now: A 0%, B 100%]_
- **A**: bet 3.0bb (pot 9.5bb, to call 0.0bb)  _[equity now: A 0%, B 100%]_
- other: fold (pot 12.5bb, to call 3.0bb)
- **B**: fold (pot 12.5bb, to call 3.0bb)  _[equity now: A 0%, B 100%]_

**What the model saw.** **B folds on the river facing A's aggression while holding the better hand** (B 100% equity vs A 0% at that moment, all cards visible). That is the type-1 signature for soft play: the value goes to the partner without a showdown.

**Benign alternative.** B cannot see A's cards. Folding 100% equity to a large bet is a normal, if tight, laydown -- the equity edge is only visible to us. A single such fold is unremarkable; what the model actually scores is how often it recurs between these two and not with their other 28 table-mates.

### Evidence 4: `H8EEE93EC74FFB5` — table 212, hand #4850, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | · | 5d Js | -1.0 | no |
| 1 | · | 6s 3d | +0.0 | no |
| 2 | · | 6h Ac | +0.0 | no |
| 3 | B | Kh Ks | +10.0 | yes |
| 4 | A | Th Td | -6.5 | yes |
| 5 | · | 7h Kc | -2.5 | no |

Board: **7s 9h 5h 5s 5c** · final pot 16.5 bb

*preflop*
- other: fold (pot 1.5bb, to call 1.0bb)
- other: fold (pot 1.5bb, to call 1.0bb)
- **B**: call 1.0bb (pot 1.5bb, to call 1.0bb)  _[equity now: A 25%, B 60%]_
- **A**: call 1.0bb (pot 2.5bb, to call 1.0bb)  _[equity now: A 25%, B 60%]_
- other: raise 2.0bb (pot 3.5bb, to call 0.5bb)
- other: fold (pot 5.5bb, to call 1.5bb)
- **B**: call 1.5bb (pot 5.5bb, to call 1.5bb)  _[equity now: A 26%, B 66%]_
- **A**: raise 5.5bb (pot 7.0bb, to call 1.5bb)  _[equity now: A 26%, B 66%]_
- other: fold (pot 12.5bb, to call 4.0bb)
- **B**: call 4.0bb (pot 12.5bb, to call 4.0bb)  _[equity now: A 21%, B 79%]_
*flop*
- **B**: check (pot 16.5bb, to call 0.0bb)  _[equity now: A 13%, B 87%]_
- **A**: check (pot 16.5bb, to call 0.0bb)  _[equity now: A 13%, B 87%]_
*turn*
- **B**: check (pot 16.5bb, to call 0.0bb)  _[equity now: A 6%, B 94%]_
- **A**: check (pot 16.5bb, to call 0.0bb)  _[equity now: A 6%, B 94%]_
*river*
- **B**: check (pot 16.5bb, to call 0.0bb)  _[equity now: A 0%, B 100%]_
- **A**: check (pot 16.5bb, to call 0.0bb)  _[equity now: A 0%, B 100%]_

**What the model saw.** **B only calls A's bet** (4.0bb into 12.5bb) and never raises, and no raise between the partners appears anywhere in the hand; B's equity at that point was 79%.

**Benign alternative.** Loose-passive is the single most common recreational style: calling stations under-raise everyone, not just one player. This only becomes evidence if the same player raises other opponents in the same spots.

### Evidence 5: `H90FC271477B452` — table 212, hand #3057, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | A | 7c 7s | +21.5 | no |
| 1 | · | Ks 5c | -1.0 | no |
| 2 | · | Tc 4s | +0.0 | no |
| 3 | B | Ah 8s | -14.0 | no |
| 4 | · | Jh 7d | +0.0 | no |
| 5 | · | Js Kc | -6.5 | no |

Board: **8h 4c 2d Qd** · final pot 79.5 bb

*preflop*
- other: call 1.0bb (pot 1.5bb, to call 1.0bb)
- other: fold (pot 2.5bb, to call 1.0bb)
- **B**: call 1.0bb (pot 2.5bb, to call 1.0bb)  _[equity now: A 26%, B 35%]_
- other: fold (pot 3.5bb, to call 1.0bb)
- other: raise 2.0bb (pot 3.5bb, to call 0.5bb)
- **A**: raise 5.5bb (pot 5.5bb, to call 1.5bb)  _[equity now: A 28%, B 37%]_
- other: fold (pot 11.0bb, to call 5.5bb)
- **B**: call 5.5bb (pot 11.0bb, to call 5.5bb)  _[equity now: A 28%, B 40%]_
- other: call 4.0bb (pot 16.5bb, to call 4.0bb)
*flop*
- other: check (pot 20.5bb, to call 0.0bb)
- **A**: check (pot 20.5bb, to call 0.0bb)  _[equity now: A 7%, B 76%]_
- **B**: bet 7.5bb (pot 20.5bb, to call 0.0bb)  _[equity now: A 7%, B 76%]_
- other: fold (pot 28.0bb, to call 7.5bb)
- **A**: call 7.5bb (pot 28.0bb, to call 7.5bb)  _[equity now: A 7%, B 93%]_
*turn*
- **A**: bet 44.0bb (pot 35.5bb, to call 0.0bb)  _[equity now: A 3%, B 97%]_
- **B**: fold (pot 79.5bb, to call 44.0bb)  _[equity now: A 3%, B 97%]_

**What the model saw.** **B folds on the turn facing A's aggression while holding the better hand** (B 97% equity vs A 3% at that moment, all cards visible). That is the type-1 signature for soft play: the value goes to the partner without a showdown.

**Benign alternative.** B cannot see A's cards. Folding 97% equity to a large bet is a normal, if tight, laydown -- the equity edge is only visible to us. A single such fold is unremarkable; what the model actually scores is how often it recurs between these two and not with their other 28 table-mates.

## Case 3 — pair `P88F46B09F1D2` · predicted **coordinated isolation**

- Rank **30 of 112,540** by submitted risk score (0.99973); the score is a rank within the evaluation set, so the top of the list is where review effort should go.
- Players: **A = `UA7C9B1557591`**, **B = `UFC3467ABF7FB`**, sharing 231 hands in the scored period.
- Net over those shared hands: A +181 bb, B -63 bb.
- Submitted evidence: `HA0B5BC8A1D70CD`, `H7ADA9BC5C64558`, `H6D0600BDF290CD`, `H3B9ADE5F4DD8B7`, `H23A864724614E6`

### Evidence 1: `HA0B5BC8A1D70CD` — table 172, hand #4126, blinds 2/4

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | · | 3d 6c | +0.0 | no |
| 1 | · | 6s Kd | +8.2 | no |
| 2 | · | 7s 5s | -1.0 | no |
| 3 | A | Kh 4h | -2.2 | no |
| 4 | · | 8h Tc | +0.0 | no |
| 5 | B | 9h 3h | -5.0 | no |

Board: **Th Kc 8c** · final pot 22.2 bb

*preflop*
- **A**: raise 2.2bb (pot 1.5bb, to call 1.0bb)  _[equity now: A 16%, B 13%]_
- other: fold (pot 3.8bb, to call 2.2bb)
- **B**: raise 5.0bb (pot 3.8bb, to call 2.2bb)  _[equity now: A 21%, B 18%]_
- other: fold (pot 8.8bb, to call 5.0bb)
- other: call 4.5bb (pot 8.8bb, to call 4.5bb)
- other: fold (pot 13.2bb, to call 4.0bb)
- **A**: fold (pot 13.2bb, to call 2.8bb)  _[equity now: A 34%, B 27%]_
*flop*
- other: bet 9.0bb (pot 13.2bb, to call 0.0bb)
- **B**: fold (pot 22.2bb, to call 9.0bb)  _[equity now: A folded, B 9%]_

**What the model saw.** The pair's first preflop raise came with **0 player(s) already folded** ahead of it (it was action #1 of the hand), so under the host's own listing convention this is a type-1 coordinated-isolation event. The point of the pattern is that the raise is made into a field that still contains the partner, who then gets out of the way cheaply.

**Benign alternative.** A wide early-position opening range is a real (if unprofitable) style, and the partner folding behind is ordinary tight play. Nothing here is visible to the raiser at the table: the two of them cannot see each other's cards, so the pattern only becomes notable across many hands.

### Evidence 2: `H7ADA9BC5C64558` — table 172, hand #4444, blinds 2/4

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | B | Ks 5d | +0.0 | no |
| 1 | · | 2c 3s | -0.5 | no |
| 2 | · | 3h 2d | -1.0 | no |
| 3 | · | 7s Ad | -3.0 | no |
| 4 | A | Qs 4s | +4.5 | no |
| 5 | · | Jd 4d | +0.0 | no |

Board: **(no flop)** · final pot 13.5 bb

*preflop*
- other: raise 3.0bb (pot 1.5bb, to call 1.0bb)
- **A**: raise 9.0bb (pot 4.5bb, to call 3.0bb)  _[equity now: A 22%, B 24%]_
- other: fold (pot 13.5bb, to call 9.0bb)
- **B**: fold (pot 13.5bb, to call 9.0bb)  _[equity now: A 23%, B 24%]_
- other: fold (pot 13.5bb, to call 8.5bb)
- other: fold (pot 13.5bb, to call 8.0bb)
- other: fold (pot 13.5bb, to call 6.0bb)

**What the model saw.** The pair's first preflop raise came with **0 player(s) already folded** ahead of it (it was action #2 of the hand), so under the host's own listing convention this is a type-1 coordinated-isolation event. The point of the pattern is that the raise is made into a field that still contains the partner, who then gets out of the way cheaply.

**Benign alternative.** A wide early-position opening range is a real (if unprofitable) style, and the partner folding behind is ordinary tight play. Nothing here is visible to the raiser at the table: the two of them cannot see each other's cards, so the pattern only becomes notable across many hands.

### Evidence 3: `H6D0600BDF290CD` — table 172, hand #4161, blinds 2/4

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | · | Ts Qs | +169.2 | yes |
| 1 | · | 7s 8d | -0.5 | no |
| 2 | · | 6d Qc | -1.0 | no |
| 3 | A | 5c 4c | -167.8 | yes |
| 4 | · | 3d 2c | +0.0 | no |
| 5 | B | Qh 2d | +0.0 | no |

Board: **Ac As Kc Td 7h** · final pot 346.5 bb

*preflop*
- **A**: raise 2.2bb (pot 1.5bb, to call 1.0bb)  _[equity now: A 18%, B 6%]_
- other: fold (pot 3.8bb, to call 2.2bb)
- **B**: fold (pot 3.8bb, to call 2.2bb)  _[equity now: A 24%, B 7%]_
- other: call 2.2bb (pot 3.8bb, to call 2.2bb)
- other: fold (pot 6.0bb, to call 1.8bb)
- other: fold (pot 6.0bb, to call 1.2bb)
*flop*
- **A**: bet 3.5bb (pot 6.0bb, to call 0.0bb)  _[equity now: A 50%, B folded]_
- other: raise 13.2bb (pot 9.5bb, to call 3.5bb)
- **A**: call 9.8bb (pot 22.8bb, to call 9.8bb)  _[equity now: A 50%, B folded]_
*turn*
- **A**: bet 21.8bb (pot 32.5bb, to call 0.0bb)  _[equity now: A 17%, B folded]_
- other: raise 70.5bb (pot 54.2bb, to call 21.8bb)
- **A**: raise 119.2bb (pot 124.8bb, to call 48.8bb)  _[equity now: A 17%, B folded]_
- other: call 70.5bb (pot 244.0bb, to call 70.5bb)
*river*
- **A**: bet 7.8bb (pot 314.5bb, to call 0.0bb)  _[equity now: A 0%, B folded]_
- other: all-in 20.8bb (pot 322.2bb, to call 7.8bb)
- **A**: all-in 3.5bb (pot 343.0bb, to call 3.5bb)  _[equity now: A 0%, B folded]_

**What the model saw.** The pair's first preflop raise came with **0 player(s) already folded** ahead of it (it was action #1 of the hand), so under the host's own listing convention this is a type-1 coordinated-isolation event. The point of the pattern is that the raise is made into a field that still contains the partner, who then gets out of the way cheaply.

**Benign alternative.** A wide early-position opening range is a real (if unprofitable) style, and the partner folding behind is ordinary tight play. Nothing here is visible to the raiser at the table: the two of them cannot see each other's cards, so the pattern only becomes notable across many hands.

### Evidence 4: `H3B9ADE5F4DD8B7` — table 172, hand #4167, blinds 2/4

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | · | 8c Th | +92.8 | yes |
| 1 | · | 7c 3c | +0.0 | no |
| 2 | · | 6c 5s | +0.0 | no |
| 3 | A | As 3s | -70.8 | yes |
| 4 | · | Qh 8s | -1.0 | no |
| 5 | B | Qs 4d | -21.0 | no |

Board: **Js 9d Kc Td 4s** · final pot 163.5 bb

*preflop*
- **B**: raise 2.0bb (pot 1.5bb, to call 1.0bb)  _[equity now: A 30%, B 10%]_
- other: call 2.0bb (pot 3.5bb, to call 2.0bb)
- other: fold (pot 5.5bb, to call 2.0bb)
- other: fold (pot 5.5bb, to call 2.0bb)
- **A**: raise 6.5bb (pot 5.5bb, to call 1.5bb)  _[equity now: A 45%, B 18%]_
- other: fold (pot 12.0bb, to call 6.0bb)
- **B**: raise 19.0bb (pot 12.0bb, to call 5.0bb)  _[equity now: A 40%, B 26%]_
- other: call 19.0bb (pot 31.0bb, to call 19.0bb)
- **A**: raise 40.2bb (pot 50.0bb, to call 14.0bb)  _[equity now: A 40%, B 26%]_
- **B**: fold (pot 90.2bb, to call 26.2bb)  _[equity now: A 40%, B 26%]_
- other: call 26.2bb (pot 90.2bb, to call 26.2bb)
*flop*
- **A**: check (pot 116.5bb, to call 0.0bb)  _[equity now: A 58%, B folded]_
- other: check (pot 116.5bb, to call 0.0bb)
*turn*
- **A**: bet 17.2bb (pot 116.5bb, to call 0.0bb)  _[equity now: A 14%, B folded]_
- other: all-in 23.5bb (pot 133.8bb, to call 17.2bb)
- **A**: call 6.2bb (pot 157.2bb, to call 6.2bb)  _[equity now: A 14%, B folded]_

**What the model saw.** The pair's first preflop raise came with **0 player(s) already folded** ahead of it (it was action #1 of the hand), so under the host's own listing convention this is a type-1 coordinated-isolation event. The point of the pattern is that the raise is made into a field that still contains the partner, who then gets out of the way cheaply.

**Benign alternative.** A wide early-position opening range is a real (if unprofitable) style, and the partner folding behind is ordinary tight play. Nothing here is visible to the raiser at the table: the two of them cannot see each other's cards, so the pattern only becomes notable across many hands.

### Evidence 5: `H23A864724614E6` — table 172, hand #4445, blinds 2/4

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | B | Kd 6s | -88.2 | yes |
| 1 | · | 6h Ah | -6.0 | no |
| 2 | · | 3d 5c | -0.5 | no |
| 3 | · | Jd 4c | -1.0 | no |
| 4 | A | 7c 8h | +98.2 | yes |
| 5 | · | Ac Jh | -2.5 | no |

Board: **3h 3c Qs 4h 8s** · final pot 186.5 bb

*preflop*
- **A**: raise 2.5bb (pot 1.5bb, to call 1.0bb)  _[equity now: A 23%, B 17%]_
- other: call 2.5bb (pot 4.0bb, to call 2.5bb)
- **B**: call 2.5bb (pot 6.5bb, to call 2.5bb)  _[equity now: A 23%, B 17%]_
- other: call 2.5bb (pot 9.0bb, to call 2.5bb)
- other: fold (pot 11.5bb, to call 2.0bb)
- other: fold (pot 11.5bb, to call 1.5bb)
*flop*
- **A**: check (pot 11.5bb, to call 0.0bb)  _[equity now: A 23%, B 13%]_
- other: check (pot 11.5bb, to call 0.0bb)
- **B**: check (pot 11.5bb, to call 0.0bb)  _[equity now: A 23%, B 13%]_
- other: check (pot 11.5bb, to call 0.0bb)
*turn*
- **A**: bet 3.5bb (pot 11.5bb, to call 0.0bb)  _[equity now: A 14%, B 6%]_
- other: fold (pot 15.0bb, to call 3.5bb)
- **B**: call 3.5bb (pot 15.0bb, to call 3.5bb)  _[equity now: A 14%, B 6%]_
- other: call 3.5bb (pot 18.5bb, to call 3.5bb)
*river*
- **A**: bet 27.2bb (pot 22.0bb, to call 0.0bb)  _[equity now: A 100%, B 0%]_
- **B**: raise 82.2bb (pot 49.2bb, to call 27.2bb)  _[equity now: A 100%, B 0%]_
- other: fold (pot 131.5bb, to call 80.5bb)
- **A**: call 55.0bb (pot 131.5bb, to call 55.0bb)  _[equity now: A 100%, B 0%]_

**What the model saw.** The pair's first preflop raise came with **0 player(s) already folded** ahead of it (it was action #1 of the hand), so under the host's own listing convention this is a type-1 coordinated-isolation event. The point of the pattern is that the raise is made into a field that still contains the partner, who then gets out of the way cheaply.

**Benign alternative.** A wide early-position opening range is a real (if unprofitable) style, and the partner folding behind is ordinary tight play. Nothing here is visible to the raiser at the table: the two of them cannot see each other's cards, so the pattern only becomes notable across many hands.

## Case 4 — pair `P8A8E7B5DF6FF` · predicted **directed transfer**

- Rank **2 of 112,540** by submitted risk score (0.99998); the score is a rank within the evaluation set, so the top of the list is where review effort should go.
- Players: **A = `U5264E27D18AE`**, **B = `U812E0DB2B2FD`**, sharing 145 hands in the scored period.
- Net over those shared hands: A -156 bb, B +126 bb.
- Submitted evidence: `H1B3D95F3C4FAC2`, `HD8893051FF28C9`, `HAB446445ECC1E1`, `H306218EB171AAC`, `HA74C86D575A096`

### Evidence 1: `H1B3D95F3C4FAC2` — table 362, hand #3828, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | · | Kc 7h | +0.0 | no |
| 1 | · | Kh Td | -2.5 | no |
| 2 | · | Tc 5h | -2.5 | no |
| 3 | · | 2s 9d | +0.0 | no |
| 4 | B | 4s As | +11.5 | no |
| 5 | A | 3d Jh | -6.5 | no |

Board: **Jc 4c 9s 5s** · final pot 44.5 bb

*preflop*
- other: fold (pot 1.5bb, to call 1.0bb)
- other: raise 2.5bb (pot 1.5bb, to call 1.0bb)
- other: call 2.5bb (pot 4.0bb, to call 2.5bb)
- other: fold (pot 6.5bb, to call 2.5bb)
- **B**: raise 6.0bb (pot 6.5bb, to call 2.0bb)  _[equity now: A 21%, B 40%]_
- **A**: call 5.5bb (pot 12.5bb, to call 5.5bb)  _[equity now: A 21%, B 40%]_
- other: fold (pot 18.0bb, to call 4.0bb)
- other: fold (pot 18.0bb, to call 4.0bb)
*flop*
- **B**: check (pot 18.0bb, to call 0.0bb)  _[equity now: A 71%, B 29%]_
- **A**: check (pot 18.0bb, to call 0.0bb)  _[equity now: A 71%, B 29%]_
*turn*
- **B**: bet 26.5bb (pot 18.0bb, to call 0.0bb)  _[equity now: A 64%, B 36%]_
- **A**: fold (pot 44.5bb, to call 26.5bb)  _[equity now: A 64%, B 36%]_

**What the model saw.** **A folds on the turn facing B's aggression while holding the better hand** (A 64% equity vs B 36% at that moment, all cards visible). That is the type-1 signature for directed transfer: the value goes to the partner without a showdown.

**Benign alternative.** A cannot see B's cards. Folding 64% equity to a large bet is a normal, if tight, laydown -- the equity edge is only visible to us. A single such fold is unremarkable; what the model actually scores is how often it recurs between these two and not with their other 28 table-mates.

### Evidence 2: `HD8893051FF28C9` — table 362, hand #4008, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | A | Ks 6h | -2.5 | no |
| 1 | · | 8s 5d | -0.5 | no |
| 2 | · | 2h 3h | -1.0 | no |
| 3 | B | Qh Jh | +6.5 | no |
| 4 | · | Js 4s | +0.0 | no |
| 5 | · | Ac Th | -2.5 | no |

Board: **6d 9h 5s 5h** · final pot 14.0 bb

*preflop*
- **B**: raise 2.5bb (pot 1.5bb, to call 1.0bb)  _[equity now: A 15%, B 19%]_
- other: fold (pot 4.0bb, to call 2.5bb)
- other: call 2.5bb (pot 4.0bb, to call 2.5bb)
- **A**: call 2.5bb (pot 6.5bb, to call 2.5bb)  _[equity now: A 17%, B 25%]_
- other: fold (pot 9.0bb, to call 2.0bb)
- other: fold (pot 9.0bb, to call 1.5bb)
*flop*
- **B**: check (pot 9.0bb, to call 0.0bb)  _[equity now: A 52%, B 24%]_
- other: check (pot 9.0bb, to call 0.0bb)
- **A**: check (pot 9.0bb, to call 0.0bb)  _[equity now: A 52%, B 24%]_
*turn*
- **B**: bet 5.0bb (pot 9.0bb, to call 0.0bb)  _[equity now: A 58%, B 28%]_
- other: fold (pot 14.0bb, to call 5.0bb)
- **A**: fold (pot 14.0bb, to call 5.0bb)  _[equity now: A 72%, B 28%]_

**What the model saw.** **A folds on the turn facing B's aggression while holding the better hand** (A 72% equity vs B 28% at that moment, all cards visible). That is the type-1 signature for directed transfer: the value goes to the partner without a showdown.

**Benign alternative.** A cannot see B's cards. Folding 72% equity to a large bet is a normal, if tight, laydown -- the equity edge is only visible to us. A single such fold is unremarkable; what the model actually scores is how often it recurs between these two and not with their other 28 table-mates.

### Evidence 3: `HAB446445ECC1E1` — table 362, hand #3834, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | · | 8d Kc | -1.0 | no |
| 1 | · | 4s 9h | +0.0 | no |
| 2 | · | 8c 5h | -1.0 | no |
| 3 | · | 6s 3s | +0.0 | no |
| 4 | B | 6h 9s | +4.0 | no |
| 5 | A | Jd 2s | -2.0 | no |

Board: **7h Kh Qs** · final pot 12.0 bb

*preflop*
- other: fold (pot 1.5bb, to call 1.0bb)
- other: call 1.0bb (pot 1.5bb, to call 1.0bb)
- other: fold (pot 2.5bb, to call 1.0bb)
- **B**: raise 2.0bb (pot 2.5bb, to call 1.0bb)  _[equity now: A 25%, B 21%]_
- **A**: call 1.5bb (pot 4.5bb, to call 1.5bb)  _[equity now: A 25%, B 21%]_
- other: fold (pot 6.0bb, to call 1.0bb)
- other: fold (pot 6.0bb, to call 1.0bb)
*flop*
- **A**: check (pot 6.0bb, to call 0.0bb)  _[equity now: A 72%, B 28%]_
- **B**: bet 6.0bb (pot 6.0bb, to call 0.0bb)  _[equity now: A 72%, B 28%]_
- **A**: fold (pot 12.0bb, to call 6.0bb)  _[equity now: A 72%, B 28%]_

**What the model saw.** **A folds on the flop facing B's aggression while holding the better hand** (A 72% equity vs B 28% at that moment, all cards visible). That is the type-1 signature for directed transfer: the value goes to the partner without a showdown.

**Benign alternative.** A cannot see B's cards. Folding 72% equity to a large bet is a normal, if tight, laydown -- the equity edge is only visible to us. A single such fold is unremarkable; what the model actually scores is how often it recurs between these two and not with their other 28 table-mates.

### Evidence 4: `H306218EB171AAC` — table 362, hand #3763, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | · | Th 6d | +0.0 | no |
| 1 | · | 8d 4c | +0.0 | no |
| 2 | B | Ad 9d | +88.0 | yes |
| 3 | A | 5h 2c | -86.0 | yes |
| 4 | · | 3s Qs | -1.0 | no |
| 5 | · | 7h 5s | -1.0 | no |

Board: **6h Kd 9c 9h 2d** · final pot 187.5 bb

*preflop*
- other: call 1.0bb (pot 1.5bb, to call 1.0bb)
- other: fold (pot 2.5bb, to call 1.0bb)
- other: fold (pot 2.5bb, to call 1.0bb)
- **B**: raise 2.5bb (pot 2.5bb, to call 1.0bb)  _[equity now: A 12%, B 40%]_
- **A**: call 2.0bb (pot 5.0bb, to call 2.0bb)  _[equity now: A 12%, B 40%]_
- other: fold (pot 7.0bb, to call 1.5bb)
- other: fold (pot 7.0bb, to call 1.5bb)
*flop*
- **A**: bet 8.5bb (pot 7.0bb, to call 0.0bb)  _[equity now: A 4%, B 96%]_
- **B**: call 8.5bb (pot 15.5bb, to call 8.5bb)  _[equity now: A 4%, B 96%]_
*turn*
- **A**: bet 18.0bb (pot 24.0bb, to call 0.0bb)  _[equity now: A 0%, B 100%]_
- **B**: call 18.0bb (pot 42.0bb, to call 18.0bb)  _[equity now: A 0%, B 100%]_
*river*
- **A**: all-in 70.5bb (pot 60.0bb, to call 0.0bb)  _[equity now: A 0%, B 100%]_
- **B**: all-in 57.0bb (pot 130.5bb, to call 57.0bb)  _[equity now: A 0%, B 100%]_

**What the model saw.** No fold-ahead in this hand: the chips move by **A paying B off** (A -172 chips, B +176). Under the listing rule this is a type-2 event -- a call-down or bet that transfers value without the fold signature.

**Benign alternative.** Paying off a better hand is the most ordinary loss in poker. One hand of this shape is pure variance; only the direction being consistently one-way across the pair's shared hands, and absent against everyone else at the table, makes it notable.

### Evidence 5: `HA74C86D575A096` — table 362, hand #4009, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | A | Js 5c | -72.0 | yes |
| 1 | · | 3d Jd | +0.0 | no |
| 2 | · | 8h 7d | +0.0 | no |
| 3 | B | Kd 2d | +74.5 | yes |
| 4 | · | 6d Th | +0.0 | no |
| 5 | · | 5s Kc | -2.5 | no |

Board: **Ad 6h Qh 2s Tc** · final pot 146.5 bb

*preflop*
- other: fold (pot 1.5bb, to call 1.0bb)
- other: fold (pot 1.5bb, to call 1.0bb)
- **B**: raise 2.5bb (pot 1.5bb, to call 1.0bb)  _[equity now: A 19%, B 25%]_
- other: fold (pot 4.0bb, to call 2.5bb)
- other: call 2.0bb (pot 4.0bb, to call 2.0bb)
- **A**: call 1.5bb (pot 6.0bb, to call 1.5bb)  _[equity now: A 21%, B 38%]_
*flop*
- other: check (pot 7.5bb, to call 0.0bb)
- **A**: check (pot 7.5bb, to call 0.0bb)  _[equity now: A 11%, B 47%]_
- **B**: check (pot 7.5bb, to call 0.0bb)  _[equity now: A 11%, B 47%]_
*turn*
- other: check (pot 7.5bb, to call 0.0bb)
- **A**: bet 5.5bb (pot 7.5bb, to call 0.0bb)  _[equity now: A 6%, B 89%]_
- **B**: call 5.5bb (pot 13.0bb, to call 5.5bb)  _[equity now: A 6%, B 89%]_
- other: fold (pot 18.5bb, to call 5.5bb)
*river*
- **A**: bet 12.5bb (pot 18.5bb, to call 0.0bb)  _[equity now: A 0%, B 100%]_
- **B**: raise 64.0bb (pot 31.0bb, to call 12.5bb)  _[equity now: A 0%, B 100%]_
- **A**: call 51.5bb (pot 95.0bb, to call 51.5bb)  _[equity now: A 0%, B 100%]_

**What the model saw.** No fold-ahead in this hand: the chips move by **A paying B off** (A -144 chips, B +149). Under the listing rule this is a type-2 event -- a call-down or bet that transfers value without the fold signature.

**Benign alternative.** Paying off a better hand is the most ordinary loss in poker. One hand of this shape is pure variance; only the direction being consistently one-way across the pair's shared hands, and absent against everyone else at the table, makes it notable.

## Case 5 — pair `P8E67334B0E04` · predicted **directed transfer**

- Rank **3 of 112,540** by submitted risk score (0.99997); the score is a rank within the evaluation set, so the top of the list is where review effort should go.
- Players: **A = `U56785476817A`**, **B = `UF5BDAA08411D`**, sharing 211 hands in the scored period.
- Net over those shared hands: A +274 bb, B -444 bb.
- Submitted evidence: `HC0F810438324E0`, `H9F305E58AD5929`, `HC6F0775882C4AA`, `H945017B5E5C347`, `H5FFA82F65F824E`

### Evidence 1: `HC0F810438324E0` — table 146, hand #3063, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | · | 6c 2c | +0.0 | no |
| 1 | · | 8s 7d | +0.0 | no |
| 2 | A | Qh 7s | +4.0 | no |
| 3 | B | 4s Js | -2.5 | no |
| 4 | · | 3s Kh | -0.5 | no |
| 5 | · | 5h Ah | -1.0 | no |

Board: **2d Jh Jd 9s** · final pot 14.5 bb

*preflop*
- other: fold (pot 1.5bb, to call 1.0bb)
- other: fold (pot 1.5bb, to call 1.0bb)
- **A**: raise 2.5bb (pot 1.5bb, to call 1.0bb)  _[equity now: A 21%, B 22%]_
- **B**: call 2.5bb (pot 4.0bb, to call 2.5bb)  _[equity now: A 21%, B 22%]_
- other: fold (pot 6.5bb, to call 2.0bb)
- other: fold (pot 6.5bb, to call 1.5bb)
*flop*
- **A**: check (pot 6.5bb, to call 0.0bb)  _[equity now: A 0%, B 100%]_
- **B**: check (pot 6.5bb, to call 0.0bb)  _[equity now: A 0%, B 100%]_
*turn*
- **A**: bet 8.0bb (pot 6.5bb, to call 0.0bb)  _[equity now: A 0%, B 100%]_
- **B**: fold (pot 14.5bb, to call 8.0bb)  _[equity now: A 0%, B 100%]_

**What the model saw.** **B folds on the turn facing A's aggression while holding the better hand** (B 100% equity vs A 0% at that moment, all cards visible). That is the type-1 signature for directed transfer: the value goes to the partner without a showdown.

**Benign alternative.** B cannot see A's cards. Folding 100% equity to a large bet is a normal, if tight, laydown -- the equity edge is only visible to us. A single such fold is unremarkable; what the model actually scores is how often it recurs between these two and not with their other 28 table-mates.

### Evidence 2: `H9F305E58AD5929` — table 146, hand #3056, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | · | 8s 9c | -0.5 | no |
| 1 | · | 3d Kd | -1.0 | no |
| 2 | A | 8h Th | +4.0 | no |
| 3 | B | Qh Tc | -2.5 | no |
| 4 | · | Jd Ac | +0.0 | no |
| 5 | · | 6s 9d | +0.0 | no |

Board: **3c Qs Qd** · final pot 14.5 bb

*preflop*
- **A**: raise 2.5bb (pot 1.5bb, to call 1.0bb)  _[equity now: A 12%, B 17%]_
- **B**: call 2.5bb (pot 4.0bb, to call 2.5bb)  _[equity now: A 12%, B 17%]_
- other: fold (pot 6.5bb, to call 2.5bb)
- other: fold (pot 6.5bb, to call 2.5bb)
- other: fold (pot 6.5bb, to call 2.0bb)
- other: fold (pot 6.5bb, to call 1.5bb)
*flop*
- **A**: bet 8.0bb (pot 6.5bb, to call 0.0bb)  _[equity now: A 1%, B 99%]_
- **B**: fold (pot 14.5bb, to call 8.0bb)  _[equity now: A 1%, B 99%]_

**What the model saw.** **B folds on the flop facing A's aggression while holding the better hand** (B 99% equity vs A 1% at that moment, all cards visible). That is the type-1 signature for directed transfer: the value goes to the partner without a showdown.

**Benign alternative.** B cannot see A's cards. Folding 99% equity to a large bet is a normal, if tight, laydown -- the equity edge is only visible to us. A single such fold is unremarkable; what the model actually scores is how often it recurs between these two and not with their other 28 table-mates.

### Evidence 3: `HC6F0775882C4AA` — table 146, hand #3068, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | · | 9d 7c | +0.0 | no |
| 1 | · | 8h 5h | +0.0 | no |
| 2 | A | Qh Ts | +3.5 | no |
| 3 | B | 6c 5c | -2.0 | no |
| 4 | · | 8c 3d | -0.5 | no |
| 5 | · | 9s 2s | -1.0 | no |

Board: **7d 5d As 6h** · final pot 11.0 bb

*preflop*
- other: fold (pot 1.5bb, to call 1.0bb)
- other: fold (pot 1.5bb, to call 1.0bb)
- **A**: raise 2.0bb (pot 1.5bb, to call 1.0bb)  _[equity now: A 41%, B 25%]_
- **B**: call 2.0bb (pot 3.5bb, to call 2.0bb)  _[equity now: A 41%, B 25%]_
- other: fold (pot 5.5bb, to call 1.5bb)
- other: fold (pot 5.5bb, to call 1.0bb)
*flop*
- **A**: check (pot 5.5bb, to call 0.0bb)  _[equity now: A 30%, B 70%]_
- **B**: check (pot 5.5bb, to call 0.0bb)  _[equity now: A 30%, B 70%]_
*turn*
- **A**: bet 5.5bb (pot 5.5bb, to call 0.0bb)  _[equity now: A 0%, B 100%]_
- **B**: fold (pot 11.0bb, to call 5.5bb)  _[equity now: A 0%, B 100%]_

**What the model saw.** **B folds on the turn facing A's aggression while holding the better hand** (B 100% equity vs A 0% at that moment, all cards visible). That is the type-1 signature for directed transfer: the value goes to the partner without a showdown.

**Benign alternative.** B cannot see A's cards. Folding 100% equity to a large bet is a normal, if tight, laydown -- the equity edge is only visible to us. A single such fold is unremarkable; what the model actually scores is how often it recurs between these two and not with their other 28 table-mates.

### Evidence 4: `H945017B5E5C347` — table 146, hand #3805, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | · | 3h Qd | +0.0 | no |
| 1 | · | 2s Kd | -0.5 | no |
| 2 | B | 6d Kc | -2.5 | no |
| 3 | · | 8c Ah | +0.0 | no |
| 4 | A | Kh As | +3.0 | no |
| 5 | · | 4s 3c | +0.0 | no |

Board: **Tc 6s Qh** · final pot 11.0 bb

*preflop*
- other: fold (pot 1.5bb, to call 1.0bb)
- **A**: raise 2.5bb (pot 1.5bb, to call 1.0bb)  _[equity now: A 34%, B 15%]_
- other: fold (pot 4.0bb, to call 2.5bb)
- other: fold (pot 4.0bb, to call 2.5bb)
- other: fold (pot 4.0bb, to call 2.0bb)
- **B**: call 1.5bb (pot 4.0bb, to call 1.5bb)  _[equity now: A 68%, B 32%]_
*flop*
- **B**: check (pot 5.5bb, to call 0.0bb)  _[equity now: A 31%, B 69%]_
- **A**: bet 5.5bb (pot 5.5bb, to call 0.0bb)  _[equity now: A 31%, B 69%]_
- **B**: fold (pot 11.0bb, to call 5.5bb)  _[equity now: A 31%, B 69%]_

**What the model saw.** **B folds on the flop facing A's aggression while holding the better hand** (B 69% equity vs A 31% at that moment, all cards visible). That is the type-1 signature for directed transfer: the value goes to the partner without a showdown.

**Benign alternative.** B cannot see A's cards. Folding 69% equity to a large bet is a normal, if tight, laydown -- the equity edge is only visible to us. A single such fold is unremarkable; what the model actually scores is how often it recurs between these two and not with their other 28 table-mates.

### Evidence 5: `H5FFA82F65F824E` — table 146, hand #3069, blinds 1/2

| seat | who | hole cards | net (bb) | showdown |
|---|---|---|---|---|
| 0 | · | Ks 5c | +0.0 | no |
| 1 | · | 9d 3h | +0.0 | no |
| 2 | A | 6d 8c | +1.5 | no |
| 3 | B | 9s Ah | -0.5 | no |
| 4 | · | 3c 2h | -1.0 | no |
| 5 | · | Ts 8d | +0.0 | no |

Board: **(no flop)** · final pot 3.5 bb

*preflop*
- other: fold (pot 1.5bb, to call 1.0bb)
- other: fold (pot 1.5bb, to call 1.0bb)
- other: fold (pot 1.5bb, to call 1.0bb)
- **A**: raise 2.0bb (pot 1.5bb, to call 1.0bb)  _[equity now: A 34%, B 44%]_
- **B**: fold (pot 3.5bb, to call 1.5bb)  _[equity now: A 34%, B 44%]_
- other: fold (pot 3.5bb, to call 1.0bb)

**What the model saw.** **B folds on the preflop facing A's aggression while holding the better hand** (B 44% equity vs A 34% at that moment, all cards visible). That is the type-1 signature for directed transfer: the value goes to the partner without a showdown.

**Benign alternative.** B cannot see A's cards. Folding 44% equity to a large bet is a normal, if tight, laydown -- the equity edge is only visible to us. A single such fold is unremarkable; what the model actually scores is how often it recurs between these two and not with their other 28 table-mates.
