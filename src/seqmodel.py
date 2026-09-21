"""exp032 -- LEARNED SEQUENCE MODEL over a hand's ordered action stream, for a pair of interest (A,B).

Hypothesis: every current feature is a hand-crafted aggregate (src/engine.py, ~90 cols per (hand,pair)); a model
that reads the ORDERED action sequence directly may capture the planted pattern with a different inductive bias.
Two downstream tasks are tested against their incumbents on the SAME rows and the SAME frozen table folds:
  (1) hand detector      4-class (clean-pair hand / DT / SP / CI planted)  vs src/handdet.py LightGBM
  (2) evidence ranker     in-pair listed-vs-not among a positive pair's hands  vs src/evidence.py stage-1
plus BLENDs (seq score as an extra LightGBM feature / rank-average) which is where a different bias usually pays.

Representation: one example = (hand, unordered pair {A,B}); A = lower player_idx (engine convention).
  token t = one action: emb(action type) + emb(street) + emb(role A/B/other) + emb(position in hand)
            + 28 continuous: 15 seat-independent (amount/to_call/pot/stack in bb, bet fraction, players_active,
            actor perceived strength hs_actor + missing flag, policy p_fold/p_check/p_call/p_agg, surprise,
            n_agg_street, all-in, to_call>0), 8 A/B-paired (omniscient eq_pre/eq_post/delta for A and B,
            "actor is facing A" / "facing B"), 5 pair-derived (actor eq_pre/delta, A-vs-B heads-up equity on this
            street, max eq_pre of the other seats).
  46 static: 10 hand-level (n_actions, board_n, pot, players dealt/showdown, blind, pre/post action counts,
            aggression count), 6 pair-symmetric (hu equity per street, both-post-active, both-showdown),
            2x15 A/B block (position vs button, folded, showdown, won_share, net bb, contribution, stack,
            preflop equity vs1/vs5, own action/aggression counts, fold street, final hand category,
            #opponents better, best-at-showdown).
Orientation carries no signal by construction: the token/static layout is built so that swapping A/B is a fixed
column permutation (SWAP_C / SWAP_S) + role remap; training uses random swap augmentation and inference averages
both orientations.

Stages:  python src/seqmodel.py cache | train_hd | train_ev | confirm
  cache     labelled-pair rows of ph_v2/phase0 + ragged action/equity/policy/strength arrays -> data/derived/seq_cache/
  train_hd  4-class detector, 5 fold models, OOF AUC + listed-hand recall@q999 per family, vs a LightGBM re-fit on
            the identical rows/folds, plus both blends (rank-average, seq margins as extra GBDT features)
  train_ev  in-pair evidence ranker (3 family heads on a shared trunk), OOF MAP@5 per family with the host AP@5
            convention (denominator min(|relevant|,5)), vs the engine stage-1 and the nested exp008 two-stage,
            with and without the seq margins; plus a rank-average against the shipped exp008 dev score file
  confirm   paired re-run of the train_ev two-stage for chosen families over both stage-2 objectives
  hd_lgb    (exp033) round-3 LightGBM hand detector + 4 seq margins -> data/derived/handdet_models_seq/
  hs_score  (exp033) re-score soft-play-gate (G0) pair-hand rows with it, others keep hs_v3 -> hs_seq/
            (SEQ_EVAL_SAMEFOLD=1 + SEQ_HS_DIR=hs_seqsf: eval tables scored by their own fold model)
  hs_agg    (exp033) handdet.agg on that dir with the mixed-score thresholds -> pfh<SEQ_PFH_TAG>_<window>
The evidence lane has no eval stage (its dev verdict did not earn one; cost in SEQMODEL.md section 5). The hand
detector lane was scored on eval by hs_score for exp033 and failed the pair-model bar (SEQMODEL.md section 7).
Doc: research/SEQMODEL.md
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "6")
os.environ.setdefault("OMP_NUM_THREADS", "6")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import polars as pl

import common as C

T0 = time.time()
c = pl.col
PH = C.DER / os.environ.get("SEQ_PH", "ph_v2")
HS_PREV = C.DER / os.environ.get("SEQ_HS_PREV", "hs_v3")   # previous-round hand scores -> self-train extras
SEQ = C.DER / "seq_cache"
FAMS = list(C.FAMILIES)
SC = ["s_dt", "s_sp", "s_ci"]
MAXLEN = 34           # max actions in any hand is 33 (99.9% <= 22)
NCONT = 28
NSTAT = 46
NSEATF = 15           # per-seat static block

# swap permutations (A <-> B)
SWAP_C = np.arange(NCONT)
for _i in (15, 17, 19, 21):
    SWAP_C[_i], SWAP_C[_i + 1] = _i + 1, _i
SWAP_S = np.arange(NSTAT)
SWAP_S[16:31], SWAP_S[31:46] = np.arange(31, 46), np.arange(16, 31)

PI = np.full((6, 6), -1, np.int64)   # seat pair -> 0..14 (i<j enumeration, matches hu_equity column order)
_k = 0
for _i in range(6):
    for _j in range(_i + 1, 6):
        PI[_i, _j] = PI[_j, _i] = _k
        _k += 1


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ============================================================================================== cache
def _rows():
    """One row per (dev hand, labelled pair) with label/family/fold/is_ev + the pair's seats."""
    lab = (C.labelled_pairs().select(c("p1").alias("pA"), c("p2").alias("pB"), "pair_id", "label",
                                     c("behavior_family").alias("family"), "fold"))
    ev = C.load("evidence").select("pair_id", "hand_idx", pl.lit(1, pl.Int8).alias("is_ev"))
    R = (pl.scan_parquet(PH / "phase0" / "*.parquet")
         .select("hand_idx", "table_idx", "hand_seq", "pA", "pB", "seatA", "seatB")
         .join(lab.lazy(), on=["pA", "pB"], how="inner").collect())
    R = (R.join(ev, on=["pair_id", "hand_idx"], how="left")
         .with_columns(c("is_ev").fill_null(0))
         .with_columns(n_rel=c("is_ev").cast(pl.Int32).sum().over("pair_id"))
         .sort("pair_id", "hand_seq"))
    assert R["is_ev"].sum() == ev.height, (R["is_ev"].sum(), ev.height)
    # self-train weak positives: unlisted hands of positive pairs whose previous-round family score > .5
    prev = (pl.scan_parquet(HS_PREV / "phase0" / "*.parquet").select("hand_idx", "pA", "pB", *SC)
            .join(R.lazy().filter(c("label") == 1).select("hand_idx", "pA", "pB"),
                  on=["hand_idx", "pA", "pB"], how="semi").collect())
    R = R.join(prev, on=["hand_idx", "pA", "pB"], how="left")
    s_fam = pl.coalesce([pl.when(c("family") == f).then(c(s)) for f, s in zip(FAMS, SC)])
    R = R.with_columns(s_fam.alias("s_fam")).with_columns(
        is_extra=((c("label") == 1) & (c("is_ev") == 0) & (c("s_fam") > 0.5)).cast(pl.Int8)).drop(SC)
    log(f"rows: {R.height:,} pair-hand rows | {R['pair_id'].n_unique()} pairs | listed {R['is_ev'].sum()} | "
        f"extras {R['is_extra'].sum()} | pos rows {int((R['label'] == 1).sum()):,}")
    return R


try:
    from numba import njit
except ImportError:                                                 # pragma: no cover
    njit = lambda **kw: (lambda f: f)


@njit(cache=True)
def _last_aggr(hrow, street, seat, is_aggr):
    """Seat of the last aggressor on the actor's current street (-1 if none yet)."""
    out = np.full(hrow.shape[0], -1, np.int8)
    ph, ps, cur = np.int64(-1), np.int8(-1), np.int8(-1)
    for i in range(hrow.shape[0]):
        if hrow[i] != ph or street[i] != ps:
            ph, ps, cur = hrow[i], street[i], np.int8(-1)
        out[i] = cur
        if is_aggr[i]:
            cur = seat[i]
    return out


def _seq_arrays(hands, phase=0, quiet=False):
    """Ragged per-hand action arrays for the given (sorted) hand_idx array of one phase (0 dev, 1 eval)."""
    ptag = "dev" if phase == 0 else "eval"
    lo, hi = int(hands[0]), int(hands[-1])
    rng_f = c("hand_idx").is_between(lo, hi)
    hm = pl.DataFrame({"hand_idx": hands, "hrow": np.arange(len(hands), dtype=np.int32)})
    pol = (pl.scan_parquet(C.DER / "action_policy.parquet").filter(rng_f)
           .select("hand_idx", "action_no", "p_fold", "p_check", "p_call", "p_agg", "surprise", "n_agg_street"))
    eqf = (pl.scan_parquet(C.DER / f"action_equity_{ptag}.parquet").filter(rng_f)
           .select("hand_idx", "action_no", "seat_no", *[f"eq_pre_s{i}" for i in range(6)],
                   *[f"eq_post_s{i}" for i in range(6)], "hs_actor"))
    A = (pl.scan_parquet(C.DER / "actions.parquet").filter(rng_f).join(hm.lazy(), on="hand_idx", how="inner")
         .join(pol, on=["hand_idx", "action_no"], how="left")
         .join(eqf, on=["hand_idx", "action_no"], how="left")
         .sort("hrow", "action_no").collect())
    if not quiet:
        log(f"actions: {A.height:,} rows for {len(hands):,} hands")
    assert A["seat_no"].null_count() == 0, "missing action_equity rows"

    hrow = A["hrow"].to_numpy()
    cnt = np.bincount(hrow, minlength=len(hands)).astype(np.int64)
    hoff = np.concatenate([[0], np.cumsum(cnt)])
    bb = (pl.scan_parquet(C.DER / "hands.parquet").filter(rng_f).join(hm.lazy(), on="hand_idx", how="inner")
          .sort("hrow").collect())
    bbv = bb["bb"].to_numpy().astype(np.float32)
    bb_a = bbv[hrow]

    f = lambda k: A[k].to_numpy().astype(np.float32)
    amount, to_call, pot, stack = f("amount"), f("to_call"), f("pot_before"), f("stack_before")
    act = A["action"].to_numpy().astype(np.int8)
    street = A["street"].to_numpy().astype(np.int8)
    seat = A["seat_no"].to_numpy().astype(np.int8)
    hs = np.nan_to_num(f("hs_actor"), nan=-1.0)
    si = np.zeros((A.height, 15), np.float32)
    si[:, 0] = np.log1p(amount / bb_a)
    si[:, 1] = np.log1p(to_call / bb_a)
    si[:, 2] = np.log1p(pot / bb_a)
    si[:, 3] = np.log1p(stack / bb_a)
    si[:, 4] = np.clip(amount / (pot + to_call + 1.0), 0, 3)
    si[:, 5] = f("players_active") / 6.0
    si[:, 6] = np.where(hs < 0, 0.0, hs)
    si[:, 7] = (hs < 0).astype(np.float32)
    si[:, 8] = np.nan_to_num(f("p_fold"))
    si[:, 9] = np.nan_to_num(f("p_check"))
    si[:, 10] = np.nan_to_num(f("p_call"))
    si[:, 11] = np.nan_to_num(f("p_agg"))
    si[:, 12] = np.clip(np.nan_to_num(f("surprise")), 0, 8) / 4.0
    si[:, 13] = np.nan_to_num(f("n_agg_street")) / 3.0
    si[:, 14] = (act == 5).astype(np.float32)
    eqpre = np.nan_to_num(A.select([f"eq_pre_s{i}" for i in range(6)]).to_numpy().astype(np.float32))
    eqpost = np.nan_to_num(A.select([f"eq_post_s{i}" for i in range(6)]).to_numpy().astype(np.float32))

    # last aggressor seat on the current street + is_aggr flag (aggr = bet/raise, or all-in above to_call)
    is_aggr = ((act == 3) | (act == 4) | ((act == 5) & (amount > to_call))).astype(np.int8)
    lastag = _last_aggr(hrow.astype(np.int64), street, seat, is_aggr)
    return dict(hoff=hoff, t_act=act, t_street=street, t_seat=seat, t_si=si, t_eqpre=eqpre, t_eqpost=eqpost,
                t_lastag=lastag, t_aggr=is_aggr, bb=bbv, hands=hands)


def _hand_static(hands, seq, phase=0):
    """Per-hand symmetric block (H,16) and per-seat block (H,6,15)."""
    H = len(hands)
    rng_f = c("hand_idx").is_between(int(hands[0]), int(hands[-1]))
    hm = pl.DataFrame({"hand_idx": hands, "hrow": np.arange(H, dtype=np.int32)})
    hd = (pl.scan_parquet(C.DER / "hands.parquet").filter(rng_f).join(hm.lazy(), on="hand_idx", how="inner")
          .select("hrow", "button_seat", "bb", "final_pot", "players_dealt", "players_at_showdown",
                  pl.col("board").list.len().alias("board_n")).sort("hrow").collect())
    bb = hd["bb"].to_numpy().astype(np.float32)
    hoff, street, seat, aggr = seq["hoff"], seq["t_street"], seq["t_seat"], seq["t_aggr"]
    nact = np.diff(hoff)
    hrow_t = np.repeat(np.arange(H), nact)
    npre = np.bincount(hrow_t, weights=(street == 0), minlength=H).astype(np.float32)
    nagg = np.bincount(hrow_t, weights=aggr, minlength=H).astype(np.float32)
    sym = np.zeros((H, 16), np.float32)
    sym[:, 0] = nact / MAXLEN
    sym[:, 1] = hd["board_n"].to_numpy() / 5.0
    sym[:, 2] = np.log1p(hd["final_pot"].to_numpy() / bb)
    sym[:, 3] = hd["players_dealt"].to_numpy() / 6.0
    sym[:, 4] = hd["players_at_showdown"].to_numpy() / 6.0
    sym[:, 5] = np.log1p(bb)
    sym[:, 6] = npre / 12.0
    sym[:, 7] = (nact - npre) / 22.0
    sym[:, 8] = nagg / 10.0
    sym[:, 9] = 0.0                                   # (filled per-example: nothing hand-level left)
    # sym[:,10..15] are pair-dependent (hu equity x4, both-post-active, both-showdown) -> filled per example

    st = (pl.scan_parquet(C.DER / "seat_strength.parquet").filter(rng_f).join(hm.lazy(), on="hand_idx", how="inner")
          .select("hrow", "seat_no", "pf_eq_vs1", "pf_eq_vs5", "fold_street", "n_opp_better_last",
                  "best_at_last_board", pl.coalesce("cat_river", "cat_turn", "cat_flop").alias("cat_last"))
          .collect())
    se = (pl.scan_parquet(C.DER / "seats.parquet").filter(rng_f).join(hm.lazy(), on="hand_idx", how="inner")
          .select("hrow", "seat_no", "starting_stack", "total_contribution", "net_chips", "folded",
                  "went_to_showdown", "won_share")
          .join(st.lazy(), on=["hrow", "seat_no"], how="left").sort("hrow", "seat_no").collect())
    assert se.height == 6 * H, (se.height, H)
    hr, sn = se["hrow"].to_numpy().astype(np.int64), se["seat_no"].to_numpy().astype(np.int64)
    assert np.array_equal(hr * 6 + sn, np.arange(6 * H)), "seats frame not complete/sorted"
    bbs = bb[hr]
    g = lambda k, fill=0.0: np.nan_to_num(se[k].to_numpy().astype(np.float32), nan=fill)
    net = g("net_chips")
    blk = np.zeros((6 * H, NSEATF), np.float32)
    blk[:, 0] = ((sn - hd["button_seat"].to_numpy()[hr]) % 6) / 5.0
    blk[:, 1] = g("folded")
    blk[:, 2] = g("went_to_showdown")
    blk[:, 3] = g("won_share")
    blk[:, 4] = np.sign(net) * np.log1p(np.abs(net) / bbs) / 5.0
    blk[:, 5] = np.log1p(g("total_contribution") / bbs)
    blk[:, 6] = np.log1p(g("starting_stack") / bbs)
    blk[:, 7] = g("pf_eq_vs1")
    blk[:, 8] = g("pf_eq_vs5")
    # per-seat action / aggression counts
    key = hrow_t * 6 + seat.astype(np.int64)
    blk[:, 9] = np.bincount(key, minlength=6 * H)[:6 * H] / 10.0
    blk[:, 10] = np.bincount(key, weights=aggr, minlength=6 * H)[:6 * H] / 5.0
    blk[:, 11] = np.nan_to_num(se["fold_street"].to_numpy().astype(np.float32), nan=4.0) / 4.0
    blk[:, 12] = np.nan_to_num(se["cat_last"].to_numpy().astype(np.float32), nan=-1.0) / 8.0
    blk[:, 13] = np.nan_to_num(se["n_opp_better_last"].to_numpy().astype(np.float32), nan=-1.0) / 5.0
    blk[:, 14] = g("best_at_last_board")
    seatblk = np.zeros((H, 6, NSEATF), np.float32)
    seatblk[hr, sn] = blk

    hu = (pl.scan_parquet(C.DER / f"hu_equity_{'dev' if phase == 0 else 'eval'}.parquet").filter(rng_f)
          .join(hm.lazy(), on="hand_idx", how="inner").sort("hrow").collect())
    assert hu.height == H, (hu.height, H)
    hucols = [f"hu_s{s}_{i}{j}" for s in range(4) for i in range(6) for j in range(i + 1, 6)]
    huv = np.nan_to_num(hu.select(hucols).to_numpy().astype(np.float32), nan=-1.0).reshape(H, 4, 15)
    return sym, seatblk, huv


def build_examples(cache, idx, out=None):
    """Padded tensors for example rows idx: cont (n,MAXLEN,NCONT) f16, tok (n,MAXLEN,3) i8, stat (n,NSTAT) f16."""
    hrow = cache["ex_hrow"][idx]
    sa, sb = cache["ex_seatA"][idx], cache["ex_seatB"][idx]
    n = len(idx)
    hoff = cache["hoff"]
    L = np.minimum(np.diff(hoff)[hrow], MAXLEN)
    pos = np.arange(MAXLEN)[None, :]
    mask = pos < L[:, None]
    gi = np.clip(hoff[hrow][:, None] + pos, 0, len(cache["t_act"]) - 1)
    m = mask.astype(np.float32)

    cont = np.zeros((n, MAXLEN, NCONT), np.float32)
    si_g = cache["t_si"][gi]
    cont[:, :, :15] = si_g
    eqpre, eqpost = cache["t_eqpre"][gi], cache["t_eqpost"][gi]          # (n,L,6)
    r = np.arange(n)[:, None]
    epa, epb = eqpre[r, pos, sa[:, None]], eqpre[r, pos, sb[:, None]]
    qpa, qpb = eqpost[r, pos, sa[:, None]], eqpost[r, pos, sb[:, None]]
    cont[:, :, 15], cont[:, :, 16] = epa, epb
    cont[:, :, 17], cont[:, :, 18] = qpa, qpb
    cont[:, :, 19], cont[:, :, 20] = qpa - epa, qpb - epb
    lastag = cache["t_lastag"][gi]
    seat = cache["t_seat"][gi]
    tocall = si_g[:, :, 1] > 0
    cont[:, :, 21] = ((lastag == sa[:, None]) & tocall & (seat != sa[:, None])).astype(np.float32)
    cont[:, :, 22] = ((lastag == sb[:, None]) & tocall & (seat != sb[:, None])).astype(np.float32)
    act_eqpre = eqpre[r, pos, seat]
    cont[:, :, 23] = act_eqpre
    cont[:, :, 24] = eqpost[r, pos, seat] - act_eqpre
    street = cache["t_street"][gi]
    cont[:, :, 25] = cache["huv"][hrow[:, None], street, PI[sa, sb][:, None]]
    other = np.ones((n, MAXLEN, 6), bool)
    other[r, pos, sa[:, None]] = False
    other[r, pos, sb[:, None]] = False
    cont[:, :, 26] = np.where(other, eqpre, -1).max(axis=2)
    cont[:, :, 27] = tocall.astype(np.float32)
    cont *= m[:, :, None]

    role = np.where(seat == sa[:, None], 1, np.where(seat == sb[:, None], 2, 0)).astype(np.int8)
    tok = (np.stack([cache["t_act"][gi], street, role], axis=2) * mask[:, :, None]).astype(np.int8)

    stat = np.zeros((n, NSTAT), np.float32)
    stat[:, :10] = cache["sym"][hrow, :10]
    stat[:, 10:14] = cache["huv"][hrow, :, PI[sa, sb]]          # (n,4) A-vs-B hu equity per street
    A_blk = cache["seatblk"][hrow, sa]
    B_blk = cache["seatblk"][hrow, sb]
    stat[:, 14] = ((A_blk[:, 1] == 0) & (B_blk[:, 1] == 0)).astype(np.float32)
    stat[:, 15] = ((A_blk[:, 2] > 0) & (B_blk[:, 2] > 0)).astype(np.float32)
    stat[:, 16:31] = A_blk
    stat[:, 31:46] = B_blk
    return cont.astype(np.float16), tok, mask, stat.astype(np.float16)


def stage_cache():
    SEQ.mkdir(parents=True, exist_ok=True)
    R = _rows()
    hands = np.sort(R["hand_idx"].unique().to_numpy())
    seq = _seq_arrays(hands)
    sym, seatblk, huv = _hand_static(hands, seq)
    hpos = pl.DataFrame({"hand_idx": hands, "hrow": np.arange(len(hands), dtype=np.int32)})
    R = R.join(hpos, on="hand_idx", how="left")
    R.write_parquet(SEQ / "rows.parquet")
    np.savez(SEQ / "seq.npz", hoff=seq["hoff"], t_act=seq["t_act"], t_street=seq["t_street"], t_seat=seq["t_seat"],
             t_si=seq["t_si"], t_eqpre=seq["t_eqpre"], t_eqpost=seq["t_eqpost"], t_lastag=seq["t_lastag"],
             t_aggr=seq["t_aggr"], sym=sym, seatblk=seatblk, huv=huv, hands=hands)
    log(f"cache written: {R.height:,} rows, {len(hands):,} hands, {len(seq['t_act']):,} tokens, "
        f"{(SEQ / 'seq.npz').stat().st_size / 1e6:.0f} MB")


def load_cache():
    z = np.load(SEQ / "seq.npz")
    R = pl.read_parquet(SEQ / "rows.parquet")
    cache = {k: z[k] for k in z.files}
    cache["ex_hrow"] = R["hrow"].to_numpy()
    cache["ex_seatA"] = R["seatA"].to_numpy().astype(np.int64)
    cache["ex_seatB"] = R["seatB"].to_numpy().astype(np.int64)
    cache["R"] = R
    return cache


# ============================================================================================== model
def _torch():
    import torch
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "6")))
    return torch


def make_tensors(cache, idx, chunk=20000):
    torch = _torch()
    n = len(idx)
    cont = torch.empty((n, MAXLEN, NCONT), dtype=torch.float16)
    tok = torch.empty((n, MAXLEN, 3), dtype=torch.int8)
    msk = torch.empty((n, MAXLEN), dtype=torch.bool)
    stat = torch.empty((n, NSTAT), dtype=torch.float16)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        cn, tk, mk, st = build_examples(cache, idx[s:e])
        cont[s:e], tok[s:e], msk[s:e], stat[s:e] = (torch.from_numpy(cn), torch.from_numpy(tk),
                                                    torch.from_numpy(mk), torch.from_numpy(st))
    log(f"tensors: {n:,} examples, {cont.nbytes / 1e6:.0f} MB")
    return cont, tok, msk, stat


def build_net(nout, d=96, layers=3, heads=4, ff=None, drop=0.1, arch="tf"):
    torch = _torch()
    nn = torch.nn

    class SeqNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(NCONT, d)
            self.e_act, self.e_str = nn.Embedding(6, d), nn.Embedding(4, d)
            self.e_role, self.e_pos = nn.Embedding(3, d), nn.Embedding(MAXLEN, d)
            self.ln = nn.LayerNorm(d)
            if arch == "gru":
                self.enc = nn.GRU(d, d // 2, num_layers=layers, batch_first=True, bidirectional=True,
                                  dropout=drop if layers > 1 else 0.0)
            else:
                lay = nn.TransformerEncoderLayer(d, heads, ff or 2 * d, dropout=drop, batch_first=True,
                                                 norm_first=True, activation="gelu")
                self.enc = nn.TransformerEncoder(lay, layers)
            self.sln = nn.LayerNorm(NSTAT)
            self.head = nn.Sequential(nn.Linear(2 * d + NSTAT, 128), nn.GELU(), nn.Dropout(drop),
                                      nn.Linear(128, nout))

        def forward(self, cont, tok, msk, stat):
            pos = torch.arange(MAXLEN, device=cont.device)
            x = (self.proj(cont) + self.e_act(tok[:, :, 0]) + self.e_str(tok[:, :, 1])
                 + self.e_role(tok[:, :, 2]) + self.e_pos(pos)[None])
            x = self.ln(x)
            if arch == "gru":
                h, _ = self.enc(x)
            else:
                h = self.enc(x, src_key_padding_mask=~msk)
            m = msk.unsqueeze(-1).to(h.dtype)
            mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
            mx = h.masked_fill(~msk.unsqueeze(-1), -1e4).max(1).values
            return self.head(torch.cat([mean, mx, self.sln(stat)], dim=1))

    return SeqNet()


def _swap(cont, tok, stat, swc, sws):
    """Exchange the A/B roles: fixed column permutation of the continuous/static blocks + role remap 1<->2."""
    torch = _torch()
    role = tok[:, :, 2]
    tok2 = tok.clone()
    tok2[:, :, 2] = torch.where(role == 1, torch.full_like(role, 2),
                                torch.where(role == 2, torch.ones_like(role), role))
    return cont[:, :, swc], tok2, stat[:, sws]


def run_model(tens, y, w, fold, nout, loss_fn, metric_fn, epochs=10, bs=512, lr=2e-3, seed=0, arch="tf",
              d=96, layers=3, drop=0.1, folds=range(5), log_every=None, device=None, save_dir=None, init_dir=None):
    """Cross-fit on frozen table folds. Returns OOF raw outputs (n,nout) with orientation averaging."""
    torch = _torch()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cont, tok, msk, stat = tens
    n = cont.shape[0]
    budget = float(os.environ.get("SEQ_VRAM_MB", "1100")) * 1e6
    if dev == "cuda" and sum(t.nbytes for t in tens) < budget:
        cont, tok, msk, stat = (t.to(dev) for t in tens)       # whole population resident on the GPU
        log(f" tensors on GPU ({sum(t.nbytes for t in tens) / 1e6:.0f} MB)")
    swc = torch.as_tensor(SWAP_C, device=dev)
    sws = torch.as_tensor(SWAP_S, device=dev)
    yt = torch.as_tensor(y).to(cont.device)
    wt = torch.as_tensor(w, dtype=torch.float32).to(cont.device)
    oof = np.zeros((n, nout), np.float32)
    for k in folds:
        torch.manual_seed(seed * 100 + k)
        tr = np.flatnonzero(fold != k)
        va = np.flatnonzero(fold == k)
        net = build_net(nout, d=d, layers=layers, drop=drop, arch=arch).to(dev)
        if init_dir is not None:      # transfer: load the pretrained trunk, leave the task head fresh
            sd = torch.load(Path(init_dir) / f"fold{k}.pt", map_location=dev)
            sd = {kk: v for kk, v in sd.items() if not kk.startswith("head.")}
            miss = net.load_state_dict(sd, strict=False)
            log(f" fold {k}: trunk initialised from {Path(init_dir).name} "
                f"(missing {len(miss.missing_keys)}, unexpected {len(miss.unexpected_keys)})")
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
        steps = epochs * max(1, len(tr) // bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
        scaler = torch.amp.GradScaler(dev, enabled=(dev == "cuda"))
        rng = np.random.default_rng(seed * 1000 + k)
        step = 0
        net.train()
        for ep in range(epochs):
            perm = rng.permutation(tr)
            for s in range(0, len(perm) - bs + 1, bs):
                b = torch.as_tensor(np.sort(perm[s:s + bs]), device=cont.device)
                cb = cont[b].to(dev, non_blocking=True).float()
                tb = tok[b].to(dev).long()
                mb = msk[b].to(dev)
                sb = stat[b].to(dev).float()
                if rng.random() < 0.5:
                    cb, tb, sb = _swap(cb, tb, sb, swc, sws)
                with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                    out = net(cb, tb, mb, sb)
                    loss = loss_fn(out, yt[b].to(dev), wt[b].to(dev))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                if step < steps - 1:
                    sched.step()
                step += 1
            if log_every and (ep + 1) % log_every == 0:
                log(f"  fold {k} ep {ep + 1}/{epochs} loss {loss.item():.4f}")
        net.eval()
        with torch.no_grad():
            for s in range(0, len(va), 4096):
                b = torch.as_tensor(va[s:s + 4096], device=cont.device)
                cb = cont[b].to(dev).float()
                tb = tok[b].to(dev).long()
                mb = msk[b].to(dev)
                sb = stat[b].to(dev).float()
                o1 = net(cb, tb, mb, sb).float()
                cs, ts, ss = _swap(cb, tb, sb, swc, sws)
                o2 = net(cs, ts, mb, ss).float()
                oof[va[s:s + 4096]] = ((o1 + o2) / 2).cpu().numpy()
        log(f" fold {k}: {len(tr):,} train / {len(va):,} val | {metric_fn(oof, va)}")
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save({kk: v.cpu() for kk, v in net.state_dict().items()}, Path(save_dir) / f"fold{k}.pt")
        del net, opt
        torch.cuda.empty_cache() if dev == "cuda" else None
    return oof


# ============================================================================================ hand detector
def _engine_rows(R):
    """The ph_v2 engine features (handdet.py's orientation-free transform) for exactly the rows of R, in R order."""
    import handdet as HD
    F = (HD.hand_features(pl.scan_parquet(PH / "phase0" / "*.parquet"))
         .join(R.lazy().select("hand_idx", "pA", "pB", pl.int_range(pl.len()).alias("ord")),
               on=["hand_idx", "pA", "pB"], how="inner").collect().sort("ord"))
    assert F.height == R.height, (F.height, R.height)
    return F.select(HD.feat_names()).to_numpy().astype(np.float32), HD.feat_names()


def hd_population(cache):
    """handdet.py's training population: class 0 = negative-pair hands, 1/2/3 = listed evidence (+ self-train extras)."""
    R = cache["R"]
    sel = ((c("label") == 0) | (c("is_ev") == 1) | (c("is_extra") == 1))
    idx = np.flatnonzero(R.select(sel).to_series().to_numpy())
    S = R[idx]
    fam = S["family"].to_numpy()
    y = np.where(S["label"].to_numpy() == 0, 0,
                 np.select([fam == f for f in FAMS], [1, 2, 3], 0)).astype(np.int64)
    w = np.where(y == 0, 1.0, 25.0)
    w = np.where(S["is_extra"].to_numpy() == 1, 8.0, w).astype(np.float32)
    return idx, S, y, w


def hd_metrics(P, y, sub=None):
    """P = (n,4) probabilities. AUC of each family head vs class-0 rows, and any-vs-neg."""
    from sklearn.metrics import roc_auc_score
    if sub is not None:
        P, y = P[sub], y[sub]
    out = {}
    for i, f in enumerate(FAMS, 1):
        m = (y == 0) | (y == i)
        out[f"auc_{f}"] = round(float(roc_auc_score(y[m] == i, P[m, i])), 5)
        o = y > 0
        out[f"auc_{f}_vs_otherfam"] = round(float(roc_auc_score(y[o] == i, P[o, i])), 5)
    out["auc_any"] = round(float(roc_auc_score(y > 0, 1 - P[:, 0])), 5)
    return out


def stage_train_hd(epochs=10, arch="tf", d=96, layers=3, seed=0, with_lgb=True, tag="hd", bs=512, save=None):
    torch = _torch()
    cache = load_cache()
    idx, S, y, w = hd_population(cache)
    fold = S["fold"].to_numpy()
    log(f"hand detector population: {len(idx):,} rows | neg {int((y == 0).sum()):,} | "
        f"DT {int((y == 1).sum())} SP {int((y == 2).sum())} CI {int((y == 3).sum())} | folds {np.bincount(fold)}")
    tens = make_tensors(cache, idx)

    def loss_fn(out, yy, ww):
        return (torch.nn.functional.cross_entropy(out, yy, reduction="none") * ww).sum() / ww.sum()

    def met(oof, va):
        P = torch.softmax(torch.as_tensor(oof[va]), dim=1).numpy()
        return {k: v for k, v in hd_metrics(P, y[va]).items() if not k.endswith("otherfam")}

    t = time.time()
    oof = run_model(tens, y, w, fold, 4, loss_fn, met, epochs=epochs, arch=arch, d=d, layers=layers, seed=seed,
                    bs=bs, save_dir=(SEQ / save) if save else None)
    P = torch.softmax(torch.as_tensor(oof), dim=1).numpy()
    res = {"seq": hd_metrics(P, y), "seq_runtime_s": round(time.time() - t, 1),
           "n_rows": int(len(idx)), "epochs": epochs, "arch": arch, "d": d, "layers": layers, "seed": seed,
           "bs": bs,
           "vram_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if torch.cuda.is_available() else 0}
    np.save(SEQ / f"oof_{tag}_seq.npy", P)
    np.save(SEQ / f"oof_{tag}_y.npy", y)
    np.save(SEQ / f"oof_{tag}_idx.npy", idx)
    log("SEQ  " + json.dumps(res["seq"]))

    if with_lgb:
        import lightgbm as lgb
        t = time.time()
        X, fn = _engine_rows(S)
        params = dict(objective="multiclass", num_class=4, learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
                      feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=6,
                      verbose=-1, seed=7)
        Pl = np.zeros((len(y), 4))
        for k in range(5):
            tr = fold != k
            m = lgb.train(params, lgb.Dataset(X[tr], y[tr], weight=w[tr], feature_name=fn), 400)
            Pl[fold == k] = m.predict(X[fold == k])
        res["lgb"] = hd_metrics(Pl, y)
        res["lgb_runtime_s"] = round(time.time() - t, 1)
        np.save(SEQ / f"oof_{tag}_lgb.npy", Pl)
        log("LGB  " + json.dumps(res["lgb"]))
        # blends: LGBM with the seq probabilities as extra features, and a rank average
        from scipy.stats import rankdata
        rk = lambda a: rankdata(a) / len(a)
        res["rankavg"] = {}
        for wgt in (0.25, 0.5):
            Pb = np.zeros_like(Pl)
            for i in range(4):
                Pb[:, i] = (1 - wgt) * rk(Pl[:, i]) + wgt * rk(P[:, i])
            res["rankavg"][str(wgt)] = hd_metrics(Pb, y)
            log(f"RANKAVG w={wgt}  " + json.dumps(res['rankavg'][str(wgt)]))
        t = time.time()
        X2 = np.hstack([X, np.log(np.clip(P, 1e-6, 1)) - np.log(np.clip(1 - P, 1e-6, 1))]).astype(np.float32)
        fn2 = fn + [f"seq_logit_{i}" for i in range(4)]
        Pc = np.zeros((len(y), 4))
        for k in range(5):
            tr = fold != k
            m = lgb.train(params, lgb.Dataset(X2[tr], y[tr], weight=w[tr], feature_name=fn2), 400)
            Pc[fold == k] = m.predict(X2[fold == k])
        res["lgb_plus_seq"] = hd_metrics(Pc, y)
        res["lgb_plus_seq_runtime_s"] = round(time.time() - t, 1)
        np.save(SEQ / f"oof_{tag}_lgbseq.npy", Pc)
        log("LGB+SEQ  " + json.dumps(res["lgb_plus_seq"]))
    (SEQ / f"res_{tag}.json").write_text(json.dumps(res, indent=1))
    return res


# ========================================================================================== evidence ranker
def _ev_align(cache):
    """The positive-pair rows of the seq cache, aligned 1:1 with evidence.py's dev_pos frame."""
    import evidence as EV
    R = cache["R"]
    pos = np.flatnonzero(R["label"].to_numpy() == 1)
    S = R[pos]
    F = EV.load_dev()
    assert F.height == S.height, (F.height, S.height)
    assert np.array_equal(F["hand_idx"].to_numpy(), S["hand_idx"].to_numpy()), "row order mismatch vs dev_pos"
    assert (F["pair_id"].to_numpy() == S["pair_id"].to_numpy()).all()
    return pos, S, F, EV.Dev(F)


def _map5_rows(EV, dev, score, rows):
    """Per-family MAP@5 over the pairs whose rows are in `rows` (folds are by table, pairs never split)."""
    m = EV.map5_frame(dev.pair[rows], dev.fam[rows], dev.y[rows], dev.n_rel[rows], score[rows], dev.tb[rows])
    return {EV.SHORT[f]: round(float(m.filter(c("family") == f)["ap"].mean()), 4) for f in FAMS}


def _two_stage(EV, dev, F, X, s1_rounds=200, s2_rounds=350, report=(100, 200, 350), obj2="binary", label="",
               fams=None):
    """exp008-style nested two-stage. Stage-2 is TRAINED on inner (3-fold) stage-1 scores and SERVED the sharper
    4-fold stage-1 (the protocol note in src/evidence_v4.py). Returns {rounds: {fam: MAP@5}} + stage-1 MAP."""
    sig = EV.sigmoid
    fams = fams or FAMS
    n = len(dev.y)
    s1 = np.zeros(n)
    for f in fams:
        fm = dev.fam == f
        for k in range(5):
            va = fm & (dev.fold == k)
            m = EV.fit(dev, X, fm & (dev.fold != k), "binary", s1_rounds, False)
            s1[va] = EV.predict(m, X[va], "binary", False, f)
    s1_map = _map5_rows(EV, dev, s1, np.arange(n))
    log(f"  [{label}] stage-1 MAP@5 {json.dumps(s1_map)}")
    s2 = {r: np.zeros(n) for r in report}
    for f in fams:
        fm = dev.fam == f
        for k in range(5):
            s1_tr = s1.copy()
            for j in range(5):
                if j == k:
                    continue
                rows = fm & (dev.fold == j)
                mj = EV.fit(dev, X, fm & (dev.fold != k) & (dev.fold != j), "binary", s1_rounds, False)
                s1_tr[rows] = EV.predict(mj, X[rows], "binary", False, f)
            X2tr = np.hstack([X, EV.chrono_features(F, sig(s1_tr), f).to_numpy().astype(np.float32)])
            X2va = np.hstack([X, EV.chrono_features(F, sig(s1), f).to_numpy().astype(np.float32)])
            m2 = EV.fit(dev, X2tr, fm & (dev.fold != k), obj2, s2_rounds, False)
            va = fm & (dev.fold == k)
            for r in report:
                s2[r][va] = EV.predict(m2, X2va[va], obj2, False, f, r)
        log(f"  [{label}] {EV.SHORT[f]} done ({time.time() - T0:.0f}s)")
    out = {"stage1": s1_map, "stage2": {str(r): _map5_rows(EV, dev, s2[r], np.arange(n)) for r in report}}
    out["stage2_rowmean"] = {EV.SHORT[f]: round(float(np.mean([out["stage2"][str(r)][EV.SHORT[f]] for r in report])), 4)
                             for f in fams}
    return out, s1, s2


def stage_train_ev(epochs=14, arch="tf", d=96, layers=3, seed=0, bs=256, tag="ev", posw=5.0, twostage=True,
                   init=None, lr=1.5e-3):
    import evidence as EV
    torch = _torch()
    cache = load_cache()
    pos, S, F, dev = _ev_align(cache)
    famcode = np.select([dev.fam == f for f in FAMS], [0, 1, 2]).astype(np.int64)
    y2 = np.stack([dev.y.astype(np.float32), famcode.astype(np.float32)], axis=1)
    w = np.ones(len(pos), np.float32)
    fold = dev.fold
    log(f"evidence population: {len(pos):,} rows | {dev.F['pair_id'].n_unique()} pairs | listed {int(dev.y.sum())}")
    tens = make_tensors(cache, pos)
    pwt = torch.tensor(posw)

    def loss_fn(out, yy, ww):
        tgt, fc = yy[:, 0], yy[:, 1].long()
        lg = out.gather(1, fc[:, None]).squeeze(1)
        return torch.nn.functional.binary_cross_entropy_with_logits(lg, tgt, pos_weight=pwt.to(out.device))

    def met(oof, va):
        own = oof[np.arange(len(oof)), famcode]
        return _map5_rows(EV, dev, own, va)

    t = time.time()
    oof = run_model(tens, y2, w, fold, 3, loss_fn, met, epochs=epochs, arch=arch, d=d, layers=layers, seed=seed,
                    bs=bs, lr=lr, init_dir=(SEQ / init) if init else None)
    own = oof[np.arange(len(oof)), famcode]
    res = {"seq": _map5_rows(EV, dev, own, np.arange(len(own))), "seq_runtime_s": round(time.time() - t, 1),
           "epochs": epochs, "arch": arch, "d": d, "layers": layers, "seed": seed, "bs": bs, "posw": posw,
           "init": init, "lr": lr,
           "vram_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if torch.cuda.is_available() else 0}
    np.save(SEQ / f"oof_{tag}_seq.npy", oof)
    log("SEQ (no chronology) MAP@5  " + json.dumps(res["seq"]))

    # ---- incumbent stage-1 on identical rows, and the same + seq logits as extra features
    base = EV.base_feature_names(F)
    X = F.select([c(k).cast(pl.Float32) for k in base]).to_numpy()
    Xs = np.hstack([X, oof.astype(np.float32)]).astype(np.float32)   # seq head margins as 3 extra features
    from scipy.stats import rankdata
    res["blend_rankavg_exp008"] = {}
    inc = (pl.read_parquet(C.DER / "evidence_scores_dev_exp008.parquet")
           .select("pair_id", "hand_idx", *[f"s_{f}" for f in FAMS]))
    J = F.select("pair_id", "hand_idx").join(inc, on=["pair_id", "hand_idx"], how="left")
    assert J.height == F.height and J[f"s_{FAMS[0]}"].null_count() == 0
    inc_own = J.select([c(f"s_{f}") for f in FAMS]).to_numpy()[np.arange(len(own)), famcode]
    res["exp008_dev_file"] = _map5_rows(EV, dev, inc_own, np.arange(len(own)))
    log("exp008 dev score file MAP@5  " + json.dumps(res["exp008_dev_file"]))
    rk = lambda a: rankdata(a) / len(a)
    for wgt in (0.15, 0.3, 0.5):
        b = (1 - wgt) * rk(inc_own) + wgt * rk(own)
        res["blend_rankavg_exp008"][str(wgt)] = _map5_rows(EV, dev, b, np.arange(len(own)))
        log(f"RANKAVG exp008 + seq w={wgt}  " + json.dumps(res['blend_rankavg_exp008'][str(wgt)]))

    if twostage:
        res["base"], _, _ = _two_stage(EV, dev, F, X, label="base")
        log("TWO-STAGE base       " + json.dumps(res["base"]))
        res["base_plus_seq"], _, _ = _two_stage(EV, dev, F, Xs, label="base+seq")
        log("TWO-STAGE base+seq   " + json.dumps(res["base_plus_seq"]))
    (SEQ / f"res_{tag}.json").write_text(json.dumps(res, indent=1))
    return res


def _per_pair_ap(EV, dev, score, fam):
    """pair_id -> host AP@5 for one family, same convention as evidence.py."""
    m = EV.map5_frame(dev.pair, dev.fam, dev.y, dev.n_rel, score, dev.tb).filter(c("family") == fam)
    return dict(zip(m["pair_id"].to_list(), m["ap"].to_list()))


def stage_confirm(tag="ev", fams=("coordinated_isolation",), objs=("binary", "lambdarank")):
    """Paired confirmation of the train_ev finding for chosen families: same nested two-stage protocol, same folds,
    with and without the seq head margins, over the stage-2 objectives the incumbent actually uses."""
    import evidence as EV
    from scipy.stats import spearmanr
    cache = load_cache()
    pos, S, F, dev = _ev_align(cache)
    famcode = np.select([dev.fam == f for f in FAMS], [0, 1, 2]).astype(np.int64)
    oof = np.load(SEQ / f"oof_{tag}_seq.npy")
    own = oof[np.arange(len(oof)), famcode]
    base = EV.base_feature_names(F)
    X = F.select([c(k).cast(pl.Float32) for k in base]).to_numpy()
    Xs = np.hstack([X, oof.astype(np.float32)]).astype(np.float32)
    res = {"fams": list(fams), "objs": list(objs)}
    for obj in objs:
        b, s1b, _s2b = _two_stage(EV, dev, F, X, obj2=obj, label=f"base|{obj}", fams=list(fams))
        p_, s1p, _s2p = _two_stage(EV, dev, F, Xs, obj2=obj, label=f"base+seq|{obj}", fams=list(fams))
        res[obj] = {"base": b, "base_plus_seq": p_,
                    "delta_rowmean": {k: round(p_["stage2_rowmean"][k] - b["stage2_rowmean"][k], 4)
                                      for k in b["stage2_rowmean"]}}
        log(f"CONFIRM {obj}: base {json.dumps(b['stage2_rowmean'])} | +seq {json.dumps(p_['stage2_rowmean'])} "
            f"| delta {json.dumps(res[obj]['delta_rowmean'])}")
        res[obj]["rank_corr_seq_vs_stage1"] = {
            EV.SHORT[f]: round(float(spearmanr(own[dev.fam == f], s1b[dev.fam == f]).statistic), 4) for f in fams}
        log(f"  rank corr(seq, engine stage-1) {json.dumps(res[obj]['rank_corr_seq_vs_stage1'])}")
        # paired bootstrap over the family's pairs (the campaign's decision statistic) per stage-2 rounds setting
        res[obj]["paired_boot"] = {}
        for r, sb in _s2b.items():
            sp = _s2p[r]
            for f in fams:
                ab = _per_pair_ap(EV, dev, sb, f)
                apv = _per_pair_ap(EV, dev, sp, f)
                keys = sorted(set(ab) & set(apv))
                db = np.array([apv[k] - ab[k] for k in keys])
                rng = np.random.default_rng(0)
                bs = db[rng.integers(0, len(db), (4000, len(db)))].mean(axis=1)
                res[obj]["paired_boot"][f"{EV.SHORT[f]}_r{r}"] = dict(
                    n_pairs=len(db), delta=round(float(db.mean()), 4), p_better=round(float((bs > 0).mean()), 3))
        log(f"  paired bootstrap {json.dumps(res[obj]['paired_boot'])}")
    (SEQ / f"res_confirm_{tag}.json").write_text(json.dumps(res, indent=1))
    return res


# =========================================================== exp033: SP recall lead -> hand scores -> pair model
# The seq margins become 4 extra columns of the round-3 LightGBM hand detector (population = handdet.py SELFTRAIN=1,
# W_EXTRA=8, extras from hs_v2 -- exactly how hs_v3 was made). The new detector is applied ONLY on soft-play gate G0
# rows (neither partner's first preflop action is a non-facing fold: ~22% of pair-hand rows, 100% of SP evidence);
# every other row keeps its hs_v3 score. Output hs_seq/ has the hs_v3 schema; thresholds for the aggregation are
# recomputed on the MIXED negative-hand OOF scores and live in handdet_models_seq/ (incumbent files untouched).
HD_SEQ = C.DER / "handdet_models_seq"
HS_SEQ = C.DER / os.environ.get("SEQ_HS_DIR", "hs_seq")
# SEQ_EVAL_SAMEFOLD=1: score each EVAL table with its own frozen-fold model (the one that never saw that table's dev
# rows), exactly like dev rows, instead of the 5-model mean -- removes the double-averaging tail shrinkage.
SAMEFOLD = os.environ.get("SEQ_EVAL_SAMEFOLD") == "1"
HS3 = C.DER / "hs_v3"
HS2 = C.DER / "hs_v2"
TRUNK = SEQ / "trunk8"
HD_PARAMS = dict(objective="multiclass", num_class=4, learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
                 feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=6,
                 verbose=-1, seed=7)                                  # = src/handdet.py


def _load_nets(dev="cuda"):
    torch = _torch()
    nets = []
    for k in range(5):
        net = build_net(4).to(dev)
        net.load_state_dict(torch.load(TRUNK / f"fold{k}.pt", map_location=dev))
        net.eval()
        nets.append(net)
    return nets


def _seq_probs(cache, idx, nets, rowfold=None, chunk=None):
    """(n,4) softmax of the orientation-averaged logits. rowfold given -> the row's own fold model (dev OOF),
    else the mean of the 5 fold models' probabilities (eval; handdet.py convention)."""
    torch = _torch()
    dev = next(nets[0].parameters()).device
    chunk = chunk or int(os.environ.get("SEQ_CHUNK", "4096"))
    swc, sws = torch.as_tensor(SWAP_C, device=dev), torch.as_tensor(SWAP_S, device=dev)
    out = np.zeros((len(idx), 4), np.float32)
    with torch.no_grad():
        for s in range(0, len(idx), chunk):
            sub = idx[s:s + chunk]
            cn, tk, mk, st = build_examples(cache, sub)
            cb, tb = torch.from_numpy(cn).to(dev).float(), torch.from_numpy(tk).to(dev).long()
            mb, sb = torch.from_numpy(mk).to(dev), torch.from_numpy(st).to(dev).float()
            cs, ts, ss = _swap(cb, tb, sb, swc, sws)
            if rowfold is None:
                groups = [(k, None) for k in range(5)]
            else:
                rf = rowfold[s:s + chunk]
                groups = [(k, np.flatnonzero(rf == k)) for k in np.unique(rf)]
            for k, rows in groups:
                if rows is None:
                    a = (cb, tb, mb, sb, cs, ts, ss)
                else:
                    r = torch.as_tensor(rows, device=dev)
                    a = (cb[r], tb[r], mb[r], sb[r], cs[r], ts[r], ss[r])
                with torch.autocast("cuda", dtype=torch.float16, enabled=str(dev).startswith("cuda")):
                    o = (nets[k](a[0], a[1], a[2], a[3]).float() + nets[k](a[4], a[5], a[2], a[6]).float()) / 2
                p = torch.softmax(o.float(), dim=1).cpu().numpy()
                if rows is None:
                    out[s:s + len(sub)] += p / 5
                else:
                    out[s + rows] = p
    return out


def _logit4(P):
    P = np.clip(P, 1e-6, 1 - 1e-6)
    return (np.log(P) - np.log(1 - P)).astype(np.float32)


def _join_scores(keys, hsdir):
    """phase-0 hs scores of `keys` (hand_idx, pA, pB), returned in keys order."""
    H = (pl.scan_parquet(hsdir / "phase0" / "*.parquet").select("hand_idx", "pA", "pB", *SC, "s_any")
         .join(keys.lazy().select("hand_idx", "pA", "pB", pl.int_range(pl.len()).alias("ord")),
               on=["hand_idx", "pA", "pB"], how="inner").collect().sort("ord"))
    assert H.height == keys.height, (H.height, keys.height)
    return H


def _recall_listed(S4, y, isev, q=0.999):
    """listed-only recall at the negative-hand quantile, per family (S4 = (n,4) class scores)."""
    return [round(float((S4[(y == i) & (isev == 1), i] > np.quantile(S4[y == 0, i], q)).mean()), 4)
            for i in (1, 2, 3)]


def stage_hd_lgb():
    """Round-3 LightGBM detector + 4 seq margins: fold models, mixed-score thresholds, diagnostics."""
    import lightgbm as lgb
    HD_SEQ.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    R = cache["R"]
    pos = R.filter(c("label") == 1).select("hand_idx", "pA", "pB", "family")
    h2 = _join_scores(pos, HS2)
    s_fam2 = np.select([pos["family"].to_numpy() == f for f in FAMS], [h2[k].to_numpy() for k in SC], np.nan)
    ext2 = pos.select("hand_idx", "pA", "pB").with_columns(pl.Series("s_fam2", s_fam2))
    R = R.join(ext2, on=["hand_idx", "pA", "pB"], how="left", maintain_order="left").with_columns(
        is_extra2=((c("label") == 1) & (c("is_ev") == 0) & (c("s_fam2") > 0.5)).cast(pl.Int8))
    assert np.array_equal(R["hrow"].to_numpy(), cache["ex_hrow"]), "row order changed by join"
    sel = R.select((c("label") == 0) | (c("is_ev") == 1) | (c("is_extra2") == 1)).to_series().to_numpy()
    idx = np.flatnonzero(sel)
    S = R[idx]
    fam = S["family"].to_numpy()
    y = np.where(S["label"].to_numpy() == 0, 0, np.select([fam == f for f in FAMS], [1, 2, 3], 0)).astype(np.int64)
    isev = S["is_ev"].to_numpy()
    w = np.where(y == 0, 1.0, 25.0)
    w = np.where(S["is_extra2"].to_numpy() == 1, 8.0, w).astype(np.float32)
    fold = S["fold"].to_numpy()
    log(f"round-3 population: {len(idx):,} rows | extras (hs_v2 > .5) {int(S['is_extra2'].sum())} "
        f"(exp028_train log: 2097) | listed {int(isev.sum())}")

    nets = _load_nets()
    t = time.time()
    P = _seq_probs(cache, idx, nets, rowfold=fold)
    log(f"seq OOF probs for {len(idx):,} rows in {time.time() - t:.1f}s")
    old_idx, old_P = np.load(SEQ / "oof_hd_idx.npy"), np.load(SEQ / "oof_hd_seq.npy")
    _, ia, ib = np.intersect1d(idx, old_idx, return_indices=True)
    log(f"sanity vs train_hd OOF (same trunk8 models, fp32) on {len(ia):,} shared rows: "
        f"max |dP| {np.abs(P[ia] - old_P[ib]).max():.4f}")

    X, fn = _engine_rows(S)
    X2 = np.hstack([X, _logit4(P)])
    fn2 = fn + [f"seq_logit_{i}" for i in range(4)]
    gate = X[:, fn.index("mx_first_fold_nonfacing")] == 0
    oof = np.zeros((len(y), 4))
    oof_base = np.zeros((len(y), 4))
    for k in range(5):
        tr = fold != k
        m = lgb.train(HD_PARAMS, lgb.Dataset(X2[tr], y[tr], weight=w[tr], feature_name=fn2), 400)
        m.save_model(str(HD_SEQ / f"fold{k}.txt"))
        oof[fold == k] = m.predict(X2[fold == k])
        mb = lgb.train(HD_PARAMS, lgb.Dataset(X[tr], y[tr], weight=w[tr], feature_name=fn), 400)
        oof_base[fold == k] = mb.predict(X[fold == k])
        log(f" fold {k} done")
    h3 = _join_scores(S, HS3)
    H3 = np.stack([1 - h3["s_any"].to_numpy()] + [h3[k].to_numpy() for k in SC], axis=1)
    mixed = np.where(gate[:, None], oof, H3)
    res = {"n_rows": int(len(y)), "n_extra": int(S["is_extra2"].sum()),
           "gate_pass_rate_neg": round(float(gate[y == 0].mean()), 4),
           "gate_pass_rate_listed": {FAMS[i - 1]: round(float(gate[(y == i) & (isev == 1)].mean()), 4)
                                     for i in (1, 2, 3)}}
    for nm, M in (("hs_v3", H3), ("lgb_refit", oof_base), ("lgb_seq_all_rows", oof), ("mixed_gated", mixed)):
        res[nm] = {**hd_metrics(M, y), "recall_listed_q999": _recall_listed(M, y, isev),
                   "recall_listed_q99": _recall_listed(M, y, isev, 0.99)}
        log(f"{nm:18s} " + json.dumps(res[nm]))
    log("gate pass rates " + json.dumps({k: res[k] for k in ("gate_pass_rate_neg", "gate_pass_rate_listed")}))
    thr = {}
    for i, s_ in enumerate(SC, 1):
        neg = mixed[y == 0, i]
        thr[s_] = {"0.99": float(np.quantile(neg, 0.99)), "0.999": float(np.quantile(neg, 0.999))}
    negany = 1 - mixed[y == 0, 0]
    thr["s_any"] = {"0.99": float(np.quantile(negany, 0.99)), "0.999": float(np.quantile(negany, 0.999))}
    (HD_SEQ / "thresholds.json").write_text(json.dumps(thr, indent=1))
    (HD_SEQ / "oof_metrics.json").write_text(json.dumps(res, indent=1))
    np.savez(HD_SEQ / "oof_population.npz", idx=idx, y=y, isev=isev, gate=gate, oof=oof, oof_base=oof_base,
             hs3=H3, seq=P)
    log("thresholds " + json.dumps(thr))


def stage_hs_score(phases=(0, 1)):
    import lightgbm as lgb
    import handdet as HD
    nets = _load_nets()
    models = [lgb.Booster(model_file=str(HD_SEQ / f"fold{k}.txt")) for k in range(5)]
    fn = HD.feat_names()
    folds = C.get_table_folds()
    tab_fold = np.zeros(400, dtype=np.int64)
    tab_fold[folds["table_idx"].to_numpy()] = folds["fold"].to_numpy()
    gate_expr = (c("first_fold_nonfacing_ab") == 0) & (c("first_fold_nonfacing_ba") == 0)
    keycols = ["hand_idx", "pA", "pB"]
    for phase in phases:
        outdir = HS_SEQ / f"phase{phase}"
        outdir.mkdir(parents=True, exist_ok=True)
        for part in sorted((PH / f"phase{phase}").glob("part_*.parquet")):
            if (outdir / part.name).exists():
                log(f"skip phase {phase} {part.name} (exists)")
                continue
            t = time.time()
            K = pl.read_parquet(part, columns=keycols + ["seatA", "seatB", "table_idx",
                                                         "first_fold_nonfacing_ab", "first_fold_nonfacing_ba"])
            h3 = pl.read_parquet(HS3 / f"phase{phase}" / part.name)
            assert h3.height == K.height and all((h3[k] == K[k]).all() for k in keycols), "hs_v3/ph_v2 row mismatch"
            gi = np.flatnonzero(K.select(gate_expr).to_series().to_numpy())
            G = K[gi]
            hands = np.unique(G["hand_idx"].to_numpy())
            seq = _seq_arrays(hands, phase, quiet=True)
            sym, seatblk, huv = _hand_static(hands, seq, phase)
            cache = dict(seq, sym=sym, seatblk=seatblk, huv=huv,
                         ex_hrow=np.searchsorted(hands, G["hand_idx"].to_numpy()),
                         ex_seatA=G["seatA"].to_numpy().astype(np.int64),
                         ex_seatB=G["seatB"].to_numpy().astype(np.int64))
            t1 = time.time()
            rowfold = tab_fold[G["table_idx"].to_numpy()] if (phase == 0 or SAMEFOLD) else None
            P = _seq_probs(cache, np.arange(len(gi)), nets, rowfold)
            t2 = time.time()
            F = HD.hand_features(pl.scan_parquet(part).filter(gate_expr)).collect()
            assert F.height == len(gi) and all((F[k] == G[k]).all() for k in keycols)
            X2 = np.hstack([F.select(fn).to_numpy().astype(np.float32), _logit4(P)])
            del F
            Q = np.zeros((len(gi), 4))
            if rowfold is not None:
                for k in range(5):
                    m = rowfold == k
                    if m.any():
                        Q[m] = models[k].predict(X2[m], num_threads=6)
            else:
                for k in range(5):
                    Q += models[k].predict(X2, num_threads=6) / 5
            cols = {k: h3[k].to_numpy().astype(np.float32).copy() for k in SC + ["s_any"]}
            for i, k in enumerate(SC, 1):
                cols[k][gi] = Q[:, i]
            cols["s_any"][gi] = 1 - Q[:, 0]
            out = h3.select("hand_idx", "pA", "pB", "table_idx", "hand_seq").with_columns(
                [pl.Series(k, cols[k]) for k in SC + ["s_any"]])
            out.write_parquet(outdir / part.name)
            log(f"phase {phase} {part.name}: {K.height:,} rows, gated {len(gi):,} ({len(gi) / K.height:.1%}), "
                f"{len(hands):,} hands | cache {t1 - t:.0f}s seq {t2 - t1:.0f}s lgb {time.time() - t2:.0f}s")


def stage_hs_agg():
    """handdet.py's own aggregation on hs_seq, with the mixed-score thresholds (MOD dir monkeypatched)."""
    os.environ["HS_IN"] = HS_SEQ.name
    os.environ["HS_OUT"] = HS_SEQ.name
    os.environ["PFH_TAG"] = os.environ.get("SEQ_PFH_TAG", "seq")
    import importlib
    import handdet as HD
    HD = importlib.reload(HD)
    assert HD.HS_OUT == HS_SEQ, HD.HS_OUT
    HD.MOD = HD_SEQ
    HD.agg()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="*", default=["cache"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--arch", default="tf")
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-lgb", action="store_true")
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--no-twostage", action="store_true")
    ap.add_argument("--save", default=None, help="train_hd: subdir of seq_cache to persist the fold trunks into")
    ap.add_argument("--init", default=None, help="train_ev: subdir of seq_cache holding pretrained fold trunks")
    ap.add_argument("--lr", type=float, default=1.5e-3)
    a = ap.parse_args()
    for st in a.stage:
        log(f"=== stage {st}")
        if st == "cache":
            stage_cache()
        elif st == "hd_lgb":
            stage_hd_lgb()
        elif st == "hs_score":
            stage_hs_score(tuple(int(x) for x in os.environ.get("SEQ_PHASES", "0,1").split(",")))
        elif st == "hs_agg":
            stage_hs_agg()
        elif st == "confirm":
            stage_confirm(tag=a.tag or "ev")
        elif st == "train_ev":
            stage_train_ev(epochs=a.epochs, arch=a.arch, d=a.d, layers=a.layers, seed=a.seed,
                           tag=a.tag or "ev", bs=a.bs, twostage=not a.no_twostage, init=a.init, lr=a.lr)
        elif st == "train_hd":
            stage_train_hd(epochs=a.epochs, arch=a.arch, d=a.d, layers=a.layers, seed=a.seed,
                           with_lgb=not a.no_lgb, tag=a.tag or "hd", bs=a.bs, save=a.save)
