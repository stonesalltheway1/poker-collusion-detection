"""Is the in-blend null for the seed bag robust to the blend weight? Flat w grid, bag vs each single seed."""
import os, sys
os.environ.setdefault("POLARS_MAX_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"F:\kaggle competitions\suspicious poker\src")
import numpy as np, polars as pl
import evbag_harness as H
import common as C, evidence as EV, evidence_v2 as V2
V2.HS = C.DER / "hs_v2"
CACHE = C.DER / "evidence_cache"
FAMS, c = H.FAMS, pl.col
SEEDS = [42, 202, 777]
F, _ = V2.build_dev()
KEY = F.select("pair_id", "hand_idx", "pot_bb")
sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
Z = {sd: np.load(CACHE / f"exp050_seed{sd}.npz") for sd in SEEDS}
prob = {sd: {f: sig(Z[sd][f"cal|{f}"][0] * Z[sd][f"oof|{f}"] + Z[sd][f"cal|{f}"][1]) for f in FAMS} for sd in SEEDS}
bag = {f: np.mean([prob[sd][f] for sd in SEEDS], axis=0) for f in FAMS}
def to_partner(S):
    df = KEY.with_columns([pl.Series(f"s_{f}", S[f].astype(np.float64)) for f in FAMS])
    df = df.with_columns(tbk=(c("pot_bb").rank("average").over("pair_id") / pl.len().over("pair_id")) * 1e-7)
    return df.with_columns([(c(f"s_{f}") + c("tbk")).alias(f"s_{f}") for f in FAMS]).drop("tbk", "pot_bb")
ship = H.load_partner(str(C.DER / "evidence_scores_dev_exp027ev.parquet"))
print(f"{'flat w':>7} | {'seed42':>8} {'seed202':>8} {'seed777':>8} | {'BAG':>8} {'bag-mean':>9} | {'shipped':>8}")
for w in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    vals = []
    for S in [prob[42], prob[202], prob[777], bag]:
        G = H.blend(to_partner(S), use_ramp=False, w_flat=w)
        vals.append(H.summarize(H.per_pair_ap(G, "b_"))["overall"])
    Gs = H.blend(ship, use_ramp=False, w_flat=w)
    sv = H.summarize(H.per_pair_ap(Gs, "b_"))["overall"]
    print(f"{w:7.2f} | {vals[0]:8.5f} {vals[1]:8.5f} {vals[2]:8.5f} | {vals[3]:8.5f} {vals[3]-np.mean(vals[:3]):+9.5f} | {sv:8.5f}")
