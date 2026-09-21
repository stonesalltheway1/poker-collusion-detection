"""Which protocol produced the SHIPPED evidence_scores_dev_exp027ev.parquet?

My seed-42 rebuild of `evidence_v4.run_eval(s1_mode="inner")` gives dt .7236 / sp .7072 / ci .6837 -- exactly
logs/exp027ev_eval3.log -- while the shipped file measures dt .7222 / sp .7191 / ci .6907 (= exp027ev_eval_final.log).
Hypothesis: the shipped file came from the OTHER branch, s1_mode != "inner", where the stage-2 model is trained AND
served on the same cached 80% outer-OOF stage-1 (X2_tr == X2_va).  That branch needs no inner_s1 loop, which also
explains why eval_final ran 2.5x faster per family than eval3.

This script tests it with the cached v4_store.npz stage-1/segment OOF (seed 42), so it only has to do the 5 stage-2
fits per family.  Read-only.
"""
import os, sys, json, time
os.environ.setdefault("POLARS_MAX_THREADS", "2")
sys.path.insert(0, r"F:\kaggle competitions\suspicious poker\src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, polars as pl
import common as C, evidence as EV, evidence_v2 as V2, evidence_v4 as V4
from evidence import FAMS, CACHE, chrono_features, fit, predict, platt, sigmoid
import evbag_harness as H

EV.NTHR = 3
V2.HS = C.DER / "hs_v2"
t0 = time.time()
F, _ = V2.build_dev()
dev = EV.Dev(F)
res2 = json.loads((CACHE / "cv_results_v2.json").read_text())
XS = V4.feature_sets(F, res2["features_base"])
z = np.load(CACHE / "v4_store.npz")
Sdev, out = {}, {}
for f, cfg in V4.CHOSEN.items():
    s, use_seg, obj, rounds = cfg["s"], cfg["use_seg"], cfg["obj"], cfg["rounds"]
    s1 = z[f"s1|{s}|{f}"]
    p2o = z[f"p2|{s}|{f}"]
    bl = [XS[s], chrono_features(F, s1, f).to_numpy().astype(np.float32)]
    if use_seg:
        bl.append(V2.seg_features(F, s1, p2o).to_numpy().astype(np.float32))
    X2 = np.hstack(bl)
    oof = np.zeros(len(dev.y))
    for k in range(5):
        mm = fit(dev, X2, (dev.fold != k) & (dev.fam == f), obj, rounds, False)
        oof[dev.fold == k] = predict(mm, X2[dev.fold == k], obj, False, f)
    a, b_ = platt(oof[dev.fam == f], dev.y[dev.fam == f])
    Sdev[f] = sigmoid(a * oof + b_)
    out[EV.SHORT[f]] = round(dev.map5(oof, f), 4)
    print(f"  {EV.SHORT[f]}: OOF MAP5 = {out[EV.SHORT[f]]}  ({time.time() - t0:.0f}s)", flush=True)

print("\ns1_mode='outer' (train and serve on the cached 80% stage-1):", out)
print("shipped evidence_scores_dev_exp027ev.parquet :  dt 0.7222 sp 0.7191 ci 0.6907  (exp027ev_eval_final.log)")
print("s1_mode='inner' rebuild (this session, seed 42): dt 0.7236 sp 0.7072 ci 0.6837  (exp027ev_eval3.log)")

# per-row agreement with the shipped file
keys = F.select("pair_id", "hand_idx", "pot_bb")
P = EV.write_scores(keys, Sdev, None, keep_all=True).select("pair_id", "hand_idx", *[f"s_{f}" for f in FAMS])
ship = pl.read_parquet(C.DER / "evidence_scores_dev_exp027ev.parquet").select(
    "pair_id", "hand_idx", *[pl.col(f"s_{f}").alias(f"t_{f}") for f in FAMS])
j = P.join(ship, on=["pair_id", "hand_idx"])
print(f"\nrows joined {j.height} of {P.height}")
for f in FAMS:
    d = (j[f"s_{f}"] - j[f"t_{f}"]).abs()
    print(f"  {EV.SHORT[f]}: max |diff| vs shipped = {float(d.max()):.3e}   mean {float(d.mean()):.3e}")

G = H.blend(P, use_ramp=True)
print("\nstandalone", H.summarize(H.per_pair_ap(G, "s_")))
print("in-blend  ", H.summarize(H.per_pair_ap(G, "b_")))
