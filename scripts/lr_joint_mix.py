"""Lane M4, mixture tier: E[AP@5]-optimal ordering under the POSTERIOR-WEIGHTED MIXTURE of the three
family listing DPs -- the one regime where the joint law is genuinely multimodal.

compose_evidence ranks by mix_i = sum_f P_f s_f(i).  Under a flat posterior each family's DP peaks on a
DIFFERENT set of ~5 hands, so the mixture law is bimodal/trimodal and marginal ranking can be strictly
suboptimal for AP (hedge vs commit).  This is also the documented weakness of the shipped scorer
(listing_rule.md section 4, "Caveat: uniform posteriors": mixture blend 0.611 vs partner 0.627) and the
reason the exp037/exp042 strength gate exists.

Measured with the real OOF behaviour posteriors and with uniform posteriors, on the 372 dev positives.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "scripts"))
import numpy as np
import polars as pl
import common as C
import lr_joint as JT

c = pl.col
FAMS, SH = JT.FAMS, JT.SH
SC = Path(os.environ["SCRATCH"]) / "m4"
POST = sys.argv[1] if len(sys.argv) > 1 else "exp035_sf101"
NCAND = int(sys.argv[2]) if len(sys.argv) > 2 else 10

K = pl.read_parquet(SC / "dev_keys.parquet")
Z = np.load(SC / "dev_qpi.npz")
B = pl.read_parquet(C.DER / "evidence_scores_dev_exp042_ramp_exp027ev.parquet").drop("hand_id")
K = K.join(B, on=["pair_id", "hand_idx"], how="left", maintain_order="left")
pair_ids = K["pair_id"].to_numpy()
brk = np.flatnonzero(pair_ids[1:] != pair_ids[:-1]) + 1
offsets = np.concatenate([[0], brk, [len(pair_ids)]]).astype(np.int64)
G = len(offsets) - 1
pid_of_pair = pair_ids[offsets[:-1]]
fam_of_pair = K["family"].to_numpy()[offsets[:-1]]
n_rel = K["n_rel"].to_numpy()[offsets[:-1]]
is_ev = K["is_ev"].to_numpy().astype(np.int64)
pot = K["pot_bb"].to_numpy()
tb = (pl.DataFrame({"p": pair_ids, "v": pot}).with_columns(
    r=(c("v").rank("average").over("p") / pl.len().over("p")) * 1e-9)["r"].to_numpy())

q3 = np.ascontiguousarray(np.stack([Z[f"q_{SH[f]}"] for f in FAMS]))
pi3 = np.ascontiguousarray(np.stack([Z[f"pi_{SH[f]}"] for f in FAMS]))
s3 = np.ascontiguousarray(np.stack([K[f"s_{f}"].to_numpy() + tb for f in FAMS]))

P = pl.read_parquet(BASE / "oof" / f"{POST}_oof_behavior_pos.parquet")
P = pl.DataFrame({"pair_id": pid_of_pair}).join(P, on="pair_id", how="left")
real = P.select([f"P_{f}" for f in FAMS]).to_numpy()
real = real / real.sum(axis=1, keepdims=True)
onehot = np.stack([(fam_of_pair == f).astype(float) for f in FAMS], axis=1)
unif = np.full((G, 3), 1 / 3)
sharp = np.sort(real, axis=1)[:, -1]
print(f"posteriors {POST}: max-P mean {sharp.mean():.3f}, share with max-P < 0.8 = {(sharp < 0.8).mean():.3f}")

out = {"post": POST, "cand": NCAND}
for name, post in (("true_family", onehot), ("real", real), ("uniform", unif)):
    for lam in (1.0,):
        lm, lj, ob = JT.run_pairs_mix(offsets, q3, pi3, s3, np.ascontiguousarray(post), NCAND, 1, lam, 5)
        am = JT.ap_from_lists(lm, is_ev, n_rel)
        aj = JT.ap_from_lists(lj, is_ev, n_rel)
        p, d = JT.boot(am, aj)
        same = (lm == lj).all(axis=1)
        row = dict(shipped=round(float(am.mean()), 4), joint=round(float(aj.mean()), 4), delta=round(d, 5),
                   P_better=round(p, 3), model_E_AP_gain=round(float((ob[:, 1] - ob[:, 0]).mean()), 5),
                   order_differs=round(float(1 - same.mean()), 3))
        out[f"{name}_lam{lam}"] = row
        print(f"  {name:<12} lam={lam}: {json.dumps(row)}")

# where the posterior is flat, the mixture is genuinely multimodal -- isolate that cohort
lm, lj, ob = JT.run_pairs_mix(offsets, q3, pi3, s3, np.ascontiguousarray(real), NCAND, 1, 1.0, 5)
am = JT.ap_from_lists(lm, is_ev, n_rel)
aj = JT.ap_from_lists(lj, is_ev, n_rel)
for th in (0.9, 0.8, 0.6):
    s = sharp < th
    if s.sum() < 5:
        print(f"  flat cohort max-P < {th}: only {s.sum()} pairs")
        continue
    p, d = JT.boot(am[s], aj[s])
    print(f"  flat cohort max-P < {th}: n={s.sum()}  shipped {am[s].mean():.4f} -> joint {aj[s].mean():.4f}  "
          f"delta {d:+.5f}  P(better) {p:.3f}  model E[AP] gain {(ob[s, 1] - ob[s, 0]).mean():+.5f}")
lmu, lju, obu = JT.run_pairs_mix(offsets, q3, pi3, s3, np.ascontiguousarray(unif), NCAND, 1, 1.0, 5)
print(f"  uniform-posterior model E[AP] gain: mean {(obu[:, 1] - obu[:, 0]).mean():+.5f}  "
      f"p95 {np.percentile(obu[:, 1] - obu[:, 0], 95):+.5f}  max {(obu[:, 1] - obu[:, 0]).max():+.5f}")
(SC / f"mix_{POST}.json").write_text(json.dumps(out, indent=1))
