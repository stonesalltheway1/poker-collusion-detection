"""Lane M3 side result: rescale the SOFT-PLAY DP's implied event count by 0.9 (a single multiplicative logit
shift per pair, solved by bisection), everything else identical to exp042_ramp (W_DP 0.7, ramp [0.45,0.65],
partner exp027ev).  Dev OOF MAP@5 .7748 -> .7775 (+.0027, paired bootstrap P(better) .882; SP pairs
.7660 -> .7736).  DT and CI are untouched -- their fold-honest optimum is exactly 1.0.

  python scripts/lane_m3_eval_sp09.py dev    -> data/derived/evidence_scores_dev_exp060_sp09_exp027ev.parquet
  python scripts/lane_m3_eval_sp09.py eval   -> data/derived/evidence_scores_eval_exp060_sp09_exp027ev.parquet
"""
import os, sys, time
os.environ["POLARS_MAX_THREADS"] = "2"
os.environ["NUMBA_NUM_THREADS"] = "2"
os.environ.setdefault("LR_GATE_RAMP", "0.45,0.65")
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts")); sys.path.insert(0, str(BASE / "src"))
import numpy as np, polars as pl
import listing_rule as LR
import lane_m3_exposure as M3
import common as C
c = pl.col
FAMS = LR.FAMS
FAC = float(os.environ.get("M3_SP_FAC", 0.9))
TAG = os.environ.get("M3_TAG", "exp060_sp09")
_orig_dp = LR.dp_scores


def dp_scores_sp(E, qlogit, tau, fam):
    if fam != "soft_play":
        return _orig_dp(E, qlogit, tau, fam)
    g = E.select(LR.gate(fam).fill_null(False)).to_series().to_numpy()
    ql = np.clip(np.asarray(qlogit, float), -40, 40).copy()
    ql[~g] = -1e6
    off = LR.group_offsets(E["pair_id"].to_numpy())
    S = np.add.reduceat(np.where(g, 1.0 / (1.0 + np.exp(-ql)), 0.0), off[:-1])
    q, _ = M3.solve_shift(off, ql, FAC * S)
    pi = LR.type_matrix(E, fam, tau)
    return LR.listed_prob_groups(off, q, np.ascontiguousarray(pi), LR.CAP)


LR.dp_scores = dp_scores_sp
LR.TAG = TAG
os.environ["LR_PARTNERS"] = "exp027ev"
LR.PARTNER = "exp027ev"

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if stage == "dev":                      # dev OOF score file straight from the cached fold models
        d = M3.Dev()
        dp = d.dp({f: d.S_raw[f] * (FAC if f == FAMS[1] else 1.0) for f in FAMS})
        G = d.score(dp, "ramp")
        hid = C.load("hands").select("hand_idx", "hand_id")
        out = (G.join(hid, on="hand_idx", how="left")
               .select("pair_id", "hand_id", "hand_idx",
                       *[c(f"b_{f}").cast(pl.Float64).alias(f"s_{f}") for f in FAMS]).sort("pair_id", "hand_idx"))
        p = C.DER / f"evidence_scores_dev_{TAG}_exp027ev.parquet"
        out.write_parquet(p)
        ap, _ = d.ap(dp)
        print(f"wrote {p} ({out.height:,} rows); dev MAP@5 true-family routing = {ap.mean():.4f}")
    else:
        t = time.time()
        LR.run_eval()
        print(f"eval done in {time.time()-t:.0f}s")
