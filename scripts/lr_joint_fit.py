"""Lane M4 stage 1: re-run the exp042_ramp OOF event/type heads and CACHE (q, tau) per row.

The shipped dev components file keeps only q and dp, not tau, and the joint law needs the full
(q, pi) pair.  This reproduces scripts/listing_rule.py:run_cv's fits exactly (same folds, same
PRM/PRM_T, same rounds) and writes everything the joint-law optimiser needs to a scratch npz.
Gameplay-only: identical feature set to exp036/exp042.
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src")); sys.path.insert(0, str(BASE / "scripts"))
import numpy as np, polars as pl
import common as C
import listing_rule as LR

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ["SCRATCH"]) / "m4"
OUT.mkdir(parents=True, exist_ok=True)
c = pl.col
FAMS = LR.FAMS
t0 = time.time()

F = LR.load_dev_features()
feats = LR.feature_names()
X = F.select([c(k).cast(pl.Float32) for k in feats]).to_numpy()
fold = F["fold"].to_numpy()
qlog = {f: np.zeros(len(F)) for f in FAMS}
tau = {f: np.zeros(len(F)) for f in FAMS}
for k in range(5):
    tr, va = fold != k, fold == k
    for f in FAMS:
        m = LR.fit_event(F, X, f, tr)
        qlog[f][va] = m.predict(X[va], raw_score=True)
        if f != "coordinated_isolation":
            tau[f][va] = LR.fit_type(F, X, f, tr).predict(X[va])
    LR.log(f"fold {k} done ({time.time()-t0:.0f}s)")
del X

d = {}
for f in FAMS:
    sh = LR.SH[f]
    q = 1.0 / (1.0 + np.exp(-np.clip(qlog[f], -40, 40)))
    q = np.where(F.select(LR.gate(f).fill_null(False)).to_series().to_numpy(), q, 0.0)
    pi = LR.type_matrix(F, f, tau[f])
    d[f"q_{sh}"] = q
    d[f"pi_{sh}"] = pi
    d[f"dp_{sh}"] = LR.listed_prob_groups(LR.group_offsets(F["pair_id"].to_numpy()), q, np.ascontiguousarray(pi), 5)
np.savez_compressed(OUT / "dev_qpi.npz", **d)
F.select("pair_id", "hand_idx", "hand_seq", "family", "is_ev", "n_rel", "pot_bb", "fold").write_parquet(OUT / "dev_keys.parquet")
LR.log(f"wrote {OUT/'dev_qpi.npz'} and dev_keys.parquet ({time.time()-t0:.0f}s)")
# sanity: reproduce the shipped dp column
S = pl.read_parquet(C.DER / "evidence_cache" / "exp042_ramp_exp027ev_dev_components.parquet")
S = F.select("pair_id", "hand_idx").join(S, on=["pair_id", "hand_idx"], how="left", maintain_order="left")
for f in FAMS:
    a, b = d[f"dp_{LR.SH[f]}"], S[f"dp_{f}"].to_numpy()
    LR.log(f"dp reproduce {LR.SH[f]}: max|diff| = {np.abs(a-b).max():.3e}  corr = {np.corrcoef(a,b)[0,1]:.6f}")
