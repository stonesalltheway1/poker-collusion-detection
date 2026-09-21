"""Lane M3 step 0: cache the exp036 OOF event/type model outputs + the fold models, so the exposure
correction can be iterated without refitting. Writes only to the session scratch dir."""
import os, sys, time
os.environ["POLARS_MAX_THREADS"] = "2"
os.environ["NUMBA_NUM_THREADS"] = "2"
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts")); sys.path.insert(0, str(BASE / "src"))
import numpy as np, polars as pl
import listing_rule as LR
import common as C
c = pl.col
SC = Path(os.environ["M3_SCRATCH"]); SC.mkdir(parents=True, exist_ok=True)
T0 = time.time()

F = LR.load_dev_features()
feats = LR.feature_names()
X = F.select([c(k).cast(pl.Float32) for k in feats]).to_numpy()
fold = F["fold"].to_numpy()
FAMS = LR.FAMS
qlog = {f: np.zeros(len(F)) for f in FAMS}
tau = {f: np.zeros(len(F)) for f in FAMS}
for k in range(5):
    tr, va = fold != k, fold == k
    for f in FAMS:
        m = LR.fit_event(F, X, f, tr); m.save_model(str(SC / f"ev_{LR.SH[f]}_f{k}.txt"))
        qlog[f][va] = m.predict(X[va], raw_score=True)
        if f != "coordinated_isolation":
            mt = LR.fit_type(F, X, f, tr); mt.save_model(str(SC / f"ty_{LR.SH[f]}_f{k}.txt"))
            tau[f][va] = mt.predict(X[va])
    print(f"[{time.time()-T0:7.1f}s] fold {k} done", flush=True)
allm = np.ones(len(F), dtype=bool)
for f in FAMS:
    LR.fit_event(F, X, f, allm).save_model(str(SC / f"ev_{LR.SH[f]}_full.txt"))
    if f != "coordinated_isolation":
        LR.fit_type(F, X, f, allm).save_model(str(SC / f"ty_{LR.SH[f]}_full.txt"))
np.savez_compressed(SC / "dev_oof.npz", **{f"q_{LR.SH[f]}": qlog[f] for f in FAMS},
                    **{f"t_{LR.SH[f]}": tau[f] for f in FAMS})
keep = ["pair_id", "hand_idx", "hand_seq", "family", "fold", "is_ev", "n_rel", "pot_bb", "evidence_rank",
        "flow", "sp_g0", "ci_fb"]
F.select(keep).write_parquet(SC / "dev_keys.parquet")
print(f"[{time.time()-T0:7.1f}s] wrote cache to {SC}", flush=True)
