"""Endgame selection v2 (2026-08-30): paired bootstrap over the D-1 candidate set.

For each candidate: OOF AUC (where an honest OOF exists), private-sized paired bootstrap
P(A beats B), test-side rank correlation, and per-fold AUC (stability).  Candidates are
(oof_source, test_source); sources may be .npy (row-aligned) or id-csv.
Usage: python src/endgame_select2.py [extra_name=oof.npy:test.csv ...]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE, TARGET, load_train_test, get_folds

train, test = load_train_test(); y = train[TARGET].to_numpy(); folds = get_folds(train)
N_PRIV = int(round(len(test) * 0.8))
D, S, L = BASE / "data", BASE / "submissions", BASE / "data" / "oof_library"
P3 = BASE / "data" / "pub3"
CANDS = {
    "exp049_full":    (D / "exp049_full_oof.npy", S / "exp049_full.csv"),
    "naji_19":        (L / "19_blend_oof_predictions.csv", L / "19_blend_submission.csv.csv"),
    "atakan_v13":     (P3 / "atakanaldemir_s6e8-v13-diversity-anchor-lb-0-97124" / "v13_diversity_anchor_oof.csv",
                       P3 / "atakanaldemir_s6e8-v13-diversity-anchor-lb-0-97124" / "v13_diversity_anchor_lb97124.csv"),
    "exp055_B2_w50":      (D / "exp055_B2_w50_oof.npy", S / "exp055_B2_w50.csv"),
    "exp055_D_full_hill": (D / "exp055_D_full_hill_oof.npy", S / "exp055_D_full_hill.csv"),
    "exp056_HILL":        (D / "exp056_HILL_oof.npy", S / "exp056_HILL.csv"),
    "exp057_LR":          (D / "exp057_LR_oof.npy", S / "exp057_LR.csv"),
    "exp057_HILL":        (D / "exp057_HILL_oof.npy", S / "exp057_HILL.csv"),
}
for a in sys.argv[1:]:
    k, v = a.split("="); o, t = v.split(":"); CANDS[k] = (Path(o), Path(t))

def load(p, ids):
    p = Path(p)
    if not p.exists(): return None
    if p.suffix == ".npy": return np.load(p).astype(float)
    df = pd.read_csv(p).set_index("id").reindex(ids)
    col = [c for c in df.columns if df[c].dtype.kind == "f"]
    col = [c for c in col if c != TARGET] or col
    return df[col[0]].to_numpy(float)

names, oofs, tests = [], {}, {}
for k, (op, tp) in CANDS.items():
    o, t = load(op, train["id"]), load(tp, test["id"])
    if t is None: continue
    names.append(k); oofs[k] = o; tests[k] = t
have = [k for k in names if oofs[k] is not None]
print("candidate OOF AUC (overall | per fold):")
for k in have:
    pf = [roc_auc_score(y[folds == f], oofs[k][folds == f]) for f in range(5)]
    print(f"  {k:22s} {roc_auc_score(y, oofs[k]):.6f} | " + " ".join(f"{a:.5f}" for a in pf))
print("\ntest-side rank correlation:")
R = np.column_stack([rankdata(tests[k]) for k in names]); C = np.corrcoef(R.T)
print(pd.DataFrame(C, index=names, columns=[k[:9] for k in names]).round(4).to_string())
# exact-copy check against public frontier files
for k in names:
    for j in names:
        if k < j and np.allclose(rankdata(tests[k]), rankdata(tests[j])):
            print(f"  IDENTICAL ranking: {k} == {j}")
rng = np.random.default_rng(42); B = 300; n = len(y)
aucs = {k: np.empty(B) for k in have}
for b in range(B):
    idx = rng.choice(n, size=N_PRIV, replace=True); yb = y[idx]
    for k in have: aucs[k][b] = roc_auc_score(yb, oofs[k][idx])
print(f"\npaired bootstrap B={B}, n={N_PRIV:,}: mean / sigma")
for k in have: print(f"  {k:22s} {aucs[k].mean():.6f}  {aucs[k].std():.6f}")
print("\nP(row beats col):")
M = pd.DataFrame(0.0, index=have, columns=[k[:9] for k in have])
for i, a in enumerate(have):
    for j, b_ in enumerate(have):
        if i != j: M.iloc[i, j] = (aucs[a] - aucs[b_] > 0).mean()
print(M.round(3).to_string())
