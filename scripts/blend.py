"""Blending utilities: rank-average, logit-average, bagged Caruana hill-climb.

Bagging (member + row subsampling per bag, weights = selection frequency) is the
overfit-resistant variant — plain greedy on OOF over-selects optimistic members
(TabArena A.6; PLAN.md WS-D). Use `load_registry()` to pull our members;
public-library members can be appended as extra columns by the caller.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OOF_DIR


def load_registry(exclude=()):
    """Returns (names, oof_matrix [n_train, m], test_matrix [n_test, m])."""
    names, oofs, tests = [], [], []
    for meta_f in sorted(OOF_DIR.glob("*.json")):
        name = meta_f.stem
        if name in exclude:
            continue
        names.append(name)
        oofs.append(np.load(OOF_DIR / f"{name}_oof.npy"))
        tests.append(np.load(OOF_DIR / f"{name}_test.npy"))
    return names, np.column_stack(oofs), np.column_stack(tests)


def rank_norm(M):
    return np.column_stack([rankdata(M[:, j]) / len(M) for j in range(M.shape[1])])


def logit(M, eps=1e-6):
    M = np.clip(M, eps, 1 - eps)
    return np.log(M / (1 - M))


def hill_climb(M, y, n_bags=20, bag_members=0.7, bag_rows=300_000, iters=40,
               seed=42, verbose=True):
    """Bagged greedy selection with replacement. Returns weights (sum to 1)."""
    rng = np.random.default_rng(seed)
    m = M.shape[1]
    counts = np.zeros(m)
    for b in range(n_bags):
        cols = rng.choice(m, size=max(2, int(m * bag_members)), replace=False)
        rows = rng.choice(len(y), size=min(bag_rows, len(y)), replace=False)
        Mb, yb = M[np.ix_(rows, cols)], y[rows]
        best_j = int(np.argmax([roc_auc_score(yb, Mb[:, j])
                                for j in range(len(cols))]))
        sel = [best_j]
        run = Mb[:, best_j].copy()
        best_auc, best_len = roc_auc_score(yb, run), 1
        # Caruana: always add the argmax (with replacement — re-picking the
        # dominant member is how later members enter at weight 1/(n+1)),
        # run a fixed number of steps, keep the best-scoring prefix.
        for _ in range(iters):
            scores = [roc_auc_score(yb, (run * len(sel) + Mb[:, j]) / (len(sel) + 1))
                      for j in range(len(cols))]
            j = int(np.argmax(scores))
            sel.append(j)
            run = (run * (len(sel) - 1) + Mb[:, j]) / len(sel)
            if scores[j] > best_auc:
                best_auc, best_len = scores[j], len(sel)
        for j in sel[:best_len]:
            counts[cols[j]] += 1
        if verbose:
            print(f"bag {b}: {best_len} picks, auc {best_auc:.6f}", flush=True)
    w = counts / counts.sum()
    return w


def evaluate(names, M, y, w):
    blend = M @ w
    print(f"\nblend AUC: {roc_auc_score(y, blend):.6f}")
    for n, wi in sorted(zip(names, w), key=lambda t: -t[1]):
        if wi > 0:
            print(f"  {wi:.3f}  {n}")
    return blend
