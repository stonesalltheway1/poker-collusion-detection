"""Shared infrastructure for the Slash poker-collusion comp.

Conventions (everything downstream depends on them):
- Data comes from the int-coded cache data/derived/*.parquet (src/build_cache.py).
- Local CV = labelled dev pairs, folds by TABLE (data/folds_tables_5.csv, table_idx -> fold),
  frozen once and never regenerated. One table per player, so table folds share no players.
- Every experiment writes oof/<exp>_dev.parquet (labelled dev pairs: pair_id, risk_score,
  predicted_behavior, evidence_hand_1..5 as hand_id strings or NO_EVIDENCE), optionally
  oof/<exp>_eval.parquet (same columns, all eval pairs), and oof/<exp>.json (meta + CV parts).
- CV is scored with the host metric (src/metric.py, exact copy) plus a component breakdown.
- Rules: infer from poker activity only; never use ID formats or row/file ordering as signal.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
DER = BASE / "data" / "derived"
OOF_DIR = BASE / "oof"
SUB_DIR = BASE / "submissions"
FOLDS_CSV = BASE / "data" / "folds_tables_5.csv"

FAMILIES = ("directed_transfer", "soft_play", "coordinated_isolation")
BEHAVIORS = ("none", *FAMILIES, "other_coordination")
EV_COLS = [f"evidence_hand_{k}" for k in range(1, 6)]
NO_EV = "NO_EVIDENCE"
SUB_COLS = ["pair_id", "risk_score", "predicted_behavior", *EV_COLS]

ACTION = {"fold": 0, "check": 1, "call": 2, "bet": 3, "raise": 4, "all_in": 5}
STREET = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
RANKS, SUITS = "23456789TJQKA", "cdhs"


def card_str(code):
    return RANKS[code // 4] + SUITS[code % 4]


def scan(name):
    """Lazy scan of a cache table: players, tables, hands, seats, actions, labels, evidence, eval_pairs."""
    return pl.scan_parquet(DER / f"{name}.parquet")


def load(name):
    return pl.read_parquet(DER / f"{name}.parquet")


def player_table():
    """player_idx -> table_idx (each player sits at exactly one table)."""
    return (scan("seats").select("hand_idx", "player_idx").unique("player_idx")
            .join(scan("hands").select("hand_idx", "table_idx"), on="hand_idx")
            .select("player_idx", "table_idx").collect())


def get_table_folds(n_splits=5, seed=42):
    """Frozen table_idx -> fold. Created once; never regenerated."""
    if FOLDS_CSV.exists():
        return pl.read_csv(FOLDS_CSV)
    rng = np.random.default_rng(seed)
    tables = np.arange(400)
    rng.shuffle(tables)
    folds = pl.DataFrame({"table_idx": tables.astype(np.int16),
                          "fold": (np.arange(400) % n_splits).astype(np.int8)}).sort("table_idx")
    folds.write_csv(FOLDS_CSV)
    return folds


def labelled_pairs():
    """labels + table_idx + fold."""
    pt = player_table()
    return (load("labels").join(pt.rename({"player_idx": "p1"}), on="p1")
            .join(get_table_folds(), on="table_idx"))


# ----------------------------------------------------------------------------- scoring
def _ap(y, s):
    """Host AP: stable mergesort on -score (ties keep input order; host sorts by pair_id)."""
    y = np.asarray(y, dtype=int)
    pos = y.sum()
    if pos == 0:
        return 0.0
    order = np.argsort(-np.asarray(s, dtype=float), kind="mergesort")
    r = y[order]
    return float(np.sum(np.cumsum(r) / np.arange(1, len(r) + 1) * r) / pos)


def dev_solution(pairs=None):
    """Solution frame in host format for labelled dev pairs (evidence = public planted hands)."""
    lab = load("labels").to_pandas()
    if pairs is not None:
        lab = lab[lab.pair_id.isin(set(pairs))]
    ev = load("evidence").to_pandas().sort_values(["pair_id", "evidence_rank"])
    wide = ev.groupby("pair_id")["hand_id"].apply(list)
    sol = pd.DataFrame({"pair_id": lab.pair_id.values, "risk_score": lab.label.values,
                        "predicted_behavior": lab.behavior_family.values})
    for k, c in enumerate(EV_COLS):
        sol[c] = [(wide.get(p, []) + [NO_EV] * 5)[k] for p in sol.pair_id]
    return sol


def score_parts(solution, submission):
    """Host score plus components. Both frames in host submission format."""
    import metric  # src/metric.py (exact host copy)
    total = metric.score(solution.copy(), submission.copy(), "pair_id")
    truth = solution.set_index("pair_id").sort_index()
    pred = submission.set_index("pair_id").loc[truth.index]
    y = truth.risk_score.to_numpy(int)
    r = pred.risk_score.to_numpy(float)
    pair_ap = _ap(y, r)
    beh = [(_ap((truth.predicted_behavior == f).to_numpy(int),
                np.where(pred.predicted_behavior == f, r, 0.0))
            if (truth.predicted_behavior == f).any() else 0.0) for f in FAMILIES]
    ev_scores = []
    for i in np.flatnonzero(y == 1):
        rel = {h for h in truth.iloc[i][EV_COLS] if h != NO_EV}
        sub = [h for h in pred.iloc[i][EV_COLS] if h != NO_EV][:5]
        hits, ps = 0, 0.0
        for k, h in enumerate(sub, 1):
            if h in rel:
                hits += 1
                ps += hits / k
        ev_scores.append(ps / min(len(rel), 5) if rel else 0.0)
    return {"score": round(total, 5), "pair_ap": round(pair_ap, 5),
            "evidence_map5": round(float(np.mean(ev_scores)) if ev_scores else 0.0, 5),
            "behavior_map": round(float(np.mean(beh)), 5),
            **{f"beh_ap_{f}": round(b, 5) for f, b in zip(FAMILIES, beh)}}


# ----------------------------------------------------------------------------- registry
def save_experiment(exp_id, dev_pred, eval_pred=None, meta=None):
    """dev_pred: host-format frame for labelled dev pairs (OOF). Returns CV parts."""
    OOF_DIR.mkdir(exist_ok=True)
    dev_pred = finalize(dev_pred)
    parts = score_parts(dev_solution(dev_pred.pair_id), dev_pred)
    dev_pred.to_parquet(OOF_DIR / f"{exp_id}_dev.parquet", index=False)
    if eval_pred is not None:
        finalize(eval_pred).to_parquet(OOF_DIR / f"{exp_id}_eval.parquet", index=False)
    out = {"exp_id": exp_id, "cv": parts, **(meta or {})}
    (OOF_DIR / f"{exp_id}.json").write_text(json.dumps(out, indent=2))
    return parts


def finalize(df):
    """Coerce to host format: fill evidence with NO_EVIDENCE and clip risk. Does NOT break ties
    (write_submission asserts none). Never leave predicted_behavior as 'none': argmax a family for every pair."""
    df = df.copy()
    for c in EV_COLS:
        if c not in df:
            df[c] = NO_EV
        df[c] = df[c].fillna(NO_EV).astype(str).replace({"": NO_EV, "None": NO_EV, "nan": NO_EV})
    if "predicted_behavior" not in df:
        df["predicted_behavior"] = "none"
    df["risk_score"] = np.clip(df["risk_score"].astype(float), 0.0, 1.0)
    return df[SUB_COLS]


def validate_submission(sub, check_shared=True):
    """Hard checks mirroring host rules + evidence sanity (eval phase, both players seated)."""
    ep = load("eval_pairs")
    assert list(sub.columns) == SUB_COLS, sub.columns
    assert len(sub) == ep.height and sub.pair_id.is_unique
    assert set(sub.pair_id) == set(ep["pair_id"].to_list())
    assert sub.risk_score.between(0, 1).all() and not sub.isna().any().any()
    assert set(sub.predicted_behavior) <= set(BEHAVIORS)
    ev = sub[EV_COLS].to_numpy()
    for row in ev:
        hs = [h for h in row if h != NO_EV]
        assert len(hs) == len(set(hs)), row
    stats = {"rows": len(sub), "pairs_with_evidence": int((ev[:, 0] != NO_EV).sum()),
             "risk_ties": int(sub.risk_score.duplicated().sum())}
    if check_shared:
        long = (pl.from_pandas(sub[["pair_id", *EV_COLS]])
                .unpivot(index="pair_id", value_name="hand_id").filter(pl.col("hand_id") != NO_EV)
                .join(ep.select("pair_id", "p1", "p2"), on="pair_id")
                .join(load("hands").select("hand_id", "hand_idx", "phase"), on="hand_id", how="left"))
        assert long["hand_idx"].null_count() == 0, "unknown hand ids"
        seats = scan("seats").select("hand_idx", "player_idx").collect()
        both = (long.join(seats.rename({"player_idx": "p1"}), on=["hand_idx", "p1"], how="semi")
                .join(seats.rename({"player_idx": "p2"}), on=["hand_idx", "p2"], how="semi"))
        stats.update(evidence_rows=long.height, non_eval_phase=int((long["phase"] != 1).sum()),
                     not_shared=long.height - both.height)
    return stats


def write_submission(exp_id, sub):
    SUB_DIR.mkdir(exist_ok=True)
    sub = finalize(sub)
    stats = validate_submission(sub)
    # host breaks risk ties by pair_id (an ID artifact) -> ties must be broken by a legitimate score upstream
    assert stats["risk_ties"] == 0, f"{stats['risk_ties']} tied risk scores; break ties with a secondary model score"
    assert stats["non_eval_phase"] == 0 and stats["not_shared"] == 0, stats
    path = SUB_DIR / f"{exp_id}.csv"
    sub.to_csv(path, index=False)
    return path, stats
