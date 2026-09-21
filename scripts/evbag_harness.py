"""Evidence measurement harness: standalone + in-blend (exp042_ramp weights) dev MAP@5 for any partner file.
Reuses the CACHED exp042_ramp DP components -- the DP/event/type heads are unchanged by a partner reseed."""
import os, sys, json
os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import numpy as np, polars as pl
BASE = r"F:\kaggle competitions\suspicious poker"
DER = os.path.join(BASE, "data", "derived")
c = pl.col
FAMS = ["directed_transfer", "soft_play", "coordinated_isolation"]
SH = {"directed_transfer": "dt", "soft_play": "sp", "coordinated_isolation": "ci"}
W_DP, LOW, HIGH = 0.7, 0.45, 0.65

KEYS = pl.read_parquet(os.path.join(DER, "evidence_cache", "v4_dev_keys.parquet")).select(
    "pair_id", "hand_idx", "family", "is_ev", "n_rel", "pot_bb", "table_idx", "fold")
COMP = pl.read_parquet(os.path.join(DER, "evidence_cache", "exp042_ramp_exp027ev_dev_components.parquet"))


def load_partner(path_or_df):
    if isinstance(path_or_df, str):
        d = pl.read_parquet(path_or_df)
    else:
        d = path_or_df
    return d.select("pair_id", "hand_idx", *[f"s_{f}" for f in FAMS])


def ramp_w(P):
    st = P.group_by("pair_id").agg(*[c(f"s_{f}").fill_null(0).top_k(5).mean().alias(f"t5_{f}") for f in FAMS])
    st = st.with_columns(strength=pl.max_horizontal([c(f"t5_{f}") for f in FAMS])).select("pair_id", "strength")
    st = st.with_columns(w=(W_DP * ((c("strength") - LOW) / (HIGH - LOW)).clip(0.0, 1.0)).cast(pl.Float64))
    return st


def blend(P, use_ramp=True, w_flat=W_DP):
    G = KEYS.join(P, on=["pair_id", "hand_idx"], how="left").join(
        COMP.select("pair_id", "hand_idx", *[f"dp_{f}" for f in FAMS]), on=["pair_id", "hand_idx"], how="left")
    if use_ramp:
        G = G.join(ramp_w(P).select("pair_id", "w"), on="pair_id", how="left")
    else:
        G = G.with_columns(w=pl.lit(float(w_flat)))
    for f in FAMS:
        G = G.with_columns(((1 - c("w")) * c(f"s_{f}").fill_null(0) + c("w") * c(f"dp_{f}").fill_null(0)).alias(f"b_{f}"))
    return G


def per_pair_ap(G, col_prefix):
    """returns DataFrame pair_id, family, table_idx, ap  using the TRUE family's score column."""
    out = []
    for f in FAMS:
        d = G.filter(c("family") == f)
        t = (d.select("pair_id", "table_idx", "is_ev", "n_rel", c(f"{col_prefix}{f}").alias("s"), "pot_bb")
             .sort(["pair_id", "s", "pot_bb"], descending=[False, True, True])
             .with_columns(k=pl.int_range(pl.len()).over("pair_id") + 1).filter(c("k") <= 5)
             .with_columns(h=c("is_ev").cast(pl.Int32).cum_sum().over("pair_id")))
        a = t.group_by("pair_id").agg(((c("h") / c("k")) * c("is_ev")).sum().alias("num"),
                                      c("n_rel").first(), c("table_idx").first())
        a = a.with_columns(ap=c("num") / c("n_rel").clip(upper_bound=5), family=pl.lit(f))
        out.append(a.select("pair_id", "family", "table_idx", "ap"))
    return pl.concat(out)


def summarize(ap):
    d = {SH[f]: round(float(ap.filter(c("family") == f)["ap"].mean()), 5) for f in FAMS}
    d["overall"] = round(float(ap["ap"].mean()), 5)
    d["n_pairs"] = ap.height
    return d


def paired_boot(ap_a, ap_b, n=4000, seed=0, by="pair"):
    """P(B better than A). Resample pairs (or tables) with replacement."""
    j = ap_a.join(ap_b.select("pair_id", ap_b=c("ap")), on="pair_id").rename({"ap": "ap_a"})
    d = (j["ap_b"] - j["ap_a"]).to_numpy()
    rng = np.random.default_rng(seed)
    if by == "pair":
        n_obs = len(d)
        idx = rng.integers(0, n_obs, size=(n, n_obs))
        means = d[idx].mean(axis=1)
    else:
        tabs = j["table_idx"].to_numpy()
        ut = np.unique(tabs)
        groups = [np.flatnonzero(tabs == t) for t in ut]
        means = np.empty(n)
        for b in range(n):
            pick = rng.integers(0, len(ut), size=len(ut))
            means[b] = d[np.concatenate([groups[i] for i in pick])].mean()
    return float(d.mean()), float((means > 0).mean()), float(np.std(means))
