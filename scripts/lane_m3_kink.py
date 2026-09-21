"""Lane M3, the decisive test: does the event model's implied event count S kink at the edge of its TRAINING
exposure support, and does de-kinking it recover MAP on held-out low-exposure pairs?

Refit the exp036 event/type models on dev pairs with n_shared >= T only (T = 86 = the dev 25th percentile),
keeping the frozen table folds. Pairs with n_shared < T are then genuinely out of the model's exposure support,
exactly as eval's 26% below 57 are for the production model, AND they carry ground-truth evidence lists.
"""
import os, sys, time, json
os.environ["POLARS_MAX_THREADS"] = "2"
os.environ["NUMBA_NUM_THREADS"] = "2"
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts")); sys.path.insert(0, str(BASE / "src"))
import numpy as np, polars as pl
import listing_rule as LR
import lane_m3_exposure as M3
import common as C
c = pl.col
SC = M3.SC
FAMS, SH = LR.FAMS, LR.SH
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

T_SUP = float(os.environ.get("M3_T", 86))
OUT = SC / f"dev_oof_T{int(T_SUP)}.npz"

if not OUT.exists():
    F = LR.load_dev_features()
    feats = LR.feature_names()
    X = F.select([c(k).cast(pl.Float32) for k in feats]).to_numpy()
    ns = F.group_by("pair_id", maintain_order=True).agg(pl.len().alias("n")).join(
        F.select("pair_id"), on="pair_id", how="right")["n"].to_numpy()
    fold = F["fold"].to_numpy()
    qlog = {f: np.zeros(len(F)) for f in FAMS}; tau = {f: np.zeros(len(F)) for f in FAMS}
    for k in range(5):
        tr = (fold != k) & (ns >= T_SUP); va = fold == k
        for f in FAMS:
            m = LR.fit_event(F, X, f, tr); qlog[f][va] = m.predict(X[va], raw_score=True)
            if f != "coordinated_isolation":
                mt = LR.fit_type(F, X, f, tr); tau[f][va] = mt.predict(X[va])
        log(f"restricted fold {k} done (train pairs n_shared>={T_SUP:.0f})")
    np.savez_compressed(OUT, **{f"q_{SH[f]}": qlog[f] for f in FAMS}, **{f"t_{SH[f]}": tau[f] for f in FAMS})
    log("wrote", OUT.name)

d = M3.Dev(npz=OUT.name)
qt_ns = d.n_shared
below = qt_ns < T_SUP
print(f"\nrestricted model (trained on n_shared >= {T_SUP:.0f}): {below.sum()} held-out low-exposure pairs, "
      f"{(~below).sum()} in-support")

# ---- 1. the kink: slope of log S vs log n, inside vs outside the training support (own-family pairs)
print("\n[1] implied event count S(n): log-log slope inside vs outside the model's training support")
slopes = {}
for f in FAMS:
    own = d.fam == f
    S = d.S_raw[f]
    hi = own & (~below) & (S > 0); lo = own & below & (S > 0)
    b_in = np.polyfit(np.log(qt_ns[hi]), np.log(S[hi]), 1)[0]
    # slope implied by the model between the two regimes (compare geometric means)
    g_in = np.exp(np.log(S[hi]).mean()); g_lo = np.exp(np.log(S[lo]).mean())
    n_in = np.exp(np.log(qt_ns[hi]).mean()); n_lo = np.exp(np.log(qt_ns[lo]).mean())
    b_cross = np.log(g_in / g_lo) / np.log(n_in / n_lo)
    slopes[f] = dict(b_in=b_in, b_cross=b_cross, g_in=g_in, g_lo=g_lo, n_in=n_in, n_lo=n_lo, n_lo_pairs=int(lo.sum()))
    print(f"  {SH[f]}: in-support slope {b_in:+.3f} | crossing slope (in-support -> held-out low) {b_cross:+.3f}"
          f"  [S {g_in:.2f}@n={n_in:.0f} -> {g_lo:.2f}@n={n_lo:.0f}, {lo.sum()} pairs]")
# reference: the same slopes for the FULL (production) model
d_full = M3.Dev()
print("  reference, production model on the same split:")
for f in FAMS:
    own = d_full.fam == f; S = d_full.S_raw[f]
    hi = own & (~below) & (S > 0); lo = own & below & (S > 0)
    b_in = np.polyfit(np.log(d_full.n_shared[hi]), np.log(S[hi]), 1)[0]
    b_cross = np.log(np.exp(np.log(S[hi]).mean()) / np.exp(np.log(S[lo]).mean())) / \
              np.log(np.exp(np.log(d_full.n_shared[hi]).mean()) / np.exp(np.log(d_full.n_shared[lo]).mean()))
    print(f"    {SH[f]}: in-support {b_in:+.3f} | crossing {b_cross:+.3f}")

# ---- 2. does de-kinking recover MAP on the held-out low-exposure pairs?
dp0 = d.dp(); ap0, _ = d.ap(dp0)
print(f"\n[2] restricted-model dev MAP@5 (true-family routing, ramp): overall {ap0.mean():.4f}, "
      f"held-out low {ap0[below].mean():.4f}, in-support {ap0[~below].mean():.4f}")
res = {"T": T_SUP, "slopes": slopes, "base_low": float(ap0[below].mean()), "arms": {}}
law = M3.fit_law(d.n_shared[~below], d.n_listed[~below], d.fam[~below])
print(f"  law fitted on the in-support pairs only: b={law['b']:.3f}")
arms = {"model_slope_dekink": None, "physical_law_b": law["b"], "flat_x1.15": None, "flat_x1.35": None}
for name in ["model_slope_dekink", "physical_law_b", "flat_x1.15", "flat_x1.35"]:
    tgt = {}
    for f in FAMS:
        S = d.S_raw[f].copy()
        if name.startswith("flat"):
            fac = float(name.split("x")[1])
            t = np.where(below, S * fac, S)
        else:
            b_t = slopes[f]["b_in"] if name == "model_slope_dekink" else law["b"]
            ref = slopes[f]["g_in"] / (slopes[f]["n_in"] ** slopes[f]["b_in"])   # anchor on the in-support fit
            # target below support: continue the chosen slope from the support boundary
            S_bound = ref * (T_SUP ** slopes[f]["b_in"])
            t = np.where(below, S_bound * (d.n_shared / T_SUP) ** b_t, S)
            t = np.maximum(t, S)      # only ever inflate: the kink is a collapse
        tgt[f] = t
    ap, _ = d.ap(d.dp(tgt))
    res["arms"][name] = M3.report(d, ap0, ap, name, below)
json.dump(res, open(SC / f"kink_T{int(T_SUP)}.json", "w"), indent=1, default=float)
log("done")
