"""Sub-bag scaling for the exp050 seed bag: standalone + in-blend dev MAP@5 for every subset of seeds and
every aggregation, plus the paired bootstrap vs the shipped single-seed exp027ev partner.
Works straight off the per-seed OOF margins (no parquet round trip)."""
import os, sys, json, itertools
os.environ.setdefault("POLARS_MAX_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"F:\kaggle competitions\suspicious poker\src")
import numpy as np, polars as pl
import evbag_harness as H

DER = r"F:\kaggle competitions\suspicious poker\data\derived"
CACHE = os.path.join(DER, "evidence_cache")
FAMS = H.FAMS
c = pl.col
SEEDS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "42,202,777").split(",")]

# --- dev row keys, in the exact order evidence_v4 produced the OOF arrays (V2.build_dev sort order)
import common as C, evidence as EV, evidence_v2 as V2, evidence_v4 as V4          # noqa: E402
V2.HS = C.DER / "hs_v2"
F, _ = V2.build_dev()
dev = EV.Dev(F)
KEY = F.select("pair_id", "hand_idx", "pot_bb")
y = dev.y
fam = dev.fam


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def platt(m, yy):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(m.reshape(-1, 1), yy)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


Z = {}
for sd in SEEDS:
    z = np.load(os.path.join(CACHE, f"exp050_seed{sd}.npz"))
    Z[sd] = dict(oof={f: z[f"oof|{f}"] for f in FAMS}, cal={f: list(z[f"cal|{f}"]) for f in FAMS})


def agg_scores(sub, mode):
    S = {}
    for f in FAMS:
        probs = [sigmoid(Z[sd]["cal"][f][0] * Z[sd]["oof"][f] + Z[sd]["cal"][f][1]) for sd in sub]
        if mode == "probmean":
            S[f] = np.mean(probs, axis=0)
        elif mode == "margmean":
            mm = np.mean([Z[sd]["oof"][f] for sd in sub], axis=0)
            a, b = platt(mm[fam == f], y[fam == f])
            S[f] = sigmoid(a * mm + b)
        elif mode == "replatt":
            pm = np.mean(probs, axis=0)
            zz = np.log(np.clip(pm, 1e-9, 1 - 1e-9) / np.clip(1 - pm, 1e-9, 1))
            a, b = platt(zz[fam == f], y[fam == f])
            S[f] = sigmoid(a * zz + b)
    return S


def to_partner(S):
    """apply the same 1e-7 pot tie-break evidence.write_scores uses, then hand to the harness."""
    df = KEY.with_columns([pl.Series(f"s_{f}", S[f].astype(np.float64)) for f in FAMS])
    df = df.with_columns(tbk=(c("pot_bb").rank("average").over("pair_id") / pl.len().over("pair_id")) * 1e-7)
    return df.with_columns([(c(f"s_{f}") + c("tbk")).alias(f"s_{f}") for f in FAMS]).drop("tbk", "pot_bb")


base_P = H.load_partner(os.path.join(DER, "evidence_scores_dev_exp027ev.parquet"))
base_G = H.blend(base_P, use_ramp=True)
base_std, base_bl = H.per_pair_ap(base_G, "s_"), H.per_pair_ap(base_G, "b_")
print("BASE exp027ev (shipped, seed 42-as-built)")
print("   standalone", H.summarize(base_std))
print("   in-blend  ", H.summarize(base_bl))
print("   mean ramp w", round(float(H.ramp_w(base_P)["w"].mean()), 5))

rows = []
for k in (1, 2, 3):
    for sub in itertools.combinations(SEEDS, k):
        for mode in (("probmean",) if k == 1 else ("probmean", "margmean", "replatt")):
            P = to_partner(agg_scores(sub, mode))
            G = H.blend(P, use_ramp=True)
            s, b = H.summarize(H.per_pair_ap(G, "s_")), H.summarize(H.per_pair_ap(G, "b_"))
            apb = H.per_pair_ap(G, "b_")
            db, pb, _ = H.paired_boot(base_bl, apb)
            _, pbt, _ = H.paired_boot(base_bl, apb, by="table")
            ds, ps, _ = H.paired_boot(base_std, H.per_pair_ap(G, "s_"))
            rows.append(dict(k=k, seeds=list(sub), mode=mode, standalone=s, blend=b,
                             d_std=ds, P_std=ps, d_blend=db, P_blend=pb, P_blend_table=pbt))
            wmean = float(H.ramp_w(P)["w"].mean())
            rows[-1]["w_mean"] = wmean
            print(f"k={k} {str(sub):20s} {mode:9s}  w~{wmean:.4f}  standalone {s['overall']:.5f} ({ds:+.5f}, P {ps:.3f})   "
                  f"in-blend {b['overall']:.5f} ({db:+.5f}, P {pb:.3f}, Ptab {pbt:.3f})   "
                  f"dt {b['dt']:.4f} sp {b['sp']:.4f} ci {b['ci']:.4f}")

out = dict(base=dict(standalone=H.summarize(base_std), blend=H.summarize(base_bl)), rows=rows, seeds=SEEDS)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "subbag.json"), "w"), indent=1)

# --- seed spread summary
for key, lbl in (("standalone", "standalone"), ("blend", "in-blend")):
    v = np.array([r[key]["overall"] for r in rows if r["k"] == 1])
    print(f"\nsingle-seed {lbl} overall: {[round(x,5) for x in v]}  mean {v.mean():.5f} "
          f"sd {v.std(ddof=1):.5f} spread {v.max()-v.min():.5f}")
    for k in (2, 3):
        vv = np.array([r[key]["overall"] for r in rows if r["k"] == k and r["mode"] == "probmean"])
        print(f"  bag k={k} probmean: n={len(vv)} mean {vv.mean():.5f} "
              f"sd {vv.std(ddof=1) if len(vv) > 1 else float('nan'):.5f} min {vv.min():.5f} max {vv.max():.5f}")

# --- seed-to-seed diversity: within-pair Spearman of the partner scores, per family
print("\nseed-to-seed within-pair rank correlation of the partner score (mean over pairs):")
pid = KEY["pair_id"].to_numpy()
import itertools as it
for f in FAMS:
    for a, b in it.combinations(SEEDS, 2):
        sa = sigmoid(Z[a]["cal"][f][0] * Z[a]["oof"][f] + Z[a]["cal"][f][1])
        sb = sigmoid(Z[b]["cal"][f][0] * Z[b]["oof"][f] + Z[b]["cal"][f][1])
        d = pl.DataFrame({"pair_id": pid, "a": sa, "b": sb, "fam": fam}).filter(c("fam") == f)
        r = (d.with_columns(ra=c("a").rank().over("pair_id"), rb=c("b").rank().over("pair_id"))
             .group_by("pair_id").agg(pl.corr("ra", "rb").alias("r"))["r"].mean())
        print(f"  {H.SH[f]}  seed {a} vs {b}: rho = {r:.4f}")
