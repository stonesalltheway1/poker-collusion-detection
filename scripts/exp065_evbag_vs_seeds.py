"""The CONTROLLED test: 3-seed bag vs each single seed of the SAME pipeline (protocol held fixed).
Standalone and in-blend, with paired bootstraps over pairs and over tables."""
import os, sys, json, itertools
os.environ.setdefault("POLARS_MAX_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"F:\kaggle competitions\suspicious poker\src")
import numpy as np, polars as pl
import evbag_harness as H
import common as C, evidence as EV, evidence_v2 as V2, evidence_v4 as V4

V2.HS = C.DER / "hs_v2"
DER = r"F:\kaggle competitions\suspicious poker\data\derived"
CACHE = os.path.join(DER, "evidence_cache")
FAMS, c = H.FAMS, pl.col
SEEDS = [42, 202, 777]
F, _ = V2.build_dev()
dev = EV.Dev(F)
KEY = F.select("pair_id", "hand_idx", "pot_bb")
y, fam = dev.y, dev.fam


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


Z = {sd: np.load(os.path.join(CACHE, f"exp050_seed{sd}.npz")) for sd in SEEDS}
prob = {sd: {f: sig(Z[sd][f"cal|{f}"][0] * Z[sd][f"oof|{f}"] + Z[sd][f"cal|{f}"][1]) for f in FAMS} for sd in SEEDS}


def to_partner(S):
    df = KEY.with_columns([pl.Series(f"s_{f}", S[f].astype(np.float64)) for f in FAMS])
    df = df.with_columns(tbk=(c("pot_bb").rank("average").over("pair_id") / pl.len().over("pair_id")) * 1e-7)
    return df.with_columns([(c(f"s_{f}") + c("tbk")).alias(f"s_{f}") for f in FAMS]).drop("tbk", "pot_bb")


def aps(S):
    G = H.blend(to_partner(S), use_ramp=True)
    return H.per_pair_ap(G, "s_"), H.per_pair_ap(G, "b_")


bag = {f: np.mean([prob[sd][f] for sd in SEEDS], axis=0) for f in FAMS}
bag_s, bag_b = aps(bag)
print("3-seed bag (probmean):      standalone", H.summarize(bag_s), "\n                            in-blend  ", H.summarize(bag_b))
res = []
for sd in SEEDS:
    s_ap, b_ap = aps(prob[sd])
    ds, ps, _ = H.paired_boot(s_ap, bag_s)
    db, pb, sdb = H.paired_boot(b_ap, bag_b)
    _, pbt, _ = H.paired_boot(b_ap, bag_b, by="table")
    ss, bb = H.summarize(s_ap), H.summarize(b_ap)
    print(f"seed {sd:4d}: standalone {ss['overall']:.5f}  bag-minus-seed {ds:+.5f} P {ps:.3f} | "
          f"in-blend {bb['overall']:.5f}  bag-minus-seed {db:+.5f} P {pb:.3f} (Ptab {pbt:.3f}, boot sd {sdb:.5f})")
    res.append(dict(seed=sd, standalone=ss, blend=bb, d_std=ds, P_std=ps, d_blend=db, P_blend=pb, P_blend_tab=pbt))

mv_s = np.mean([r["standalone"]["overall"] for r in res])
mv_b = np.mean([r["blend"]["overall"] for r in res])
print(f"\nbag vs MEAN single seed: standalone {H.summarize(bag_s)['overall'] - mv_s:+.5f}  "
      f"in-blend {H.summarize(bag_b)['overall'] - mv_b:+.5f}")
print(f"bag vs BEST single seed: standalone "
      f"{H.summarize(bag_s)['overall'] - max(r['standalone']['overall'] for r in res):+.5f}  in-blend "
      f"{H.summarize(bag_b)['overall'] - max(r['blend']['overall'] for r in res):+.5f}")

# --- why the standalone gain dies in the blend: the partner's WITHIN-PAIR spread
print("\nwithin-pair spread of the partner score (mean over pairs of the top-5 sd), per family:")
pid = KEY["pair_id"].to_numpy()
for f in FAMS:
    row = []
    for lbl, S in [("seed42", prob[42]), ("seed202", prob[202]), ("seed777", prob[777]), ("bag", bag)]:
        d = pl.DataFrame({"pair_id": pid, "s": S[f], "fam": fam}).filter(c("fam") == f)
        v = d.group_by("pair_id").agg(c("s").top_k(5).std().alias("sd"))["sd"].mean()
        row.append(f"{lbl} {v:.4f}")
    print(f"  {H.SH[f]}: " + "  ".join(row))

# --- DP magnitude for scale
COMP = pl.read_parquet(os.path.join(CACHE, "exp042_ramp_exp027ev_dev_components.parquet"))
for f in FAMS:
    v = (COMP.join(KEY.with_columns(fam=pl.Series(fam)), on=["pair_id", "hand_idx"]).filter(c("fam") == f)
         .group_by("pair_id").agg(c(f"dp_{f}").top_k(5).std().alias("sd"))["sd"].mean())
    print(f"  {H.SH[f]}: DP top-5 sd {v:.4f}")
json.dump(dict(bag=dict(standalone=H.summarize(bag_s), blend=H.summarize(bag_b)), seeds=res),
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bag_vs_seeds.json"), "w"), indent=1)
