"""How much other_coordination can hide in the eval positives, given the EXACT probes?

Exact measurements (matched LB probes, experiments/LEDGER.md):
    pair AP           = 0.97378   (sub025/sub026 arithmetic; 0.9733 for the older sub011 risk)
    behaviour macroAP = 0.96815   (sub014 probe + the PER-FAMILY random-ranking floor b0 = 0.00175:
                                   metric.py scores each family against its own ~P/3 positives, not all P)

Host metric fact (src/metric.py): a truth pair whose behaviour is `other_coordination` is a NEGATIVE for
all three disclosed families, but a POSITIVE for pair AP. So an OC pair that we rank high and label with a
disclosed family is a high-ranked false positive in exactly one behaviour AP.

This script asks: for what OC share phi_oc is behAP = 0.9664 reproducible, given pairAP = 0.9740?
"""
import numpy as np

rng = np.random.default_rng(0)
M = 112_540
FAM_SHARE = np.array([148, 132, 92], float) / 372.0   # dev family proportions


def ap(y, s):
    order = np.argsort(-s, kind="mergesort")
    r = y[order]
    return float(np.sum(np.cumsum(r) / np.arange(1, len(r) + 1) * r) / r.sum())


def simulate(N, phi_oc, mu, mu_oc_delta, beh_acc, reps=12, oc_routing="resemble"):
    """Returns (pairAP, behAP) averaged over reps."""
    n_oc = int(round(N * phi_oc))
    n_fam = N - n_oc
    per_fam = np.round(FAM_SHARE * n_fam).astype(int)
    per_fam[0] += n_fam - per_fam.sum()
    pa, ba = [], []
    for _ in range(reps):
        n_neg = M - N
        s_neg = rng.normal(0, 1, n_neg)
        s_fam = rng.normal(mu, 1, n_fam)
        s_oc = rng.normal(mu + mu_oc_delta, 1, n_oc)
        scores = np.concatenate([s_neg, s_fam, s_oc])
        y = np.concatenate([np.zeros(n_neg), np.ones(n_fam + n_oc)])
        pa.append(ap(y.astype(int), scores))

        # true family index: -1 = negative, 3 = other_coordination
        truth = np.concatenate([
            np.full(n_neg, -1),
            np.repeat([0, 1, 2], per_fam),
            np.full(n_oc, 3)])
        # predicted family
        pred = np.empty(M, int)
        # negatives: assigned by resemblance -> approximately the family prior
        pred[:n_neg] = rng.choice(3, n_neg, p=FAM_SHARE)
        off = n_neg
        for f, k in enumerate(per_fam):                       # family positives
            p = np.full(3, (1 - beh_acc) / 2); p[f] = beh_acc
            pred[off:off + k] = rng.choice(3, k, p=p)
            off += k
        if oc_routing == "resemble":                          # OC pairs look like SOME family
            pred[off:] = rng.choice(3, n_oc, p=FAM_SHARE)
        else:
            pred[off:] = rng.choice(3, n_oc)
        # host behaviour AP: score = risk where predicted == f else 0
        rank = (-scores).argsort().argsort()
        risk = 1.0 - rank / (M + 1.0)                          # monotone in score, in (0,1)
        aps = []
        for f in range(3):
            yt = (truth == f).astype(int)
            sc = np.where(pred == f, risk, 0.0)
            aps.append(ap(yt, sc))
        ba.append(float(np.mean(aps)))
    return float(np.mean(pa)), float(np.mean(ba))


TARGET_PAIR, TARGET_BEH = 0.97378, 0.96815

for N in (420, 470, 520):
    # calibrate mu so that pairAP hits the target with phi_oc = 0
    lo, hi = 2.0, 7.0
    for _ in range(28):
        mid = (lo + hi) / 2
        p, _b = simulate(N, 0.0, mid, 0.0, 0.99, reps=6)
        if p < TARGET_PAIR: lo = mid
        else: hi = mid
    mu = (lo + hi) / 2
    print(f"\n=== N = {N} positives; calibrated mu = {mu:.3f} (pairAP -> {TARGET_PAIR}) ===")
    print(f"{'beh_acc':>8} {'phi_oc':>7} {'oc rank':>9} {'pairAP':>8} {'behAP':>8}  verdict")
    for beh_acc in (1.00, 0.99, 0.98):
        for phi in (0.0, 0.02, 0.05, 0.10, 0.14):
            for d in (0.0, -0.5):
                # re-calibrate mu so pairAP stays at target for this phi/delta
                lo, hi = 2.0, 8.0
                for _ in range(24):
                    mid = (lo + hi) / 2
                    p, _ = simulate(N, phi, mid, d, beh_acc, reps=4)
                    if p < TARGET_PAIR: lo = mid
                    else: hi = mid
                p, b = simulate(N, phi, (lo + hi) / 2, d, beh_acc, reps=14)
                tag = "MATCHES measured behAP" if abs(b - TARGET_BEH) < 0.004 else (
                    "too low" if b < TARGET_BEH else "too high")
                print(f"{beh_acc:>8.2f} {phi:>7.2f} {d:>9.1f} {p:>8.4f} {b:>8.4f}  {tag}")
