"""Figures for the Kaggle solution write-up.

Every number is sourced from a repo file or from a scored leaderboard submission; the source is named in the
caption note under each figure. Run: python scripts/make_figures.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "research" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#161d1a"; MUTED = "#69756f"; GRID = "#dde1dc"; PAPER = "#ffffff"
C_DT, C_SP, C_CI = "#35608f", "#c2733a", "#3f7f62"
C_HI, C_LO = "#35608f", "#b9c0bd"

plt.rcParams.update({
    "figure.dpi": 170, "savefig.dpi": 170, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
})


def finish(fig, name, note=None):
    if note:
        fig.text(0.005, -0.035, note, fontsize=6.6, color=MUTED, ha="left", va="top")
    fig.savefig(OUT / (name + ".png"))
    plt.close(fig)
    print("wrote", (OUT / (name + ".png")).name)


# ------------------------------------------------------------------ 1. phase locality
res = json.loads((BASE / "research/forensics/structure_results.json").read_text())
E2 = res["E2_persistence_dev_vs_eval_auc"]
names = {"both_vpip_rate": "both players\nvoluntarily in pot", "both_flop_rate": "both see\nthe flop",
         "sd_tog_rate": "showdown\ntogether", "gross_flow_per100": "gross chip flow\nper 100 hands",
         "abs_net_flow_per100": "net chip flow\nper 100 hands"}
keys = [k for k in names if k in E2]
dev = [E2[k]["dev_auc_pos_vs_neg"] for k in keys]
ev = [E2[k]["eval_auc_pos_vs_neg(eval_shared>=38)"] for k in keys]
fig, ax = plt.subplots(figsize=(6.9, 3.2))
x = np.arange(len(keys)); w = 0.38
ax.bar(x - w / 2, dev, w, color=C_HI, label="measured on development-period hands")
ax.bar(x + w / 2, ev, w, color=C_LO, label="measured on evaluation-period hands")
ax.axhline(0.5, color=INK, lw=1, ls=(0, (4, 3)), zorder=4)
for xi, (d, e) in enumerate(zip(dev, ev)):
    ax.text(xi - w / 2, d + .013, "%.2f" % d, ha="center", fontsize=7.8, color=INK)
    ax.text(xi + w / 2, e - .022, "%.2f" % e, ha="center", fontsize=7.8, color="white", zorder=5)
ax.set_xticks(x); ax.set_xticklabels([names[k] for k in keys], fontsize=7.6)
ax.set_ylim(0.4, 1.02)
ax.set_ylabel("AUC, 281 labelled colluding pairs\nvs 1,149 confirmed-clean pairs")
ax.set_title("The same colluding pairs are inert in the period they are not labelled in")
ax.annotate("AUC 0.50 = indistinguishable from a clean pair", xy=(4.2, 0.5), xytext=(4.45, 0.60),
            fontsize=7.4, color=INK, ha="center", arrowprops=dict(arrowstyle="-", color=INK, lw=.8))
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.yaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)
finish(fig, "fig1_phase_locality",
       "Source: research/forensics/structure_results.json, E2_persistence_dev_vs_eval_auc.")

# ------------------------------------------------------------------ 2. the listing rule
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw={"width_ratios": [1.3, 1]})
fig.subplots_adjust(wspace=0.40)
ax = axes[0]
fams = ["directed\ntransfer", "soft\nplay", "coordinated\nisolation"]
chrono = [0.384, 0.391, 0.316]; seg1 = [0.498, 0.499, 0.411]; seg2 = [0.248, 0.309, 0.172]
x = np.arange(3); w = 0.26
ax.bar(x - w, chrono, w, color="#b9c0bd", label="lists with no inversion")
ax.bar(x, seg1, w, color=C_HI, label="segment 1 (type 1)")
ax.bar(x + w, seg2, w, color="#d98f4a", label="segment 2 (type 2)")
ax.axhline(0.5, color=INK, lw=1, ls=(0, (4, 3)), zorder=4)
ax.text(-0.42, 0.515, "0.50 = spread evenly over the period", fontsize=7, color=INK)
ax.set_xticks(x); ax.set_xticklabels(fams, fontsize=8); ax.set_ylim(0, 0.70)
ax.set_ylabel("mean position in the pair's timeline\n(0 = first shared hand, 1 = last)")
ax.set_title("Segment 1 is the complete type-1 set,\nsegment 2 only the earliest type-2 events", fontsize=9.5)
ax.legend(frameon=False, fontsize=7.4, loc="upper center")
ax.yaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)
ax = axes[1]
groups = ["listed, no\ninversion\n(405 hands)", "segment 1\n(43)", "segment 2\n(12)"]
fb0 = np.array([405, 43, 0]); fb1 = np.array([0, 0, 12])
x = np.arange(3)
ax.bar(x, fb0, 0.55, color=C_HI, label="0 players folded\nbefore the raise")
ax.bar(x, fb1, 0.55, bottom=fb0, color="#d98f4a", label="exactly 1 folded")
for xi, tot in enumerate(fb0 + fb1):
    ax.text(xi, tot + 12, str(tot), ha="center", fontsize=8.5, color=INK, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=7.4); ax.set_ylim(0, 540)
ax.set_ylabel("listed coordinated-isolation hands")
ax.set_title("For coordinated isolation the type key\nis exact, and mechanical", fontsize=9.5)
ax.legend(frameon=False, fontsize=7.2, loc="upper right")
ax.yaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)
finish(fig, "fig2_listing_rule",
       "Source: research/forensics/listing_rule.md sections 1b and 2, over the 1,817 listed hands of the 372 labelled positive pairs.")

# ------------------------------------------------------------------ 3. pipeline
fig, ax = plt.subplots(figsize=(8.6, 4.0))
ax.set_xlim(0, 100); ax.set_ylim(2, 101); ax.axis("off")


def box(x, y, w_, h_, text, fc, ec, fs=8, bold=False, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w_, h_, boxstyle="round,pad=0.6,rounding_size=1.6",
                                fc=fc, ec=ec, lw=1.1, zorder=2))
    ax.text(x + w_ / 2, y + h_ / 2, text, ha="center", va="center", fontsize=fs, zorder=3,
            color=tc or INK, fontweight="bold" if bold else "normal", linespacing=1.45)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                 color="#9aa5a0", lw=1.0, zorder=1,
                                 shrinkA=1, shrinkB=1))


box(2, 84, 27, 11, "actions · seats · hands\nall hole cards visible", "#f2f5f3", GRID)
box(36, 84, 28, 11, "omniscient equity\nat every decision", "#f2f5f3", GRID)
box(70, 84, 28, 11, "label-free normal-play policy\nP(action | state, own cards)", "#f2f5f3", GRID)
arrow(29, 89.5, 36, 89.5); arrow(64, 89.5, 70, 89.5)

box(20, 66, 60, 11, "PAIR-HAND ENGINE  (numba)\n~90 features for all 15 seat pairs, one pass per hand",
    "#e7eef4", "#9db8cf", fs=8.5, bold=True)
arrow(15, 84, 30, 77.5); arrow(50, 84, 50, 77.5); arrow(84, 84, 70, 77.5)

box(1.5, 44, 30, 15, "PAIR FEATURES\nrates, binomial z,\nown-baseline directed lift", "#f2f5f3", GRID)
box(35, 44, 30, 15, "HAND DETECTOR\nplanted vs clean-pair hands\n-> per-pair top-k aggregates", "#f2f5f3", GRID)
box(68.5, 44, 30, 15, "EVIDENCE\nlisting-rule DP\n+ in-pair ranker", "#fbf0e6", "#d9ab86")
arrow(35, 66, 17, 59); arrow(50, 66, 50, 59); arrow(65, 66, 83, 59)

box(10, 25, 55, 12, "PAIR MODEL   LightGBM GOSS  x  GPU XGBoost   (PU learning)",
    "#e7eef4", "#9db8cf", fs=8.5, bold=True)
arrow(16.5, 44, 25, 37); arrow(50, 44, 50, 37)

box(1.5, 4, 26, 12, "risk_score\npair AP  ·  70%", "#dfeae2", "#8fb3a0", bold=True)
box(31, 4, 26, 12, "predicted_behavior\nbehaviour macro-AP  ·  10%", "#dfeae2", "#8fb3a0", bold=True)
box(60.5, 4, 38, 12, "5 evidence hands per pair\nevidence MAP@5  ·  20%", "#fbf0e6", "#d9ab86", bold=True)
arrow(26, 25, 15, 16); arrow(45, 25, 44, 16); arrow(83, 44, 79, 16)

ax.text(50, 98, "Pipeline", ha="center", fontsize=11.5, fontweight="bold", color=INK)
finish(fig, "fig3_pipeline",
       "Everything is computed inside the period being scored; the other period is used only as a clean per-player baseline.")

# ------------------------------------------------------------------ 4. attribution
steps = [
    ("phase-local pair-hand engine", 0.61704, 0.78984),
    ("learned evidence rankers", 0.80414, 0.85923),
    ("chronology inside the scored period", 0.85923, 0.88153),
    ("hand detectors + generic pair features", 0.88153, 0.90310),
    ("normal-play policy features", 0.78984, 0.80414),
    ("the listing-rule dynamic program", 0.91117, 0.91773),
    ("suspected hidden positives out of the negatives", 0.90310, 0.90908),
    ("LightGBM x XGBoost pair fusion", 0.91845, 0.91954),
    ("self-training rounds", 0.90908, 0.91018),
    ("a stronger in-pair evidence ranker", 0.91018, 0.91117),
    ("same-fold evaluation-hand scoring", 0.91773, 0.91820),
    ("smooth family-strength gate", 0.91820, 0.91845),
]
labs = [s[0] for s in steps]
d = np.array([s[2] - s[1] for s in steps])
fig, ax = plt.subplots(figsize=(7.6, 4.0))
y = np.arange(len(labs))[::-1]
ax.barh(y, d, 0.62, color=[C_HI if v >= 0.005 else "#9fb6cd" for v in d])
ax.set_xscale("log"); ax.set_xlim(1.5e-4, 0.45)
for yi, v in zip(y, d):
    ax.text(v * 1.18, yi, ("+%.4f" % v).rstrip("0").rstrip("."), va="center", fontsize=7.6, color=INK)
ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8)
ax.set_xlabel("gain on the public leaderboard, from a matched A/B that changed exactly one component (log scale)")
ax.set_title("What actually moved the score: 0.617 to 0.91954 in twelve measured steps")
ax.xaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
finish(fig, "fig4_attribution",
       "Source: experiments/LEDGER.md. Each bar is the difference between two submissions that changed exactly one component.")

# ------------------------------------------------------------------ 5. score anatomy, private
fig, ax = plt.subplots(figsize=(7.6, 2.8))
comp = [("pair average precision", 0.70, 0.98070, C_DT),
        ("evidence MAP@5", 0.20, 0.72180, C_SP),
        ("behaviour macro-AP", 0.10, 0.96919, C_CI)]
ypos = np.arange(3)[::-1]
for (name, wgt, val, col), yp in zip(comp, ypos):
    got, lost = wgt * val, wgt * (1 - val)
    ax.barh(yp, got, 0.52, color=col)
    ax.barh(yp, lost, 0.52, left=got, color="#eceeed")
    ax.text(got / 2, yp, "%.4f" % val, va="center", ha="center", color="white", fontsize=9.5, fontweight="bold")
    ax.text(got + lost + 0.007, yp, "%.4f of the final score still on the table" % lost,
            va="center", fontsize=8, color=MUTED)
    ax.text(-0.012, yp, "%s\nweight %d%%" % (name, wgt * 100), va="center", ha="right", fontsize=8)
ax.set_xlim(0, 0.93); ax.set_ylim(-0.6, 2.6); ax.set_yticks([])
ax.set_xlabel("contribution to the final private score")
ax.set_title("Private score 0.92817, decomposed exactly with matched probe submissions")
ax.spines["left"].set_visible(False)
ax.xaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)
finish(fig, "fig5_score_anatomy",
       "Components solved from the private scores of sub011 / sub012 / sub014 / sub026 / sub031, each a copy of a scored file with exactly one field blanked.")

# ------------------------------------------------------------------ 6. what limits it
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw={"width_ratios": [1.05, 1]})
fig.subplots_adjust(wspace=0.34)
ax = axes[0]
lab = ["shipped\nmodel", "+ true\ntype", "+ true\nevent", "+ both"]
val = [0.7748, 0.7930, 0.9322, 0.9420]
bars = ax.bar(np.arange(4), val, 0.58, color=[C_HI, "#8aa9c4", "#c2733a", "#d9a880"])
for i, v in enumerate(val):
    ax.text(i, v + .008, "%.3f" % v, ha="center", fontsize=8.5, color=INK, fontweight="bold")
ax.annotate("", xy=(2, 0.9322), xytext=(0, 0.7748),
            arrowprops=dict(arrowstyle="->", color="#c2733a", lw=1.4,
                            connectionstyle="arc3,rad=-0.28"))
ax.text(1.0, 0.885, "+0.157", fontsize=9, color="#c2733a", fontweight="bold", ha="center")
ax.text(1, 0.748, "+0.018", fontsize=8.2, color="white", ha="center", fontweight="bold")
ax.set_xticks(np.arange(4)); ax.set_xticklabels(lab, fontsize=8)
ax.set_ylim(0.70, 1.0); ax.set_ylabel("evidence MAP@5, out of fold")
ax.set_title("Knowing which hands were planted is worth\nnine times more than knowing their type", fontsize=9.5)
ax.yaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)

ax = axes[1]
k = [1, 3, 5, 10, 20]
rec = [0.352, 0.681, 0.8199, 0.9696, 0.9978]
ax.plot(k, rec, color=C_HI, lw=1.8, marker="o", ms=5, mfc="white", mew=1.6, zorder=3)
ax.axhline(0.7748, color="#c2733a", lw=1.2, ls=(0, (4, 3)))
ax.text(20, 0.762, "our AP@5 = 0.775", fontsize=7.8, color="#c2733a", ha="right", va="top")
for kk, rr in [(5, 0.8199), (10, 0.9696)]:
    ax.annotate("%.3f" % rr, (kk, rr), textcoords="offset points", xytext=(6, -12),
                fontsize=8.2, color=INK, fontweight="bold")
ax.set_xticks(k); ax.set_xlabel("k"); ax.set_ylim(0.30, 1.03)
ax.set_ylabel("recall@k of the host's listed hands")
ax.set_title("Retrieval is solved. The listed hands are\nalmost always inside our top 10", fontsize=9.5)
ax.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)
finish(fig, "fig6_what_limits",
       "Oracle study and recall curve over the 372 labelled positive pairs, same folds and same pipeline as the shipped scorer.")

print("done")
