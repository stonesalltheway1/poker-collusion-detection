"""Kaggle write-up card / thumbnail, rendered at exactly 560 x 280 px.

Run: python scripts/make_card.py   ->  research/figures/kaggle_card_560x280.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent.parent / "research" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FELT = "#0c1613"        # deep table green, almost black
FELT2 = "#12211c"       # panel
INK = "#eef2ef"
MUTED = "#8b9b93"
ACCENT = "#5fc08f"      # type 1 / headline green
AMBER = "#d9964e"       # type 2
RULE = "#22332c"

W, H, DPI = 560, 280, 100
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
fig.patch.set_facecolor(FELT)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
ax.add_patch(Rectangle((0, 0), W, H, fc=FELT, ec="none", zorder=0))

# soft vignette panel on the right half
ax.add_patch(FancyBboxPatch((243, 26), 292, 228, boxstyle="round,pad=0,rounding_size=10",
                            fc=FELT2, ec=RULE, lw=1, zorder=1))

# accent hairline top-left
ax.add_patch(Rectangle((26, 246), 52, 3, fc=ACCENT, ec="none", zorder=2))

# ---------------------------------------------------------------- left: the result
ax.text(26, 224, "KAGGLE  ·  DETECT SUSPICIOUS", color=MUTED, fontsize=6.6,
        fontfamily="DejaVu Sans", fontweight="bold", zorder=3)
ax.text(26, 210, "VALUE TRANSFERS IN POKER", color=MUTED, fontsize=6.6,
        fontfamily="DejaVu Sans", fontweight="bold", zorder=3)

ax.text(22, 156, "8th", color=INK, fontsize=52, fontweight="bold",
        fontfamily="DejaVu Sans", va="center", zorder=3)
ax.text(28, 112, "of 370 teams", color=MUTED, fontsize=12, fontweight="bold",
        va="center", zorder=3)

ax.add_patch(Rectangle((26, 92), 175, 1, fc=RULE, ec="none", zorder=2))
ax.text(26, 68, "private", color=MUTED, fontsize=8.5, va="center", zorder=3)
ax.text(72, 68, "0.92817", color=ACCENT, fontsize=15, fontweight="bold",
        fontfamily="DejaVu Sans Mono", va="center", zorder=3)
ax.text(26, 44, "pair AP .981  ·  evidence .722", color=MUTED, fontsize=6.9,
        fontfamily="DejaVu Sans Mono", va="center", zorder=3)

# ---------------------------------------------------------------- right: the finding
ax.text(267, 224, "THE FINDING", color=ACCENT, fontsize=6.8, fontweight="bold", zorder=3)
ax.text(267, 200, "The evidence list is not a ranking.", color=INK, fontsize=10.5,
        fontweight="bold", zorder=3)
ax.text(267, 182, "It is a sorting rule.", color=INK, fontsize=10.5, fontweight="bold", zorder=3)

ax.add_patch(FancyBboxPatch((265, 132), 250, 34, boxstyle="round,pad=0,rounding_size=5",
                            fc="#0a1210", ec=RULE, lw=1, zorder=2))
ax.text(277, 149, "sorted(EVENTS,", color=ACCENT, fontsize=9.2,
        fontfamily="DejaVu Sans Mono", va="center", zorder=3)
ax.text(385, 149, "key=(type, seq))[:5]", color=INK, fontsize=9.2,
        fontfamily="DejaVu Sans Mono", va="center", zorder=3)

# the five listed hands: three type-1 then two type-2
x0, y0, w, h = 267, 78, 30, 26
for i in range(5):
    c = ACCENT if i < 3 else AMBER
    ax.add_patch(FancyBboxPatch((x0 + i * (w + 8), y0), w, h,
                                boxstyle="round,pad=0,rounding_size=4",
                                fc=c, ec="none", alpha=0.92, zorder=3))
    ax.text(x0 + i * (w + 8) + w / 2, y0 + h / 2, str(i + 1), color="#0c1613",
            fontsize=10, fontweight="bold", ha="center", va="center", zorder=4)

ax.text(267, 62, "type 1, oldest first", color=ACCENT, fontsize=7, va="top", zorder=3)
ax.text(382, 62, "then earliest type 2", color=AMBER, fontsize=7, va="top", zorder=3)

ax.text(267, 38, "Exact for coordinated isolation on all 460 listed hands.",
        color=MUTED, fontsize=7.2, va="center", zorder=3)

fig.savefig(OUT / "kaggle_card_560x280.png", dpi=DPI, facecolor=FELT)
plt.close(fig)

from PIL import Image  # noqa: E402
im = Image.open(OUT / "kaggle_card_560x280.png")
print("wrote", OUT / "kaggle_card_560x280.png", "size:", im.size)
