"""Generate the paper's data figures from paper/ledger.csv.

fig_margins.pdf   - improvement margin vs prior record holder (log strip plot),
                    mechanism encoded by color + marker shape (CVD-validated
                    palette; grayscale-safe via shapes)
fig_attrition.pdf - credited vs standing record counts over the six scrapes

Run:  polygon-packer/.venv/Scripts/python paper/make_figures.py
"""
import csv
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
FIG.mkdir(exist_ok=True)

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
MECH_COLOR = {"harvest": "#2a78d6", "search": "#eb6834",
              "closed-form": "#1baf7a", "drop-one": "#eda100"}
MECH_MARK = {"harvest": "o", "search": "s", "closed-form": "D", "drop-one": "^"}
MECH_ORDER = ["harvest", "search", "closed-form", "drop-one"]

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})

rows = list(csv.DictReader(open(ROOT / "paper" / "ledger.csv", encoding="utf-8")))

# ------------------------------------------------------------- fig_margins ----
pts = []
for r in rows:
    if "withdrawn" in r["status_2026-08-16"]:
        continue  # pen_in_hex 6: quoted floor was stale, margin not meaningful
    mech = r["mechanism"].replace("search+drop-one", "search")
    holder = r["prior_holder"].replace("Trivial (10−5√2)", "Trivial lattice")
    pts.append((holder, float(r["margin_lb"]), mech))

holders = {}
for h, m, k in pts:
    holders.setdefault(h, []).append(m)
order = sorted(holders, key=lambda h: -float(np.median(holders[h])))

fig, ax = plt.subplots(figsize=(6.3, 3.0))
rng = np.random.default_rng(11)
for h, m, k in pts:
    y = order.index(h) + rng.uniform(-0.16, 0.16)
    ax.scatter(m, y, s=16, marker=MECH_MARK[k], facecolor=MECH_COLOR[k],
               edgecolor="white", linewidth=0.4, zorder=3)
for i, h in enumerate(order):
    med = float(np.median(holders[h]))
    ax.plot([med, med], [i - 0.28, i + 0.28], color=INK2, lw=1.1, zorder=2)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{h}  ({len(holders[h])})" for h in order])
ax.set_xscale("log")
ax.set_xlim(2e-7, 2e-2)
ax.set_xlabel("improvement margin over displayed record floor (side-length units, log scale)")
ax.invert_yaxis()
ax.grid(axis="x", color=MUTED, alpha=0.22, lw=0.5)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
handles = [plt.Line2D([], [], marker=MECH_MARK[k], color="none", markersize=5,
                      markerfacecolor=MECH_COLOR[k], markeredgecolor="white",
                      markeredgewidth=0.4, label=k) for k in MECH_ORDER]
handles.append(plt.Line2D([], [], color=INK2, lw=1.1, label="median"))
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7.5,
          handletextpad=0.2, borderaxespad=0.1)
fig.tight_layout()
fig.savefig(FIG / "fig_margins.pdf")
fig.savefig(FIG / "fig_margins.png", dpi=220)
plt.close(fig)

# ----------------------------------------------------------- fig_attrition ----
scrapes = [date(2026, 7, 4), date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 7),
           date(2026, 7, 9), date(2026, 8, 16)]
standing = [0, 48, 48, 48, 49, 41]
credited = [0, 48, 48, 48, 52, 52]

fig, ax = plt.subplots(figsize=(6.3, 2.4))
ax.step(scrapes, credited, where="post", color="#2a78d6", lw=1.8)
ax.step(scrapes, standing, where="post", color="#eb6834", lw=1.8)
ax.scatter(scrapes, credited, s=13, color="#2a78d6", zorder=3)
ax.scatter(scrapes, standing, s=13, color="#eb6834", zorder=3)
ax.annotate("problems credited (cumulative): 52", (scrapes[-1], 52),
            xytext=(-4, 6), textcoords="offset points", ha="right",
            color="#184f95", fontsize=8)
ax.annotate("still standing: 41", (scrapes[-1], 41), xytext=(-4, -12),
            textcoords="offset points", ha="right", color="#d95926", fontsize=8)
ax.annotate("batches 1–6 processed", (date(2026, 7, 6), 48), xytext=(6, -14),
            textcoords="offset points", color=INK2, fontsize=7.5)
ax.annotate("batch 7 processed;\nfirst re-takes", (date(2026, 7, 9), 49),
            xytext=(10, -26), textcoords="offset points", color=INK2, fontsize=7.5)
ax.set_ylim(0, 60)
ax.set_ylabel("record count")
ax.grid(axis="y", color=MUTED, alpha=0.22, lw=0.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig_attrition.pdf")
fig.savefig(FIG / "fig_attrition.png", dpi=220)
plt.close(fig)

print("wrote", FIG / "fig_margins.pdf", "and", FIG / "fig_attrition.pdf")
