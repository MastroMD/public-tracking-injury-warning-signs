"""Figures 1-4 of the full paper, drawn from paper/figure_data.json (written by export_figure_data.py).

Run where the files are kept:
    python3 make_figures.py            (matplotlib and numpy only)
Writes figures/Figure1_trajectory_horizon.png ... Figure4_command_trajectory.png, each also as .pdf.
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
D = json.loads((HERE / "figure_data.json").read_text())

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK})
META = {"Software": None}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, facecolor="white", metadata=META)
    fig.savefig(FIG / f"{name}.pdf", facecolor="white", metadata={"Creator": None, "Producer": None})
    plt.close(fig)
    print("wrote", name)


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def label_ends(ax, ends, x=0, gap=0.11):
    ends = sorted(ends, key=lambda e: e[0])
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    for (y0, lab), y in zip(ends, ys):
        ax.annotate(lab, (x, y), xytext=(6, 0), textcoords="offset points", color=INK, fontsize=7, va="center")


# ------------------------------------------------------------------ Figure 1
def figure1():
    T, K0 = D["fig1"]["trajectory_h30"], D["fig1"]["k0_by_horizon"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), dpi=300, gridspec_kw=dict(width_ratios=[1.3, 1.3, 0.75]))
    panels = [(axes[0], [("pitcher_all", BLUE, "any IL"), ("pitcher_shoulder", ORANGE, "shoulder"), ("pitcher_elbow", AQUA, "elbow")],
               "A  Pitchers: four-seam velocity", "Outings before the placement"),
              (axes[1], [("batter_msk", BLUE, "any IL"), ("batter_oblique", ORANGE, "oblique")],
               "B  Batters: bat speed", "Games before the placement")]
    for ax, series, title, xlab in panels:
        ends = []
        for key, col, short in series:
            t = T[key]; ref = t["reference"]; k = np.arange(len(t["mean"]))
            m, lo, hi = np.array(t["mean"]) - ref, np.array(t["lo"]) - ref, np.array(t["hi"]) - ref
            ax.fill_between(k, lo, hi, color=col, alpha=0.14, linewidth=0)
            ax.plot(k, m, color=col, linewidth=2, marker="o", markersize=3.2, markeredgecolor="white", markeredgewidth=0.7)
            ends.append((m[0], f"{short} ({t['n'][0]:,})"))
        label_ends(ax, ends)
        ax.axhline(0, color=MUTED, linewidth=0.9)
        ax.set_xlim(10.4, -4.6)
        ax.spines["bottom"].set_bounds(10, 0)
        ax.set_xticks(range(0, 11, 2)); ax.set_xticklabels([str(i) if i else "last" for i in range(0, 11, 2)])
        ax.set_xlabel(xlab, x=0.36); ax.set_title(title)
        ax.set_ylim(-0.95, 0.45)
        style(ax)
    axes[0].set_ylabel("Deviation from own baseline,\nrelative to uninjured rows (SD)", fontsize=8)
    # panel C: last outing by horizon, pitchers
    ax = axes[2]
    hs = ["7", "14", "30", "60"]
    for j, (key, col) in enumerate([("pitcher_all", BLUE), ("pitcher_shoulder", ORANGE), ("pitcher_elbow", AQUA)]):
        x = np.arange(4) + (j - 1) * 0.2
        m = [K0[key][h][0] for h in hs]; lo = [K0[key][h][1] for h in hs]; hi = [K0[key][h][2] for h in hs]
        ax.vlines(x, lo, hi, color=col, linewidth=1.6)
        ax.plot(x, m, "o", color=col, markersize=4.2, markeredgecolor="white", markeredgewidth=0.7)
    ax.axhline(0, color=MUTED, linewidth=0.9)
    ax.set_xticks(range(4)); ax.set_xticklabels(hs)
    ax.set_xlabel("Window, days")
    ax.set_title("C  Last outing")
    ax.set_ylim(-0.95, 0.45)
    style(ax)
    ax.legend(handles=[Line2D([], [], color=c, marker="o", linewidth=0, markersize=4.2, label=l)
                       for c, l in [(BLUE, "any IL"), (ORANGE, "shoulder"), (AQUA, "elbow")]],
              loc="upper right", frameon=False, fontsize=7, handletextpad=0.2)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.88, wspace=0.28)
    save(fig, "Figure1_trajectory_horizon")


# ------------------------------------------------------------------ Figure 2
def figure2():
    rows = D["fig2"]
    groups = []
    for r in rows:
        if not groups or groups[-1][0] != r["group"]:
            groups.append((r["group"], []))
        groups[-1][1].append(r)
    ypos, labels, heads, y = [], [], [], 0.0
    for g, rs in groups:
        heads.append((y, g)); y -= 1.0
        for r in rs:
            ypos.append(y); labels.append(r["label"]); y -= 1.15
        y -= 0.25
    fig, ax = plt.subplots(figsize=(7.2, 7.6), dpi=300)
    slots = [("twin", BLUE, "o", True, 0.33), ("twin_abl", BLUE, "o", False, 0.11),
             ("sealed", ORANGE, "s", True, -0.11), ("sealed_abl", ORANGE, "s", False, -0.33)]
    yi = 0
    for g, rs in groups:
        for r in rs:
            yy = ypos[yi]; yi += 1
            present = [s for s in slots if s[0] in r]
            offs = np.linspace(0.12 * (len(present) - 1), -0.12 * (len(present) - 1), len(present)) if len(present) > 1 else [0.0]
            for (key, col, mk, filled, _), off in zip(present, offs):
                d, lo, hi = r[key]
                ax.hlines(yy + off, lo, hi, color=col, linewidth=1.4)
                ax.plot(d, yy + off, mk, color=col, markersize=4.6, markerfacecolor=col if filled else "white",
                        markeredgecolor=col, markeredgewidth=1.2)
    ax.axvline(0, color=MUTED, linewidth=0.9)
    ax.axvline(0.01, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.text(0.0105, heads[0][0] + 0.2, "escalation bar (+0.01)", fontsize=6.8, color=MUTED, va="bottom")
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=7.6, color=INK)
    for y0, g in heads:
        ax.text(-0.0005, y0, g, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.2,
                fontweight="bold", color=INK)
    ax.set_xlim(-0.065, 0.065)
    ax.set_ylim(min(ypos) - 0.8, heads[0][0] + 0.9)
    ax.set_xlabel("Gain in AUC over the workload model (95% CI)")
    ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    handles = [Line2D([], [], color=BLUE, marker="o", markersize=4.6, linewidth=1.4, label="Public twin, all games"),
               Line2D([], [], color=BLUE, marker="o", markersize=4.6, markerfacecolor="white", linewidth=1.4, label="Public twin, last game removed"),
               Line2D([], [], color=ORANGE, marker="s", markersize=4.6, linewidth=1.4, label="Registry, all games"),
               Line2D([], [], color=ORANGE, marker="s", markersize=4.6, markerfacecolor="white", linewidth=1.4, label="Registry, last game removed")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=2, frameon=False, fontsize=7.4)
    fig.subplots_adjust(left=0.40, right=0.98, bottom=0.07, top=0.93)
    save(fig, "Figure2_ablation_forest")


# ------------------------------------------------------------------ Figure 3
def figure3():
    F = D["fig3"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), dpi=300)
    spec = [("P2", "all", "any IL", 0.30, axes[0]), ("P3", "shoulder", "shoulder IL", 0.13, axes[1])]
    thr = F["thresholds_f2"]
    for model, lab, name, tmax, (axc, axd) in spec:
        dec = F["deciles"][model]
        for who, col, mk in (("null", ORANGE, "s"), ("model", BLUE, "o")):
            x, y = np.array(dec[who]["mean_pred"]), np.array(dec[who]["observed"])
            axc.plot(x, y, marker=mk, color=col, linewidth=1.4, markersize=4, markeredgecolor="white", markeredgewidth=0.6)
        hi = max(max(dec["model"]["observed"]), max(dec["model"]["mean_pred"]), max(dec["null"]["observed"])) * 1.08
        axc.plot([0, hi], [0, hi], color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
        axc.set_xlim(0, hi); axc.set_ylim(0, hi)
        axc.set_xlabel("Predicted risk (decile mean)"); axc.set_ylabel("Observed rate")
        s = F["slope"][model]
        axc.set_title(f"{'A' if model == 'P2' else 'C'}  Calibration, {name}")
        axc.text(0.04 * hi, 0.92 * hi, f"slope {s[0]:.2f} ({s[1]:.2f} to {s[2]:.2f})", fontsize=7, color=INK, va="top")
        style(axc)
        rows = [r for r in F["dca"] if r["model"] == model and float(r["t"]) <= tmax + 1e-9]
        t = np.array([float(r["t"]) for r in rows])
        for key, col, ls, lw in (("nb_treat_all", MUTED, (0, (3, 2)), 1.0), ("nb_workload", ORANGE, "-", 1.6), ("nb_model", BLUE, "-", 1.6)):
            axd.plot(t, [float(r[key]) for r in rows], color=col, linestyle=ls, linewidth=lw)
        axd.axhline(0, color=INK, linewidth=0.8)
        nb_top = max(float(r["nb_workload"]) for r in rows) + 0.004
        axd.set_ylim(-0.01 if model == "P2" else -0.003, max(nb_top, 0.006))
        for tt in thr[lab]:
            if tt <= tmax:
                axd.axvline(tt, color=MUTED, linewidth=0.7, linestyle=(0, (1, 2)))
        axd.set_xlim(t.min(), tmax)
        axd.set_xlabel("Threshold probability for acting"); axd.set_ylabel("Net benefit")
        axd.set_title(f"{'B' if model == 'P2' else 'D'}  Decision curve, {name}")
        style(axd)
    handles = [Line2D([], [], color=BLUE, linewidth=1.6, marker="o", markersize=4, label="Tracking model"),
               Line2D([], [], color=ORANGE, linewidth=1.6, marker="s", markersize=4, label="Workload model"),
               Line2D([], [], color=MUTED, linewidth=1.0, linestyle=(0, (3, 2)), label="Act on every pitcher"),
               Line2D([], [], color=INK, linewidth=0.8, label="Act on none"),
               Line2D([], [], color=MUTED, linewidth=0.7, linestyle=(0, (1, 2)), label="Rest-decision thresholds")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.09, right=0.965, bottom=0.085, top=0.9, wspace=0.3, hspace=0.5)
    save(fig, "Figure3_calibration_decision")


# ------------------------------------------------------------------ Figure 4
def figure4():
    F = D["fig4"]
    fig, ax = plt.subplots(figsize=(3.8, 2.8), dpi=300)
    k = np.array(F["k"]); ref = F["reference"]
    m, lo, hi = np.array(F["mean"]) - ref, np.array(F["lo"]) - ref, np.array(F["hi"]) - ref
    ax.fill_between(k, lo, hi, color=BLUE, alpha=0.14, linewidth=0)
    ax.plot(k, m, color=BLUE, linewidth=2, marker="o", markersize=3.2, markeredgecolor="white", markeredgewidth=0.7)
    ax.annotate(f"any IL ({F['n'][0]:,})", (0, m[0]), xytext=(6, 0), textcoords="offset points", fontsize=7, color=INK, va="center")
    ax.axhline(0, color=MUTED, linewidth=0.9)
    ax.set_xlim(len(k) - 0.6, -0.4)
    ax.set_xticks(range(0, len(k), 2)); ax.set_xticklabels([str(i) if i else "last" for i in range(0, len(k), 2)])
    ax.set_xlabel("Outings before the placement")
    ax.set_ylabel("Miss from target vs own baseline,\nrelative to uninjured outings (SD)", fontsize=8)
    ax.set_title("Command, 2024–2026 (descriptive)")
    style(ax)
    fig.subplots_adjust(left=0.2, right=0.83, bottom=0.17, top=0.88)
    save(fig, "Figure4_command_trajectory")


if __name__ == "__main__":
    figure1(); figure2(); figure3(); figure4()
