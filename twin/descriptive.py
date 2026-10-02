"""Descriptive trajectories and a simple-rule check — POST HOC, no model is fit (DEVIATIONS.md #6).

From the twin's feature panels (already built by batter.py and pitcher.py) and the public census:
  A. mean own-baseline z-score by number of outings (pitchers) or games (batters) before an IL placement,
     k = 0 for the last outing/game before the placement (<= 30 days), k = 1.. the ones before it in the
     same season; reference = rows with no placement in the following 60 days. Player-cluster bootstrap CIs.
  B. the simplest possible alert: rows with z <= -1 (and <= -2) — what fraction are followed by a placement
     within 30 days (PPV), against the base rate, and what fraction of placements such a rule would catch.
Writes out/descriptive_trajectory.json and figures/Figure1_trajectory.png.

Usage: python3 -m twin.descriptive
"""
import json

import numpy as np
import pandas as pd

from twin import common as C

K = 10
B = 1000


def trajectories(X, pid_col, z_col, starts_by_pid, tag):
    X = X.sort_values([pid_col, "season", "game_date"]).reset_index(drop=True)
    pids, dates, z = X[pid_col].values, X.game_date.values, X[z_col].values.astype(float)
    seasons = X.season.values
    rows_by = {}
    for i, p in enumerate(pids):
        rows_by.setdefault(p, []).append(i)
    rec = []  # (pid, k, z)
    within60 = np.zeros(len(X), bool)
    for p, a in starts_by_pid.items():
        rows = rows_by.get(p)
        if not rows:
            continue
        rows = np.array(rows)
        for s0 in a:
            d = (s0 - dates[rows]) / np.timedelta64(1, "D")
            within60 |= np.isin(np.arange(len(X)), rows[(d > 0) & (d <= 60)])
            before = rows[(d > 0)]
            if not len(before):
                continue
            last = before[-1]
            if (s0 - dates[last]) / np.timedelta64(1, "D") > 30:
                continue
            same = before[seasons[before] == seasons[last]]
            for k, r in enumerate(same[::-1][:K + 1]):
                if np.isfinite(z[r]):
                    rec.append((p, k, z[r]))
    R = pd.DataFrame(rec, columns=["pid", "k", "z"])
    ref = z[~within60 & np.isfinite(z)]
    ref_p = pids[~within60 & np.isfinite(z)]
    rng = np.random.default_rng(C.SEED)
    out = {"k": list(range(K + 1)), "mean": [], "lo": [], "hi": [], "n": [], "n_players": []}
    for k in range(K + 1):
        g = R[R.k == k]
        out["mean"].append(float(g.z.mean())); out["n"].append(int(len(g))); out["n_players"].append(int(g.pid.nunique()))
        u = g.pid.unique(); idx = {p: g.z.values[g.pid.values == p] for p in u}
        bs = [np.concatenate([idx[p] for p in rng.choice(u, len(u))]).mean() for _ in range(B)]
        out["lo"].append(float(np.quantile(bs, .025))); out["hi"].append(float(np.quantile(bs, .975)))
    u = np.unique(ref_p); idx = {p: ref[ref_p == p] for p in u}
    bs = [np.concatenate([idx[p] for p in rng.choice(u, len(u))]).mean() for _ in range(200)]
    out["reference"] = dict(mean=float(ref.mean()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), n=int(len(ref)))
    out["tag"] = tag
    return out


def simple_rule(X, pid_col, z_col, starts_by_pid):
    y = C.label_within(X[pid_col].values, X.game_date.values, starts_by_pid)
    z = X[z_col].values.astype(float)
    ok = np.isfinite(z)
    res = {"base_rate": float(y[ok].mean()), "n_rows": int(ok.sum()), "n_pos": int(y[ok].sum())}
    for thr in (-1.0, -2.0):
        f = ok & (z <= thr)
        res[f"z_le_{abs(thr):.0f}"] = dict(n_flagged=int(f.sum()), share_of_rows=float(f.sum() / ok.sum()),
                                           ppv=float(y[f].mean()) if f.sum() else None,
                                           sensitivity=float((y[f] == 1).sum() / max(y[ok].sum(), 1)))
    return res


def main():
    P = pd.read_csv(C.OUT / "placements_public.csv", parse_dates=["il_start"])
    mk = lambda d: {int(k): np.sort(g.il_start.values) for k, g in d.groupby("mlbam")}
    Pp = P[~P.covid_era_blank]
    L = dict(pitcher_all=mk(Pp), pitcher_shoulder=mk(Pp[Pp.site == "shoulder"]),
             batter_msk=mk(P[P.msk]), batter_oblique=mk(P[P.site == "core_oblique"]))
    Xp = pd.read_parquet(C.OUT / "pitcher_primary_post_features.parquet")
    Xp["game_date"] = pd.to_datetime(Xp.game_date)
    Xb = pd.read_parquet(C.OUT / "batter_primary_public_features.parquet")
    Xb["game_date"] = pd.to_datetime(Xb.game_date)
    Xb = Xb[Xb.eligible10].reset_index(drop=True)
    out = dict(
        trajectories=dict(
            pitcher_all=trajectories(Xp, "player_id", "z_velocity", L["pitcher_all"], "pitchers, any IL"),
            pitcher_shoulder=trajectories(Xp, "player_id", "z_velocity", L["pitcher_shoulder"], "pitchers, shoulder IL"),
            batter_msk=trajectories(Xb, "batter", "z_bs_dev", L["batter_msk"], "batters, any musculoskeletal IL"),
            batter_oblique=trajectories(Xb, "batter", "z_bs_dev", L["batter_oblique"], "batters, oblique IL")),
        simple_rule=dict(
            pitcher_velocity_any_il=simple_rule(Xp, "player_id", "z_velocity", L["pitcher_all"]),
            batter_batspeed_any_msk=simple_rule(Xb, "batter", "z_bs_dev", L["batter_msk"]),
            batter_batspeed_oblique=simple_rule(Xb, "batter", "z_bs_dev", L["batter_oblique"])),
        note="post hoc descriptive; no model fit; DEVIATIONS.md #6")
    json.dump(out, open(C.OUT / "descriptive_trajectory.json", "w"), indent=1)
    for k, t in out["trajectories"].items():
        print(k, "k0..3:", [round(m, 3) for m in t["mean"][:4]], "ref", round(t["reference"]["mean"], 3), "n_k0", t["n"][0])
    print(json.dumps(out["simple_rule"], indent=1))
    figure(out)


def figure(out):
    """Each line is centred on its own reference (rows with no placement in the next 60 days), so zero is
    'no different from an uninjured outing/game'. Raw means are in the JSON."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = C.REPO / "figures"; figdir.mkdir(exist_ok=True)
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), dpi=300, sharey=True)
    panels = [("pitcher_all", "pitcher_shoulder", "Pitchers: four-seam velocity", "Outings before the placement"),
              ("batter_msk", "batter_oblique", "Batters: bat speed", "Games before the placement")]
    for ax, (a, b, title, xlab) in zip(axes, panels):
        ends = []
        for key, col in [(a, BLUE), (b, ORANGE)]:
            t = out["trajectories"][key]; ref = t["reference"]["mean"]
            k = np.array(t["k"]); m = np.array(t["mean"]) - ref; lo = np.array(t["lo"]) - ref; hi = np.array(t["hi"]) - ref
            ax.fill_between(k, lo, hi, color=col, alpha=0.15, linewidth=0)
            ax.plot(k, m, color=col, linewidth=2, marker="o", markersize=3.5, markeredgecolor="white", markeredgewidth=0.8)
            short = {"pitcher_all": "any IL", "pitcher_shoulder": "shoulder", "batter_msk": "any IL", "batter_oblique": "oblique"}[key]
            ends.append((m[0], f"{short} (n = {t['n'][0]:,})"))
        # direct labels at the placement end, nudged apart if they would overlap
        (y0, l0), (y1, l1) = ends
        if abs(y0 - y1) < 0.09:
            mid = (y0 + y1) / 2; y0, y1 = (mid + 0.045, mid - 0.045) if y0 >= y1 else (mid - 0.045, mid + 0.045)
        ax.annotate(l0, (0, y0), xytext=(5, 0), textcoords="offset points", color=INK, fontsize=7, va="center")
        ax.annotate(l1, (0, y1), xytext=(5, 0), textcoords="offset points", color=INK, fontsize=7, va="center")
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xlim(K + 0.4, -0.4)
        ax.set_xticks(range(0, K + 1, 2)); ax.set_xticklabels([str(i) if i else "last" for i in range(0, K + 1, 2)])
        ax.set_xlabel(xlab, color=INK); ax.set_title(title, loc="left", fontsize=9.5, color=INK, fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
    axes[0].set_ylabel("Deviation from own baseline,\nrelative to uninjured outings (SD)", color=INK, fontsize=8)
    axes[0].set_ylim(-0.85, 0.35)
    fig.subplots_adjust(left=0.11, right=0.86, bottom=0.17, top=0.88, wspace=0.45)
    fig.savefig(figdir / "Figure1_trajectory.png", dpi=300)
    fig.savefig(figdir / "Figure1_trajectory.pdf")
    print("wrote", figdir / "Figure1_trajectory.png")


if __name__ == "__main__":
    import sys
    if "--figure-only" in sys.argv:
        # redraw Figure 1 from the saved descriptive JSON (no data, no model); for regenerating the PNG on another machine
        figure(json.load(open(C.OUT / "descriptive_trajectory.json")))
    else:
        main()
