"""F2 — "what it would take" (PREREGISTRATION_FRAMING_2026-09-30.md §F2).

(a) Minimum detectable ΔAUC (80% power, two-sided α = .05) for every contrast in an arms CSV
    (columns: arm, dauc, ci_lo, ci_hi): MDE = 2.80 × (ci_hi − ci_lo) / 3.92. For the batter oblique arm,
    MDE as an odds ratio per −1 SD from the R1 logistic CI.
(b) Decision threshold for resting a flagged player: PPV* = r / (e × D) on the declared grid, converted to
    the positive likelihood ratio needed at base rate π and to the AUC an equal-variance binormal score
    would need to reach it at a 10% alert fraction; compared with what was observed.

Usage: python3 -m twin.framing_f2 ARMS_CSV
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from twin import common as C

arms = pd.read_csv(sys.argv[1])
arms["se"] = (arms.ci_hi - arms.ci_lo) / 3.92
arms["mde_dauc"] = 2.80 * arms.se
out = {"a_mde": arms[["arm", "dauc", "ci_lo", "ci_hi", "mde_dauc"]].to_dict(orient="records")}
try:
    r1 = json.load(open(C.OUT / "r1_public.json"))
    se = r1["b_OR_zbsdev_on_I"]["se_log"]
    out["a_mde_or_oblique_per_sd"] = dict(se_log=se, mde_or=float(np.exp(2.80 * se)))
except FileNotFoundError:
    pass

# ---------------------------------------------------------------- (b) decision threshold
P = pd.read_csv(C.OUT / "placements_public.csv", parse_dates=["il_start", "il_end"])
P = P[~P.covid_era_blank & P.il_end.notna()].copy()
P["days"] = (P.il_end - P.il_start).dt.days
pit = set(pd.read_parquet(C.OUT / "pitcher_primary_post_features.parquet", columns=["player_id"]).player_id)
bat = set(pd.read_parquet(C.OUT / "batter_primary_public_features.parquet", columns=["batter"]).batter)
D = {"pitcher_all": float(P[P.mlbam.isin(pit)].days.median()),
     "pitcher_shoulder": float(P[P.mlbam.isin(pit) & (P.site == "shoulder")].days.median()),
     "batter_oblique": float(P[P.mlbam.isin(bat) & ~P.mlbam.isin(pit) & (P.site == "core_oblique")].days.median())}
pr = pd.read_csv(C.OUT / "pitcher_primary_results.csv").set_index("test")
br = pd.read_csv(C.OUT / "batter_primary_public_fdisc.csv").set_index("contrast")
ps = json.load(open(C.OUT / "pitcher_primary_summary.json"))
obs = {"pitcher_all": dict(pi=ps["base_rates"]["all"], auc=float(pr.loc["P2", "auc_a"]), auc_null=float(pr.loc["P2", "auc_b"]),
                           auc_ablated=float(pr.loc["P2_ablated", "auc_a"]), auc_null_ablated=float(pr.loc["P2_ablated", "auc_b"]), label="W0+M3 (P2)"),
       "pitcher_shoulder": dict(pi=ps["base_rates"]["shoulder"], auc=float(pr.loc["P3", "auc_a"]), auc_null=float(pr.loc["P3", "auc_b"]), label="W0 (P3)"),
       "batter_oblique": dict(pi=float(br.loc["B_oblique", "n_pos"] / br.loc["B_oblique", "n"]),
                              auc=float(br.loc["B_oblique", "auc_full"]), auc_null=float(br.loc["B_oblique", "auc_null"]),
                              label="bat-speed deviation (B)")}
Z90 = norm.ppf(0.90)


def lr_at_10(auc):
    d = np.sqrt(2) * norm.ppf(auc)
    return float((1 - norm.cdf(Z90 - d)) / 0.10)


def auc_for_lr(lr):
    tpr = lr * 0.10
    if tpr >= 1:
        return float("nan")
    d = Z90 - norm.ppf(1 - tpr)
    return float(norm.cdf(d / np.sqrt(2)))


grid = []
for lab, Dm in D.items():
    pi = obs[lab]["pi"]
    for r in (5, 15):
        for e in (0.25, 0.5, 1.0):
            ppv = r / (e * Dm)
            if ppv >= 1:
                grid.append(dict(label=lab, r=r, e=e, D=Dm, ppv_star=ppv, lr_star=None, auc_star=None, note="unattainable (PPV* ≥ 1)"))
                continue
            lr = (ppv / (1 - ppv)) / (pi / (1 - pi))
            grid.append(dict(label=lab, r=r, e=e, D=Dm, pi=pi, ppv_star=ppv, lr_star=lr, auc_star=auc_for_lr(lr)))
out["b_grid"] = grid
out["b_observed"] = {k: dict(v, lr_at_10pct_binormal=lr_at_10(v["auc"]), lr_null_at_10pct_binormal=lr_at_10(v["auc_null"]),
                             ppv_at_10pct=float(lr_at_10(v["auc"]) * v["pi"] / (lr_at_10(v["auc"]) * v["pi"] + (1 - v["pi"]))))
                     for k, v in obs.items()}
out["b_sealed_top10_ppv_lift_pitcher_all"] = 1.27
met = [dict(g, met_by_workload_null_alone=bool(out["b_observed"][g["label"]]["lr_null_at_10pct_binormal"] >= g["lr_star"]))
       for g in grid if g.get("lr_star") and out["b_observed"][g["label"]]["lr_at_10pct_binormal"] >= g["lr_star"]]
out["b_any_cell_met"] = bool(met)
out["b_cells_met"] = met
out["median_il_days"] = D
json.dump(out, open(C.OUT / "f2_results.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "a_mde"}, indent=1, default=float))
print(pd.DataFrame(out["a_mde"]).round(4).to_string(index=False))
