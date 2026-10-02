"""F1 — WAR lost to the injured list per team-season from the public transaction feed
(PREREGISTRATION_FRAMING_2026-09-30.md §F1).

Episodes from OUT/placements_public.csv (twin census). fWAR from FanGraphs season exports, schedule and
standings from the MLB Stats API. Inputs other than the pinned feed/Statcast files are hashed at load and
the hashes written to the output (they were not pinned in the framing prereg; see DEVIATIONS.md #2).

Usage: python3 -m twin.framing_f1 FG_WAR_CSV GAMES_FLAT_CSV STANDINGS_CSV [THIRD_PARTY_TEAM_SEASON_CSV]
"""
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from twin import common as C

FG_CSV, GAMES, STAND = sys.argv[1:4]
TP = sys.argv[4] if len(sys.argv) > 4 else None
OUTCOME = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
W = (5, 4, 3)
K = 100

inputs = {p: C.sha16(p) for p in [FG_CSV, GAMES, STAND] + ([TP] if TP else [])}

G = pd.read_csv(GAMES)
G = G[(G.gameType == "R") & G.status.isin(["Final", "Completed Early"])].copy()
G["date"] = pd.to_datetime(G.date)
tg = pd.concat([G[["season", "date", "home_id", "home"]].rename(columns={"home_id": "tid", "home": "team"}),
                G[["season", "date", "away_id", "away"]].rename(columns={"away_id": "tid", "away": "team"})])
TEAM_DATES = {k: np.sort(v.date.to_numpy()) for k, v in tg.groupby(["season", "tid"])}
SPAN = {s: (g.date.min(), g.date.max()) for s, g in G.groupby("season")}
NAME = tg.sort_values("season").groupby("tid").team.last().to_dict()


def games_between(season, tid, a, b):
    d = TEAM_DATES.get((season, tid))
    if d is None:
        return np.nan
    return int(np.searchsorted(d, np.datetime64(b), side="right") - np.searchsorted(d, np.datetime64(a), side="left"))


# ---------------------------------------------------------------- episodes
P = pd.read_csv(C.OUT / "placements_public.csv", parse_dates=["il_start", "il_end"])
P = P[~P.covid_era_blank].copy()
P["club_id"] = P.club_id.astype(int)
rows = []
for s in range(2012, 2026):
    if s not in SPAN:
        continue
    a, b = SPAN[s]
    x = P[(P.il_start <= b) & (P.il_end.fillna(b) >= a) & P.il_start.dt.year.between(s - 1, s)].copy()
    x["start"] = x.il_start.clip(lower=a)
    x["end"] = (x.il_end.fillna(b + pd.Timedelta(days=1)) - pd.Timedelta(days=1)).clip(upper=b)  # [start, end)
    x = x[x.end >= x.start].sort_values(["mlbam", "club_id", "start"])
    for (pid, tid), g in x.groupby(["mlbam", "club_id"]):
        cs = ce = None; site = None
        for st, en, si in zip(g.start, g.end, g.site):
            if cs is None:
                cs, ce, site = st, en, si
            elif st <= ce + pd.Timedelta(days=1):
                ce = max(ce, en)
            else:
                rows.append((s, pid, tid, cs, ce, site)); cs, ce, site = st, en, si
        rows.append((s, pid, tid, cs, ce, site))
S = pd.DataFrame(rows, columns=["season", "mlbid", "tid", "start", "end", "site"])
S["days"] = (S.end - S.start).dt.days + 1
S["games_missed"] = [games_between(s, t, a, b) for s, t, a, b in zip(S.season, S.tid, S.start, S.end)]

# ---------------------------------------------------------------- WAR and projections
fg = pd.read_csv(FG_CSV).rename(columns={"mlbamid": "player_id"})
fg = fg.dropna(subset=["player_id"]); fg["player_id"] = fg.player_id.astype(int)
war = fg.groupby(["player_id", "season"]).agg(war=("war", "sum"), g_played=("g", "max")).reset_index()
gm = S.groupby(["mlbid", "season"]).games_missed.sum().rename("gm").reset_index().rename(columns={"mlbid": "player_id"})
war = war.merge(gm, on=["player_id", "season"], how="left").fillna({"gm": 0})
war["exposure"] = (np.where(war.season == 2020, 60, 162) - war.gm).clip(lower=1)
Wd = {(p, s): (w, e) for p, s, w, e in zip(war.player_id, war.season, war.war, war.exposure)}
Pd = {(p, s): (w, g) for p, s, w, g in zip(war.player_id, war.season, war.war, war.g_played)}


def rate(pid, s):
    num = den = 0.0
    for w, k in zip(W, (1, 2, 3)):
        v = Wd.get((pid, s - k))
        if v is not None:
            num += w * v[0]; den += w * v[1]
    return num / (den + K)


def rate_zeros(pid, s):  # lower bracket: Marcel with missing seasons = 0, no exposure denominator
    num = sum(w * Wd.get((pid, s - k), (0.0, 0))[0] for w, k in zip(W, (1, 2, 3)))
    return num / (sum(W) * 162)


def pace(pid, s):
    v = Pd.get((pid, s))
    if v is None or not np.isfinite(v[1]) or v[1] < 10:
        return np.nan
    return v[0] / v[1]


S = S[S.season.isin(OUTCOME)].copy()
for tag, f in [("", rate), ("_zeros", rate_zeros), ("_pace", pace)]:
    S["war_lost" + tag] = np.clip([f(p, s) for p, s in zip(S.mlbid, S.season)], 0, None) * S.games_missed
T = S.groupby(["season", "tid"]).agg(stints=("mlbid", "size"), games_missed=("games_missed", "sum"),
                                     war_lost=("war_lost", "sum"), war_lost_zeros=("war_lost_zeros", "sum"),
                                     war_lost_pace=("war_lost_pace", "sum")).reset_index()
st = pd.read_csv(STAND)[["season", "team_id", "W", "L"]].rename(columns={"team_id": "tid"})
T = T.merge(st, on=["season", "tid"], how="left")
n_unmatched = int(T.W.isna().sum())
T = T.dropna(subset=["W"])
# roster projection (preseason) — FanGraphs team abbreviations to club ids via the schedule's names
ABBR = {"ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox",
        "CHC": "Chicago Cubs", "CHW": "Chicago White Sox", "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
        "COL": "Colorado Rockies", "DET": "Detroit Tigers", "HOU": "Houston Astros", "KCR": "Kansas City Royals",
        "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers",
        "MIN": "Minnesota Twins", "NYM": "New York Mets", "NYY": "New York Yankees", "OAK": "Athletics", "ATH": "Athletics",
        "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates", "SDP": "San Diego Padres", "SEA": "Seattle Mariners",
        "SFG": "San Francisco Giants", "STL": "St. Louis Cardinals", "TBR": "Tampa Bay Rays", "TEX": "Texas Rangers",
        "TOR": "Toronto Blue Jays", "WSN": "Washington Nationals", "FLA": "Miami Marlins"}
ID_OF = {v: k for k, v in NAME.items()}
fgp = fg[fg.season.isin(OUTCOME)].copy()
fgp["tid"] = fgp.team.map(ABBR).map(ID_OF)
fgp = fgp.dropna(subset=["tid"]).drop_duplicates(["player_id", "season", "tid"])
fgp["proj_war"] = [rate(p, s) * 162 for p, s in zip(fgp.player_id, fgp.season)]
rost = fgp.groupby(["season", "tid"]).proj_war.sum().rename("team_proj").reset_index()
T = T.merge(rost, on=["season", "tid"], how="left")
T = T.sort_values(["tid", "season"]).reset_index(drop=True)
T["war_lost_prev"] = T.groupby("tid").war_lost.shift(1)
T["W_prev"] = T.groupby("tid").W.shift(1)


def dist(col):
    g = T.groupby("season")[col]
    return dict(mean=float(T[col].mean()), sd_within_season=float(g.std().mean()),
                p10=float(T[col].quantile(.1)), p90=float(T[col].quantile(.9)), max=float(T[col].max()))


R = dict(inputs_sha16=inputs, n_team_seasons=int(len(T)), n_unmatched_dropped=n_unmatched, n_episodes=int(len(S)),
         war_lost=dist("war_lost"), war_lost_zeros=dist("war_lost_zeros"), war_lost_pace=dist("war_lost_pace"))
# year-to-year R^2 (S on S-1, pooled; consecutive outcome seasons only, 2021 follows 2019)
Y = T.dropna(subset=["war_lost_prev"])
f = sm.OLS(Y.war_lost, sm.add_constant(Y.war_lost_prev)).fit()
R["yoy_r2"] = float(f.rsquared); R["yoy_slope"] = float(f.params.iloc[1]); R["yoy_n"] = int(len(Y))
# wins per WAR lost
M = T.dropna(subset=["W_prev", "team_proj"]).copy()
Xm = pd.concat([M[["war_lost", "team_proj", "W_prev"]], pd.get_dummies(M.season, prefix="s", drop_first=True, dtype=float)], axis=1)
fm = sm.OLS(M.W, sm.add_constant(Xm)).fit(cov_type="cluster", cov_kwds={"groups": M.tid})
R["wins_per_war_lost"] = dict(est=float(fm.params["war_lost"]), se=float(fm.bse["war_lost"]),
                              lo=float(fm.conf_int().loc["war_lost", 0]), hi=float(fm.conf_int().loc["war_lost", 1]),
                              n=int(len(M)), r2=float(fm.rsquared))
R["wins_one_sd"] = float(-R["wins_per_war_lost"]["est"] * R["war_lost"]["sd_within_season"])
if TP:
    tp = pd.read_csv(TP)[["season", "team", "stints"]].rename(columns={"stints": "tp_stints"})
    tp["tid"] = tp.team.map(ID_OF)
    J = T.merge(tp.dropna(subset=["tid"]), on=["season", "tid"], how="inner")
    R["agreement_r_stint_counts_vs_third_party"] = float(np.corrcoef(J.stints, J.tp_stints)[0, 1])
    R["agreement_n"] = int(len(J))
# medians for F2 (public episodes, days on IL)
P2 = P.dropna(subset=["il_end"]).copy()
P2["days"] = (P2.il_end - P2.il_start).dt.days
R["median_il_days"] = dict(all=float(P2.days.median()), shoulder=float(P2[P2.site == "shoulder"].days.median()),
                          core_oblique=float(P2[P2.site == "core_oblique"].days.median()))
T.to_csv(C.OUT / "f1_team_season.csv", index=False)
json.dump(R, open(C.OUT / "f1_results.json", "w"), indent=1)
print(json.dumps(R, indent=1))
