"""Reconciliation R1 (twin prereg §8): the batter-arm oblique estimate vs the 2026-09-30 exploratory
cold-swing OR, on one set of hitter-games under one label rule.

Frames
  S  the exploratory split's decision points: hitter-game, recent 50 PA vs regressed prior history,
     bat speed park- and week-adjusted (hot_streak_study.py, 2024+ data), season >= 300 PA,
     no IL return in the prior 45 days, 2026 points only when the 30-day window closes by 2026-07-21.
  W  the batter arm's eligible batter-games (>= 10 prior tracked games, >= 5 tracked swings), 2024+,
     same 2026 rule.
  I  = S ∩ W on (batter, game_date).
Label (one rule for every row): oblique placement with start in (g, g + 30].

Reported for each label set: (a) ΔAUC of the batter-arm model vs its workload null evaluated on I
(OOF predictions from the full batter-arm frame); (b) logistic OR per −1 SD of z_bs_dev on I;
(c) logistic OR per −1 SD of the split feature on I; (d) (b) on all of W and (c) on all of S;
(e) the split's own label rule on S (first placement within [d, d + 30], family = oblique).

Usage: python3 -m twin.r1 public|strict|CSV_PATH
  CSV_PATH: any case list with columns mlbam, season, start, end, family (family 'oblique' is used).
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from twin import common as C

sys.path.insert(0, str(C.REPO / "twin"))
import hot_streak_study as H  # noqa: E402  (vendored copy of the hot-hand repo module)

LABEL_END = pd.Timestamp("2026-07-21")
MODE = sys.argv[1] if len(sys.argv) > 1 else "public"


def case_list():
    if MODE in ("public", "strict"):
        p = pd.read_csv(C.OUT / "placements_public.csv", parse_dates=["il_start", "il_end"])
        p = p[p.msk].rename(columns={"il_start": "start", "il_end": "end"})
        dx = p.dx.fillna("").str.lower()
        ob = (p.site == "core_oblique") if MODE == "public" else ((p.site == "core_oblique") & dx.str.contains(r"obliqu|oliqu|intercostal"))
        p["family"] = np.where(ob, "oblique", p.site)
        return p[["mlbam", "season", "start", "end", "family"]]
    c = pd.read_csv(MODE, parse_dates=["start", "end"])
    return c[["mlbam", "season", "start", "end", "family"]]


ALL = case_list()
ALL = ALL[ALL.start <= LABEL_END].copy()
ALL["mlbam"] = ALL.mlbam.astype(int)
# labels for the batter-arm frame use every season (2023 rows train the 2024 fold)
OB = {p: np.sort(g.start.values) for p, g in ALL[ALL.family == "oblique"].groupby("mlbam")}
# the split's own frame used 2024+ stints only (recent-return filter and its own label rule)
CS = ALL[ALL.season >= 2024].copy()


def split_frame():
    import pyarrow.parquet as pq
    d = pd.concat([pq.read_table(C.DATA / f"statcast_season/statcast_{y}.parquet").to_pandas() for y in (2024, 2025, 2026)])
    d = d[d.game_type == "R"].copy()
    for c in ["batter", "pitcher", "game_pk", "game_year"]:
        d[c] = d[c].astype("int64")
    d["game_date"] = pd.to_datetime(d.game_date)
    d = d.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
    d = d.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    wk = d.game_date.dt.isocalendar().week.astype(int)
    for c in ["bat_speed", "swing_length", "launch_speed"]:
        x = pd.to_numeric(d[c], errors="coerce")
        ymean = x.groupby(d.game_year).transform("mean")
        park = x.groupby([d.home_team, d.game_year]).transform("mean") - ymean
        week = x.groupby([d.game_year, wk]).transform("mean") - ymean
        d[c] = x - park - week
    d = H.pitch_features(d)
    d = H.context(d)
    bg = H.batter_games(d)
    lg = H.league_means(bg)
    W = H.build_windows(bg, 50, 50, lg)
    W = W[W.season_pa >= 300].copy()
    W["game_date"] = pd.to_datetime(W.game_date)
    W = W[(W.game_date + pd.Timedelta(days=30) <= LABEL_END) | (W.game_year < 2026)]
    ok = (W.rec_bat_speed__n >= 30) & (W.orc_bat_speed__n >= 90) & (W.fwd_bat_speed__n >= 30)
    D = W[ok].copy()
    # recent return (45 d) from the same case list's end dates
    ends = {b: np.sort(g.end.dropna().values) for b, g in CS.groupby("mlbam")}
    rr = np.zeros(len(D), bool)
    for i, (dt, b) in enumerate(zip(D.game_date.values, D.batter.values)):
        e = ends.get(b)
        if e is not None and ((e <= dt) & (e >= dt - np.timedelta64(45, "D"))).any():
            rr[i] = True
    D["bs_dev_prior"] = D.rec_bat_speed - D.prior_bat_speed
    D["z_split"] = D.bs_dev_prior / D.bs_dev_prior.std()   # standardised before the return filter, as in the split
    D = D[~rr].copy()
    # the split's own label rule: first placement (any family) within [d, d+30] is oblique
    byb = {b: g.sort_values("start") for b, g in CS.groupby("mlbam")}
    own = np.zeros(len(D), int)
    for i, (dt, b) in enumerate(zip(D.game_date, D.batter)):
        g = byb.get(b)
        if g is None:
            continue
        nxt = g[(g.start >= dt) & (g.start <= dt + pd.Timedelta(days=30))]
        if len(nxt) and nxt.family.iloc[0] == "oblique":
            own[i] = 1
    D["y_own_rule"] = own
    return D.drop_duplicates(["batter", "game_date"])[["batter", "game_year", "game_date", "z_split", "y_own_rule"]]


def logit_or(y, x, g):
    f = sm.Logit(y, sm.add_constant(x)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": g})
    b, se = float(np.asarray(f.params)[1]), float(np.asarray(f.bse)[1])
    return dict(OR_per_minus1SD=float(np.exp(-b)), lo=float(np.exp(-b - 1.96 * se)), hi=float(np.exp(-b + 1.96 * se)),
                p=float(np.asarray(f.pvalues)[1]), n=int(len(y)), events=int(y.sum()), se_log=se)


if __name__ == "__main__":
    C.assert_pins(only=["2023", "2024", "2025", "2026", "txns"])
    S = split_frame()
    Wf = pd.read_parquet(C.OUT / "batter_extended_public_features.parquet")
    Wf["game_date"] = pd.to_datetime(Wf.game_date)
    Wf = Wf[Wf.eligible10 & ((Wf.game_date + pd.Timedelta(days=30) <= LABEL_END) | (Wf.season < 2026))].reset_index(drop=True)
    y_full = C.label_within(Wf.batter.values, Wf.game_date.values, OB)
    NULLF = ["n_swings_g", "days_since_prev", "cum_swings_season", "games_so_far"]
    pa, folds = C.oof(Wf, NULLF + ["z_bs_dev", "streak_low", "var_dev"], y_full, 1)
    pb, _ = C.oof(Wf, NULLF, y_full, 1)
    Wf["pa"], Wf["pb"], Wf["y"] = pa, pb, y_full
    W24 = Wf[Wf.season >= 2024].drop_duplicates(["batter", "game_date"])
    I = W24.merge(S, on=["batter", "game_date"], how="inner")
    I = I[np.isfinite(I.pa) & np.isfinite(I.pb)].reset_index(drop=True)
    out = dict(mode=MODE if MODE in ("public", "strict") else "custom_case_list", folds=str(folds),
               n_S=int(len(S)), n_W2024=int(len(W24)), n_I=int(len(I)), events_I=int(I.y.sum()),
               clusters_I=int(I.batter.nunique()))
    yy, cl = I.y.values, I.batter.values
    lo, hi, p = C.boot_dauc(yy, I.pa.values, I.pb.values, cl)
    out["a_dauc_on_I"] = dict(dauc=C.auc(yy, I.pa.values) - C.auc(yy, I.pb.values), lo=lo, hi=hi, p=p)
    zI = (I.z_bs_dev / I.z_bs_dev.std()).values
    out["b_OR_zbsdev_on_I"] = logit_or(yy, zI, cl)
    out["c_OR_split_on_I"] = logit_or(yy, (I.z_split / I.z_split.std()).values, cl)
    Wv = W24[np.isfinite(W24.z_bs_dev)]
    out["d_OR_zbsdev_on_W"] = logit_or(Wv.y.values, (Wv.z_bs_dev / Wv.z_bs_dev.std()).values, Wv.batter.values)
    yS = C.label_within(S.batter.values, S.game_date.values, OB)
    out["d_OR_split_on_S"] = logit_or(yS, S.z_split.values, S.batter.values)
    out["e_OR_split_on_S_own_rule"] = logit_or(S.y_own_rule.values, S.z_split.values, S.batter.values)
    out["corr_features_on_I"] = float(np.corrcoef(zI, I.z_split.values)[0, 1])
    tag = MODE if MODE in ("public", "strict") else "custom"
    json.dump(out, open(C.OUT / f"r1_{tag}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
