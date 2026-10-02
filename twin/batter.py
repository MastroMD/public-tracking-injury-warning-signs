"""Batter arm of the public twin — replica of WS-010 (sealed prereg 881bb2f8e35038fd) with public labels.

Twin prereg §5. Feature construction, eligibility, OOF scheme, bootstrap and ablation are copied from
the sealed script; only the label source (public census, sites.py families) and the window option differ.

Usage: python3 -m twin.batter [primary|extended] [public|strict]
Writes OUT/batter_{window}_{labels}_{fdesc,fdisc}.csv and a JSON summary.
"""
import json
import sys

import numpy as np
import pandas as pd

from twin import common as C

WINDOW = sys.argv[1] if len(sys.argv) > 1 else "primary"
LABELS = sys.argv[2] if len(sys.argv) > 2 else "public"
if WINDOW == "primary":
    COVEND_LABELS, CUTOFF = pd.Timestamp("2026-06-30"), pd.Timestamp("2026-05-31")
    COVER = {2023: ("2023-07-14", "2023-11-04"), 2024: ("2024-04-03", "2024-10-31"),
             2025: ("2025-03-27", "2025-11-02"), 2026: ("2026-03-26", "2026-08-11")}
else:
    COVEND_LABELS, CUTOFF = pd.Timestamp("2026-09-29"), pd.Timestamp("2026-08-30")
    COVER = {2023: ("2023-07-14", "2023-11-04"), 2024: ("2024-04-03", "2024-10-31"),
             2025: ("2025-03-27", "2025-11-02"), 2026: ("2026-03-26", "2026-09-29")}
CONF = ["oblique", "hamstring", "hand_wrist"]
SWING_DESCS = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play",
               "hit_into_play_no_out", "hit_into_play_score", "foul_bunt", "missed_bunt",
               "bunt_foul_tip", "swinging_pitchout", "foul_pitchout"}
NULLF = ["n_swings_g", "days_since_prev", "cum_swings_season", "games_so_far"]
DEVF = ["z_bs_dev", "streak_low", "var_dev"]
TAG = f"batter_{WINDOW}_{LABELS}"


def cases():
    p = pd.read_csv(C.OUT / "placements_public.csv", parse_dates=["il_start"])
    p = p[p.msk & (p.il_start <= COVEND_LABELS)].copy()
    fam = pd.Series(pd.NA, index=p.index, dtype="object")
    if LABELS == "public":
        fam[p.site == "core_oblique"] = "oblique"
        fam[p.site == "hamstring_quad"] = "hamstring"
        fam[p.site == "hand_wrist_finger"] = "hand_wrist"
    else:  # declared strict-text sensitivity
        dx = p.dx.fillna("").str.lower()
        fam[(p.site == "core_oblique") & dx.str.contains(r"obliqu|oliqu|intercostal")] = "oblique"
        fam[dx.str.contains("hamstring")] = "hamstring"
        fam[p.site == "hand_wrist_finger"] = "hand_wrist"
    p["family"] = fam
    p = p.rename(columns={"il_start": "start"})
    return p[["mlbam", "season", "start", "family", "site"]]


CASES = cases()


def build_features():
    import pyarrow.parquet as pq
    cols = ["game_date", "game_pk", "game_type", "batter", "description", "bat_speed"]
    S = pd.concat([pq.read_table(C.DATA / f"statcast_season/statcast_{y}.parquet", columns=cols).to_pandas()
                   for y in (2023, 2024, 2025, 2026)])
    S = S[(S.game_type == "R") & S.description.isin(SWING_DESCS)].copy()
    S["game_date"] = pd.to_datetime(S.game_date)
    S["season"] = S.game_date.dt.year
    keep = np.zeros(len(S), bool)
    for yr, (a, b) in COVER.items():
        keep |= ((S.season == yr) & (S.game_date >= a) & (S.game_date <= b)).values
    S = S[keep]
    S["is_bunt"] = S.description.str.contains("bunt", na=False)
    tracked = S[S.bat_speed.notna() & ~S.is_bunt]
    n_all = S.groupby(["batter", "season", "game_pk", "game_date"]).size().rename("n_swings_g")
    g = (tracked.groupby(["batter", "season", "game_pk", "game_date"])
         .agg(g_bs_mean=("bat_speed", "mean"), n_tracked=("bat_speed", "size"))
         .join(n_all, how="right").reset_index())
    g["n_tracked"] = g.n_tracked.fillna(0)
    g = g.sort_values(["batter", "season", "game_date", "game_pk"]).reset_index(drop=True)
    funnel = dict(batter_games=len(g), ge5_tracked=0, ge10_prior=0, ge5_prior=0)
    rows = []
    for (b_, s_), grp in g.groupby(["batter", "season"]):
        grp = grp.reset_index(drop=True)
        gm = grp.g_bs_mean.values; nt = grp.n_tracked.values; dts = grp.game_date.values
        cum_sw = np.concatenate([[0], np.cumsum(grp.n_swings_g.values)])[:-1]
        pm, low_hist = [], []
        for i in range(len(grp)):
            n_prior = len(pm)
            base = np.array(pm[-25:], float)
            ok5 = nt[i] >= 5
            funnel["ge5_tracked"] += int(ok5)
            funnel["ge5_prior"] += int(n_prior >= 5 and ok5)
            feat = dict(batter=b_, season=s_, game_pk=grp.game_pk[i], game_date=dts[i],
                        n_swings_g=grp.n_swings_g[i],
                        days_since_prev=(dts[i] - dts[i - 1]) / np.timedelta64(1, "D") if i else np.nan,
                        cum_swings_season=cum_sw[i], games_so_far=i, n_prior_tracked=n_prior, ok5=ok5,
                        g_bs_mean=gm[i])
            if n_prior >= 10 and ok5 and len(base) >= 2 and np.isfinite(gm[i]):
                mu, sd = base.mean(), base.std(ddof=1)
                z = (gm[i] - mu) / sd if sd > 0 else np.nan
                low = 1 if (np.isfinite(gm[i]) and gm[i] < mu - 0.5 * sd and sd > 0) else 0
                streak = 0
                if low:
                    streak = 1
                    for lh in reversed(low_hist):
                        if lh:
                            streak += 1
                        else:
                            break
                if n_prior >= 20:
                    last10 = np.array((pm + [gm[i]])[-10:], float)
                    prev10 = np.array(pm[-19:-9], float)
                    var_dev = last10.std(ddof=1) - prev10.std(ddof=1)
                else:
                    var_dev = np.nan
                feat.update(z_bs_dev=z, streak_low=streak, var_dev=var_dev, eligible10=True)
                funnel["ge10_prior"] += 1
            else:
                feat.update(z_bs_dev=np.nan, streak_low=np.nan, var_dev=np.nan, eligible10=False)
            rows.append(feat)
            if np.isfinite(gm[i]) and nt[i] >= 1:
                if n_prior >= 10 and len(base) >= 2 and base.std(ddof=1) > 0:
                    low_hist.append(1 if gm[i] < base.mean() - 0.5 * base.std(ddof=1) else 0)
                else:
                    low_hist.append(0)
                pm.append(gm[i])
    X = pd.DataFrame(rows)
    stop = CASES.groupby(["mlbam", "season"]).start.min()
    stopd = pd.Series([stop.get(k, pd.NaT) for k in zip(X.batter, X.season)], index=X.index, dtype="datetime64[ns]")
    post = (stopd.notna() & (X.game_date >= stopd)).values
    funnel["w11_dropped"] = int(post.sum())
    X = X[~post].reset_index(drop=True)
    X = X[X.game_date <= CUTOFF].reset_index(drop=True)
    funnel["rows_after_cutoff"] = len(X)
    X.to_parquet(C.OUT / f"{TAG}_features.parquet", index=False)
    return funnel


def load_X():
    X = pd.read_parquet(C.OUT / f"{TAG}_features.parquet")
    X["game_date"] = pd.to_datetime(X.game_date)
    X = X[X.eligible10].reset_index(drop=True)
    assert X.game_date.max() <= CUTOFF, "§9d fail"
    return X


def starts(fam):
    cc = CASES[CASES.family == fam]
    return {p: np.sort(g.start.values) for p, g in cc.groupby("mlbam")}


def contrast(name, X, y, mask=None):
    pa, fa = C.oof(X, NULLF + DEVF, y, 1, mask)
    pb, fb = C.oof(X, NULLF, y, 1, mask)
    ok = np.isfinite(pa) & np.isfinite(pb)
    if mask is not None:
        ok &= mask
    yy, cl = y[ok], X.batter.values[ok]
    aa, ab = C.auc(yy, pa[ok]), C.auc(yy, pb[ok])
    lo, hi, p = C.boot_dauc(yy, pa[ok], pb[ok], cl)
    return dict(contrast=name, n=int(ok.sum()), n_pos=int(yy.sum()), n_clusters=int(pd.unique(cl).size),
                n_pos_clusters=int(pd.unique(cl[yy == 1]).size), auc_full=aa, auc_null=ab,
                dauc=aa - ab, ci_lo=lo, ci_hi=hi, p=p, folds=str(fa))


def run_fdisc(X):
    res = []
    for fam in CONF:
        st = starts(fam)
        y = C.label_within(X.batter.values, X.game_date.values, st)
        ncl = int(pd.unique(X.batter.values[y == 1]).size)
        if ncl < 30:
            res.append(dict(contrast=f"B_{fam}", n_pos_clusters=ncl, verdict="GATE-CLOSED"))
            continue
        res.append(contrast(f"B_{fam}", X, y))
        drop = C.index_mask(X.batter.values, X.game_date.values, st)
        r = contrast(f"B_{fam}_INDEX_ABLATED", X, y, mask=~drop)
        r["n_index_rows_removed"] = int(drop.sum())
        res.append(r)
        print(res[-2]["contrast"], round(res[-2]["dauc"], 4), res[-2]["p"], "| ablated", round(r["dauc"], 4), r["p"], flush=True)
    df = pd.DataFrame(res)
    conf = ~df.contrast.str.endswith("ABLATED") & df.get("p", pd.Series(dtype=float)).notna()
    df.loc[conf, "q_bh"] = C.bh(df.loc[conf, "p"].values)
    df.to_csv(C.OUT / f"{TAG}_fdisc.csv", index=False)
    return df


def run_fdesc(X):
    r = np.random.default_rng(C.SEED)
    case_keys = set(zip(CASES.mlbam, CASES.season))
    clean = X[[k not in case_keys for k in zip(X.batter, X.season)]]
    res = []
    for fam in CONF:
        idx = C.index_mask(X.batter.values, X.game_date.values, starts(fam))
        cs = X[idx]
        months = cs.game_date.dt.month.value_counts(normalize=True)
        picks = []
        for (b_, s_), grp in clean.groupby(["batter", "season"]):
            gm = grp[grp.game_date.dt.month.isin(months.index)]
            pool = gm if len(gm) else grp
            w = pool.game_date.dt.month.map(months).fillna(months.min()).values
            picks.append(pool.iloc[r.choice(len(pool), p=w / w.sum())])
        ctrl = pd.DataFrame(picks)
        for met in ["z_bs_dev", "var_dev"]:
            a = cs[met].dropna().values; acl = cs.loc[cs[met].notna(), "batter"].values
            b_ = ctrl[met].dropna().values; bcl = ctrl.loc[ctrl[met].notna(), "batter"].values
            rr = np.random.default_rng(C.SEED); ds = []
            ua, ub = np.unique(acl), np.unique(bcl)
            ia = {u: np.where(acl == u)[0] for u in ua}; ib = {u: np.where(bcl == u)[0] for u in ub}
            for _ in range(C.B):
                sa = np.concatenate([ia[u] for u in rr.choice(ua, len(ua))])
                sb = np.concatenate([ib[u] for u in rr.choice(ub, len(ub))])
                ds.append(a[sa].mean() - b_[sb].mean())
            ds = np.array(ds)
            res.append(dict(contrast=f"FDESC_{fam}_{met}", n_case=len(a), n_case_clusters=len(ua),
                            n_ctrl=len(b_), n_ctrl_clusters=len(ub), diff=float(a.mean() - b_.mean()),
                            ci_lo=float(np.quantile(ds, .025)), ci_hi=float(np.quantile(ds, .975)),
                            p=max(float(2 * min((ds <= 0).mean(), (ds >= 0).mean())), 1 / C.B)))
    df = pd.DataFrame(res)
    df["q_bh"] = C.bh(df.p.values)
    df.to_csv(C.OUT / f"{TAG}_fdesc.csv", index=False)
    print(df[["contrast", "diff", "p"]].to_string(index=False))
    return df


if __name__ == "__main__":
    C.assert_pins(only=["2023", "2024", "2025", "2026", "txns"])
    funnel = build_features()
    X = load_X()
    funnel.update(eligible_rows=len(X), batters=int(X.batter.nunique()),
                  by_season=X.groupby("season").size().to_dict())
    print(funnel, flush=True)
    fdesc = run_fdesc(X)
    fdisc = run_fdisc(X)
    json.dump(dict(window=WINDOW, labels=LABELS, funnel=funnel, cases_by_family=CASES.family.value_counts().to_dict()),
              open(C.OUT / f"{TAG}_summary.json", "w"), indent=1, default=str)
