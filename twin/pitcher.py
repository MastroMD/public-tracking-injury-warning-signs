"""Pitcher arm of the public twin — replicas of WS-002 (D-LEAN), WS-003 (shoulder), WS-008 (composite).

Twin prereg §6. Outing panel from public Statcast pitch-level data (four-seam means per pitcher-game,
matching the sealed Savant name-date aggregation); features copied from the sealed builders
(within_season_002_fdisc.py for W0/ladder, within_season_004_build_m2345.py for M3); labels from the
public census. Confirmatory family P = {P1, P2, P3, P4}, BH across the four; P3-ablated is twin-new.

Usage: python3 -m twin.pitcher [primary|extended]
"""
import json
import sys

import numpy as np
import pandas as pd

from twin import common as C

WINDOW = sys.argv[1] if len(sys.argv) > 1 else "primary"
CUTOFF = pd.Timestamp("2026-07-12") if WINDOW == "primary" else pd.Timestamp("2026-08-30")
METRICS = ["velocity", "spin_rate", "release_extension", "arm_angle", "abs_release_pos_x", "release_pos_z"]
COV = ["cov_pitches_outing", "cov_days_rest", "cov_cum_pitches", "cov_n_out_so_far"]
ELBOW = {"elbow_other", "forearm_flexor", "ulnar_nerve", "ucl_graft"}
TAG = f"pitcher_{WINDOW}"


def outing_panel():
    import pyarrow.parquet as pq
    cols = ["game_date", "game_pk", "game_type", "pitcher", "pitch_type", "release_speed", "release_spin_rate",
            "release_extension", "arm_angle", "release_pos_x", "release_pos_z"]
    parts = []
    for y in range(2015, 2027):
        t = pq.read_table(C.DATA / f"statcast_season/statcast_{y}.parquet", columns=cols).to_pandas()
        t = t[t.game_type == "R"]
        tot = t.groupby(["pitcher", "game_pk", "game_date"]).size().rename("total_pitches")
        ff = t[t.pitch_type == "FF"]
        a = ff.groupby(["pitcher", "game_pk", "game_date"]).agg(
            pitches=("release_speed", "size"), velocity=("release_speed", "mean"),
            spin_rate=("release_spin_rate", "mean"), release_extension=("release_extension", "mean"),
            arm_angle=("arm_angle", "mean"), release_pos_x=("release_pos_x", "mean"),
            release_pos_z=("release_pos_z", "mean"))
        a = a.join(tot, how="left").reset_index()
        a["season"] = y
        parts.append(a)
    o = pd.concat(parts, ignore_index=True).rename(columns={"pitcher": "player_id"})
    o = o[(o.total_pitches >= 20) & (o.pitches >= 5)].copy()
    o["abs_release_pos_x"] = o.release_pos_x.abs()
    o["ff_share"] = o.pitches / o.total_pitches
    o["game_date"] = pd.to_datetime(o.game_date)
    return o.sort_values(["player_id", "season", "game_date", "game_pk"]).reset_index(drop=True)


def slope(y):
    n = len(y)
    if n < 2 or not np.isfinite(y).all():
        return np.nan
    return float(np.polyfit(np.arange(n), y, 1)[0])


def _z(v, base):
    bm, bs = np.nanmean(base), np.nanstd(base, ddof=1)
    return (v - bm) / bs if np.isfinite(v) and bs and bs > 0 else np.nan


def build(o, timing):
    prev = {}
    for (pid, s), g in o.groupby(["player_id", "season"]):
        if len(g) >= 6:
            d = {f"prev_mean_{m}": float(g[m].mean()) for m in METRICS}
            d.update({f"prev_dS1_{m}": float(g.tail(3)[m].mean() - g.head(3)[m].mean()) for m in METRICS})
            d["prev_n_out"] = float(len(g)); d["has_prev"] = 1.0
            prev[(pid, s)] = d
    rows = []
    for (pid, season), g in o.groupby(["player_id", "season"], sort=False):
        g = g.reset_index(drop=True); n = len(g)
        pf = prev.get((pid, season - 1), {"has_prev": 0.0})
        V = {m: g[m].values.astype(float) for m in METRICS + ["ff_share", "release_pos_x", "release_pos_z"]}
        dates = g.game_date.values; tp = g.total_pitches.values.astype(float)
        for i in range(5, n):
            w_end = i if timing == "post" else i - 1
            r = dict(player_id=pid, season=season, game_date=dates[i], outing_idx=i,
                     cov_pitches_outing=tp[i], cov_days_rest=float((dates[i] - dates[i - 1]) / np.timedelta64(1, "D")),
                     cov_cum_pitches=float(tp[:i].sum()), cov_n_out_so_far=float(i))
            for m in METRICS:
                v = V[m]; tb = v[:w_end]; w = v[:w_end + 1]
                bm = np.nanmean(tb)
                r[f"z_{m}"] = _z(v[w_end], tb)
                r[f"lvl_{m}"] = bm
                r[f"d3_{m}"] = (np.nanmean(w[-3:]) - bm) if len(w) >= 3 else np.nan
                r[f"slope5_{m}"] = slope(w[-5:])
            x, zz = V["release_pos_x"], V["release_pos_z"]
            r["release_drift"] = float(np.hypot(x[w_end] - np.nanmean(x[:w_end]), zz[w_end] - np.nanmean(zz[:w_end])))
            fs = V["ff_share"]; sd = np.nanstd(fs[:w_end], ddof=1)
            r["z_ff_share"] = (fs[w_end] - np.nanmean(fs[:w_end])) / (sd or np.nan)
            if timing == "post":  # M3 (WS-004 builder, POST only)
                vel = V["velocity"]; streak = 0
                for j in range(i, 4, -1):
                    zj = _z(vel[j], vel[:j])
                    if np.isfinite(zj) and zj <= -0.5:
                        streak += 1
                    else:
                        break
                gaps = np.diff(dates[:i + 1]) / np.timedelta64(1, "D")
                rest = float(gaps[-1]); med = float(np.median(gaps[:-1])) if len(gaps) > 1 else rest
                zv = _z(vel[i], vel[:i])
                r["m3_streak_low_velo"] = float(streak); r["m3_rest_dev"] = rest - med
                r["m3_z_velo_x_rest"] = zv * (rest - med) if np.isfinite(zv) else np.nan
            r.update(pf)
            rows.append(r)
    X = pd.DataFrame(rows)
    X["game_date"] = pd.to_datetime(X.game_date)
    return X


def label_starts():
    p = pd.read_csv(C.OUT / "placements_public.csv", parse_dates=["il_start"])
    allc = p[~p.covid_era_blank]
    mk = lambda d: {int(k): np.sort(g.il_start.values) for k, g in d.groupby("mlbam")}
    return dict(all=mk(allc), all_asis=mk(p), shoulder=mk(p[p.site == "shoulder"]), elbow=mk(p[p.site.isin(ELBOW)]))


def contrast(name, X, y, fa, fb, mask=None):
    pa, folds = C.oof(X, COV + fa, y, 2, mask)
    pb, _ = C.oof(X, COV + fb, y, 2, mask)
    ok = np.isfinite(pa) & np.isfinite(pb)
    if mask is not None:
        ok &= mask
    yy, cl = y[ok], X.player_id.values[ok]
    aa, ab = C.auc(yy, pa[ok]), C.auc(yy, pb[ok])
    lo, hi, p = C.boot_dauc(yy, pa[ok], pb[ok], cl)
    rec = dict(contrast=name, n=int(ok.sum()), n_pos=int(yy.sum()), n_clusters=int(pd.unique(cl).size),
               n_pos_clusters=int(pd.unique(cl[yy == 1]).size), auc_a=aa, auc_b=ab, dauc=aa - ab,
               ci_lo=lo, ci_hi=hi, p=p, folds=str(folds))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != "folds"}, flush=True)
    return rec


if __name__ == "__main__":
    C.assert_pins()
    o = outing_panel()
    print(f"outings {len(o)} pitchers {o.player_id.nunique()} arm_angle non-null by season "
          f"{o.groupby('season').arm_angle.apply(lambda s: round(s.notna().mean(), 2)).to_dict()}", flush=True)
    L = label_starts()
    res = []
    Xs = {}
    for timing in ("post", "pre"):
        X = build(o, timing)
        X = X[X.game_date <= CUTOFF].sort_values(["season", "game_date"]).reset_index(drop=True)
        X.to_parquet(C.OUT / f"{TAG}_{timing}_features.parquet", index=False)
        Xs[timing] = X
        print(f"[{timing}] scored-panel rows {len(X)} pitchers {X.player_id.nunique()}", flush=True)
    X = Xs["post"]
    Z = [c for c in X.columns if c.startswith("z_")]
    W0 = Z + ["release_drift"]
    LVL = [c for c in X.columns if c.startswith("lvl_")]
    D3 = [c for c in X.columns if c.startswith("d3_")]
    SL = [c for c in X.columns if c.startswith("slope5_")]
    PREV = [c for c in X.columns if c.startswith("prev_")] + ["has_prev"]
    M3 = ["m3_streak_low_velo", "m3_rest_dev", "m3_z_velo_x_rest"]
    pid, gd = X.player_id.values, X.game_date.values
    y_all = C.label_within(pid, gd, L["all"])
    y_sh = C.label_within(pid, gd, L["shoulder"])
    y_el = C.label_within(pid, gd, L["elbow"])
    y_asis = C.label_within(pid, gd, L["all_asis"])
    gate = {k: int(pd.unique(pid[y == 1]).size) for k, y in
            dict(all=y_all, shoulder=y_sh, elbow=y_el).items()}
    print("injured clusters on scored grid:", gate, flush=True)
    base = {"all": float(y_all.mean()), "shoulder": float(y_sh.mean()), "elbow": float(y_el.mean())}
    res.append(dict(test="P1", **contrast("P1_W0_vs_null_y30", X, y_all, W0, [])))
    res.append(dict(test="P2", **contrast("P2_W0M3_vs_null_y30", X, y_all, W0 + M3, [])))
    drop_all = C.index_mask(pid, gd, L["all"])
    res.append(dict(test="P2_ablated", n_index_removed=int(drop_all.sum()),
                    **contrast("P2_W0M3_vs_null_y30_INDEX_ABLATED", X, y_all, W0 + M3, [], mask=~drop_all)))
    res.append(dict(test="P3", **contrast("P3_W0_vs_null_y30_shoulder", X, y_sh, W0, [])))
    drop_sh = C.index_mask(pid, gd, L["shoulder"])
    res.append(dict(test="P3_ablated_TWIN_NEW", n_index_removed=int(drop_sh.sum()),
                    **contrast("P3_W0_vs_null_y30_shoulder_INDEX_ABLATED", X, y_sh, W0, [], mask=~drop_sh)))
    res.append(dict(test="P4", **contrast("P4_ladder_vs_null_y30", X, y_all, Z + LVL + D3 + SL + ["release_drift"] + PREV, [])))
    # declared sensitivities
    Xp = Xs["pre"]
    y_pre = C.label_within(Xp.player_id.values, Xp.game_date.values, L["all"])
    res.append(dict(test="SENS_P1_PRE", **contrast("P1_PRE_W0_vs_null_y30", Xp, y_pre, W0, [])))
    res.append(dict(test="SENS_P3_elbow", **contrast("W0_vs_null_y30_elbow", X, y_el, W0, [])))
    res.append(dict(test="SENS_P1_covid_as_written", **contrast("P1_W0_vs_null_y30_asis", X, y_asis, W0, [])))
    df = pd.DataFrame(res)
    conf = df.test.isin(["P1", "P2", "P3", "P4"])
    df.loc[conf, "q_bh"] = C.bh(df.loc[conf, "p"].values)
    df.to_csv(C.OUT / f"{TAG}_results.csv", index=False)
    json.dump(dict(window=WINDOW, cutoff=str(CUTOFF.date()), gate=gate, base_rates=base,
                   n_outings_panel=int(len(o)), n_scored=int(len(X))),
              open(C.OUT / f"{TAG}_summary.json", "w"), indent=1)
    print(df.drop(columns=["folds"]).round(4).to_string(index=False))
