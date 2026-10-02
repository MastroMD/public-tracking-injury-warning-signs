"""Addendum A1 — pitcher command (OpenCommand inferred-target miss) as a within-season warning sign.

Preregistration: ../opencommand/PREREGISTRATION_ADDENDUM_OPENCOMMAND_2026-10-01.md (sha16 d211f7f5dd66d630),
sealed before any OpenCommand file was opened. Data: Kim T. OpenCommand v1.2.0 (open-command.com),
CC BY-NC-SA 4.0, per-pitch inferred targets 2024-2026, downloaded by !RUN_OPENCOMMAND_PULL.command.

Usage: TWIN_DATA=... TWIN_OUT=... OPENCOMMAND_DATA=<folder with 2024_targets.csv.gz ...> python3 -m twin.command
Writes OUT/command_results.csv, OUT/command_summary.json.
"""
import json
import os
import pathlib

import numpy as np
import pandas as pd

from twin import common as C

OC = pathlib.Path(os.environ.get("OPENCOMMAND_DATA", C.REPO.parent / "opencommand" / "data"))
PREREG_A1 = "d211f7f5dd66d630"
COV = ["cov_pitches_outing", "cov_days_rest", "cov_cum_pitches", "cov_n_out_so_far"]
CMD = ["z_cmd", "d3_cmd", "slope5_cmd"]
MIN_TRACKED, MIN_PRIOR = 10, 5


def assert_seal():
    f = pathlib.Path(os.environ.get("OPENCOMMAND_PREREG", C.REPO.parent / "opencommand" / "PREREGISTRATION_ADDENDUM_OPENCOMMAND_2026-10-01.md"))
    assert C.sha16(f) == PREREG_A1, "A1 prereg hash mismatch"
    sha = {}
    for line in (OC / "SHA16.txt").read_text().splitlines():
        h, name = line.split(None, 1)
        sha[name.strip().split("/")[-1]] = h
    for y in (2024, 2025, 2026):
        for f in (f"{y}_targets.csv.gz", f"{y}_pbp_info.csv.gz"):
            assert C.sha16(OC / f) == sha[f], f"{f} hash moved since download"
    return sha


def outing_command():
    parts = []
    for y in (2024, 2025, 2026):
        t = pd.read_csv(OC / f"{y}_targets.csv.gz")
        p = pd.read_csv(OC / f"{y}_pbp_info.csv.gz")
        t = t[t["plausible"].astype(bool)]
        p = p.rename(columns={"date": "game_date"})
        cols = [c for c in ["play_id", "pitcher_id", "game_date", "game_type"] if c in p.columns]
        t = t.merge(p[cols], on="play_id", how="inner", suffixes=("", "_p"))
        if "game_type" in t.columns:
            t = t[t.game_type == "R"]
        t["miss"] = np.hypot(t.plate_x_in - t.inferred_x_in, t.plate_z_in - t.inferred_z_in)
        t["game_date"] = pd.to_datetime(t.game_date)
        g = (t.groupby(["pitcher_id", "game_date"]).agg(cmd_med=("miss", "median"), n_cmd=("miss", "size"))
             .reset_index().rename(columns={"pitcher_id": "player_id"}))
        g["season"] = y
        parts.append(g)
    o = pd.concat(parts, ignore_index=True)
    return o[o.n_cmd >= MIN_TRACKED]


def add_features(X):
    X = X.sort_values(["player_id", "season", "game_date"]).reset_index(drop=True)
    z = np.full(len(X), np.nan); d3 = np.full(len(X), np.nan); sl = np.full(len(X), np.nan)
    for (pid, s), g in X.groupby(["player_id", "season"], sort=False):
        idx = g.index.values; v = g.cmd_med.values
        for j in range(len(idx)):
            prior = v[:j]; prior = prior[np.isfinite(prior)]
            if len(prior) < MIN_PRIOR or not np.isfinite(v[j]):
                continue
            mu, sd = prior.mean(), prior.std(ddof=1)
            if sd > 0:
                z[idx[j]] = (v[j] - mu) / sd
            w = v[:j + 1]; w = w[np.isfinite(w)]
            d3[idx[j]] = w[-3:].mean() - mu
            if len(w) >= 5:
                sl[idx[j]] = float(np.polyfit(np.arange(5), w[-5:], 1)[0])
    X["z_cmd"], X["d3_cmd"], X["slope5_cmd"] = z, d3, sl
    return X


def contrast(name, X, y, fa, fb, mask=None):
    pa, folds = C.oof(X, fa, y, 1, mask)
    pb, _ = C.oof(X, fb, y, 1, mask)
    ok = np.isfinite(pa) & np.isfinite(pb)
    if mask is not None:
        ok &= mask
    yy, cl = y[ok], X.player_id.values[ok]
    aa, ab = C.auc(yy, pa[ok]), C.auc(yy, pb[ok])
    lo, hi, p = C.boot_dauc(yy, pa[ok], pb[ok], cl)
    rec = dict(contrast=name, n=int(ok.sum()), n_pos=int(yy.sum()), n_clusters=int(pd.unique(cl).size),
               n_pos_clusters=int(pd.unique(cl[yy == 1]).size), auc_a=aa, auc_b=ab, dauc=aa - ab,
               ci_lo=lo, ci_hi=hi, p=p, folds=str(folds))
    print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in rec.items() if k != "folds"}, flush=True)
    return rec


def main():
    sha = assert_seal()
    cmd = outing_command()
    X = pd.read_parquet(C.OUT / "pitcher_primary_post_features.parquet")
    X["game_date"] = pd.to_datetime(X.game_date)
    X = X[X.season >= 2024].merge(cmd[["player_id", "game_date", "cmd_med", "n_cmd"]], on=["player_id", "game_date"], how="left")
    join_rate = float(X.cmd_med.notna().mean())
    X = add_features(X)
    X = X[np.isfinite(X.z_cmd)].reset_index(drop=True)
    P = pd.read_csv(C.OUT / "placements_public.csv", parse_dates=["il_start"])
    P = P[~P.covid_era_blank]
    mk = lambda d: {int(k): np.sort(g.il_start.values) for k, g in d.groupby("mlbam")}
    L_all, L_sh = mk(P), mk(P[P.site == "shoulder"])
    pid, gd = X.player_id.values, X.game_date.values
    y_all = C.label_within(pid, gd, L_all); y_sh = C.label_within(pid, gd, L_sh)
    gate = dict(all=int(pd.unique(pid[y_all == 1]).size), shoulder=int(pd.unique(pid[y_sh == 1]).size))
    print(f"rows {len(X)} pitchers {X.player_id.nunique()} join_rate {join_rate:.3f} gate {gate}", flush=True)
    W0 = [c for c in X.columns if c.startswith("z_") and c not in CMD] + ["release_drift"]
    res = []
    if gate["all"] >= 30:
        res.append(dict(test="A1.1", **contrast("A1.1_cmd_vs_null_y30", X, y_all, COV + CMD, COV)))
        res.append(dict(test="A1.2", **contrast("A1.2_W0cmd_vs_W0_y30", X, y_all, COV + W0 + CMD, COV + W0)))
        drop = C.index_mask(pid, gd, L_all)
        res.append(dict(test="A1.1_ablated", n_index_removed=int(drop.sum()),
                        **contrast("A1.1_cmd_vs_null_y30_INDEX_ABLATED", X, y_all, COV + CMD, COV, mask=~drop)))
    else:
        res.append(dict(test="A1.1", verdict="GATE-CLOSED", n_pos_clusters=gate["all"]))
    if gate["shoulder"] >= 30:
        res.append(dict(test="SENS_shoulder", **contrast("A1.1_cmd_vs_null_y30_shoulder", X, y_sh, COV + CMD, COV)))
    else:
        res.append(dict(test="SENS_shoulder", verdict="GATE-CLOSED", n_pos_clusters=gate["shoulder"]))
    df = pd.DataFrame(res)
    conf = df.test.isin(["A1.1", "A1.2"]) & df.get("p", pd.Series(dtype=float)).notna()
    if conf.any():
        df.loc[conf, "q_bh"] = C.bh(df.loc[conf, "p"].values)
    df.to_csv(C.OUT / "command_results.csv", index=False)
    # descriptive trajectory of z_cmd (labelled descriptive)
    from twin.descriptive import trajectories
    traj = trajectories(X.assign(game_date=pd.to_datetime(X.game_date)), "player_id", "z_cmd", L_all, "pitchers, any IL (command)")
    json.dump(dict(prereg_a1=PREREG_A1, input_sha16=sha, rows=int(len(X)), pitchers=int(X.player_id.nunique()),
                   join_rate_2024_2026=join_rate, gate=gate, base_rate=float(y_all.mean()),
                   median_miss_in=float(X.cmd_med.median()), trajectory=traj),
              open(C.OUT / "command_summary.json", "w"), indent=1)
    print(df.drop(columns=["folds"], errors="ignore").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
