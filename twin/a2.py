"""Addendum A2 — estimation and description for the full paper.

Preregistration: PREREGISTRATION_ADDENDUM_A2_2026-10-02.md (sha256[:16] 5a1f00b5420b4e29, SEAL_A2.txt), sealed
before any quantity below was computed. Nothing here changes a sealed or twin estimate.

Feature panels are read from TWIN_OUT (built by `twin.pitcher primary|extended` and `twin.batter primary public`).
Committed results are read from TWIN_RESULTS (default: this repository's out/). A2 outputs are written to A2_OUT
(default: out/).

Usage:
  python3 -m twin.a2 reproduce      # A2.0  re-run contrasts in TWIN_OUT vs committed out/ -> a2_reproduction.json
  python3 -m twin.a2 horizon        # A2.2  -> a2_horizon.json
  python3 -m twin.a2 calibration    # A2.3  -> a2_calibration.json, a2_dca.csv
  python3 -m twin.a2 command        # A2.4  -> a2_command_ext.csv, a2_command_ext_summary.json (needs OPENCOMMAND_DATA)
  python3 -m twin.a2 anatomy        # A2.5  -> a2_index_anatomy.json
  python3 -m twin.a2 power          # A2.6  -> a2_power.csv, a2_f2_defect_check.json
"""
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

from twin import common as C

PREREG_A2 = "5a1f00b5420b4e29"
RES = pathlib.Path(os.environ.get("TWIN_RESULTS", C.REPO / "out"))
A2_OUT = pathlib.Path(os.environ.get("A2_OUT", C.REPO / "out"))
A2_OUT.mkdir(parents=True, exist_ok=True)
COV = ["cov_pitches_outing", "cov_days_rest", "cov_cum_pitches", "cov_n_out_so_far"]
M3 = ["m3_streak_low_velo", "m3_rest_dev", "m3_z_velo_x_rest"]
ELBOW = {"elbow_other", "forearm_flexor", "ulnar_nerve", "ucl_graft"}
HORIZONS = (7, 14, 30, 60)
K = 10
B_TRAJ = 1000


def assert_seal():
    assert C.sha16(C.REPO / "PREREGISTRATION_ADDENDUM_A2_2026-10-02.md") == PREREG_A2, "A2 prereg hash mismatch"


def dump(obj, name):
    (A2_OUT / name).write_text(json.dumps(obj, indent=1, default=float))
    print("wrote", name, flush=True)


# ------------------------------------------------------------------ shared loaders
def placements():
    P = pd.read_csv(RES / "placements_public.csv", parse_dates=["il_start", "il_end"])
    P["mlbam"] = P.mlbam.astype(int)
    return P


def starts(d):
    return {int(k): np.sort(g.il_start.values) for k, g in d.groupby("mlbam")}


def pitcher_X(window="primary"):
    X = pd.read_parquet(C.OUT / f"pitcher_{window}_post_features.parquet")
    X["game_date"] = pd.to_datetime(X.game_date)
    return X


def w0(X):
    return [c for c in X.columns if c.startswith("z_")] + ["release_drift"]


# ------------------------------------------------------------------ A2.0 reproduction check
def reproduce():
    out = {}
    pairs = {"pitcher_primary_results.csv": "test", "pitcher_extended_results.csv": "test",
             "batter_primary_public_fdisc.csv": "contrast", "batter_primary_public_fdesc.csv": "contrast",
             "command_results.csv": "test"}
    for f, key in pairs.items():
        a, b = C.OUT / f, RES / f
        if not a.exists():
            out[f] = "not re-run"
            continue
        A, Bc = pd.read_csv(a).set_index(key), pd.read_csv(b).set_index(key)
        num = [c for c in Bc.columns if c in A.columns and pd.api.types.is_numeric_dtype(Bc[c])]
        diffs = []
        for r in Bc.index:
            for c in num:
                x, y = A.loc[r, c], Bc.loc[r, c]
                if (pd.isna(x) and pd.isna(y)):
                    continue
                if pd.isna(x) != pd.isna(y) or round(float(x), 4) != round(float(y), 4):
                    diffs.append(dict(row=r, col=c, rerun=None if pd.isna(x) else float(x), committed=None if pd.isna(y) else float(y)))
        out[f] = dict(rows=int(len(Bc)), cols_checked=num, n_diffs_at_4dp=len(diffs), diffs=diffs[:50])
    out["all_equal_at_4dp"] = all(isinstance(v, dict) and v["n_diffs_at_4dp"] == 0 for v in out.values() if isinstance(v, dict))
    import platform
    out["software"] = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)
    dump(out, "a2_reproduction.json")


# ------------------------------------------------------------------ A2.2 horizon sweep (descriptive)
def trajectories_h(X, pid_col, z_col, starts_by_pid, tag, h):
    """twin/descriptive.py::trajectories with the 30-day case window replaced by h (reference fixed at 60 days)."""
    X = X.sort_values([pid_col, "season", "game_date"]).reset_index(drop=True)
    pids, dates, z = X[pid_col].values, X.game_date.values, X[z_col].values.astype(float)
    seasons = X.season.values
    rows_by = {}
    for i, p in enumerate(pids):
        rows_by.setdefault(p, []).append(i)
    rec = []
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
            if (s0 - dates[last]) / np.timedelta64(1, "D") > h:
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
        bs = [np.concatenate([idx[p] for p in rng.choice(u, len(u))]).mean() for _ in range(B_TRAJ)]
        out["lo"].append(float(np.quantile(bs, .025))); out["hi"].append(float(np.quantile(bs, .975)))
    u = np.unique(ref_p); idx = {p: ref[ref_p == p] for p in u}
    bs = [np.concatenate([idx[p] for p in rng.choice(u, len(u))]).mean() for _ in range(200)]
    out["reference"] = dict(mean=float(ref.mean()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), n=int(len(ref)))
    out["tag"] = tag
    out["h_days"] = h
    out["k0_minus_ref"] = out["mean"][0] - out["reference"]["mean"]
    out["k0_minus_ref_lo"] = out["lo"][0] - out["reference"]["mean"]
    out["k0_minus_ref_hi"] = out["hi"][0] - out["reference"]["mean"]
    out["max_abs_k1_10_minus_ref"] = float(np.nanmax(np.abs(np.array(out["mean"][1:]) - out["reference"]["mean"])))
    return out


def simple_rule_h(X, pid_col, z_col, starts_by_pid, h):
    y = C.label_within(X[pid_col].values, X.game_date.values, starts_by_pid, days=h)
    z = X[z_col].values.astype(float)
    ok = np.isfinite(z)
    f = ok & (z <= -1.0)
    return dict(h_days=h, n_rows=int(ok.sum()), n_pos=int(y[ok].sum()), base_rate=float(y[ok].mean()),
                n_flagged=int(f.sum()), share_flagged=float(f.sum() / ok.sum()), ppv=float(y[f].mean()),
                ppv_over_base=float(y[f].mean() / y[ok].mean()), sensitivity=float(y[f].sum() / max(y[ok].sum(), 1)))


def horizon():
    P = placements()
    Pp = P[~P.covid_era_blank]
    Lp = dict(pitcher_all=starts(Pp), pitcher_shoulder=starts(Pp[Pp.site == "shoulder"]),
              pitcher_elbow=starts(Pp[Pp.site.isin(ELBOW)]))
    Lb = dict(batter_msk=starts(P[P.msk]), batter_oblique=starts(P[P.site == "core_oblique"]))
    Xp = pitcher_X("primary")
    Xb = pd.read_parquet(C.OUT / "batter_primary_public_features.parquet")
    Xb["game_date"] = pd.to_datetime(Xb.game_date)
    Xb = Xb[Xb.eligible10].reset_index(drop=True)
    tags = dict(pitcher_all="pitchers, any IL", pitcher_shoulder="pitchers, shoulder IL", pitcher_elbow="pitchers, elbow-family IL",
                batter_msk="batters, any musculoskeletal IL", batter_oblique="batters, oblique IL")
    out = dict(trajectories={}, simple_rule={}, note="A2.2, descriptive, no test; reference = rows with no placement in the next 60 days")
    for h in HORIZONS:
        for k, L in Lp.items():
            out["trajectories"][f"{k}.h{h}"] = trajectories_h(Xp, "player_id", "z_velocity", L, tags[k], h)
            out["simple_rule"][f"{k}.h{h}"] = simple_rule_h(Xp, "player_id", "z_velocity", L, h)
        for k, L in Lb.items():
            out["trajectories"][f"{k}.h{h}"] = trajectories_h(Xb, "batter", "z_bs_dev", L, tags[k], h)
        print("horizon", h, {k: round(v["k0_minus_ref"], 3) for k, v in out["trajectories"].items() if k.endswith(f"h{h}")}, flush=True)
    # self-check: h = 30 must reproduce the committed Figure 1 trajectories
    D = json.load(open(RES / "descriptive_trajectory.json"))["trajectories"]
    chk = {}
    for k in ["pitcher_all", "pitcher_shoulder", "batter_msk", "batter_oblique"]:
        a, b = out["trajectories"][f"{k}.h30"], D[k]
        chk[k] = bool(np.allclose(a["mean"], b["mean"], equal_nan=True) and a["n"] == b["n"])
    out["h30_reproduces_figure1"] = chk
    print("h30 reproduces Figure 1:", chk, flush=True)
    dump(out, "a2_horizon.json")


# ------------------------------------------------------------------ A2.3 calibration and decision curve
def _irls_offset(y, off, w, slope):
    """Weighted logistic y ~ a + b*off (slope=True) or y ~ a + offset(off) (slope=False). Returns (a, b)."""
    a, b = 0.0, 1.0
    for _ in range(50):
        eta = a + b * off
        p = 1 / (1 + np.exp(-np.clip(eta, -35, 35)))
        W = w * np.maximum(p * (1 - p), 1e-10)
        r = w * (y - p)
        if slope:
            g = np.array([r.sum(), (r * off).sum()])
            H = np.array([[W.sum(), (W * off).sum()], [(W * off).sum(), (W * off * off).sum()]])
            step = np.linalg.solve(H, g)
            a, b = a + step[0], b + step[1]
            if np.abs(step).max() < 1e-9:
                break
        else:
            step = r.sum() / W.sum()
            a = a + step
            if abs(step) < 1e-10:
                break
    return a, b


def calib_stats(y, p, w=None):
    w = np.ones(len(y)) if w is None else w
    off = np.log(p / (1 - p))
    citl, _ = _irls_offset(y, off, w, slope=False)
    _, slope = _irls_offset(y, off, w, slope=True)
    oe = (w * y).sum() / (w * p).sum()
    return citl, slope, oe


def boot_calib(y, p, cl, seed=C.SEED, B=C.B):
    r = np.random.default_rng(seed)
    u, inv = np.unique(cl, return_inverse=True)
    idx = [np.where(inv == k)[0] for k in range(len(u))]
    est = calib_stats(y, p)
    bs = np.empty((B, 3))
    for b in range(B):
        take = np.concatenate([idx[k] for k in r.integers(0, len(u), len(u))])
        mlt = np.bincount(take, minlength=len(y)).astype(float)
        bs[b] = calib_stats(y, p, mlt)
    q = lambda j: (float(np.quantile(bs[:, j], .025)), float(np.quantile(bs[:, j], .975)))
    return dict(citl=est[0], citl_ci=q(0), slope=est[1], slope_ci=q(1), oe=est[2], oe_ci=q(2), n=int(len(y)),
                events=int(y.sum()), mean_pred=float(p.mean()), observed=float(y.mean()))


def net_benefit(y, p, t, w=None):
    w = np.ones(len(y)) if w is None else w
    n = w.sum()
    out = []
    for tt in t:
        f = p >= tt
        tp = (w * (f & (y == 1))).sum(); fp = (w * (f & (y == 0))).sum()
        out.append(tp / n - fp / n * tt / (1 - tt))
    return np.array(out)


def treat_all(y, t, w=None):
    w = np.ones(len(y)) if w is None else w
    pi = (w * y).sum() / w.sum()
    return np.array([pi - (1 - pi) * tt / (1 - tt) for tt in t])


def boot_dnb(y, pm, pn, cl, t, seed=C.SEED, B=C.B):
    r = np.random.default_rng(seed)
    u, inv = np.unique(cl, return_inverse=True)
    idx = [np.where(inv == k)[0] for k in range(len(u))]
    est = net_benefit(y, pm, t) - net_benefit(y, pn, t)
    bs = np.empty((B, len(t)))
    for b in range(B):
        take = np.concatenate([idx[k] for k in r.integers(0, len(u), len(u))])
        mlt = np.bincount(take, minlength=len(y)).astype(float)
        bs[b] = net_benefit(y, pm, t, mlt) - net_benefit(y, pn, t, mlt)
    return [dict(t=float(tt), dnb=float(est[i]), lo=float(np.quantile(bs[:, i], .025)), hi=float(np.quantile(bs[:, i], .975)),
                 nb_model=float(net_benefit(y, pm, [tt])[0]), nb_null=float(net_benefit(y, pn, [tt])[0]),
                 nb_all=float(treat_all(y, [tt])[0])) for i, tt in enumerate(t)]


def deciles(y, p):
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    d = pd.DataFrame(dict(y=y, p=p, q=q)).groupby("q").agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return d.reset_index().to_dict(orient="list")


def calibration():
    P = placements()
    Pp = P[~P.covid_era_blank]
    L_all, L_sh = starts(Pp), starts(Pp[Pp.site == "shoulder"])
    X = pitcher_X("primary")
    W0, pid, gd = w0(X), X.player_id.values, X.game_date.values
    y_all = C.label_within(pid, gd, L_all); y_sh = C.label_within(pid, gd, L_sh)
    drop_all, drop_sh = C.index_mask(pid, gd, L_all), C.index_mask(pid, gd, L_sh)
    f2 = json.load(open(RES / "f2_results.json"))
    thr = dict(all=sorted({round(g["ppv_star"], 6) for g in f2["b_grid"] if g["label"] == "pitcher_all" and g["ppv_star"] < 1}),
               shoulder=sorted({round(g["ppv_star"], 6) for g in f2["b_grid"] if g["label"] == "pitcher_shoulder" and g["ppv_star"] < 1}))
    grid = dict(all=np.round(np.arange(0.01, 0.5001, 0.005), 4), shoulder=np.round(np.arange(0.005, 0.2501, 0.0025), 4))
    models = dict(P2=("all", y_all, COV + W0 + M3, drop_all), P3=("shoulder", y_sh, COV + W0, drop_sh))
    committed = pd.read_csv(RES / "pitcher_primary_results.csv").set_index("test")
    out = dict(note="A2.3; OOF season-forward, first two seasons unscored; estimation only", thresholds_f2=thr, models={})
    dca_rows = []
    for name, (lab, y, feats, drop) in models.items():
        for grid_name, mask in (("full", None), ("ablated", ~drop)):
            pm, folds = C.oof(X, feats, y, 2, mask)
            pn, _ = C.oof(X, COV, y, 2, mask)
            ok = np.isfinite(pm) & np.isfinite(pn)
            if mask is not None:
                ok &= mask
            yy, cl, ss = y[ok], pid[ok], X.season.values[ok]
            a_m, a_n = C.auc(yy, pm[ok]), C.auc(yy, pn[ok])
            ref_test = {("P2", "full"): "P2", ("P2", "ablated"): "P2_ablated", ("P3", "full"): "P3",
                        ("P3", "ablated"): "P3_ablated_TWIN_NEW"}[(name, grid_name)]
            same = bool(round(a_m - a_n, 4) == round(float(committed.loc[ref_test, "dauc"]), 4))
            rec = dict(label=lab, grid=grid_name, n=int(ok.sum()), events=int(yy.sum()), auc_model=a_m, auc_null=a_n,
                       dauc=a_m - a_n, dauc_matches_committed=same)
            rec["calibration_pooled"] = dict(model=boot_calib(yy, pm[ok], cl), null=boot_calib(yy, pn[ok], cl))
            by = {}
            for s in sorted(np.unique(ss)):
                m = ss == s
                if yy[m].sum() < 5:
                    continue
                by[int(s)] = dict(model=boot_calib(yy[m], pm[ok][m], cl[m]), null=boot_calib(yy[m], pn[ok][m], cl[m]),
                                  auc_model=C.auc(yy[m], pm[ok][m]), auc_null=C.auc(yy[m], pn[ok][m]),
                                  dauc=C.auc(yy[m], pm[ok][m]) - C.auc(yy[m], pn[ok][m]), n=int(m.sum()), events=int(yy[m].sum()))
            rec["by_season"] = by
            rec["deciles"] = dict(model=deciles(yy, pm[ok]), null=deciles(yy, pn[ok]))
            rec["dnb_at_f2_thresholds"] = boot_dnb(yy, pm[ok], pn[ok], cl, thr[lab])
            t = grid[lab]
            nbm, nbn, nba = net_benefit(yy, pm[ok], t), net_benefit(yy, pn[ok], t), treat_all(yy, t)
            for i, tt in enumerate(t):
                dca_rows.append(dict(model=name, label=lab, grid=grid_name, t=float(tt), nb_model=nbm[i], nb_workload=nbn[i],
                                     nb_treat_all=nba[i], nb_treat_none=0.0))
            rec["max_dnb_on_curve"] = float(np.max(nbm - nbn))
            rec["share_thresholds_model_above_both_defaults"] = float(np.mean((nbm > np.maximum(nba, 0))))
            out["models"][f"{name}.{grid_name}"] = rec
            print(name, grid_name, "dauc", round(a_m - a_n, 4), "matches committed", same,
                  "slope", round(rec["calibration_pooled"]["model"]["slope"], 3), "citl", round(rec["calibration_pooled"]["model"]["citl"], 3),
                  "dNB", [(round(d["t"], 3), round(d["dnb"], 4)) for d in rec["dnb_at_f2_thresholds"]], flush=True)
    pd.DataFrame(dca_rows).to_csv(A2_OUT / "a2_dca.csv", index=False)
    print("wrote a2_dca.csv", flush=True)
    dump(out, "a2_calibration.json")


# ------------------------------------------------------------------ A2.4 command, extended + pitch-type sensitivity
FASTBALL = {"FF", "SI", "FC", "FA"}
BREAKING = {"SL", "ST", "SV", "CU", "KC", "CS"}


def command_by_type():
    from twin import command as CM
    parts = []
    for y in (2024, 2025, 2026):
        t = pd.read_csv(CM.OC / f"{y}_targets.csv.gz")
        p = pd.read_csv(CM.OC / f"{y}_pbp_info.csv.gz", usecols=["play_id", "pitcher_id", "date", "game_type", "pitch_type"])
        t = t[t["plausible"].astype(bool)]
        p = p.rename(columns={"date": "game_date"})
        t = t.merge(p, on="play_id", how="inner")
        t = t[t.game_type == "R"]
        t["miss"] = np.hypot(t.plate_x_in - t.inferred_x_in, t.plate_z_in - t.inferred_z_in)
        t["game_date"] = pd.to_datetime(t.game_date)
        t["grp"] = np.where(t.pitch_type.isin(FASTBALL), "fb", np.where(t.pitch_type.isin(BREAKING), "br", "other"))
        t = t[t.grp != "other"]
        g = (t.groupby(["pitcher_id", "game_date", "grp"]).agg(med=("miss", "median"), n=("miss", "size")).reset_index())
        g = g[g.n >= 5]
        w = g.pivot_table(index=["pitcher_id", "game_date"], columns="grp", values="med").reset_index()
        w.columns = [c if c in ("pitcher_id", "game_date") else f"cmd_med_{c}" for c in w.columns]
        parts.append(w)
    o = pd.concat(parts, ignore_index=True).rename(columns={"pitcher_id": "player_id"})
    for c in ("cmd_med_fb", "cmd_med_br"):
        if c not in o.columns:
            o[c] = np.nan
    return o


def add_group_features(X, col, tag):
    X = X.sort_values(["player_id", "season", "game_date"]).reset_index(drop=True)
    z = np.full(len(X), np.nan); d3 = np.full(len(X), np.nan); sl = np.full(len(X), np.nan)
    for (pid, s), g in X.groupby(["player_id", "season"], sort=False):
        idx = g.index.values; v = g[col].values.astype(float)
        for j in range(len(idx)):
            prior = v[:j]; prior = prior[np.isfinite(prior)]
            if len(prior) < 5 or not np.isfinite(v[j]):
                continue
            mu, sd = prior.mean(), prior.std(ddof=1)
            if sd > 0:
                z[idx[j]] = (v[j] - mu) / sd
            w = v[:j + 1]; w = w[np.isfinite(w)]
            d3[idx[j]] = w[-3:].mean() - mu
            if len(w) >= 5:
                sl[idx[j]] = float(np.polyfit(np.arange(5), w[-5:], 1)[0])
    X[f"z_cmd_{tag}"], X[f"d3_cmd_{tag}"], X[f"slope5_cmd_{tag}"] = z, d3, sl
    return X


def command():
    from twin import command as CM
    sha = CM.assert_seal()
    cmd = CM.outing_command()
    P = placements()
    Pp = P[~P.covid_era_blank]
    L_all, L_sh = starts(Pp), starts(Pp[Pp.site == "shoulder"])
    res, summ = [], dict(prereg_a2=PREREG_A2, prereg_a1=CM.PREREG_A1, input_sha16=sha)
    # (1) A1 on the extended window
    X = pitcher_X("extended")
    X = X[X.season >= 2024].merge(cmd[["player_id", "game_date", "cmd_med", "n_cmd"]], on=["player_id", "game_date"], how="left")
    summ["ext_join_rate"] = float(X.cmd_med.notna().mean())
    X = CM.add_features(X)
    X = X[np.isfinite(X.z_cmd)].reset_index(drop=True)
    pid, gd = X.player_id.values, X.game_date.values
    y_all = C.label_within(pid, gd, L_all); y_sh = C.label_within(pid, gd, L_sh)
    gate = dict(all=int(pd.unique(pid[y_all == 1]).size), shoulder=int(pd.unique(pid[y_sh == 1]).size))
    summ.update(ext_rows=int(len(X)), ext_pitchers=int(X.player_id.nunique()), ext_gate=gate, ext_base_rate=float(y_all.mean()))
    W0 = [c for c in X.columns if c.startswith("z_") and c not in CM.CMD] + ["release_drift"]
    print(f"extended rows {len(X)} gate {gate}", flush=True)
    if gate["all"] >= 30:
        res.append(dict(test="A1.1_ext", **CM.contrast("A1.1_cmd_vs_null_y30_EXT", X, y_all, COV + CM.CMD, COV)))
        res.append(dict(test="A1.2_ext", **CM.contrast("A1.2_W0cmd_vs_W0_y30_EXT", X, y_all, COV + W0 + CM.CMD, COV + W0)))
        drop = C.index_mask(pid, gd, L_all)
        res.append(dict(test="A1.1_ablated_ext", n_index_removed=int(drop.sum()),
                        **CM.contrast("A1.1_cmd_vs_null_y30_INDEX_ABLATED_EXT", X, y_all, COV + CM.CMD, COV, mask=~drop)))
    else:
        res.append(dict(test="A1.1_ext", verdict="GATE-CLOSED", n_pos_clusters=gate["all"]))
    if gate["shoulder"] >= 30:
        res.append(dict(test="SENS_shoulder_ext", **CM.contrast("A1.1_cmd_vs_null_y30_shoulder_EXT", X, y_sh, COV + CM.CMD, COV)))
    else:
        res.append(dict(test="SENS_shoulder_ext", verdict="GATE-CLOSED", n_pos_clusters=gate["shoulder"]))
    # (2) pitch-type sensitivity, primary window, on the A1 rows
    Xp = pitcher_X("primary")
    Xp = Xp[Xp.season >= 2024].merge(cmd[["player_id", "game_date", "cmd_med", "n_cmd"]], on=["player_id", "game_date"], how="left")
    Xp = CM.add_features(Xp)
    Xp = Xp[np.isfinite(Xp.z_cmd)].reset_index(drop=True)
    bt = command_by_type()
    Xp = Xp.merge(bt, on=["player_id", "game_date"], how="left")
    Xp = add_group_features(Xp, "cmd_med_fb", "fb")
    Xp = add_group_features(Xp, "cmd_med_br", "br")
    TYPE = ["z_cmd_fb", "d3_cmd_fb", "slope5_cmd_fb", "z_cmd_br", "d3_cmd_br", "slope5_cmd_br"]
    pid2, gd2 = Xp.player_id.values, Xp.game_date.values
    y2 = C.label_within(pid2, gd2, L_all)
    gate2 = int(pd.unique(pid2[y2 == 1]).size)
    summ.update(type_rows=int(len(Xp)), type_gate=gate2, type_share_fb=float(np.isfinite(Xp.z_cmd_fb).mean()),
                type_share_br=float(np.isfinite(Xp.z_cmd_br).mean()))
    if gate2 >= 30:
        res.append(dict(test="SENS_pitch_type", **CM.contrast("cmd_by_pitch_type_vs_null_y30", Xp, y2, COV + TYPE, COV)))
    else:
        res.append(dict(test="SENS_pitch_type", verdict="GATE-CLOSED", n_pos_clusters=gate2))
    df = pd.DataFrame(res)
    fam = df.test.isin(["A1.1_ext", "A1.2_ext"]) & df.get("p", pd.Series(dtype=float)).notna()
    if fam.any():
        df.loc[fam, "q_bh"] = C.bh(df.loc[fam, "p"].values)
    df.to_csv(A2_OUT / "a2_command_ext.csv", index=False)
    print("wrote a2_command_ext.csv", flush=True)
    print(df.drop(columns=["folds"], errors="ignore").round(4).to_string(index=False), flush=True)
    dump(summ, "a2_command_ext_summary.json")


# ------------------------------------------------------------------ A2.5 index-outing anatomy (descriptive)
def anatomy():
    argv = sys.argv
    sys.argv = [argv[0], "primary"]
    from twin import pitcher as PI
    sys.argv = argv
    o = PI.outing_panel()
    # median of the pitcher's previous in-season outings' total pitches, keyed by (player, season, outing_idx)
    med = {}
    for (pid, s), g in o.groupby(["player_id", "season"], sort=False):
        tp = g.total_pitches.values.astype(float)
        for i in range(5, len(tp)):
            med[(pid, s, i)] = float(np.median(tp[:i]))
    X = pitcher_X("primary")
    X["own_median"] = [med.get(k, np.nan) for k in zip(X.player_id, X.season, X.outing_idx)]
    X["early"] = X.cov_pitches_outing < X.own_median
    P = placements()
    Pp = P[~P.covid_era_blank].sort_values(["mlbam", "il_start"])
    fam = Pp.drop_duplicates(["mlbam", "il_start"]).set_index(["mlbam", "il_start"]).site
    L_all = starts(Pp)
    X = X.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    pid, gd, z = X.player_id.values, X.game_date.values, X.z_velocity.values.astype(float)
    rows_by = {}
    for i, p in enumerate(pid):
        rows_by.setdefault(p, []).append(i)
    rec = []
    for p, a in L_all.items():
        rows = rows_by.get(p)
        if not rows:
            continue
        rows = np.array(rows); dts = gd[rows]
        for s0 in a:
            j = np.searchsorted(dts, s0, side="left")
            if j == 0:
                continue
            gap = (s0 - dts[j - 1]) / np.timedelta64(1, "D")
            if gap > 30:
                continue
            r = rows[j - 1]
            site = fam.get((p, pd.Timestamp(s0)), "unknown")
            famname = "shoulder" if site == "shoulder" else ("elbow" if site in ELBOW else "other")
            rec.append(dict(pid=p, start=s0, days=float(gap), z=z[r], early=bool(X.early.values[r]) if np.isfinite(X.own_median.values[r]) else np.nan,
                            family=famname))
    R = pd.DataFrame(rec)
    R["slow"] = R.z <= -1
    R = R[np.isfinite(R.z)]

    def summ(d):
        dd = d.days.values
        e = d.early.dropna().astype(float)
        return dict(n=int(len(d)), n_pitchers=int(d.pid.nunique()), days_median=float(np.median(dd)),
                    days_q1=float(np.quantile(dd, .25)), days_q3=float(np.quantile(dd, .75)),
                    share_le1=float((dd <= 1).mean()), share_le3=float((dd <= 3).mean()), share_le7=float((dd <= 7).mean()),
                    share_le14=float((dd <= 14).mean()), share_early=float(e.mean()), n_early_known=int(len(e)),
                    by_family={k: int(v) for k, v in d.family.value_counts().items()})
    y30 = C.label_within(pid, gd, L_all)
    okz = np.isfinite(z) & np.isfinite(X.own_median.values)
    comp_slow = okz & (z <= -1) & (y30 == 0)
    comp_all = okz & (y30 == 0)
    out = dict(note="A2.5, descriptive; IL start is the effective date and can be backdated",
               slow_index=summ(R[R.slow]), not_slow_index=summ(R[~R.slow]), all_index=summ(R),
               share_slow_among_index=float(R.slow.mean()),
               comparator_slow_no_placement=dict(n=int(comp_slow.sum()), share_early=float(X.early.values[comp_slow].mean())),
               comparator_all_no_placement=dict(n=int(comp_all.sum()), share_early=float(X.early.values[comp_all].mean())))
    print(json.dumps({k: v for k, v in out.items() if k != "note"}, indent=1, default=float), flush=True)
    dump(out, "a2_index_anatomy.json")


# ------------------------------------------------------------------ A2.6 power and the F2(b) defect check
def mde(lo, hi):
    return 2.80 * (hi - lo) / 3.92


def power():
    rows = []
    f2 = json.load(open(RES / "f2_results.json"))
    for a in f2["a_mde"]:
        rows.append(dict(source="sealed", arm=a["arm"], dauc=a["dauc"], ci_lo=a["ci_lo"], ci_hi=a["ci_hi"], mde=mde(a["ci_lo"], a["ci_hi"])))
    pr = pd.read_csv(RES / "pitcher_primary_results.csv").set_index("test")
    names = dict(P1="Twin P1 W0 vs workload null, any IL", P2="Twin P2 W0+M3 vs workload null, any IL",
                 P2_ablated="Twin P2, index outing removed", P3="Twin P3 W0 vs workload null, shoulder",
                 P3_ablated_TWIN_NEW="Twin P3, index outing removed", P4="Twin P4 full ladder vs workload null, any IL",
                 SENS_P1_PRE="Twin P1, features through the previous outing")
    for t, nm in names.items():
        r = pr.loc[t]
        rows.append(dict(source="twin", arm=nm, dauc=r.dauc, ci_lo=r.ci_lo, ci_hi=r.ci_hi, mde=mde(r.ci_lo, r.ci_hi)))
    br = pd.read_csv(RES / "batter_primary_public_fdisc.csv").set_index("contrast")
    for fam, lab in [("oblique", "oblique"), ("hamstring", "hamstring"), ("hand_wrist", "hand/wrist")]:
        for c, suf in [(f"B_{fam}", ""), (f"B_{fam}_INDEX_ABLATED", ", index game removed")]:
            r = br.loc[c]
            rows.append(dict(source="twin", arm=f"Twin batter {lab}{suf}", dauc=r.dauc, ci_lo=r.ci_lo, ci_hi=r.ci_hi, mde=mde(r.ci_lo, r.ci_hi)))
    cr = pd.read_csv(RES / "command_results.csv").set_index("test")
    for t, nm in [("A1.1", "A1.1 command vs workload null, any IL"), ("A1.1_ablated", "A1.1, index outing removed"),
                  ("A1.2", "A1.2 command over kinematics, any IL")]:
        r = cr.loc[t]
        rows.append(dict(source="A1", arm=nm, dauc=r.dauc, ci_lo=r.ci_lo, ci_hi=r.ci_hi, mde=mde(r.ci_lo, r.ci_hi)))
    df = pd.DataFrame(rows)
    df["upper_95"] = df.ci_hi
    df.to_csv(A2_OUT / "a2_power.csv", index=False)
    print(df.round(4).to_string(index=False), flush=True)
    # F2(b) defect check: D from placements starting 2015-01-01 or later
    from scipy.stats import norm
    P = placements()
    P = P[~P.covid_era_blank & P.il_end.notna()].copy()
    P["days"] = (P.il_end - P.il_start).dt.days
    pit = set(pd.read_parquet(C.OUT / "pitcher_primary_post_features.parquet", columns=["player_id"]).player_id)
    bat = set(pd.read_parquet(C.OUT / "batter_primary_public_features.parquet", columns=["batter"]).batter)

    def Dof(Q):
        return {"pitcher_all": float(Q[Q.mlbam.isin(pit)].days.median()),
                "pitcher_shoulder": float(Q[Q.mlbam.isin(pit) & (Q.site == "shoulder")].days.median()),
                "batter_oblique": float(Q[Q.mlbam.isin(bat) & ~Q.mlbam.isin(pit) & (Q.site == "core_oblique")].days.median())}
    D_all, D_15 = Dof(P), Dof(P[P.il_start >= "2015-01-01"])
    assert D_all == f2["median_il_days"], "sealed D not reproduced"
    Z90 = norm.ppf(0.90)

    def auc_for_lr(lr):
        tpr = lr * 0.10
        if tpr >= 1:
            return float("nan")
        return float(norm.cdf((Z90 - norm.ppf(1 - tpr)) / np.sqrt(2)))
    obs = f2["b_observed"]
    grid = []
    for lab, Dm in D_15.items():
        pi = obs[lab]["pi"]
        for r_ in (5, 15):
            for e in (0.25, 0.5, 1.0):
                ppv = r_ / (e * Dm)
                if ppv >= 1:
                    grid.append(dict(label=lab, r=r_, e=e, D=Dm, ppv_star=ppv, lr_star=None, auc_star=None))
                    continue
                lr = (ppv / (1 - ppv)) / (pi / (1 - pi))
                met = obs[lab]["lr_at_10pct_binormal"] >= lr
                grid.append(dict(label=lab, r=r_, e=e, D=Dm, ppv_star=ppv, lr_star=lr, auc_star=auc_for_lr(lr), met=bool(met),
                                 met_by_workload_null_alone=bool(obs[lab]["lr_null_at_10pct_binormal"] >= lr) if met else None))
    out = dict(note="A2.6(2) defect check: median IL days from placements starting 2015-01-01 or later",
               D_sealed=D_all, D_2015_on=D_15, D_changed=D_all != D_15, grid=grid,
               cells_met=[g for g in grid if g.get("met")], n_cells_met=int(sum(bool(g.get("met")) for g in grid)))
    print(json.dumps({k: v for k, v in out.items() if k != "grid"}, indent=1, default=float), flush=True)
    dump(out, "a2_f2_defect_check.json")


if __name__ == "__main__":
    assert_seal()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"reproduce": reproduce, "horizon": horizon, "calibration": calibration, "command": command,
     "anatomy": anatomy, "power": power}[cmd]()
