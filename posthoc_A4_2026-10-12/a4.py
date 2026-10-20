"""Post hoc addendum A4 (2026-10-12): robustness and candidate analyses from the manuscript review.

Specification: POSTHOC_ADDENDUM_A4_2026-10-12.md (sha256[:16] recorded in SEAL_A4.txt before this code first ran).
Everything here is post hoc and cannot change a preregistered verdict.

Reads the twin's rebuilt feature panels and census from TWIN_OUT, raw pinned Statcast from TWIN_DATA, and uses twin.common
(estimator, AUC, cluster bootstrap, label and removal rules) unchanged. Writes to A4_OUT.

Usage: python3 a4.py [all|<id> ...]   (ids: a41 a42 a43 a44 a45 a46 a47 a48 a49 a410 a411)
"""
import hashlib
import json
import os
import pathlib
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
SPEC = HERE / "POSTHOC_ADDENDUM_A4_2026-10-12.md"
SEAL = HERE / "SEAL_A4.txt"
TWIN = pathlib.Path(os.environ.get("TWIN_REPO", HERE.parent / "public_twin"))
sys.path.insert(0, str(TWIN))
from twin import common as C  # noqa: E402

A4_OUT = pathlib.Path(os.environ.get("A4_OUT", HERE / "out")); A4_OUT.mkdir(parents=True, exist_ok=True)
N4 = ["cov_pitches_outing", "cov_days_rest", "cov_cum_pitches", "cov_n_out_so_far"]
M3 = ["m3_streak_low_velo", "m3_rest_dev", "m3_z_velo_x_rest"]
NA = ["cov_pitches_outing", "wa_days_since_app", "wa_p3", "wa_p7", "wa_p14", "wa_p28", "wa_apps7", "wa_b2b", "wa_cum",
      "wa_napps"]
ELBOW = {"elbow_other", "forearm_flexor", "ulnar_nerve", "ucl_graft"}
STRUCT = re.compile(r"\btear\b|\btorn\b|strain|sprain|fractur|ruptur|surger|reconstruct|stress reaction|dislocat|sublux", re.I)
NONSPEC = re.compile(r"inflamm|sore|fatigue|tight|tendin|tendon|discomfort|irritat|impinge|spasm|cramp|stiff", re.I)
FEED_END = pd.Timestamp("2026-09-29")
GATE = 30
ROWS = []
LOG = open(A4_OUT / "a4_run.log", "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n"); LOG.flush()


def assert_seal():
    h = hashlib.sha256(SPEC.read_bytes()).hexdigest()[:16]
    rec = re.search(r"sha256\[:16\] ([0-9a-f]{16})", SEAL.read_text()).group(1)
    assert h == rec, f"A4 spec hash {h} != sealed {rec}"
    return h


# ------------------------------------------------------------------ data
def placements():
    P = pd.read_csv(C.OUT / "placements_public.csv", parse_dates=["il_start", "il_end", "date"])
    P["mlbam"] = P.mlbam.astype(int)
    return P[~P.covid_era_blank].copy()


def starts(d):
    return {int(k): np.sort(g.il_start.values) for k, g in d.groupby("mlbam")}


def pitcher_X(window="primary"):
    X = pd.read_parquet(C.OUT / f"pitcher_{window}_post_features.parquet")
    X["game_date"] = pd.to_datetime(X.game_date)
    return X.sort_values(["season", "game_date"], kind="stable").reset_index(drop=True)


def w0(X):
    return [c for c in X.columns if c.startswith("z_")] + ["release_drift"]


_APP = {}


def appearances():
    """Every regular-season appearance (pitcher, game): date, pitches, started."""
    if "A" in _APP:
        return _APP["A"]
    import pyarrow.parquet as pq
    parts = []
    for y in range(2015, 2027):
        t = pq.read_table(C.DATA / f"statcast_season/statcast_{y}.parquet",
                          columns=["game_date", "game_pk", "game_type", "pitcher", "inning", "inning_topbot",
                                   "at_bat_number", "pitch_number"]).to_pandas()
        t = t[t.game_type == "R"]
        first = (t[t.inning == 1].sort_values(["game_pk", "inning_topbot", "at_bat_number", "pitch_number"])
                 .drop_duplicates(["game_pk", "inning_topbot"])[["game_pk", "pitcher"]])
        st = set(zip(first.game_pk, first.pitcher))
        a = t.groupby(["pitcher", "game_pk", "game_date"]).size().rename("pitches").reset_index()
        a["started"] = [(g, p) in st for g, p in zip(a.game_pk, a.pitcher)]
        a["season"] = y
        parts.append(a)
    A = pd.concat(parts, ignore_index=True)
    A["game_date"] = pd.to_datetime(A.game_date)
    A = A.sort_values(["pitcher", "game_date", "game_pk"]).reset_index(drop=True)
    _APP["A"] = A
    return A


def add_workload_all(X):
    A = appearances()
    by = {k: g for k, g in A.groupby(["pitcher", "season"])}
    cols = {c: np.full(len(X), np.nan) for c in NA if c != "cov_pitches_outing"}
    role = np.full(len(X), np.nan)
    for i, (p, s, d) in enumerate(zip(X.player_id.values, X.season.values, X.game_date.values)):
        g = by.get((p, s))
        if g is None:
            continue
        gd = g.game_date.values; pt = g.pitches.values.astype(float)
        prior = gd < d
        cols["wa_cum"][i] = pt[prior].sum(); cols["wa_napps"][i] = prior.sum()
        if prior.any():
            cols["wa_days_since_app"][i] = (d - gd[prior].max()) / np.timedelta64(1, "D")
        for k in (3, 7, 14, 28):
            w = prior & (gd >= d - np.timedelta64(k, "D"))
            cols[f"wa_p{k}"][i] = pt[w].sum()
            if k == 7:
                cols["wa_apps7"][i] = w.sum()
        cols["wa_b2b"][i] = float((gd == d - np.timedelta64(1, "D")).any())
        upto = gd <= d
        role[i] = g.started.values[upto].mean() if upto.any() else np.nan
    for c, v in cols.items():
        X[c] = v
    X["role_starter"] = role >= 0.5
    return X


# ------------------------------------------------------------------ estimation
def contrast(cid, label, X, y, fa, fb, grid, mask=None, extra=None):
    pa, _ = C.oof(X, fa, y, 2, mask)
    pb, _ = C.oof(X, fb, y, 2, mask)
    ok = np.isfinite(pa) & np.isfinite(pb)
    if mask is not None:
        ok &= mask
    yy, cl = y[ok], X.player_id.values[ok]
    npc = int(pd.unique(cl[yy == 1]).size)
    rec = dict(id=cid, label=label, grid=grid, model="+".join(fa), comparator="+".join(fb), n=int(ok.sum()),
               n_pos=int(yy.sum()), n_clusters=int(pd.unique(cl).size), n_pos_clusters=npc)
    if extra:
        rec.update(extra)
    if npc < GATE:
        rec.update(status="closed (gate)")
        log(cid, label, grid, "closed", npc)
        ROWS.append(rec)
        return rec, None
    aa, ab = C.auc(yy, pa[ok]), C.auc(yy, pb[ok])
    lo, hi, p = C.boot_dauc(yy, pa[ok], pb[ok], cl)
    rec.update(status="run", auc_a=aa, auc_b=ab, dauc=aa - ab, ci_lo=lo, ci_hi=hi, p=p)
    log(cid, label, grid, {k: round(rec[k], 4) for k in ("auc_a", "auc_b", "dauc", "ci_lo", "ci_hi", "p")}, "n_pos_cl", npc)
    ROWS.append(rec)
    return rec, (ok, pa, pb)


def subset_dauc(cid, label, grid, y, pa, pb, ok, sub, cl_all, tag):
    m = ok & sub
    yy, cl = y[m], cl_all[m]
    npc = int(pd.unique(cl[yy == 1]).size)
    rec = dict(id=cid, label=label, grid=grid, subset=tag, n=int(m.sum()), n_pos=int(yy.sum()),
               n_clusters=int(pd.unique(cl).size), n_pos_clusters=npc)
    if npc < GATE:
        rec.update(status="closed (gate)"); ROWS.append(rec); log(cid, tag, grid, "closed", npc); return rec
    aa, ab = C.auc(yy, pa[m]), C.auc(yy, pb[m])
    lo, hi, p = C.boot_dauc(yy, pa[m], pb[m], cl)
    rec.update(status="run", auc_a=aa, auc_b=ab, dauc=aa - ab, ci_lo=lo, ci_hi=hi, p=p)
    log(cid, tag, grid, {k: round(rec[k], 4) for k in ("auc_a", "auc_b", "dauc", "ci_lo", "ci_hi", "p")})
    ROWS.append(rec)
    return rec


def boot_reps(y, s_a, s_b, cl, seed=C.SEED, B=C.B):
    """Replicates of the fixed-model cluster bootstrap, identical draws to twin.common.boot_dauc."""
    r = np.random.default_rng(seed)
    u, inv = np.unique(cl, return_inverse=True)
    idx = [np.where(inv == k)[0] for k in range(len(u))]
    oa, ob = np.argsort(s_a, kind="stable"), np.argsort(s_b, kind="stable")
    ya, yb = y[oa], y[ob]; n = len(y); d = np.empty(B)
    for b in range(B):
        take = np.concatenate([idx[k] for k in r.integers(0, len(u), len(u))])
        mlt = np.bincount(take, minlength=n)
        d[b] = C._auc_w(oa, ya, mlt) - C._auc_w(ob, yb, mlt)
    return d[np.isfinite(d)]


def label_window(pids, dates, st, lo_days, hi_days):
    """(positive, excluded): a start in (d + lo, d + hi] is positive; a start in (d, d + lo] excludes the row."""
    pos = np.zeros(len(pids), int); exc = np.zeros(len(pids), bool)
    for i, (p, d) in enumerate(zip(pids, dates)):
        a = st.get(int(p))
        if a is None:
            continue
        dd = (a - d) / np.timedelta64(1, "D")
        if ((dd > 0) & (dd <= lo_days)).any():
            exc[i] = True
        elif ((dd > lo_days) & (dd <= hi_days)).any():
            pos[i] = 1
    return pos, exc


# ------------------------------------------------------------------ analyses
def a41(X, L):
    t = time.time()
    X = add_workload_all(X)
    X.to_parquet(A4_OUT / "a4_primary_with_workload_all.parquet", index=False)
    W0 = w0(X); pid, gd = X.player_id.values, X.game_date.values
    y_all = C.label_within(pid, gd, L["all"]); y_sh = C.label_within(pid, gd, L["shoulder"])
    d_all = C.index_mask(pid, gd, L["all"]); d_sh = C.index_mask(pid, gd, L["shoulder"])
    contrast("A4.1", "all", X, y_all, NA, N4, "all games, every-appearance workload vs paper workload")
    contrast("A4.1", "all", X, y_all, NA + W0 + M3, NA, "all games")
    contrast("A4.1", "all", X, y_all, NA + W0 + M3, NA, "last outing removed", mask=~d_all)
    contrast("A4.1", "shoulder", X, y_sh, NA + W0, NA, "all games")
    contrast("A4.1", "shoulder", X, y_sh, NA + W0, NA, "last outing removed", mask=~d_sh)
    log("A4.1 done", round(time.time() - t), "s")
    return X


def a42(X, L):
    W0 = w0(X); pid, gd = X.player_id.values, X.game_date.values
    for g in (7, 14):
        for lab, feats in (("all", W0 + M3), ("shoulder", W0)):
            y, exc = label_window(pid, gd, L[lab], g, g + 30)
            contrast("A4.2", lab, X, y, N4 + feats, N4, f"lead time {g} days", mask=~exc,
                     extra=dict(rows_excluded=int(exc.sum())))


def classes(P):
    P = P.copy()
    dur = (P.il_end - P.il_start).dt.days
    open_ = P.il_end.isna()
    P["long"] = (dur >= 30) | (open_ & ((FEED_END - P.il_start).dt.days >= 30))
    dx = P.dx.fillna("")
    P["struct"] = dx.str.contains(STRUCT)
    P["nonspec"] = dx.str.contains(NONSPEC) & ~P.struct
    return P


def a43(X, P):
    W0 = w0(X); pid, gd = X.player_id.values, X.game_date.values
    PC = classes(P)
    st_any = starts(PC)
    y_any = C.label_within(pid, gd, st_any)
    for cls, tag in (("long", "stint >= 30 days"), ("struct", "structural diagnosis text"),
                     ("nonspec", "nonspecific diagnosis text")):
        stc = starts(PC[PC[cls]])
        y = C.label_within(pid, gd, stc)
        exc = (y_any == 1) & (y == 0)
        drop = C.index_mask(pid, gd, stc)
        ex = dict(rows_excluded_competing=int(exc.sum()), placements_in_class=int(PC[cls].sum()))
        contrast("A4.3", tag, X, y, N4 + W0 + M3, N4, "all games", mask=~exc, extra=ex)
        contrast("A4.3", tag, X, y, N4 + W0 + M3, N4, "last outing removed", mask=~exc & ~drop, extra=ex)


def primary_preds(X, L):
    """The paper's composite and shoulder OOF predictions (all games and removed)."""
    W0 = w0(X); pid, gd = X.player_id.values, X.game_date.values
    out = {}
    for lab, feats in (("all", W0 + M3), ("shoulder", W0)):
        y = C.label_within(pid, gd, L[lab]); drop = C.index_mask(pid, gd, L[lab])
        for grid, mask in (("all games", None), ("last outing removed", ~drop)):
            pa, _ = C.oof(X, N4 + feats, y, 2, mask); pb, _ = C.oof(X, N4, y, 2, mask)
            ok = np.isfinite(pa) & np.isfinite(pb)
            if mask is not None:
                ok &= mask
            out[(lab, grid)] = (y, pa, pb, ok)
    return out


def a44_a45(X, PRED):
    cl = X.player_id.values
    era1 = X.season.values <= 2021
    for (lab, grid), (y, pa, pb, ok) in PRED.items():
        subset_dauc("A4.4", lab, grid, y, pa, pb, ok, era1, cl, "OOF seasons 2017-2021 (10-day IL)")
        subset_dauc("A4.4", lab, grid, y, pa, pb, ok, ~era1, cl, "OOF seasons 2022-2026 (15-day IL)")
        if lab == "all":
            st = X.role_starter.values.astype(bool)
            subset_dauc("A4.5", lab, grid, y, pa, pb, ok, st, cl, "starters")
            subset_dauc("A4.5", lab, grid, y, pa, pb, ok, ~st, cl, "relievers")


def a46(X, PRED):
    out = {}
    for key in (("all", "all games"), ("all", "last outing removed"), ("shoulder", "all games")):
        y, pa, pb, ok = PRED[key]
        s = X.season.values
        seasons = [t for t in sorted(np.unique(s[ok])) if y[ok & (s == t)].sum() >= 5]
        per = {}
        for t in seasons:
            m = ok & (s == t)
            per[int(t)] = dict(n=int(m.sum()), n_pos=int(y[m].sum()), dauc=C.auc(y[m], pa[m]) - C.auc(y[m], pb[m]))
        w = np.array([per[t]["n_pos"] for t in per], float); dv = np.array([per[t]["dauc"] for t in per])
        est = float((w * dv).sum() / w.sum())
        # cluster bootstrap of the weighted mean of within-season gains
        rng = np.random.default_rng(C.SEED)
        cl = X.player_id.values
        u, inv = np.unique(cl[ok], return_inverse=True)
        rows_ok = np.where(ok)[0]
        grp = [rows_ok[inv == k] for k in range(len(u))]
        pre = []
        for t in seasons:
            m = np.where(ok & (s == t))[0]
            yy = y[m]
            oa, ob = np.argsort(pa[m], kind="stable"), np.argsort(pb[m], kind="stable")
            pre.append((m, yy, oa, ob, yy[oa], yy[ob]))
        reps = []
        for _ in range(C.B):
            take = np.concatenate([grp[k] for k in rng.integers(0, len(u), len(u))])
            mlt = np.bincount(take, minlength=len(y))
            vals, ws = [], []
            for m, yy, oa, ob, ya, yb in pre:
                mm = mlt[m]
                npos = (mm * yy).sum()
                if npos == 0 or (mm * (1 - yy)).sum() == 0:
                    continue
                vals.append(C._auc_w(oa, ya, mm) - C._auc_w(ob, yb, mm)); ws.append(npos)
            reps.append(np.average(vals, weights=ws))
        reps = np.array(reps)
        out[f"{key[0]}|{key[1]}"] = dict(by_season=per, weighted_mean=est, ci_lo=float(np.quantile(reps, .025)),
                                         ci_hi=float(np.quantile(reps, .975)),
                                         p=max(float(2 * min((reps <= 0).mean(), (reps >= 0).mean())), 1 / C.B))
        log("A4.6", key, round(est, 4), round(out[f"{key[0]}|{key[1]}"]["ci_lo"], 4), round(out[f"{key[0]}|{key[1]}"]["ci_hi"], 4))
    return out


def a47(X, L, PRED):
    out = {}
    Xp = pd.read_parquet(C.OUT / "pitcher_primary_pre_features.parquet")
    Xp["game_date"] = pd.to_datetime(Xp.game_date)
    Xp = Xp.sort_values(["season", "game_date"], kind="stable").reset_index(drop=True)
    W0p = w0(Xp)
    yp = C.label_within(Xp.player_id.values, Xp.game_date.values, L["all"])
    pa, _ = C.oof(Xp, N4 + W0p, yp, 2); pb, _ = C.oof(Xp, N4, yp, 2)
    okp = np.isfinite(pa) & np.isfinite(pb)
    items = {"P1 PRE (features through the previous outing)": (yp[okp], pa[okp], pb[okp], Xp.player_id.values[okp])}
    for key, nm in ((("all", "last outing removed"), "composite, last outing removed"),
                    (("shoulder", "last outing removed"), "shoulder kinematics, last outing removed")):
        y, a, b, ok = PRED[key]
        items[nm] = (y[ok], a[ok], b[ok], X.player_id.values[ok])
    for nm, (y, a, b, cl) in items.items():
        reps = boot_reps(y, a, b, cl)
        est = C.auc(y, a) - C.auc(y, b)
        lo90, hi90 = float(np.quantile(reps, .05)), float(np.quantile(reps, .95))
        out[nm] = dict(dauc=est, ci95=[float(np.quantile(reps, .025)), float(np.quantile(reps, .975))], ci90=[lo90, hi90],
                       equivalent_within_0_005=bool(lo90 > -0.005 and hi90 < 0.005),
                       share_reps_ge_0_005=float((reps >= 0.005).mean()), share_reps_le_m0_005=float((reps <= -0.005).mean()))
        log("A4.7", nm, {k: (np.round(v, 4) if not isinstance(v, bool) else v) for k, v in out[nm].items()})
    return out


def primary_fastball_panel():
    import pyarrow.parquet as pq
    sys.argv = [sys.argv[0], "primary"]
    from twin import pitcher as PI
    cols = ["game_date", "game_pk", "game_type", "pitcher", "pitch_type", "release_speed", "release_spin_rate",
            "release_extension", "arm_angle", "release_pos_x", "release_pos_z"]
    parts = []
    for y in range(2015, 2027):
        t = pq.read_table(C.DATA / f"statcast_season/statcast_{y}.parquet", columns=cols).to_pandas()
        t = t[t.game_type == "R"]
        key = ["pitcher", "game_pk", "game_date"]
        tot = t.groupby(key).size().rename("total_pitches")
        cnt = t[t.pitch_type.isin(["FF", "SI"])].groupby(key + ["pitch_type"]).size().unstack(fill_value=0)
        for c in ("FF", "SI"):
            if c not in cnt.columns:
                cnt[c] = 0
        g = pd.concat([tot, cnt[["FF", "SI"]]], axis=1).fillna(0).reset_index()
        g["season"] = y
        g["game_date"] = pd.to_datetime(g.game_date)
        g = g.sort_values(["pitcher", "game_date", "game_pk"])
        q = g[(g.total_pitches >= 20) & (g.FF + g.SI >= 5)]
        first5 = q.groupby("pitcher").head(5).groupby("pitcher")[["FF", "SI"]].sum()
        prim = pd.Series(np.where(first5.SI > first5.FF, "SI", "FF"), index=first5.index)
        t = t[t.pitcher.isin(prim.index)]
        t = t[t.pitch_type == t.pitcher.map(prim)]
        a = t.groupby(key).agg(pitches=("release_speed", "size"), velocity=("release_speed", "mean"),
                               spin_rate=("release_spin_rate", "mean"), release_extension=("release_extension", "mean"),
                               arm_angle=("arm_angle", "mean"), release_pos_x=("release_pos_x", "mean"),
                               release_pos_z=("release_pos_z", "mean"))
        a = a.join(tot, how="left").reset_index()
        a["season"] = y
        a["primary_type"] = a.pitcher.map(prim)
        parts.append(a)
    o = pd.concat(parts, ignore_index=True).rename(columns={"pitcher": "player_id"})
    o = o[(o.total_pitches >= 20) & (o.pitches >= 5)].copy()
    o["abs_release_pos_x"] = o.release_pos_x.abs()
    o["ff_share"] = o.pitches / o.total_pitches
    o["game_date"] = pd.to_datetime(o.game_date)
    o = o.sort_values(["player_id", "season", "game_date", "game_pk"]).reset_index(drop=True)
    X = PI.build(o, "post")
    X = X[X.game_date <= pd.Timestamp("2026-07-12")].sort_values(["season", "game_date"]).reset_index(drop=True)
    ps = o.drop_duplicates(["player_id", "season"]).primary_type.value_counts().to_dict()
    return X, ps, int(len(o)), int(o.player_id.nunique())


def a48(L):
    X, ps, n_o, n_p = primary_fastball_panel()
    X.to_parquet(A4_OUT / "a4_primary_fastball_features.parquet", index=False)
    log("A4.8 panel outings", n_o, "pitchers", n_p, "pitcher-seasons by primary type", ps, "scored rows", len(X))
    W0 = w0(X); pid, gd = X.player_id.values, X.game_date.values
    y_all = C.label_within(pid, gd, L["all"]); y_sh = C.label_within(pid, gd, L["shoulder"])
    d_all = C.index_mask(pid, gd, L["all"]); d_sh = C.index_mask(pid, gd, L["shoulder"])
    ex = dict(panel_outings=n_o, panel_pitchers=n_p, pitcher_seasons_FF=int(ps.get("FF", 0)), pitcher_seasons_SI=int(ps.get("SI", 0)))
    contrast("A4.8", "all", X, y_all, N4 + W0 + M3, N4, "all games", extra=ex)
    contrast("A4.8", "all", X, y_all, N4 + W0 + M3, N4, "last outing removed", mask=~d_all, extra=ex)
    contrast("A4.8", "shoulder", X, y_sh, N4 + W0, N4, "all games", extra=ex)
    contrast("A4.8", "shoulder", X, y_sh, N4 + W0, N4, "last outing removed", mask=~d_sh, extra=ex)


def _wmetrics(y, p, w):
    o = np.argsort(-p, kind="stable"); ys, ws, ps = y[o], w[o], p[o]
    tp = np.cumsum(ws * ys); fp = np.cumsum(ws * (1 - ys)); P_, N_ = tp[-1], fp[-1]
    prec = tp / np.maximum(tp + fp, 1e-12)
    ap = float(np.sum(prec * ws * ys) / P_)
    tpr = tp / P_; fpr = fp / N_
    # partial AUC to FPR 0.10, McClish standardised
    fpr0 = np.concatenate([[0], fpr]); tpr0 = np.concatenate([[0], tpr])
    k = np.searchsorted(fpr0, 0.1, side="right")
    xs = np.concatenate([fpr0[:k], [0.1]]); ys_ = np.concatenate([tpr0[:k], [np.interp(0.1, fpr0, tpr0)]])
    pa = float(np.trapezoid(ys_, xs)); mn, mx = 0.1 ** 2 / 2, 0.1
    spa = 0.5 * (1 + (pa - mn) / (mx - mn))
    sens90 = float(np.interp(0.1, fpr0, tpr0))
    brier = float(np.sum(w * (p - y) ** 2) / w.sum())
    return ap, spa, sens90, brier


def a49(X, PRED):
    out = {}
    for key, (y, pa, pb, ok) in PRED.items():
        yy, a, b, cl = y[ok].astype(float), pa[ok], pb[ok], X.player_id.values[ok]
        w1 = np.ones(len(yy))
        ma, mb = _wmetrics(yy, a, w1), _wmetrics(yy, b, w1)
        est = [ma[0] - mb[0], ma[1] - mb[1], ma[2] - mb[2], mb[3] - ma[3]]
        rng = np.random.default_rng(C.SEED)
        u, inv = np.unique(cl, return_inverse=True)
        grp = [np.where(inv == k)[0] for k in range(len(u))]
        reps = []
        for _ in range(1000):
            take = np.concatenate([grp[k] for k in rng.integers(0, len(u), len(u))])
            w = np.bincount(take, minlength=len(yy)).astype(float)
            xa, xb = _wmetrics(yy, a, w), _wmetrics(yy, b, w)
            reps.append([xa[0] - xb[0], xa[1] - xb[1], xa[2] - xb[2], xb[3] - xa[3]])
        reps = np.array(reps)
        names = ["average_precision_gain", "std_partial_auc_fpr10_gain", "sensitivity_at_90spec_gain", "brier_improvement"]
        out[f"{key[0]}|{key[1]}"] = {nm: dict(model=float(ma[i]) if i < 3 else float(ma[3]), workload=float(mb[i]) if i < 3 else float(mb[3]),
                                              est=float(est[i]), lo=float(np.quantile(reps[:, i], .025)),
                                              hi=float(np.quantile(reps[:, i], .975))) for i, nm in enumerate(names)}
        out[f"{key[0]}|{key[1]}"]["base_rate"] = float(yy.mean())
        log("A4.9", key, {nm: (round(v["est"], 4), round(v["lo"], 4), round(v["hi"], 4)) for nm, v in out[f"{key[0]}|{key[1]}"].items() if isinstance(v, dict)})
    return out


def a410(X, P):
    W0 = w0(X); pid, gd = X.player_id.values, X.game_date.values
    st = starts(P[P.site == "ucl_graft"])
    y = C.label_within(pid, gd, st); drop = C.index_mask(pid, gd, st)
    contrast("A4.10", "ucl_graft", X, y, N4 + W0, N4, "all games")
    contrast("A4.10", "ucl_graft", X, y, N4 + W0, N4, "last outing removed", mask=~drop)


def a411(X, P):
    """Last-outing velocity deficit relative to the reference, within subsets of placements (h = 30)."""
    PC = classes(P)
    T = PC.groupby(["mlbam", "il_start"]).date.min()
    Xs = X.sort_values(["player_id", "season", "game_date"]).reset_index(drop=True)
    pid, dts, z = Xs.player_id.values, Xs.game_date.values, Xs.z_velocity.values.astype(float)
    seasons = Xs.season.values
    st_all = starts(PC)
    within60 = np.zeros(len(Xs), bool)
    rows_by = {}
    for i, p in enumerate(pid):
        rows_by.setdefault(p, []).append(i)
    rec = []
    info = PC.drop_duplicates(["mlbam", "il_start"]).set_index(["mlbam", "il_start"])
    for p, a in st_all.items():
        rows = rows_by.get(p)
        if not rows:
            continue
        rows = np.array(rows)
        for s0 in a:
            d = (s0 - dts[rows]) / np.timedelta64(1, "D")
            within60[rows[(d > 0) & (d <= 60)]] = True
            before = rows[d > 0]
            if not len(before):
                continue
            last = before[-1]
            gap = (s0 - dts[last]) / np.timedelta64(1, "D")
            if gap > 30 or not np.isfinite(z[last]):
                continue
            r = info.loc[(p, pd.Timestamp(s0))]
            t = T.get((p, pd.Timestamp(s0)))
            rec.append(dict(pid=p, z=z[last], season=int(seasons[last]), struct=bool(r.struct), nonspec=bool(r.nonspec),
                            starter=bool(Xs.role_starter.values[last]) if "role_starter" in Xs else None,
                            days_txn=float((t - pd.Timestamp(dts[last])).days) if t is not None else np.nan))
    R = pd.DataFrame(rec)
    ref = float(z[~within60 & np.isfinite(z)].mean())

    def summ(d):
        u = d.pid.unique(); idx = {q: d.z.values[d.pid.values == q] for q in u}
        rng = np.random.default_rng(C.SEED)
        bs = [np.concatenate([idx[q] for q in rng.choice(u, len(u))]).mean() for _ in range(1000)]
        return dict(n=int(len(d)), n_pitchers=int(len(u)), k0_minus_ref=float(d.z.mean() - ref),
                    lo=float(np.quantile(bs, .025) - ref), hi=float(np.quantile(bs, .975) - ref))
    out = dict(reference_mean=ref, all=summ(R))
    out["structural"] = summ(R[R.struct]); out["nonspecific"] = summ(R[R.nonspec])
    for lab, m in (("txn<=1", R.days_txn <= 1), ("txn2-3", R.days_txn.between(2, 3)), ("txn4-7", R.days_txn.between(4, 7)),
                   ("txn>7", R.days_txn > 7)):
        out[lab] = summ(R[m])
    out["era2017_2021"] = summ(R[R.season.between(2017, 2021)]); out["era2022_2026"] = summ(R[R.season >= 2022])
    out["era2015_2016"] = summ(R[R.season <= 2016])
    if R.starter.notna().any():
        out["starters"] = summ(R[R.starter == True]); out["relievers"] = summ(R[R.starter == False])  # noqa: E712
    for k, v in out.items():
        if isinstance(v, dict):
            log("A4.11", k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    return out


def main(which):
    h = assert_seal()
    log(f"A4 start {pd.Timestamp.now()} spec {h}")
    P = placements()
    L = dict(all=starts(P), shoulder=starts(P[P.site == "shoulder"]))
    X = pitcher_X("primary")
    res_json = {"spec_sha16": h}
    if "a41" in which or "all" in which or any(w in which for w in ("a44", "a45", "a411")):
        X = a41(X, L) if ("a41" in which or "all" in which) else add_workload_all(X)
    if "a42" in which or "all" in which:
        a42(X, L)
    if "a43" in which or "all" in which:
        a43(X, P)
    PRED = None
    if any(w in which for w in ("a44", "a45", "a46", "a47", "a49", "all")):
        PRED = primary_preds(X, L)
    if "a44" in which or "a45" in which or "all" in which:
        a44_a45(X, PRED)
    if "a46" in which or "all" in which:
        res_json["A4.6"] = a46(X, PRED)
    if "a47" in which or "all" in which:
        res_json["A4.7"] = a47(X, L, PRED)
    if "a49" in which or "all" in which:
        res_json["A4.9"] = a49(X, PRED)
    if "a410" in which or "all" in which:
        a410(X, P)
    if "a411" in which or "all" in which:
        res_json["A4.11"] = a411(X, P)
    if "a48" in which or "all" in which:
        a48(L)
    tag = "_".join(which)
    pd.DataFrame(ROWS).to_csv(A4_OUT / f"a4_contrasts_{tag}.csv", index=False)
    (A4_OUT / f"a4_metrics_{tag}.json").write_text(json.dumps(res_json, indent=1, default=float))
    log(f"A4 end {pd.Timestamp.now()}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["all"])
