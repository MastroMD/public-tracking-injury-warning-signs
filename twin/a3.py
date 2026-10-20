"""Addendum A3 (post hoc) — checks raised in review of the full paper.

Specification: POSTHOC_ADDENDUM_A3_2026-10-02.md (sha256[:16] 3a7d6ea5251bfb88, SEAL_A3.txt), hashed before this code
first ran. Everything here is post hoc and cannot change a preregistered verdict.

Feature panels are read from TWIN_OUT (built by `twin.pitcher primary|extended`). Committed inputs (placements) are read
from TWIN_RESULTS (default: out/). Outputs go to A3_OUT (default: out/).

Usage:
  python3 -m twin.a3 contrasts     # A3.1-A3.3, A3.4b -> a3_contrasts.csv
  python3 -m twin.a3 gaps          # A3.4a -> a3_gaps.json
  python3 -m twin.a3 refit [B]     # A3.5 -> a3_refit_bootstrap.json, a3_refit_replicates.csv (B = 1000 as specified)
"""
import json
import os
import pathlib
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from twin import common as C
from twin.sites import ARM_SITES

SPEC_A3 = "3a7d6ea5251bfb88"
SPEC_FILE = "POSTHOC_ADDENDUM_A3_2026-10-02.md"
RES = pathlib.Path(os.environ.get("TWIN_RESULTS", C.REPO / "out"))
A3_OUT = pathlib.Path(os.environ.get("A3_OUT", C.REPO / "out"))
A3_OUT.mkdir(parents=True, exist_ok=True)
N4 = ["cov_pitches_outing", "cov_days_rest", "cov_cum_pitches", "cov_n_out_so_far"]
N3 = N4[1:]
N4R = N4 + ["rel_pitches"]
M3 = ["m3_streak_low_velo", "m3_rest_dev", "m3_z_velo_x_rest"]
V = ["z_velocity"]
ELBOW = {"elbow_other", "forearm_flexor", "ulnar_nerve", "ucl_graft"}
GATE = 30
REFIT_SEED = 20261002
REFIT_B = 1000


def assert_seal():
    assert C.sha16(C.REPO / SPEC_FILE) == SPEC_A3, "A3 specification hash mismatch"


def dump(obj, name):
    (A3_OUT / name).write_text(json.dumps(obj, indent=1, default=float))
    print("wrote", name, flush=True)


def placements():
    P = pd.read_csv(RES / "placements_public.csv", parse_dates=["il_start", "il_end", "date"])
    P["mlbam"] = P.mlbam.astype(int)
    return P[~P.covid_era_blank].copy()


def starts(d):
    return {int(k): np.sort(g.il_start.values) for k, g in d.groupby("mlbam")}


def labels(P):
    return dict(all=starts(P), shoulder=starts(P[P.site == "shoulder"]), elbow=starts(P[P.site.isin(ELBOW)]),
                arm=starts(P[P.site.isin(ARM_SITES)]), nonarm=starts(P[~P.site.isin(ARM_SITES)]))


def pitcher_X(window="primary"):
    X = pd.read_parquet(C.OUT / f"pitcher_{window}_post_features.parquet")
    X["game_date"] = pd.to_datetime(X.game_date)
    X = X.sort_values(["season", "game_date"], kind="stable").reset_index(drop=True)
    X["rel_pitches"] = X.cov_pitches_outing / (X.cov_cum_pitches / X.cov_n_out_so_far)
    return X


def w0(X):
    return [c for c in X.columns if c.startswith("z_")] + ["release_drift"]


def bar_text(full, abl):
    """Escalation bar as a description only (A3 never issues a verdict)."""
    s1 = full["dauc"] >= 0.01 and full["ci_lo"] > 0
    if abl is None:
        return "would meet stage 1" if s1 else "would not meet stage 1"
    s2 = abl["dauc"] >= 0.005 and abl["ci_lo"] > 0
    return "would meet" if (s1 and s2) else "would not meet"


def contrast(cid, label, X, y, fa, fb, grid, mask=None, window="primary"):
    pa, _ = C.oof(X, fa, y, 2, mask)
    pb, _ = C.oof(X, fb, y, 2, mask)
    ok = np.isfinite(pa) & np.isfinite(pb)
    if mask is not None:
        ok &= mask
    yy, cl = y[ok], X.player_id.values[ok]
    npc = int(pd.unique(cl[yy == 1]).size)
    rec = dict(id=cid, label=label, window=window, grid=grid, model="+".join(fa), comparator="+".join(fb),
               n=int(ok.sum()), n_pos=int(yy.sum()), n_clusters=int(pd.unique(cl).size), n_pos_clusters=npc)
    if npc < GATE:
        rec.update(status="closed (gate)")
        print(cid, grid, "closed", npc, flush=True)
        return rec
    aa, ab = C.auc(yy, pa[ok]), C.auc(yy, pb[ok])
    lo, hi, p = C.boot_dauc(yy, pa[ok], pb[ok], cl)
    rec.update(status="run", auc_a=aa, auc_b=ab, dauc=aa - ab, ci_lo=lo, ci_hi=hi, p=p)
    print(cid, label, grid, window, {k: round(rec[k], 4) for k in ("dauc", "ci_lo", "ci_hi", "p")}, flush=True)
    return rec


def contrasts():
    P = placements()
    L = labels(P)
    X = pitcher_X("primary")
    W0 = w0(X)
    P2 = W0 + M3
    pid, gd = X.player_id.values, X.game_date.values
    Y = {k: C.label_within(pid, gd, v) for k, v in L.items()}
    D = {k: C.index_mask(pid, gd, v) for k, v in L.items()}
    rows = []

    def both(cid, lab, fa, fb, Xw=X, Yw=Y, Dw=D, window="primary"):
        f = contrast(cid, lab, Xw, Yw[lab], fa, fb, "all games", None, window)
        a = contrast(cid, lab, Xw, Yw[lab], fa, fb, "last outing removed", ~Dw[lab], window)
        desc = bar_text(f, a) if f.get("status") == "run" and a.get("status") == "run" else "n/a"
        f["bar"] = a["bar"] = desc
        rows.extend([f, a])

    def removed_only(cid, lab, fa, fb):
        a = contrast(cid, lab, X, Y[lab], fa, fb, "last outing removed", ~D[lab])
        a["bar"] = "stage 2 " + ("would be met" if a.get("status") == "run" and a["dauc"] >= 0.005 and a["ci_lo"] > 0
                                 else "would not be met")
        rows.append(a)

    # A3.1 dependence on the scored outing's pitch count
    both("A3.1a", "all", N3 + P2, N3)
    both("A3.1b", "all", N4, N3)
    both("A3.1c", "all", N4R, N4)
    both("A3.1d", "all", N4R + P2, N4R)
    both("A3.1e", "shoulder", N3 + W0, N3)
    # A3.2 removal-test coverage
    both("A3.2a", "shoulder", N4 + V, N4)
    Xe = pitcher_X("extended")
    pe, ge = Xe.player_id.values, Xe.game_date.values
    both("A3.2a", "shoulder", N4 + V, N4, Xw=Xe, Yw={"shoulder": C.label_within(pe, ge, L["shoulder"])},
         Dw={"shoulder": C.index_mask(pe, ge, L["shoulder"])}, window="extended")
    both("A3.2b", "all", N4 + V, N4)
    both("A3.2c", "elbow", N4 + V, N4)
    removed_only("A3.2d", "all", N4 + W0, N4)
    removed_only("A3.2e", "elbow", N4 + W0, N4)
    # A3.3 arm and non-arm labels
    both("A3.3a", "arm", N4 + W0, N4)
    both("A3.3b", "arm", N4 + P2, N4)
    both("A3.3c", "nonarm", N4 + W0, N4)
    both("A3.3d", "nonarm", N4 + P2, N4)
    # A3.4b removal only when the club's transaction came within 3 days of the last outing
    m3d = index_mask_txn(X, P, days_txn=3)
    a = contrast("A3.4b", "all", X, Y["all"], N4 + P2, N4, "last outing removed if transaction within 3 days", ~m3d)
    a["n_rows_removed"] = int(m3d.sum())
    a["n_rows_removed_standard"] = int(D["all"].sum())
    a["bar"] = "stage 2 " + ("would be met" if a.get("status") == "run" and a["dauc"] >= 0.005 and a["ci_lo"] > 0
                             else "would not be met")
    rows.append(a)
    df = pd.DataFrame(rows)
    df.to_csv(A3_OUT / "a3_contrasts.csv", index=False)
    print(df.drop(columns=["model", "comparator"]).round(4).to_string(index=False), flush=True)


def txn_dates(P):
    return P.groupby(["mlbam", "il_start"]).date.min()


def index_mask_txn(X, P, days_txn=3, days=30):
    """Like common.index_mask, but the last outing is removed only if the transaction came within days_txn of it."""
    L = starts(P)
    T = txn_dates(P)
    pid, gd = X.player_id.values, X.game_date.values
    drop = np.zeros(len(X), bool)
    by = {}
    for i, p in enumerate(pid):
        by.setdefault(p, []).append(i)
    for p, a in L.items():
        rows = by.get(p)
        if not rows:
            continue
        rows = np.array(rows); dts = gd[rows]
        o = np.argsort(dts, kind="stable"); rows, dts = rows[o], dts[o]
        for s0 in a:
            j = np.searchsorted(dts, s0, side="left")
            if j > 0 and (s0 - dts[j - 1]) / np.timedelta64(1, "D") <= days:
                t = T.get((p, pd.Timestamp(s0)))
                if t is not None and (t - pd.Timestamp(dts[j - 1])).days <= days_txn:
                    drop[rows[j - 1]] = True
    return drop


def gaps():
    """A3.4a: same base as A2.5 (non-blank placements after a primary-panel outing within 30 days, finite z)."""
    P = placements()
    L = starts(P)
    T = txn_dates(P)
    X = pitcher_X("primary").sort_values(["player_id", "game_date"]).reset_index(drop=True)
    pid, gd, z = X.player_id.values, X.game_date.values, X.z_velocity.values.astype(float)
    by = {}
    for i, p in enumerate(pid):
        by.setdefault(p, []).append(i)
    rec = []
    for p, a in L.items():
        rows = by.get(p)
        if not rows:
            continue
        rows = np.array(rows); dts = gd[rows]
        for s0 in a:
            j = np.searchsorted(dts, s0, side="left")
            if j == 0:
                continue
            g_start = (s0 - dts[j - 1]) / np.timedelta64(1, "D")
            if g_start > 30:
                continue
            t = T.get((p, pd.Timestamp(s0)))
            g_txn = float((t - pd.Timestamp(dts[j - 1])).days) if t is not None else np.nan
            rec.append(dict(pid=p, start=s0, z=z[rows[j - 1]], days_start=float(g_start), days_txn=g_txn,
                            retro=bool(t is not None and t > pd.Timestamp(s0))))
    R = pd.DataFrame(rec)
    R = R[np.isfinite(R.z)]

    def summ(d):
        out = dict(n=int(len(d)), share_retroactive=float(d.retro.mean()))
        for col, tag in (("days_start", "il_start"), ("days_txn", "transaction")):
            v = d[col].dropna().values
            out[tag] = dict(n=int(len(v)), median=float(np.median(v)), q1=float(np.quantile(v, .25)),
                            q3=float(np.quantile(v, .75)), share_le1=float((v <= 1).mean()),
                            share_le3=float((v <= 3).mean()), share_le7=float((v <= 7).mean()))
        return out

    out = dict(note="A3.4a, post hoc, descriptive. Base as A2.5.", all_index=summ(R), slow_index=summ(R[R.z <= -1]),
               not_slow_index=summ(R[R.z > -1]))
    print(json.dumps(out, indent=1, default=float), flush=True)
    dump(out, "a3_gaps.json")


# ------------------------------------------------------------------ A3.5 refit bootstrap
_G = {}


def _init():
    P = placements()
    L = starts(P)
    X = pitcher_X("primary")
    W0 = w0(X)
    pid, gd = X.player_id.values, X.game_date.values
    _G.update(X=X, y=C.label_within(pid, gd, L), drop=C.index_mask(pid, gd, L), fa=N4 + W0 + M3, fb=N4,
              idx={p: np.where(pid == p)[0] for p in np.unique(pid)})


def _dauc(Xb, yb, mask):
    pa, _ = C.oof(Xb, _G["fa"], yb, 2, mask)
    pb, _ = C.oof(Xb, _G["fb"], yb, 2, mask)
    ok = np.isfinite(pa) & np.isfinite(pb)
    if mask is not None:
        ok &= mask
    return C.auc(yb[ok], pa[ok]) - C.auc(yb[ok], pb[ok])


def _rep(b):
    if not _G:
        _init()
    X, y, drop, idx = _G["X"], _G["y"], _G["drop"], _G["idx"]
    u = np.array(sorted(idx))
    rng = np.random.default_rng([REFIT_SEED, b])
    take = np.concatenate([idx[p] for p in rng.choice(u, len(u), replace=True)])
    o = np.argsort(X.season.values[take], kind="stable")
    take = take[o]
    Xb = X.iloc[take].reset_index(drop=True)
    return b, _dauc(Xb, y[take], None), _dauc(Xb, y[take], ~drop[take])


def refit(B=REFIT_B):
    with Pool(2) as pool:
        res = []
        for k, r in enumerate(pool.imap_unordered(_rep, range(B), chunksize=4)):
            res.append(r)
            if (k + 1) % 50 == 0:
                print(k + 1, "replicates", flush=True)
    R = pd.DataFrame(sorted(res), columns=["b", "dauc_full", "dauc_removed"])
    R.to_csv(A3_OUT / "a3_refit_replicates.csv", index=False)
    fixed = pd.read_csv(RES / "pitcher_primary_results.csv").set_index("test")
    out = dict(note="A3.5, post hoc. Player-cluster resamples with the season-forward models refitted in each.",
               B=int(B), seed=REFIT_SEED)
    for col, test in (("dauc_full", "P2"), ("dauc_removed", "P2_ablated")):
        v = R[col].dropna().values
        sd_fixed = float((fixed.loc[test, "ci_hi"] - fixed.loc[test, "ci_lo"]) / 3.92)
        out[test] = dict(n=int(len(v)), mean=float(v.mean()), sd=float(v.std(ddof=1)), lo=float(np.quantile(v, .025)),
                         hi=float(np.quantile(v, .975)), share_le0=float((v <= 0).mean()),
                         share_ge_bar=float((v >= (0.01 if test == "P2" else 0.005)).mean()),
                         sd_fixed=sd_fixed, sd_ratio=float(v.std(ddof=1) / sd_fixed), estimate=float(fixed.loc[test, "dauc"]))
    print(json.dumps(out, indent=1, default=float), flush=True)
    dump(out, "a3_refit_bootstrap.json")


if __name__ == "__main__":
    assert_seal()
    C.assert_pins(only=["__none__"])  # preregistrations and sites.py; data pins are checked when panels are built
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "contrasts":
        contrasts()
    elif cmd == "gaps":
        gaps()
    elif cmd == "refit":
        refit(int(sys.argv[2]) if len(sys.argv) > 2 else REFIT_B)
    else:
        sys.exit(__doc__)
