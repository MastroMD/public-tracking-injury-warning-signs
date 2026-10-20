"""Amendment 1 to post hoc addendum A4 (2026-10-12): A4.3b, A4.5b, A4.6b against the every-appearance workload model.

Specification: POSTHOC_ADDENDUM_A4_AMENDMENT1_2026-10-12.md (hash in SEAL_A4_AMENDMENT1.txt, checked before running).
Reuses a4.py unchanged.
"""
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd

import a4
from a4 import C, M3, NA

C_LOG = a4.log
_auc_w = None
SPEC = a4.HERE / "POSTHOC_ADDENDUM_A4_AMENDMENT1_2026-10-12.md"
SEAL = a4.HERE / "SEAL_A4_AMENDMENT1.txt"


def a46b(X, PRED, keys, tag="A4.6b"):
    out = {}
    for key in keys:
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
        a4.log(tag, key, round(est, 4), round(out[f"{key[0]}|{key[1]}"]["ci_lo"], 4), round(out[f"{key[0]}|{key[1]}"]["ci_hi"], 4))
    return out




def main():
    h = hashlib.sha256(SPEC.read_bytes()).hexdigest()[:16]
    assert h == re.search(r"sha256\[:16\] ([0-9a-f]{16})", SEAL.read_text()).group(1), "amendment hash mismatch"
    a4.assert_seal()
    a4.log(f"A4 amendment 1 start {pd.Timestamp.now()} spec {h}")
    X = pd.read_parquet(a4.A4_OUT / "a4_primary_with_workload_all.parquet")
    X["game_date"] = pd.to_datetime(X.game_date)
    X = X.sort_values(["season", "game_date"], kind="stable").reset_index(drop=True)
    P = a4.placements()
    L = dict(all=a4.starts(P))
    W0 = a4.w0(X); pid, gd = X.player_id.values, X.game_date.values
    # A4.3b
    PC = a4.classes(P)
    y_any = C.label_within(pid, gd, a4.starts(PC))
    stc = a4.starts(PC[PC.nonspec])
    y = C.label_within(pid, gd, stc); exc = (y_any == 1) & (y == 0); drop = C.index_mask(pid, gd, stc)
    a4.contrast("A4.3b", "nonspecific diagnosis text", X, y, NA + W0 + M3, NA, "all games", mask=~exc)
    a4.contrast("A4.3b", "nonspecific diagnosis text", X, y, NA + W0 + M3, NA, "last outing removed", mask=~exc & ~drop)
    # predictions against NA
    y = C.label_within(pid, gd, L["all"]); drop = C.index_mask(pid, gd, L["all"])
    PRED = {}
    for grid, mask in (("all games", None), ("last outing removed", ~drop)):
        pa, _ = C.oof(X, NA + W0 + M3, y, 2, mask); pb, _ = C.oof(X, NA, y, 2, mask)
        ok = np.isfinite(pa) & np.isfinite(pb)
        if mask is not None:
            ok &= mask
        PRED[("all", grid)] = (y, pa, pb, ok)
    # A4.5b
    st = X.role_starter.values.astype(bool); cl = X.player_id.values
    for (lab, grid), (yy, pa, pb, ok) in PRED.items():
        a4.subset_dauc("A4.5b", lab, grid, yy, pa, pb, ok, st, cl, "starters (NA comparator)")
        a4.subset_dauc("A4.5b", lab, grid, yy, pa, pb, ok, ~st, cl, "relievers (NA comparator)")
    # A4.6b
    out = a46b(X, PRED, [("all", "all games"), ("all", "last outing removed")])
    pd.DataFrame(a4.ROWS).to_csv(a4.A4_OUT / "a4_contrasts_amend1.csv", index=False)
    (a4.A4_OUT / "a4_metrics_amend1.json").write_text(json.dumps({"spec_sha16": h, "A4.6b": out}, indent=1, default=float))
    a4.log(f"A4 amendment 1 end {pd.Timestamp.now()}")


if __name__ == "__main__":
    main()
