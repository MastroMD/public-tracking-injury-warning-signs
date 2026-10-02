"""Shared paths, input pins, estimator and inference helpers for the public twin.

Every analysis script imports this module; importing it asserts the sealed preregistration and every
pinned input hash (PREREGISTRATION_PUBLIC_TWIN_2026-09-30.md §2). Exit non-zero on any mismatch.

Set TWIN_DATA to the folder holding the public inputs:
  TWIN_DATA/txns_live.jsonl
  TWIN_DATA/statcast_season/statcast_YYYY.parquet   (2015..2026)
"""
import hashlib
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA = pathlib.Path(os.environ.get("TWIN_DATA", REPO / "data"))
OUT = pathlib.Path(os.environ.get("TWIN_OUT", REPO / "out"))
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260930
B = 2000

PREREG = {"PREREGISTRATION_PUBLIC_TWIN_2026-09-30.md": "48619f924703fe98",
          "PREREGISTRATION_FRAMING_2026-09-30.md": "1898c1f6e60769bd"}
PINS = {"txns_live.jsonl": "f65ea265e77b5735",
        "statcast_season/statcast_2015.parquet": "0194b45414693021",
        "statcast_season/statcast_2016.parquet": "3bf68ba977729bea",
        "statcast_season/statcast_2017.parquet": "51f2e8cf8310a200",
        "statcast_season/statcast_2018.parquet": "5deeeb6c779b217a",
        "statcast_season/statcast_2019.parquet": "aaa20bfc8c30bc04",
        "statcast_season/statcast_2020.parquet": "f82966c8965d5465",
        "statcast_season/statcast_2021.parquet": "6cf746129913c1fb",
        "statcast_season/statcast_2022.parquet": "a013eadc63c547ac",
        "statcast_season/statcast_2023.parquet": "a162588175e5958d",
        "statcast_season/statcast_2024.parquet": "65b82081ea38f1ea",
        "statcast_season/statcast_2025.parquet": "3c686658c711097d",
        "statcast_season/statcast_2026.parquet": "88660e24fe4a7b00"}
SITES_SHA = "bd025c4f2a307bba"


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def assert_pins(only=None):
    bad = []
    for f, s in PREREG.items():
        if sha16(REPO / f) != s:
            bad.append(f)
    if sha16(REPO / "twin" / "sites.py") != SITES_SHA:
        bad.append("sites.py")
    for f, s in PINS.items():
        if only and not any(o in f for o in only):
            continue
        p = DATA / f
        if not p.exists() or sha16(p) != s:
            bad.append(f)
    if bad:
        sys.exit(f"PIN FAIL: {bad}")


MLB_CLUBS = {108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 133, 134, 135, 136,
             137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 158}


# ------------------------------------------------------------------ estimator
def fit_l2_logistic(X, y, C=1.0, iters=100, tol=1e-8):
    """IRLS solving sklearn LogisticRegression(penalty='l2', C=C) with an unpenalized intercept."""
    n, k = X.shape
    Z = np.column_stack([np.ones(n), X])
    w = np.zeros(k + 1)
    pen = np.ones(k + 1) / C
    pen[0] = 0.0

    def obj(wv):
        e = Z @ wv
        return (np.logaddexp(0, e) - y * e).sum() + 0.5 * np.sum(pen * wv * wv)

    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -35, 35)))
        g = Z.T @ (p - y) + pen * w
        H = (Z * np.maximum(p * (1 - p), 1e-10)[:, None]).T @ Z + np.diag(pen)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        f0, t = obj(w), 1.0
        while t > 1e-6 and obj(w - t * step) > f0 - 1e-4 * t * (g @ step):
            t *= 0.5
        w_new = w - t * step
        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w = w_new
    return w


def predict_proba(w, X):
    return 1.0 / (1.0 + np.exp(-np.clip(w[0] + X @ w[1:], -35, 35)))


def oof(X, feats, y, seasons_skip, mask=None):
    """Season-forward OOF: train on seasons < t, predict t. Median-fill and standardise on train."""
    m_all = np.ones(len(X), bool) if mask is None else mask
    pred = np.full(len(X), np.nan)
    folds = []
    for t in sorted(X.season.unique())[seasons_skip:]:
        tr = (X.season < t).values & m_all
        te = (X.season == t).values & m_all
        if tr.sum() < 500 or te.sum() == 0 or y[tr].sum() < 20:
            folds.append((int(t), "skipped", int(tr.sum()), int(y[tr].sum())))
            continue
        A, Bm = X.loc[tr, feats], X.loc[te, feats]
        med = A.median()
        Ai, Bi = A.fillna(med).fillna(0), Bm.fillna(med).fillna(0)
        mu, sd = Ai.mean(), Ai.std(ddof=0).replace(0, 1)
        w = fit_l2_logistic(((Ai - mu) / sd).values, y[tr].astype(float))
        pred[np.where(te)[0]] = predict_proba(w, ((Bi - mu) / sd).values)
        folds.append((int(t), "ran", int(tr.sum()), int(y[tr].sum())))
    return pred, folds


# ------------------------------------------------------------------ AUC and cluster bootstrap
def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 < 5 or n0 < 5:
        return np.nan
    order = np.argsort(s, kind="stable")
    r = np.empty(len(s), float); r[order] = np.arange(1, len(s) + 1)
    ss = s[order]; i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _auc_w(order, y_o, mlt):
    mo = mlt[order]; neg = mo * (1 - y_o); pos = mo * y_o
    n1, n0 = pos.sum(), neg.sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    cnb = np.cumsum(neg) - neg
    return float((np.sum(pos * cnb) + 0.5 * np.sum(pos * neg)) / (n1 * n0))


def boot_dauc(y, s_a, s_b, cl, seed=SEED):
    r = np.random.default_rng(seed)
    u, inv = np.unique(cl, return_inverse=True)
    idx = [np.where(inv == k)[0] for k in range(len(u))]
    oa, ob = np.argsort(s_a, kind="stable"), np.argsort(s_b, kind="stable")
    ya, yb = y[oa], y[ob]; n = len(y); d = np.empty(B)
    for b in range(B):
        take = np.concatenate([idx[k] for k in r.integers(0, len(u), len(u))])
        mlt = np.bincount(take, minlength=n)
        d[b] = _auc_w(oa, ya, mlt) - _auc_w(ob, yb, mlt)
    d = d[np.isfinite(d)]
    return (float(np.quantile(d, .025)), float(np.quantile(d, .975)),
            max(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / B))


def bh(p):
    p = np.asarray(p, float); m = len(p)
    order = np.argsort(p); q = np.empty(m); prev = 1.0
    for rank, i in list(enumerate(order))[::-1]:
        prev = min(prev, p[i] * m / (rank + 1)); q[i] = prev
    return q


def label_within(pids, dates, starts_by_pid, days=30):
    """1 if a start falls in (d, d+days] for that player."""
    lab = np.zeros(len(pids), int)
    for i, (p, d) in enumerate(zip(pids, dates)):
        a = starts_by_pid.get(p)
        if a is None:
            continue
        j = np.searchsorted(a, d, side="right")
        if j < len(a) and (a[j] - d) / np.timedelta64(1, "D") <= days:
            lab[i] = 1
    return lab


def index_mask(pids, dates, starts_by_pid, days=30):
    """Each case's last scored row before each start (<= days before) -> True."""
    drop = np.zeros(len(pids), bool)
    by = {}
    for i, p in enumerate(pids):
        by.setdefault(p, []).append(i)
    for p, a in starts_by_pid.items():
        rows = by.get(p)
        if not rows:
            continue
        rows = np.array(rows); dts = dates[rows]
        o = np.argsort(dts, kind="stable"); rows, dts = rows[o], dts[o]
        for s0 in a:
            j = np.searchsorted(dts, s0, side="left")
            if j > 0 and (s0 - dts[j - 1]) / np.timedelta64(1, "D") <= days:
                drop[rows[j - 1]] = True
    return drop
