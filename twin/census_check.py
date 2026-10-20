"""Post hoc census check (2026-10-19, extended 2026-10-20; DEVIATIONS #15). Not preregistered: written after every
preregistered, addendum and A3 result was seen. It refits nothing and changes no committed output.

It asks three things of the public census (out/placements_public.csv):
  1. How many placement records share a player and an IL start date with another record? The census removes a duplicate
     only when the full transaction text matches (twin/census.py), so a re-entered or corrected transaction stays.
  2. Why do the horizon and anatomy outputs of addendum A2 (a2.py horizon and anatomy; the paper's Supplementary Tables S4
     and S5) count shoulder and elbow placements differently by one? The script repeats both selections and asserts
     that it reproduces the committed counts in out/a2_horizon.json and out/a2_index_anatomy.json.
  3. What do the repeated records do to the outcome labels? It rebuilds every 30-day label and last-outing removal flag
     with the twin's own helpers (common.label_within, common.index_mask), and the batter season stop (batter.py), from
     every record as committed, from each stint's first record only, with each probable double posting merged into its
     earlier stint, and with thumb UCL records coded to the hand rather than to the UCL graft family (sites.py).

Feature panels are read from TWIN_OUT (built by `twin.pitcher primary|extended` and `twin.batter primary|extended
public|strict`). Committed inputs are read from TWIN_RESULTS (default: out/). The output goes to CENSUS_CHECK_OUT
(default: out/), as census_check.json.

Usage: python3 -m twin.census_check
"""
import hashlib
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

from twin import common as C

RES = pathlib.Path(os.environ.get("TWIN_RESULTS", C.REPO / "out"))
CHECK_OUT = pathlib.Path(os.environ.get("CENSUS_CHECK_OUT", C.REPO / "out"))
CHECK_OUT.mkdir(parents=True, exist_ok=True)
TW = RES
PANEL = C.OUT / "pitcher_primary_post_features.parquet"
# each panel: its file, and for batters the label cutoff and label set of batter.py
PANELS = {"pitcher_primary": (PANEL, None, None),
          "pitcher_extended": (C.OUT / "pitcher_extended_post_features.parquet", None, None),
          **{f"batter_{w}_{lab}": (C.OUT / f"batter_{w}_{lab}_features.parquet", cov, lab)
             for w, cov in (("primary", "2026-06-30"), ("extended", "2026-09-29")) for lab in ("public", "strict")}}
ELBOW = {"elbow_other", "forearm_flexor", "ulnar_nerve", "ucl_graft"}  # a2.py
H = 30


def sha16(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


missing = [str(f) for f, _, _ in PANELS.values() if not f.exists()]
if missing:
    sys.exit(f"feature panels not found: {missing}; run twin.pitcher and twin.batter first (TWIN_OUT)")
P = pd.read_csv(TW / "placements_public.csv", parse_dates=["il_start", "il_end", "date"])
P["mlbam"] = P.mlbam.astype(int)
X = pd.read_parquet(PANEL)
X["game_date"] = pd.to_datetime(X.game_date)

# ---------------------------------------------------------------- the whole census
g = P.groupby(["mlbam", "il_start"])
size = g.txn_id.transform("size")
multi = P[size > 1]
kinds = dict(same_dx_later_date=0, same_dx_same_date=0, different_dx=0)
n_blank = 0
for _, h in multi.groupby(["mlbam", "il_start"]):
    dx = h.dx.fillna("").str.strip().tolist()
    n_blank += any(x == "" for x in dx)
    if len(set(dx)) > 1:
        kinds["different_dx"] += 1
    elif h.date.nunique() > 1:
        kinds["same_dx_later_date"] += 1
    else:
        kinds["same_dx_same_date"] += 1
n_sites = multi.groupby(["mlbam", "il_start"]).site.nunique()
census = dict(records=int(len(P)), stints=int(g.ngroups), stints_with_two_or_more_records=int(multi.groupby(["mlbam", "il_start"]).ngroups),
              stints_whose_records_carry_different_site_codes=int((n_sites > 1).sum()),
              second_records=int(len(multi) - multi.groupby(["mlbam", "il_start"]).ngroups),
              same_il_end_in_every_such_stint=bool((multi.groupby(["mlbam", "il_start"]).il_end.nunique(dropna=False) == 1).all()),
              kinds=kinds, stints_with_a_record_without_diagnosis=int(n_blank),
              covid_era_blank_records_among_them=int(multi.covid_era_blank.sum()))

# ---------------------------------------------------------------- the 30-day index placements of A2 horizon and anatomy
Pp = P[~P.covid_era_blank]


def starts(d):
    return {int(k): np.sort(v.il_start.values) for k, v in d.groupby("mlbam")}


def index_hits(L):
    """a2.py trajectories_h, k = 0: the last outing before each listed start, within H days, with a finite z."""
    Xs = X.sort_values(["player_id", "season", "game_date"]).reset_index(drop=True)
    pid, dts, z = Xs.player_id.values, Xs.game_date.values, Xs.z_velocity.values.astype(float)
    rows_by = {}
    for i, p in enumerate(pid):
        rows_by.setdefault(p, []).append(i)
    out = []
    for p, a in L.items():
        rows = rows_by.get(p)
        if not rows:
            continue
        rows = np.array(rows)
        for s0 in a:
            before = rows[(s0 - dts[rows]) / np.timedelta64(1, "D") > 0]
            if len(before) and (s0 - dts[before[-1]]) / np.timedelta64(1, "D") <= H and np.isfinite(z[before[-1]]):
                out.append((int(p), pd.Timestamp(s0)))
    return out


s4 = {f: index_hits(starts(d)) for f, d in (("all", Pp), ("shoulder", Pp[Pp.site == "shoulder"]),
                                             ("elbow", Pp[Pp.site.isin(ELBOW)]))}
first = Pp.sort_values(["mlbam", "il_start"]).drop_duplicates(["mlbam", "il_start"]).set_index(["mlbam", "il_start"]).site


def fam(site):
    return "shoulder" if site == "shoulder" else ("elbow" if site in ELBOW else "other")


s5 = pd.Series([fam(first.get(k, "unknown")) for k in s4["all"]]).value_counts().to_dict()
A2H = json.loads((TW / "a2_horizon.json").read_text())["trajectories"]
A2A = json.loads((TW / "a2_index_anatomy.json").read_text())["all_index"]
committed_s4 = {f: A2H[f"pitcher_{f}.h{H}"]["n"][0] for f in ("all", "shoulder", "elbow")}
committed_s5 = A2A["by_family"]
assert {f: len(v) for f, v in s4.items()} == committed_s4, ({f: len(v) for f, v in s4.items()}, committed_s4)
assert s5 == committed_s5, (s5, committed_s5)
idx = pd.DataFrame(s4["all"], columns=["mlbam", "il_start"]).drop_duplicates()
two = Pp.merge(idx, on=["mlbam", "il_start"]).groupby(["mlbam", "il_start"]).filter(lambda d: len(d) > 1)
detail = []
for (m, s), d in two.groupby(["mlbam", "il_start"]):
    sites = list(d.sort_values("date").site)
    detail.append(dict(mlbam=int(m), il_start=str(s.date()), sites=sites,
                       table_s4=dict(shoulder=sum(x == "shoulder" for x in d.site), elbow=sum(x in ELBOW for x in d.site)),
                       table_s5_family=fam(first.get((m, s))), table_s5_count=int(len(d))))
index_h30 = dict(records=len(s4["all"]), starts=int(len(idx)), starts_with_two_records=len(detail),
                 table_s4=committed_s4, table_s5=committed_s5,
                 shoulder_table_s4_minus_table_s5=committed_s4["shoulder"] - committed_s5["shoulder"],
                 elbow_table_s5_minus_table_s4=committed_s5["elbow"] - committed_s4["elbow"],
                 starts_with_two_records_detail=detail)
# ---------------------------------------------------------------- the stints whose records carry different site codes
blank_dx = P.dx.fillna("").str.strip().eq("")
sites_detail = []
for (m, s0), h in multi.groupby(["mlbam", "il_start"]):
    if h.site.nunique() < 2:
        continue
    named = sorted(set(h.site) - {"unspecified"})
    kind = ("two_named_site_codes" if len(named) > 1 else
            "one_record_without_diagnosis" if (blank_dx[h.index] & h.site.eq("unspecified")).any() else "unspecified_text")
    sites_detail.append(dict(mlbam=int(m), il_start=str(s0.date()), sites=sorted(set(h.site)), kind=kind,
                             dx=list(h.sort_values(["date", "txn_id"]).dx.fillna(""))))
kinds_sites = pd.Series([d["kind"] for d in sites_detail]).value_counts().to_dict()
different_site_codes = dict(stints=len(sites_detail), **{k: int(kinds_sites.get(k, 0)) for k in
                            ("one_record_without_diagnosis", "unspecified_text", "two_named_site_codes")},
                            two_named_site_codes_detail=[d for d in sites_detail if d["kind"] == "two_named_site_codes"])
assert different_site_codes["stints"] == census["stints_whose_records_carry_different_site_codes"]
# 11a: the taxonomy codes a thumb UCL as a UCL graft (elbow family), census-wide
THUMB = P.dx.fillna("").str.contains("thumb", case=False) & (P.site == "ucl_graft")
mixed = multi.groupby(["mlbam", "il_start"]).covid_era_blank.agg(lambda b: bool(b.any() and not b.all()))
first_blank = multi.sort_values(["mlbam", "il_start", "date", "txn_id"]).groupby(["mlbam", "il_start"]).covid_era_blank.first()
covid_era_blank_mix = dict(stints_pairing_a_covid_era_blank_with_another_record=int(mixed.sum()),
                           of_which_the_blank_is_the_first_record=int((mixed & first_blank).sum()),
                           of_the_117_stints_whose_records_are_all_covid_era_blanks=int(
                               multi.groupby(["mlbam", "il_start"]).covid_era_blank.all().sum()),
                           census_stints_whose_records_are_all_covid_era_blanks=int(g.covid_era_blank.all().sum()))

# ---------------------------------------------------------------- what the repeated records do to the labels
def family_starts(d, name):
    """Per-family start lists as the twin builds them: pitcher.py label_starts, and batter.py cases with the
    panel's label cutoff and label set (public: by site; strict: by diagnosis text)."""
    mk = lambda e: {int(k): np.sort(v.il_start.values) for k, v in e.groupby("mlbam")}
    _, cov, lab = PANELS[name]
    if cov is None:
        return dict(all=mk(d[~d.covid_era_blank]), shoulder=mk(d[d.site == "shoulder"]), elbow=mk(d[d.site.isin(ELBOW)]))
    b = d[d.msk & (d.il_start <= pd.Timestamp(cov))]
    if lab == "public":
        return dict(oblique=mk(b[b.site == "core_oblique"]), hamstring=mk(b[b.site == "hamstring_quad"]),
                    hand_wrist=mk(b[b.site == "hand_wrist_finger"]))
    dx = b.dx.fillna("").str.lower()
    return dict(oblique=mk(b[(b.site == "core_oblique") & dx.str.contains(r"obliqu|oliqu|intercostal")]),
                hamstring=mk(b[dx.str.contains("hamstring")]), hand_wrist=mk(b[b.site == "hand_wrist_finger"]))


PAN = {}
for name, (f, _, _) in PANELS.items():
    Z = pd.read_parquet(f, columns=["player_id" if name.startswith("pitcher") else "batter", "game_date"])
    PAN[name] = (Z.iloc[:, 0].values, pd.to_datetime(Z.game_date).values)
FULL = {name: family_starts(P, name) for name in PANELS}


def labels_against(alt):
    """Rows of each twin panel whose 30-day label, or last-outing removal flag, would differ if the census held `alt`."""
    res = {}
    for name, (pid, gd) in PAN.items():
        A = family_starts(alt, name)
        for fam, L in FULL[name].items():
            y0, y1 = C.label_within(pid, gd, L), C.label_within(pid, gd, A[fam])
            r0, r1 = C.index_mask(pid, gd, L), C.index_mask(pid, gd, A[fam])
            res[f"{name}.{fam}"] = dict(panel_rows=int(len(pid)), positive_rows=int(y0.sum()),
                                        label_rows_changed=int((y0 != y1).sum()), removed_rows=int(r0.sum()),
                                        removed_rows_changed=int((r0 != r1).sum()))
    return res


def stops_against(alt):
    """Batter-seasons of the twin's batter panels (2023-2026) whose stop, the first musculoskeletal placement of the season
    after which a batter's games are dropped (batter.py), would move if the census held `alt`."""
    res = {}
    for name, cov in (("batter_primary", "2026-06-30"), ("batter_extended", "2026-09-29")):
        bat = set(PAN[f"{name}_public"][0])

        def stop(d):
            c = d[d.msk & (d.il_start <= pd.Timestamp(cov)) & d.mlbam.isin(bat) & d.season.between(2023, 2026)]
            return c.groupby(["mlbam", "season"]).il_start.min()
        a, b = stop(P), stop(alt)
        moved = sorted(k for k in set(a.index) | set(b.index) if a.get(k) != b.get(k))
        res[name] = dict(batter_seasons_with_a_stop=int(len(a)), stops_moved=len(moved),
                         detail=[dict(mlbam=int(m), season=int(y), stop=str(a[(m, y)].date()) if (m, y) in a.index else None,
                                      stop_alt=str(b[(m, y)].date()) if (m, y) in b.index else None) for m, y in moved])
    return res


def total(lab):
    return int(sum(r["label_rows_changed"] + r["removed_rows_changed"] for r in lab.values()))


first_only = P.sort_values(["mlbam", "il_start", "date", "txn_id"]).drop_duplicates(["mlbam", "il_start"])
labels_if_first_record_only = labels_against(first_only)
stops_if_first_record_only = stops_against(first_only)
Q = P.copy()
Q.loc[THUMB, "site"] = "hand_wrist_finger"
thumb_ucl = dict(records_coded_ucl_graft=int(THUMB.sum()),
                 in_the_117_stints=int(THUMB[multi.index].sum()),
                 label_or_removal_rows_changed_if_coded_hand=total(labels_against(Q)))

# ---------------------------------------------------------------- stints that are probably one stint posted twice
St = first_only.sort_values(["mlbam", "il_start"]).copy()
for c in ("il_start", "il_end", "club_id", "dx"):
    St["next_" + c] = St.groupby("mlbam")[c].shift(-1)
norm = lambda x: x.fillna("").str.lower().str.replace(r"[^a-z]", "", regex=True)
gap = (St.next_il_start - St.il_start).dt.days
near = St[gap.between(1, 3) & St.il_end.notna() & (St.next_il_start < St.il_end) & (St.club_id == St.next_club_id)
          & ((norm(St.dx) == norm(St.next_dx)) | (norm(St.dx) == "") | (norm(St.next_dx) == ""))]
P_merged = P.copy()
for m, a, b in zip(near.mlbam.astype(int), near.il_start, near.next_il_start):
    P_merged.loc[(P_merged.mlbam == m) & (P_merged.il_start == b), "il_start"] = a
in_index = set(s4["all"])
near_duplicates = dict(
    pairs=int(len(near)), same_il_end=int((near.il_end == near.next_il_end).sum()),
    pairs_with_both_starts_among_the_table_s4_s5_records=int(sum((int(m), pd.Timestamp(a)) in in_index and (int(m), pd.Timestamp(b)) in in_index
                                                                 for m, a, b in zip(near.mlbam, near.il_start, near.next_il_start))),
    pairs_whose_earlier_stint_has_no_site_code=int((near.site == "unspecified").sum()),
    labels_if_merged=labels_against(P_merged),
    stops_if_merged=stops_against(P_merged),
    detail=[dict(mlbam=int(m), il_starts=[str(a.date()), str(b.date())]) for m, a, b in zip(near.mlbam, near.il_start, near.next_il_start)])
near_duplicates["label_or_removal_rows_changed_if_merged"] = total(near_duplicates["labels_if_merged"])

# ---------------------------------------------------------------- the median IL stint (framing_f2.py, a2.py f2 check)
PIT, BAT = set(PAN["pitcher_primary"][0]), set(PAN["batter_primary_public"][0])


def median_days(d):
    q = d[~d.covid_era_blank & d.il_end.notna()].assign(days=lambda e: (e.il_end - e.il_start).dt.days)

    def of(Q):
        return {"pitcher_all": float(Q[Q.mlbam.isin(PIT)].days.median()),
                "pitcher_shoulder": float(Q[Q.mlbam.isin(PIT) & (Q.site == "shoulder")].days.median()),
                "batter_oblique": float(Q[Q.mlbam.isin(BAT) & ~Q.mlbam.isin(PIT) & (Q.site == "core_oblique")].days.median())}
    return dict(all_seasons=of(q), from_2015=of(q[q.il_start >= "2015-01-01"]))


median_il_days = dict(records=median_days(P), first_record_only=median_days(first_only),
                      near_duplicates_merged=median_days(P_merged))
assert median_il_days["records"]["all_seasons"] == json.loads((TW / "f2_results.json").read_text())["median_il_days"]
assert median_il_days["records"]["from_2015"] == json.loads((TW / "a2_f2_defect_check.json").read_text())["D_2015_on"]

out = dict(
    note="Post hoc, 2026-10-19, extended 2026-10-20 (DEVIATIONS #15). No committed value is refit or changed. Records that "
         "share a player and an IL start date are records of one stint that the census kept because their transaction text "
         "differed (its duplicate rule matches the full text). Outcome labels are built from IL start dates (pitcher.py "
         "label_starts, batter.py cases), so a stint counts once in a label however many records it has, and it enters a "
         "label if any of its records qualifies: a family's label if any record is coded to that family, and the all-cause "
         "label unless every record is a COVID-era blank. Descriptive values include the repeated records. The A2 horizon "
         "output (the paper's Table S4) counts an IL start once for each of its records in the row's family (every record, "
         "for any IL), and the A2 anatomy output (Table S5) gives every record of a start the family of the start's first "
         "record. labels_if_first_record_only counts the panel rows whose label or removal flag would differ if only each "
         "stint's first record (by posting date, then transaction id) were kept, and stops_if_first_record_only the "
         "batter-seasons whose stop (the first musculoskeletal placement of the season, after which a batter's games are "
         "dropped) would move; the stop also takes any record. near_duplicates are stints of one player that start 1 to 3 "
         "days apart, with the same club and the same or a blank diagnosis, the later one starting before the earlier one "
         "ends: probably one stint posted twice, which the census counts as two; the alternative merges the later stint's "
         "records into the earlier stint. thumb_ucl: sites.py codes a thumb UCL injury as a UCL graft (elbow family); the "
         "alternative codes those records to the hand. median_il_days repeats the median IL stint of framing_f2.py (all "
         "seasons) and of the addendum A2 check (from 2015) from every record, as committed, and under both census "
         "alternatives.",
    inputs={"out/placements_public.csv": sha16(TW / "placements_public.csv"),
            **{f.name: sha16(f) for f, _, _ in PANELS.values()},
            "out/a2_horizon.json": sha16(TW / "a2_horizon.json"),
            "out/a2_index_anatomy.json": sha16(TW / "a2_index_anatomy.json"),
            "out/f2_results.json": sha16(TW / "f2_results.json"),
            "out/a2_f2_defect_check.json": sha16(TW / "a2_f2_defect_check.json")},
    census=census, index_h30=index_h30, different_site_codes=different_site_codes, thumb_ucl=thumb_ucl,
    covid_era_blank_mix=covid_era_blank_mix, labels_if_first_record_only=labels_if_first_record_only,
    label_or_removal_rows_changed_if_first_record_only=total(labels_if_first_record_only),
    stops_if_first_record_only=stops_if_first_record_only, near_duplicates=near_duplicates, median_il_days=median_il_days)
(CHECK_OUT / "census_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "note"}, indent=1))
