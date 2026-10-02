"""Public injured-list census from the MLB Stats API transaction feed (twin prereg §3; framing §F1).

Port of the defect-repaired feed parser used across the MLB Total Research library (defects 2, 3, 4, 8
and the 60-day-transfer diagnosis fill), with three differences, all declared in the prereg:
  * MLB clubs only (a placement whose club is not one of the 30 MLB clubs is dropped);
  * no adjudicated case table or curated override is read, ever;
  * appearances (for the defect-5 start repair, the defect-8 continuation check and first-game-back
    end dates) come from public Statcast pitch-level data, batter or pitcher.

Outputs (OUT/): placements_public.csv, activations_public.csv, census_cascade.json
Usage: python3 -m twin.census
"""
import json
import re

import numpy as np
import pandas as pd

from twin import common as C
from twin.sites import classify_site, HAND_CORRECTIONS

LIST_RE = re.compile(r"(?:injured|disabled) list", re.I)
PLACED_RE = re.compile(r"\bplaced\b", re.I)
ACT_RE = re.compile(r"\b(activated|reinstated)\b", re.I)
RETRO_RE = re.compile(r"retroactive to ([A-Z][a-z]+ \d{1,2}, \d{4})", re.I)
DAYS_RE = re.compile(r"\b(7|10|15|60)[\s-]day\b", re.I)
OTHERLIST_RE = re.compile(r"paternity|bereavement|restricted|reserve list|inactive|suspen|military", re.I)
NONIL_RE = re.compile(r"paternity|bereavement|restricted list|family medical|military|suspend", re.I)
COVID_TEXT_RE = re.compile(r"covid|coronavirus|health\s*(?:and|&)\s*safety", re.I)
TRANSFER_RE = re.compile(r"transferred .* to the 60-day", re.I)
MSK_EXCLUDE = {"illness_other", "unspecified"}


def parse_dx(s):
    m = re.search(r"(?:injured|disabled) list(?:\s*retroactive to [A-Z][a-z]+ \d{1,2}, \d{4})?\.?\s*(.*)$", s, re.I)
    return (m.group(1) if m else "").strip()


def load_feed():
    rows = []
    with open(C.DATA / "txns_live.jsonl") as fh:
        for line in fh:
            t = json.loads(line)
            p = t.get("person") or {}
            rows.append(dict(txn_id=t.get("id"), mlbam=p.get("id"), name=p.get("fullName"),
                             to_team_id=(t.get("toTeam") or {}).get("id"),
                             from_team_id=(t.get("fromTeam") or {}).get("id"),
                             to_team=(t.get("toTeam") or {}).get("name"), date=t.get("date"),
                             effective_date=t.get("effectiveDate"), type_desc=t.get("typeDesc"),
                             description=t.get("description") or ""))
    return pd.DataFrame(rows)


def appearances():
    """Regular-season game dates per player (batter or pitcher) from public Statcast."""
    import pyarrow.parquet as pq
    frames = []
    for y in range(2015, 2027):
        t = pq.read_table(C.DATA / f"statcast_season/statcast_{y}.parquet",
                          columns=["game_date", "game_pk", "game_type", "batter", "pitcher"]).to_pandas()
        t = t[t.game_type == "R"]
        a = pd.concat([t[["game_date", "batter"]].rename(columns={"batter": "mlbam"}),
                       t[["game_date", "pitcher"]].rename(columns={"pitcher": "mlbam"})]).drop_duplicates()
        frames.append(a)
    A = pd.concat(frames).drop_duplicates()
    A["game_date"] = pd.to_datetime(A.game_date)
    return {int(m): np.sort(g.game_date.values) for m, g in A.groupby("mlbam")}


def main():
    C.assert_pins()
    casc = {}
    raw = load_feed()
    casc["raw_records"] = len(raw)
    raw["_hp"] = raw.mlbam.notna()
    d = (raw.sort_values(["txn_id", "_hp"], ascending=[True, False])
         .drop_duplicates("txn_id").drop(columns="_hp").reset_index(drop=True))
    casc["unique_transaction_ids"] = len(d)
    for c in ("date", "effective_date"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    # defect 4: whole-record year typos, repaired from the id sequence
    d = d.sort_values("txn_id").reset_index(drop=True)
    ordn = d.date.map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    nbr = ordn.rolling(401, center=True, min_periods=50).median()
    gap = ordn - nbr
    yrs = (gap / 365.25).round()
    typo = ((gap.abs() > 300) & ((gap - yrs * 365.25).abs() <= 40) & (yrs != 0)).fillna(False)
    for i in d.index[typo]:
        nb = pd.Timestamp.fromordinal(int(nbr[i]))
        for col in ("date", "effective_date"):
            v = d.at[i, col]
            if pd.isna(v):
                continue
            g = (v - nb).days; k = round(g / 365.25)
            if abs(g) > 300 and abs(g - k * 365.25) <= 40 and k != 0:
                try:
                    d.at[i, col] = v.replace(year=v.year - int(k))
                except ValueError:
                    d.at[i, col] = v - pd.DateOffset(years=int(k))
    casc["defect4_year_repairs"] = int(typo.sum())
    sc = d[d.type_desc == "Status Change"].copy()
    pl = d[d.description.str.contains(PLACED_RE) & d.description.str.contains(LIST_RE)].copy()
    casc["raw_IL_placement_records"] = len(pl)
    pl["dx"] = pl.description.apply(parse_dx)
    pl["retro"] = pd.to_datetime(pl.description.str.extract(RETRO_RE, expand=False), format="%B %d, %Y", errors="coerce")
    pl["il_start"] = pl.effective_date.fillna(pl.retro).fillna(pl.date)
    lag = (pl.date - pl.il_start).dt.days
    n3 = 0
    for i in pl.index[lag > 60]:  # defect 3: retroactive date with the wrong year
        r, dte = pl.at[i, "il_start"], pl.at[i, "date"]
        try:
            cand = r.replace(year=dte.year)
        except ValueError:
            cand = pd.NaT
        if pd.notna(cand) and 0 <= (dte - cand).days <= 60:
            pl.at[i, "il_start"] = cand; n3 += 1
        else:
            pl.at[i, "il_start"] = dte
    casc["defect3_retro_year_repairs"] = n3
    pl = pl.sort_values(["mlbam", "il_start", "date"])
    dup = pl.duplicated(subset=["mlbam", "il_start", "description"], keep="first")
    casc["defect2_duplicates_removed"] = int(dup.sum())
    pl = pl[~dup].copy()
    pl["il_len_code"] = pd.to_numeric(pl.description.str.extract(DAYS_RE, expand=False), errors="coerce")
    # 60-day transfer text fills a blank placement diagnosis
    tr = sc[sc.description.str.contains(TRANSFER_RE)].copy()
    tr["dx"] = tr.description.str.replace(r"^.*to the 60-day (?:injured|disabled) list\.?\s*", "", regex=True).str.strip()
    tr = tr[tr.dx.str.len() > 0]
    tr["il_start"] = tr.effective_date.fillna(tr.date)
    pl = pl.sort_values(["mlbam", "il_start"]).reset_index(drop=True)
    byp = {m: g.index.tolist() for m, g in pl.groupby("mlbam")}
    nfill = 0
    for _, r in tr.iterrows():
        cand = [i for i in byp.get(r.mlbam, []) if r.il_start - pd.Timedelta(days=120) <= pl.at[i, "il_start"] <= r.il_start]
        if cand:
            i = max(cand, key=lambda j: pl.at[j, "il_start"])
            if not str(pl.at[i, "dx"]).strip():
                pl.at[i, "dx"] = r.dx; nfill += 1
    casc["blank_dx_filled_from_60day_transfer"] = nfill
    act = sc[sc.description.str.contains(ACT_RE) & ~sc.description.str.contains(OTHERLIST_RE)].copy()
    act = act[["txn_id", "mlbam", "date", "description"]].sort_values(["mlbam", "date"])
    acts = {m: np.sort(g.date.values) for m, g in act.groupby("mlbam")}
    # club: MLB clubs only (toTeam on placements is the placing club)
    pl["club_id"] = pl.to_team_id.fillna(pl.from_team_id)
    casc["placements_non_mlb_club_dropped"] = int((~pl.club_id.isin(C.MLB_CLUBS)).sum())
    pl = pl[pl.club_id.isin(C.MLB_CLUBS)].copy()
    pl["covid"] = pl.description.str.contains(COVID_TEXT_RE) | pl.dx.str.contains(COVID_TEXT_RE)
    pl["nonil"] = pl.description.str.contains(NONIL_RE)
    casc["excluded_covid"] = int(pl.covid.sum()); casc["excluded_nonil"] = int(pl.nonil.sum())
    pl = pl[~pl.covid & ~pl.nonil].copy()
    pl = pl[pl.mlbam.notna()].copy(); pl["mlbam"] = pl.mlbam.astype(int)
    # appearances -> defect 8 (continuation) and defect 5 (start after last appearance)
    APP = appearances()
    pl = pl.sort_values(["mlbam", "il_start"]).reset_index(drop=True)
    pl["continuation"] = False
    byp = {m: g.index.tolist() for m, g in pl.groupby("mlbam")}
    n8 = 0
    for i in pl.index[pl.il_len_code == 60]:
        r = pl.loc[i]
        prior = [j for j in byp.get(r.mlbam, []) if r.il_start - pd.Timedelta(days=120) <= pl.at[j, "il_start"] < r.il_start
                 and not pl.at[j, "continuation"]]
        if not prior:
            continue
        j = max(prior, key=lambda k: pl.at[k, "il_start"])
        a = acts.get(r.mlbam)
        if a is not None and ((a >= np.datetime64(pl.at[j, "il_start"])) & (a <= np.datetime64(r.il_start))).any():
            continue
        g = APP.get(int(r.mlbam))
        if g is not None and ((g > np.datetime64(pl.at[j, "il_start"])) & (g < np.datetime64(r.il_start))).any():
            continue
        pl.at[i, "continuation"] = True; n8 += 1
        if not str(pl.at[j, "dx"]).strip() and str(r.dx).strip():
            pl.at[j, "dx"] = r.dx
    casc["defect8_continuations"] = n8
    pl = pl[~pl.continuation].copy()
    n5 = 0
    for i in pl.index:
        g = APP.get(int(pl.at[i, "mlbam"]))
        if g is None:
            continue
        post, st = pl.at[i, "date"], pl.at[i, "il_start"]
        pg = g[(g <= np.datetime64(post)) & (g >= np.datetime64(st))]
        if len(pg):
            new = pd.Timestamp(pg.max()) + pd.Timedelta(days=1)
            if st < new <= post:
                pl.at[i, "il_start"] = new; n5 += 1
    casc["defect5_start_after_last_appearance"] = n5
    # site (public taxonomy, hand readings keyed on text; no adjudicated override)
    pl["dx"] = pl.dx.fillna("")
    pl["site"] = [classify_site(x) for x in pl.dx]
    for i in pl.index:
        k = pl.at[i, "dx"].lower().strip()
        if k in HAND_CORRECTIONS:
            pl.at[i, "site"] = HAND_CORRECTIONS[k][0]
    pl["msk"] = ~pl.site.isin(MSK_EXCLUDE)
    pl["season"] = pl.il_start.dt.year
    # DEVIATIONS.md #1: inherited covid_era_unattributable flag (blank dx, 2020-2022)
    pl["covid_era_blank"] = pl.dx.str.strip().eq("") & pl.season.isin([2020, 2021, 2022])
    casc["covid_era_blank_flagged"] = int(pl.covid_era_blank.sum())
    # end date: first activation after start validated against first game back
    ends, srcs = [], []
    for r in pl.itertuples():
        g = APP.get(int(r.mlbam))
        fgb = pd.Timestamp(g[g > np.datetime64(r.il_start)][0]) if g is not None and (g > np.datetime64(r.il_start)).any() else pd.NaT
        a = acts.get(r.mlbam)
        nxt = pd.Timestamp(a[a > np.datetime64(r.il_start)][0]) if a is not None and (a > np.datetime64(r.il_start)).any() else pd.NaT
        if pd.notna(nxt) and (pd.isna(fgb) or nxt <= fgb + pd.Timedelta(days=3)):
            ends.append(nxt); srcs.append("activation")
        elif pd.notna(fgb):
            ends.append(fgb); srcs.append("first_game_back")
        else:
            ends.append(pd.NaT); srcs.append("open")
    pl["il_end"] = ends; pl["il_end_src"] = srcs
    casc["placements_final"] = len(pl)
    casc["by_season"] = pl.groupby("season").size().to_dict()
    casc["site_counts"] = pl.site.value_counts().to_dict()
    casc["end_src"] = pl.il_end_src.value_counts().to_dict()
    cols = ["txn_id", "mlbam", "name", "club_id", "date", "il_start", "il_end", "il_end_src", "il_len_code",
            "season", "dx", "site", "msk", "covid_era_blank"]
    pl[cols].to_csv(C.OUT / "placements_public.csv", index=False)
    act.to_csv(C.OUT / "activations_public.csv", index=False)
    json.dump(casc, open(C.OUT / "census_cascade.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in casc.items() if k not in ("site_counts",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
