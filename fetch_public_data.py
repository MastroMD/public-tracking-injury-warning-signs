"""Pull the two public inputs the twin uses into ./data (resumable).

  1. MLB Stats API transactions, sportId 1, 2010-01-01 .. END   -> data/txns_live.jsonl (one JSON record per line)
  2. Baseball Savant pitch-level search export, one request per game day, all game types
     (regular season, postseason, spring) 2015-03-01 .. END      -> data/statcast/YYYY-MM-DD.parquet
     then one file per season                                    -> data/statcast_season/statcast_YYYY.parquet

Usage: python3 fetch_public_data.py [--end 2026-09-29] [--workers 4]
Days already on disk are skipped. The feeds are revised over time, so a fresh pull will not match the hashes
pinned in twin/common.py byte for byte; update the pins to your pull (the analysis asserts them).
"""
import argparse
import datetime as dt
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SC, SEAS = os.path.join(DATA, "statcast"), os.path.join(DATA, "statcast_season")
for d in (DATA, SC, SEAS):
    os.makedirs(d, exist_ok=True)
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (research; public-twin)"})
KEEP = ["game_date", "game_pk", "game_type", "game_year", "home_team", "away_team", "inning", "pitcher", "player_name",
        "p_throws", "batter", "stand", "pitch_type", "pitch_name", "release_speed", "effective_speed", "pfx_x", "pfx_z",
        "api_break_x_arm", "api_break_x_batter_in", "api_break_z_with_gravity", "release_spin_rate", "spin_axis",
        "release_pos_x", "release_pos_y", "release_pos_z", "release_extension", "arm_angle", "at_bat_number",
        "pitch_number", "n_thruorder_pitcher", "pitcher_days_since_prev_game", "pitcher_days_until_next_game",
        "description", "events", "type", "launch_speed", "launch_angle", "launch_speed_angle", "bb_type", "zone",
        "plate_x", "plate_z", "sz_top", "sz_bot", "estimated_woba_using_speedangle", "woba_value", "woba_denom",
        "bat_speed", "swing_length", "balls", "strikes", "outs_when_up", "inning_topbot",
        "estimated_ba_using_speedangle", "hit_distance_sc", "delta_run_exp"]


def get(url, tries=5):
    for i in range(tries):
        try:
            r = S.get(url, timeout=240)
            if r.status_code == 200:
                return r.text
            time.sleep(20 * (i + 1) if r.status_code in (403, 429) else 5)
        except requests.RequestException:
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"GET failed: {url[:120]}")


def pull_transactions(end):
    out = os.path.join(DATA, "txns_live.jsonl")
    cur = dt.date(2010, 1, 1)
    with open(out, "w") as fh:
        while cur <= end:
            nxt = min((cur.replace(day=1) + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1), end)
            u = f"https://statsapi.mlb.com/api/v1/transactions?startDate={cur}&endDate={nxt}&sportId=1"
            for t in json.loads(get(u)).get("transactions", []):
                t["_win"] = f"{cur}_{nxt}"
                fh.write(json.dumps(t) + "\n")
            cur = nxt + dt.timedelta(days=1)


def sc_url(day, home_road="", gt="R%7CPO%7CS%7C"):
    return ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium="
            "&hfBBL=&hfNewZones=&hfGT=" + gt + "=&hfSea=&hfSit=&player_type=pitcher&hfOuts=&opponent="
            f"&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={day}&game_date_lt={day}&team=&position=&hfRO="
            f"&home_road={home_road}&hfFlag=&metric_1=&hfInn=&min_pitches=0&min_results=0&group_by=name&sort_col=pitches"
            "&player_event_sort=h_launch_speed&sort_order=desc&min_abs=0&type=details&")


def parse(txt):
    return pd.read_csv(io.StringIO(txt), low_memory=False) if txt and len(txt) > 50 else pd.DataFrame()


def pull_day(day):
    out = os.path.join(SC, f"{day}.parquet")
    if os.path.exists(out):
        return
    df = parse(get(sc_url(day)))
    if len(df) >= 20000:  # near Savant's row cap: rebuild from home/road x game-type slices
        parts = [parse(get(sc_url(day, hr, gt))) for hr in ("Home", "Road") for gt in ("R%7C", "PO%7C", "S%7C")]
        df = pd.concat([p for p in parts if len(p)], ignore_index=True).drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
    for c in KEEP:
        if c not in df.columns:
            df[c] = pd.NA
    df[KEEP].to_parquet(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-09-29")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    end = dt.date.fromisoformat(a.end)
    pull_transactions(end)
    days = [dt.date(y, 2, 20) + dt.timedelta(days=i) for y in range(2015, end.year + 1) for i in range(0, 260)]
    days = [d for d in days if d <= end]
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(pull_day, days))
    for y in range(2015, end.year + 1):
        fs = sorted(f for f in os.listdir(SC) if f.startswith(str(y)))
        pd.concat([pd.read_parquet(os.path.join(SC, f)) for f in fs], ignore_index=True).to_parquet(
            os.path.join(SEAS, f"statcast_{y}.parquet"), index=False)


if __name__ == "__main__":
    main()
