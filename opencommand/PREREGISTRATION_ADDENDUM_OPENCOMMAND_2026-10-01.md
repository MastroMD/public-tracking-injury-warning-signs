# Preregistration addendum A1 — pitcher command (OpenCommand) as a within-season warning sign

**Written 2026-10-01 22:55 EDT, before any OpenCommand file was opened and before any command feature was
computed.** Extends the sealed twin preregistration (`48619f924703fe98`). sha256[:16] of this file is recorded
in `SEAL_A1.txt` and asserted by `twin/command.py` at load.

## Why

Stuff+, Location+ and Pitching+ (FanGraphs) were tested in the sealed programme and carried no within-season
signal, but they are licensed outputs the public twin cannot reproduce. OpenCommand (Kim, 2026; CC BY-NC-SA 4.0)
publishes per-pitch inferred catcher targets from broadcast video for 2024–2026, so a pitcher's command, the
distance between where the pitch was aimed and where it went, can be tested as a warning sign on open data.
The prior, from the plus-metric nulls, is null.

## Input

`data/<year>_targets.csv.gz` and `data/<year>_pbp_info.csv.gz`, 2024–2026, as downloaded by
`!RUN_OPENCOMMAND_PULL.command`; hashes recorded in `data/SHA16.txt` at download and asserted at load. Joined
to the twin's pitch-level Statcast rows on `game_pk` and `play_id` where available, otherwise on
(`game_pk`, `at_bat_number`, `pitch_number`). No other OpenCommand output is used.

## Feature, fixed here

Per pitch, `miss` = Euclidean distance (inches) from the inferred target to the actual plate location, after
OpenCommand's own plausibility filter. Per outing, `cmd_med` = median miss over all tracked pitches in the
outing, requiring ≥ 10 tracked pitches. Per outing, `z_cmd` = (cmd_med − mean of the pitcher's prior
in-season outings' cmd_med) / their SD, with ≥ 5 prior outings, exactly as the W0 z-scores. Also `d3_cmd`
(mean of the last 3 outings minus baseline) and `slope5_cmd`, as in the sealed ladder. These rows are the
subset of the twin's primary pitcher panel (`pitcher_primary_post_features.parquet`) with a command value,
2024–2026.

## Tests (family A1, BH across the two confirmatory contrasts)

| test | contrast | label |
|---|---|---|
| A1.1 | workload null + [`z_cmd`, `d3_cmd`, `slope5_cmd`] vs workload null | y30 all-cause |
| A1.2 | workload null + W0 + command features vs workload null + W0 | y30 all-cause |
| A1.1-ablated | A1.1 with each pitcher's last outing before the placement removed | y30 all-cause |
| sensitivity | A1.1 on y30 shoulder | y30 shoulder |

Season-forward out-of-fold L2 logistic as in the sealed design; with only 2024–2026 available, the training
set for the 2025 fold is 2024 and for the 2026 fold is 2024–2025, and the first available season is never
scored. Cluster bootstrap by pitcher, B = 2000, seed 20260930. Escalation bar as sealed (full ≥ +0.01 with
CI > 0 and ablated ≥ +0.005 with CI > 0). Verdict wording follows the twin prereg §7. A descriptive
trajectory (mean `z_cmd` by outings before placement) is drawn as in Figure 1, labelled descriptive.

## What is not done

No other command definition, window, threshold or pitch-type split is tried. If fewer than 30 injured pitchers
fall on the scored grid, the test is reported as gate-closed, not relaxed. Nothing here changes any sealed or
twin result.
