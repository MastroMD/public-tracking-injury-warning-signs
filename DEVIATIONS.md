# Deviations from the sealed twin preregistration (48619f924703fe98)

Each entry: date, what, why, and whether any feature or model output had been seen.

1. **2026-09-30, before any feature was computed or model fit — COVID-era blank placements.**
   Prereg §3 excludes "COVID-labelled moves". The census shows 25–31% of 2020–2022 placements carry no
   diagnosis text (vs 1–5% in other seasons); these are overwhelmingly COVID-protocol placements that the
   feed did not label. The library's inherited parser flags them as `covid_era_unattributable` and drops
   them; the prereg wording under-specified that rule. **Primary:** the pitcher all-cause label excludes
   blank-diagnosis placements in 2020–2022. **Sensitivity (as written):** they are kept. Site-family labels
   are unaffected (a blank diagnosis is `unspecified` and belongs to no family). Only label counts had been
   looked at when this was decided.

2. **2026-09-30 — framing inputs not pinned.** The framing prereg names the F1 inputs (FanGraphs season
   fWAR exports, Stats API schedule and standings) but does not pin their hashes. They are hashed at load
   and written to `out/f1_results.json` (`inputs_sha16`). No change to method.

3. **2026-09-30 — R1 code defect fixed before any R1 number was used or reported.** The first R1 run
   restricted the case list to 2024+ before building the oblique labels for the batter-arm frame, so the
   2023 training rows had no positives and the 2024 fold was skipped. Labels now use every season; the
   exploratory split's frame keeps its own 2024+ rule (recent-return filter and its own label rule), as the
   split did. The odds-ratio legs (b)–(e) were unaffected; leg (a) is recomputed.

4. **2026-09-30 — F2(b) observed operating point.** The framing prereg compares PPV\*/LR\* with "the best
   observed operating point". Twin OOF predictions were not saved, so the twin's LR at a 10% alert fraction
   is derived from its AUC under the same equal-variance binormal model used to compute AUC\*; the sealed
   empirical top-10% operating point (PPV lift 1.27×) is reported beside it.

5. **2026-09-30 — post hoc, descriptive (added after F2 output was seen).** F2 found one grid cell met
   (pitcher all-cause; r = 5 days; e = 1.0). The script was then extended to report whether the workload
   null alone meets the same cell (`met_by_workload_null_alone`). It does. This is labelled post hoc
   wherever it is quoted.

6. **2026-09-30 — post hoc, descriptive, no model fit (added after all confirmatory results were seen).**
   `twin/descriptive.py` computes, from the feature panels already built, (a) the mean own-baseline z-score by
   number of outings/games before an IL placement, relative to outings/games not followed by one, with
   player-clustered bootstrap intervals, and (b) the PPV and sensitivity of the simplest alert (z ≤ −1, z ≤ −2).
   These describe where the confirmatory signal sits; they are not tests and carry no P values. Figure 1 of
   the abstract is drawn from them.

7. **2026-10-01 — addendum A1 (OpenCommand), implementation notes, decided before results were seen.** The
   addendum says the join is on `play_id` or (`game_pk`, `at_bat_number`, `pitch_number`); the twin's outing
   panel carries neither, so per-pitch misses were aggregated to (pitcher, game date) and joined on that
   (a doubleheader day pools both games). The `date` column of `pbp_info` is the game date. The shoulder
   sensitivity had 35 injured pitchers on the full grid but 21 on the out-of-fold scored grid, below the
   30-cluster gate, and is reported as gate-closed.

8. **2026-10-02 — addendum A2 adds one pitch-type split of the command feature, declared before any A2 output.**
   Addendum A1 says no pitch-type split would be tried. A2.4 (sealed `5a1f00b5420b4e29`, `SEAL_A2.txt`) adds exactly
   one: fastball vs breaking-ball miss, one contrast, reported outside the confirmatory family and labelled as a
   sensitivity added after the A1 results were known.

9. **2026-10-02 — framing values quoted from the companion's defect-repaired build.** The companion injury-luck
   analysis (same public feed, same census) found and repaired an episode-end defect after this twin's F1 was run:
   placements from 2010–14 could stay open until the player's first 2015 game (its DEVIATIONS #10). Its repaired
   primary is the value the two papers share. The paper quotes that repaired value, copied with its source hash, and
   shows this twin's F1 (`out/f1_results.json`, unchanged) beside it. F2(b)'s median IL days are checked against the
   same defect under A2.6.

10. **2026-10-02 — A2.3 threshold list (clerical; found when the output was read).** A2.3 defines the decision-curve
    points as "the thresholds of the F2 grid that are below 1, as computed in `out/f2_results.json`", then lists four
    per label. The grid has five below 1 per label (the list omitted r = 15, e = 0.5: 0.75 all-cause, 0.714 shoulder).
    The code applies the rule as written, so all five are reported.

11. **2026-10-02 — A2.4 shoulder sensitivity on the extended window, gate.** As in #7, the gate is read on the
    out-of-fold scored grid: 38 injured pitchers on the full grid, 24 scored. Reported as gate-closed; its estimate is
    in `out/a2_command_ext.csv` and is not interpreted.
