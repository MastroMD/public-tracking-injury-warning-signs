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

12. **2026-10-02 — addendum A1 seal time (clerical; found in review, nothing changed).** `SEAL_A1.txt` records
    `sealed_at_utc: 2026-10-02T02:48:19Z` (22:48 EDT), while the addendum's own first line says "Written 2026-10-01 22:55
    EDT". One of the two times is a typing error. The file is not edited, since editing it would change its hash. The seals in
    this repository are hashes recorded in text files, not third-party timestamps, and the paper says so.

13. **2026-10-02 — post hoc addendum A3, written after the independent review of the full paper.** Mike approved (10:50 ET)
    turning the review's informal checks into twin outputs. `POSTHOC_ADDENDUM_A3_2026-10-02.md` (sha16
    `3a7d6ea5251bfb88`, `SEAL_A3.txt`) specifies them, and `twin/a3.py` runs them (`out/a3_*`). They are post hoc by
    construction, because the review's approximate values were known when the specification was written. The hash fixes
    the specification, not the order of knowledge. No A3 value changes a preregistered verdict, and both papers label every
    A3 value post hoc. Two A3 contrasts would meet the escalation bar (the composite against a workload model without the
    outing's pitch count, and velocity alone for shoulder placements on the primary window). Both are reported as
    observations for a future preregistered study.

14. **2026-10-02 — addendum A3 wording (clerical; found in the independent review of the journal manuscript, nothing
    rerun).** Three points, recorded so the A3 outputs can be read against the text that specified them. The addendum is
    not edited, since editing it would change its hash.
    (a) A3 defines W0 as "the eight kinematic z-scores plus release drift". The sealed W0 is seven z-scores (velocity,
    spin rate, extension, arm angle, horizontal and vertical release, four-seam share) plus release drift, eight features
    in all. `twin/a3.py` uses the sealed set (every `z_` column plus `release_drift`), so the outputs match the twin's
    preregistered W0, not the miscount.
    (b) A3 defines non-arm as "every other non-blank placement". The only blank its shared definitions name is the
    COVID-era blank, and the code reads it that way, so the non-arm label includes placements whose text names no site
    outside the COVID era (`unspecified`). A reader who takes "non-blank" to mean "with a site" would exclude them and get
    a smaller non-arm cohort. The non-arm rows are not interpreted in either paper, and the Supplementary Table S11 note
    says which placements they include.
    (c) `twin/a3.py` asserts the A3 seal and reads the twin's derived panels in `out/`, which were built by the pinned
    scripts. It does not re-check the raw-input pins itself.

15. **2026-10-19, extended 2026-10-20 — post hoc census check, written after every other result was seen.**
    `twin/census_check.py` (`out/census_check.json`) explains why the A2 horizon and anatomy outputs (the paper's
    Supplementary Tables S4 and S5) count shoulder and elbow placements differently by one, and checks what repeated
    placement records do to the labels. It refits nothing and changes no committed output, label or verdict. Like A3, it
    reads the derived files in `out/` and does not re-check the raw-input pins.
    (a) The census removes a duplicate only when the full transaction text matches (`twin/census.py`). So 119 of its
    11,604 records, in 117 stints, repeat a player and IL start date under different text, and the census holds 11,485
    distinct player and IL start dates.
    (b) The two A2 outputs assign such starts to families by different rules. `horizon` counts a start once for each of its
    records in the row's family (every record, for any IL), and `anatomy` gives every record the family of the start's
    first record. Of the 1,736 placement records with a last outing within 30 days, 10 starts have two records, and two
    of them explain the difference. The script reproduces both committed counts exactly.
    (c) Labels are built from IL start dates, so a stint counts once in a label however many records it has. It enters a
    family's label if any of its records is coded to that family, and the all-cause label unless every record is a
    COVID-era blank (#1). Keeping only each stint's first record would change the 30-day label of 2 of the 58,387 outings
    in the pitcher panel for any IL and 6 for shoulder placements, no batter label, and the season stop of 2 batters. The
    extended window gives the same counts.
    (d) 22 further pairs of stints of one player start 1 to 3 days apart, with the same club, the same or a blank diagnosis
    and the same IL end, and are probably one stint posted twice. Merging each pair changes no label, removed outing or
    batter stop. The median IL stint used by F2 is the same from every record, from first records only and with the pairs
    merged.
    (e) `sites.py` codes a thumb UCL injury as a UCL graft, in the elbow family (9 records). Coding them to the hand instead
    changes no label.

16. **2026-09-30, recorded 2026-10-20 — post hoc convention for agreement with the sealed registry.** Section 7 calls
    the twin in agreement with a sealed verdict when both fail the escalation bar or both pass it. The registry never ran
    the removal test for the shoulder cell, so its verdict on the full bar is undefined there. The comparison of the twin
    with the sealed registry, written on 2026-09-30 after the twin's outputs existed, judges agreement on the full
    contrast wherever only one of the two ran the removal test. Under that convention both pass the first stage for the
    shoulder cell, which counts as agreement although the twin then fails its removal test, and the twin agrees with the
    registry on 9 of 9 shared tests. The convention was found in the accuracy audit of 2026-10-20 and is labelled post hoc
    in the papers.

17. **2026-10-12, published 2026-10-20 — post hoc addendum A4 and its amendment 1, written after every preregistered, A1,
    A2 and A3 result was known.** `posthoc_A4_2026-10-12/POSTHOC_ADDENDUM_A4_2026-10-12.md` (sha16 `f6b63d099d7a5c1a`,
    `SEAL_A4.txt`) specifies robustness checks and candidate analyses raised in a review of the manuscript, and `a4.py` runs
    them. `SEAL_A4.txt` records the hash at 18:50 EDT on 2026-10-12, and the run log (`out/a4_run.log`) begins at 18:53.
    Amendment 1 (`742e33fd3ee4c4b2`, `SEAL_A4_AMENDMENT1.txt`, sealed at 18:58) was written after A4.1 to A4.7 and A4.9 to
    A4.11 were seen. It repeats three of them against the workload model built from every appearance, and `a4_amend1.py`
    runs it. As with A3, the hashes fix the specifications, not the order of knowledge. The decision to report A4, in one
    version of the paper that labels every A4 value post hoc, was made on 2026-10-16, after its results were seen. No A4
    value changes a preregistered verdict. Neither specification is edited, since editing it would change its hash. Three
    points, recorded so the outputs can be read against the text.
    (a) The specification names the outputs `a4_contrasts.csv`, `a4_metrics.json` and `a4_descriptive.json`. The script adds
    the run's argument to each name and writes the descriptive results (A4.11) into the metrics file, so the outputs are
    `out/a4_contrasts_all.csv` (A4.1 to A4.5, A4.8 and A4.10) and `out/a4_metrics_all.json` (A4.6, A4.7, A4.9 and A4.11).
    The amendment writes `out/a4_contrasts_amend1.csv` and `out/a4_metrics_amend1.json`.
    (b) The specification says a findings report states which results would change a sentence of the manuscript. That report
    is an internal note and is not in this repository.
    (c) `a4.py` asserts the A4 seal, and `a4_amend1.py` asserts both seals. `a4.py` reads the census and the pitcher panels in
    `out/`, which the pinned scripts built, and reads the Statcast files directly for appearance-level workload, starts and
    the primary-fastball panel without re-checking their pins. `a4_amend1.py` reads the census and a panel that `a4.py`
    writes. Both scripts are published as they ran, except one line of `a4.py`: its default path to the twin, which named a
    folder on the machine that ran it, was made relative on 2026-10-20, and nothing was rerun. That default expects the twin
    in a folder beside the A4 folder, so in this repository run both scripts with `TWIN_REPO` set to the repository root
    (README, Run order).
