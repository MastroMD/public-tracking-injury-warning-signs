# Post hoc addendum A3 to the public-twin preregistration — checks raised in review

Written 2026-10-02, after every preregistered twin result, every addendum A1 and A2 output, and the external review of
the full paper had been seen. **Everything in this addendum is post hoc.** The review ran most of these checks
informally on the twin's feature panels, so their approximate values were known when this file was written. The addendum
exists so that each check is specified once, run by the twin's own code from the pinned inputs, and reported with an
interval in the results file. It does not replace the review's numbers with better ones, and it cannot change a
preregistered verdict. Every A3 value is labelled post hoc wherever it is reported.

The file is hashed before `twin/a3.py` is first run, and the hash is recorded in `SEAL_A3.txt`. The hash fixes the
specification. It is not evidence that the results were unknown.

## Shared definitions

- Panel, labels, scoring and inference are those of the twin (`PREREGISTRATION_PUBLIC_TWIN_2026-09-30.md`).
  - Pitcher panel: the primary window (`pitcher_primary_post_features.parquet`). The extended window is used only where named.
  - Placements: `placements_public.csv` without COVID-era blanks, with the 30-day label from `common.label_within`.
  - Removal test: `common.index_mask`, which removes each player's last scored outing within 30 days before each placement.
  - Models are season-forward out-of-fold L2 logistic regression (`common.oof`, first two seasons unscored).
  - Intervals come from the player-clustered bootstrap with the fitted models held fixed (`common.boot_dauc`), B = 2,000,
    seed 20260930, unless A3.5 says otherwise.
- Comparators:
  - **N4** is the preregistered workload model: pitches in the scored outing, days of rest, season-to-date pitches before
    the outing, and outings so far.
  - **N3** is N4 without pitches in the scored outing.
  - **N4R** is N4 plus `rel_pitches`, the pitches in the scored outing divided by the pitcher's mean pitches in his
    earlier outings that season (`cov_pitches_outing / (cov_cum_pitches / cov_n_out_so_far)`).
- Feature sets are as sealed:
  - **W0** is the eight kinematic z-scores plus release drift.
  - **M3** is the three velocity-persistence terms.
  - **P2** is W0 + M3.
  - **V** is `z_velocity` alone.
- Labels:
  - **all** is any placement.
  - **shoulder** is site `shoulder`.
  - **elbow** is the elbow family: `elbow_other`, `forearm_flexor`, `ulnar_nerve` and `ucl_graft`.
  - **arm** is any site in `sites.ARM_SITES`: shoulder, the elbow family, `lat_teres`, `pec`, `tos` and `arm_other`.
  - **non-arm** is every other non-blank placement.
- **Gate.** A contrast is run only if at least 30 injured players have a positive row on the scored grid, counted after
  removal for the removed grid. Otherwise it is reported as closed.
- **Reporting.** Each contrast is reported with all games and with the last outing removed, wherever both are listed.
  The escalation bar is printed beside each one as a description ("would meet", "would not meet"), never as a verdict.
  No multiplicity correction is applied, because nothing here is a test of the preregistered hypotheses.

## A3.1 Dependence on the scored outing's pitch count

| id | label | model | comparator |
|---|---|---|---|
| A3.1a | all | N3 + P2 | N3 |
| A3.1b | all | N4 | N3 (what pitches in the scored outing add) |
| A3.1c | all | N4R | N4 (what the relative pitch count adds) |
| A3.1d | all | N4R + P2 | N4R |
| A3.1e | shoulder | N3 + W0 | N3 |

Each is run with all games and with the last outing removed.

## A3.2 Removal-test coverage

| id | label | model | comparator | grid |
|---|---|---|---|---|
| A3.2a | shoulder | N4 + V | N4 | primary window, both grids; extended window, both grids |
| A3.2b | all | N4 + V | N4 | both |
| A3.2c | elbow | N4 + V | N4 | both |
| A3.2d | all | N4 + W0 (P1) | N4 | last outing removed (the full contrast is the sealed P1) |
| A3.2e | elbow | N4 + W0 | N4 | last outing removed (the full contrast is the sealed elbow sensitivity) |

## A3.3 Arm and non-arm labels

| id | label | model | comparator |
|---|---|---|---|
| A3.3a | arm | N4 + W0 | N4 |
| A3.3b | arm | N4 + P2 | N4 |
| A3.3c | non-arm | N4 + W0 | N4 |
| A3.3d | non-arm | N4 + P2 | N4 |

Each is run with all games and with the last outing removed.

## A3.4 Timing from the transaction date

- **A3.4a (descriptive).** The base is every non-blank placement that followed a primary-panel outing within 30 days,
  the same base as A2.5. For each one, compute the days from the last outing to (i) the IL start and (ii) the
  transaction date, the `date` field of the transaction that opened the stint.
  - Report the median and IQR, and the shares within 1, 3 and 7 days.
  - Report the share of placements whose transaction came after their IL start (retroactive).
  - Report the same figures for the subset whose last outing had `z_velocity` ≤ −1.
- **A3.4b.** The removal test for P2 (N4 + P2 against N4, all), with a narrower rule. A last outing is removed only
  if the transaction came within 3 days of it. Last outings more than 3 days before the club's transaction stay in.
  - Report the number of rows removed.
  - Report the contrast.

## A3.5 Refit bootstrap for P2

- Use the P2 contrast (N4 + P2 against N4, all), with all games and with the last outing removed.
- Draw B = 1,000 resamples of pitchers with replacement, seed 20261002.
- In each resample, rebuild the season-forward out-of-fold predictions for both models from scratch on the resampled
  panel, and compute the gain in AUC.
- Report the mean, the SD, the 2.5th and 97.5th percentiles, and the ratio of the refit SD to the fixed-model bootstrap
  SD. The fixed-model SD is (upper − lower) / 3.92 from the sealed intervals.

## Outputs

`out/a3_contrasts.csv` (one row per contrast and grid), `out/a3_gaps.json`, `out/a3_refit_bootstrap.json`,
`out/a3_refit_replicates.csv`. Code: `twin/a3.py`, which asserts this file's hash before running.

## What A3 cannot do

- It cannot rescue or overturn a preregistered verdict.
- It cannot add a feature or a model class.
- It cannot use adjudicated labels.
- If an A3 contrast would meet the escalation bar, the paper says so as a post hoc observation that a future
  preregistered study would have to test.
