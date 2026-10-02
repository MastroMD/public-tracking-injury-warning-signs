# Preregistration addendum A2: estimation and description for the full paper

**Written 2026-10-02, before any A2 output was computed.** Extends the sealed twin preregistration
(`48619f924703fe98`), the framing preregistration (`1898c1f6e60769bd`) and addendum A1 (`d211f7f5dd66d630`).
The sha256[:16] of this file is recorded in `SEAL_A2.txt` and asserted by `twin/a2.py` at load. Nothing here
changes after sealing; departures go to `DEVIATIONS.md` with the date and reason.

## What had been seen when this was written

Every confirmatory output of the twin, the framing analyses and A1 (`out/`, the submitted abstract), and the
companion injury-luck repository's defect repair of the episode end (its DEVIATIONS #10). No feature panel had
been rebuilt in this round, no out-of-fold prediction had been saved, and none of the quantities below had been
computed. Everything in A2 is estimation with intervals or plain description unless it is marked as a test.

## A2.0 Reproduction check (a check, not an analysis)

The feature panels are not kept in `out/` (they are regenerated). Before any A2 quantity is computed, the pitcher
and batter panels are rebuilt from the pinned inputs and the confirmatory pitcher (P1–P4, P2-ablated,
P3-ablated, the three declared sensitivities) and batter (primary, public labels) contrasts are re-run into a
scratch output folder. Each re-run value must equal the committed `out/` value to four decimals. Any difference
is reported in `out/a2_reproduction.json`; the committed values are not replaced.

## A2.1 Secondary windows and labels (no new fitting)

The declared extended windows (`pitcher_extended_*`, `batter_extended_*`) and the strict-text label sensitivity
(`batter_*_strict_*`), already run under the twin preregistration, are tabulated in the supplement beside the
primary. Nothing is fitted.

## A2.2 Horizon sweep of the pre-placement trajectory (descriptive, no test)

The trajectory of the abstract's Figure 1 (`twin/descriptive.py::trajectories`) is recomputed with the case
window h in {7, 14, 30, 60} days: a placement is a case when the last scored outing (game) of the same season
precedes its start by at most h days; k = 0 is that outing, k = 1..10 the ones before it in the same season. The
reference is fixed at rows with no placement in the following 60 days, so the lines are comparable across h.
Families: pitchers all-cause (COVID-era blank placements excluded, as in Figure 1), shoulder, and the elbow family
(`elbow_other`, `forearm_flexor`, `ulnar_nerve`, `ucl_graft`), metric `z_velocity`; batters any musculoskeletal
placement and oblique (`core_oblique`), metric `z_bs_dev`. Player-clustered bootstrap, B = 1000, as Figure 1.
Beside each trajectory, the simple rule z <= -1 at label horizons of 7, 14, 30 and 60 days: share of rows
flagged, PPV, base rate and sensitivity, for the three pitcher families. Primary windows, POST timing.

## A2.3 Calibration and decision curve for P2 and P3

Models, exactly as in the twin: P2 = workload null + W0 + M3 on y30 all-cause; P3 = workload null + W0 on y30
shoulder; each with its workload null. Season-forward out-of-fold predictions (first two seasons not scored),
primary window, saved this time. Each is also evaluated on its index-ablated grid (twin §6 mask).

- **Calibration**, pooled and by out-of-fold season: calibration-in-the-large as the intercept of
  `y ~ 1 + offset(logit p)`, and the observed/expected ratio; calibration slope as `b` in `y ~ a + b logit p`.
  Player-clustered bootstrap 95% CIs, B = 2000. Pooled calibration by decile of predicted risk is saved for the
  figure.
- **Discrimination by season** (supplement "per-season folds"): out-of-fold AUC of each model and its null by
  season, and the ΔAUC, with no test.
- **Decision curve.** Net benefit NB(t) = TP/n − (FP/n)·t/(1 − t) for each model, the workload null, treat-all and
  treat-none; t from 0.01 to 0.50 by 0.005 (all-cause) and 0.005 to 0.25 by 0.0025 (shoulder). The difference
  ΔNB = NB(model) − NB(workload null) is reported with player-clustered bootstrap 95% CIs (B = 2000) at the
  thresholds of the F2 grid that are below 1, as computed in `out/f2_results.json`: all-cause 0.125, 0.25,
  0.375, 0.5; shoulder 5/(e·42) and 15/(e·42) for the cells below 1 (0.119, 0.238, 0.357, 0.476). The F2
  threshold PPV* = r/(e·D) is the threshold probability of a decision curve, which is why these points are used.
  Estimation only; no P values.

## A2.4 Command, extended window, and one pitch-type sensitivity

1. **A1 on the extended window.** Addendum A1 exactly (features, ≥ 10 tracked pitches, ≥ 5 prior outings,
   OOF with the first available season unscored, cluster bootstrap B = 2000, seed 20260930, the 30-cluster gate,
   index ablation, shoulder sensitivity), on the twin's extended pitcher panel (scored rows <= 2026-08-30, labels
   from the census through 2026-09-29). Family A2.4 = {A1.1-ext, A1.2-ext}, BH within it.
2. **Per-pitch-type miss (one declared sensitivity; A1 said no pitch-type split would be tried, so this is
   declared here as new and logged in DEVIATIONS).** Pitch type from OpenCommand's own `pbp_info.pitch_type`.
   Fastball = FF, SI, FC, FA; breaking = SL, ST, SV, CU, KC, CS. Per outing and group, the median miss over
   plausible tracked pitches with ≥ 5 in that group (else missing). For each group, `z`, `d3` and `slope5` as in A1
   (≥ 5 prior outings with a value). Contrast: workload null + the six group features vs workload null, y30
   all-cause, primary window, on the A1 rows; missing values median-filled inside each fold as everywhere else.
   Reported beside A1.1, outside the family. No other split, threshold or definition.

## A2.5 Anatomy of the index outing (descriptive, no test)

Index outing = the last scored outing of the same season at most 30 days before an all-cause placement (COVID-era
blanks excluded); slow = `z_velocity` <= -1 at that outing. For slow and not-slow index outings: the number of
placements; days from the outing to the IL start (median, IQR, share at 0–1, ≤ 3, ≤ 7 and ≤ 14 days); and the
share removed early, defined as total pitches in the outing below the median of the pitcher's previous in-season
outings. Comparators: the same "removed early" share among scored outings with z <= -1 that were not followed by
a placement within 30 days, and among all scored outings not followed by one. Counts of slow index outings by
family (shoulder, elbow family, other). The IL start is the effective date and can be backdated, which is stated
wherever the days are quoted. Nothing here is a claim about mechanism.

## A2.6 What the study can and cannot rule out

1. **MDE table.** MDE_ΔAUC = 2.80 × (CI_upper − CI_lower)/3.92 (framing prereg F2a) for every sealed arm (from
   `out/f2_results.json`) and for every twin and A1 contrast (P1, P2, P2-ablated, P3, P3-ablated, P4, the PRE
   sensitivity, batter oblique/hamstring/hand-wrist full and ablated, A1.1, A1.1-ablated, A1.2), beside the
   observed estimate and its upper 95% bound (the largest effect the data are compatible with).
2. **Defect check on F2(b).** The companion injury-luck build found that the census can carry a 2010–14 placement
   to the player's first 2015 game. F2(b)'s median IL days D are recomputed from placements starting on or after
   2015-01-01 only, and the grid, the cells met and `met_by_workload_null_alone` are reported beside the sealed
   F2(b). The sealed F2(b) stays the reported primary unless D changes; if it does, both are shown.

## A2.7 Twin against sealed

`results/twin_vs_sealed.csv` (nine verdicts, both estimates, both intervals) becomes a table. No computation.

## Not done

No new feature, model class, site family, wearable or training proxy, adjudicated-label run, window or threshold
beyond those named above. Logistic only. Seed 20260930 for every bootstrap and sampling step.

## Outputs

`out/a2_reproduction.json`, `out/a2_horizon.json`, `out/a2_calibration.json`, `out/a2_dca.csv`,
`out/a2_command_ext.csv`, `out/a2_command_ext_summary.json`, `out/a2_index_anatomy.json`, `out/a2_power.csv`,
`out/a2_f2_defect_check.json`.
