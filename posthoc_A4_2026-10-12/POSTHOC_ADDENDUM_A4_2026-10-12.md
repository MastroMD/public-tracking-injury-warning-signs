# Post hoc addendum A4 — robustness and candidate analyses from the 2026-10-12 manuscript review

Written 2026-10-12 (ET) during the comprehensive review of the tracking manuscript (Paper J, OJSM revision), after every
preregistered, A1, A2 and A3 result was known. Everything below is **post hoc**. Nothing here can change a sealed or
preregistered verdict, and nothing here is in a manuscript until the senior author decides it should be. The file is hashed
(sha256[:16]) into `SEAL_A4.txt` before `a4.py` first runs; `a4.py` refuses to run if the hash differs.

## Inputs

* Pitcher feature panels rebuilt from the pinned public inputs by `twin.pitcher primary` (results bit-identical to the
  committed `out/pitcher_primary_results.csv`, checked 2026-10-12) and the public census `placements_public.csv` (rebuilt,
  byte-identical to the committed file).
* Raw pinned Statcast pitch-level files (2015–2026) for appearance-level workload, starts and the primary-fastball panel.
* Estimator, AUC, cluster bootstrap (B = 2000, seed 20260930), label and removal rules: `twin.common` unchanged. Gate: a
  contrast with fewer than 30 injured pitchers on the scored grid is closed. Bar language ("would meet") is descriptive only.
* N4 = the paper's workload model (pitches in the scored outing, days since the previous qualifying outing, season-to-date
  pitches and outings in qualifying outings). W0 = the eight kinematic features. M3 = the three persistence terms.
  Labels exclude the COVID-era blank-diagnosis placements, as in the paper.

## Analyses

**A4.1 Workload from every appearance.** NA = pitches in the scored outing; days since the previous appearance of any
length; pitches thrown in the previous 3, 7, 14 and 28 days; appearances in the previous 7 days; whether the previous
appearance was the day before; season-to-date pitches and appearances, all computed from every regular-season appearance
(any pitch count) strictly before the scored outing in the same season. Contrasts: NA + W0 + M3 vs NA (any IL), NA + W0 vs NA
(shoulder), each with all outings and with the last outing removed; and NA vs N4 (any IL), full grid.

**A4.2 Lead time.** Positive if a placement of the label starts in (d + g, d + g + 30] for g = 7 (primary) and 14
(secondary); rows with a placement of the label in (d, d + g] are excluded. Contrasts N4 + W0 + M3 vs N4 (any IL) and
N4 + W0 vs N4 (shoulder).

**A4.3 Severity and diagnosis specificity.** (a) Long stints: IL duration ≥ 30 days (end minus start; an open stint counts
if the feed end, 2026-09-29, is ≥ 30 days after its start). (b) Structural text: diagnosis matches
`\btear\b|\btorn\b|strain|sprain|fractur|ruptur|surger|reconstruct|stress reaction|dislocat|sublux`. (c) Nonspecific text:
matches `inflamm|sore|fatigue|tight|tendin|tendon|discomfort|irritat|impinge|spasm|cramp|stiff` and not (b). For each class
the label is a placement of that class within 30 days; rows followed within 30 days only by placements outside the class are
excluded. Contrast N4 + W0 + M3 vs N4, all outings and last outing removed.

**A4.4 IL-rule era.** The primary composite and shoulder OOF predictions (all outings and last outing removed), with AUCs
computed separately in out-of-fold seasons 2017–2021 (10-day minimum stint for pitchers) and 2022–2026 (15-day).

**A4.5 Role.** Starter if at least half of the pitcher's appearances in the season through the scored outing were starts
(the first pitcher his club used in the game). The primary composite predictions, AUCs within starters and within relievers,
all outings and last outing removed.

**A4.6 Season-stratified gain.** For the primary composite (all outings, last outing removed) and shoulder (all outings), the
gain in AUC within each out-of-fold season and their mean weighted by the season's positives, with the cluster bootstrap on
the weighted mean.

**A4.7 Equivalence.** For the previous-outing sensitivity (P1 PRE), the composite with the last outing removed and the
shoulder cell with the last outing removed: the 90% percentile interval of the fixed-model cluster bootstrap of the gain;
equivalence to zero within ±0.005 if the whole interval lies inside (−0.005, +0.005).

**A4.8 Primary fastball.** For each pitcher-season, the primary fastball is FF or SI, whichever he threw more often in his
first 5 outings of the season with ≥ 20 pitches and ≥ 5 pitches of FF and SI combined. Panel: outings with ≥ 20 total
pitches and ≥ 5 of the primary type; the eight kinematic features and persistence terms rebuilt on that type exactly as in the
paper (its share replacing four-seam share); N4 rebuilt on this panel. Contrasts composite vs N4 (any IL, all and removed)
and kinematics vs N4 (shoulder, all and removed).

**A4.9 Clinically oriented metrics.** For the primary composite and shoulder predictions (all outings and removed): gain in
average precision, in standardised partial AUC at a false-positive rate ≤ 0.10, in sensitivity at 90% specificity, and in
Brier score (workload minus model), each with a cluster bootstrap interval (B = 1000).

**A4.10 UCL placements.** Label: public site `ucl_graft`. Contrast N4 + W0 vs N4, all outings and last outing removed (gate
30).

**A4.11 Descriptive velocity deficit.** The last-outing four-seam velocity z minus the reference mean (rows with no placement
in the next 60 days), as in A2.2 (h = 30), within subsets of placements: diagnosis class (A4.3 b and c), days from the last
outing to the transaction (≤ 1, 2–3, 4–7, > 7), era (A4.4) and role (A4.5); player-cluster bootstrap, B = 1000.

## Reporting

Outputs: `out/a4_contrasts.csv`, `out/a4_metrics.json`, `out/a4_descriptive.json`, `out/a4_run.log`. Every contrast is
reported, closed or not. A findings report states which results would change a sentence of the manuscript.
