# Preregistration — public-data twin of the within-season warning-sign tests

**Written and sealed 2026-09-30, before any feature is computed from the pitch-level data and before any
model is fit.** The sha256[:16] of this file is recorded in `SEAL.txt` and asserted by every analysis
script at load. Nothing in this file changes after sealing; deviations are logged in `DEVIATIONS.md`
with the date and reason.

## 1. Purpose

A sealed programme of preregistered analyses (WS-002 → WS-010) asked whether a within-season change in
a player's public Statcast tracking metrics identifies the player about to be placed on the injured
list (IL). Those analyses used a hand-adjudicated injury registry for some labels (unpublished). This
twin rebuilds the pitcher and batter tests from **public inputs only** — the MLB Stats API transaction
feed and Baseball Savant pitch-level data — with open code, so that anyone can rerun them.

The twin's claim is limited to replication: the synthesis paper's conclusion ("no usable within-season
warning sign at game-aggregate grain") survives only if the twin reaches the same verdict under the
same decision rules. Where twin and sealed estimates differ, the difference is reported with its
likely source (label definition), not reconciled by changing either analysis.

## 2. Inputs, pinned before any feature is computed

| input | what | sha256[:16] |
|---|---|---|
| `txns_live.jsonl` | MLB Stats API `/api/v1/transactions`, sportId 1, 2010-01-01 → 2026-09-29, pulled 2026-09-29 | `f65ea265e77b5735` |
| `statcast_2015.parquet` … `statcast_2026.parquet` | Baseball Savant pitch-level search export, all game types, one file per season, pulled 2026-09-30 (9,330,122 rows; 0 failed days; every Final regular-season game present) | 2015 `0194b45414693021` · 2016 `3bf68ba977729bea` · 2017 `51f2e8cf8310a200` · 2018 `5deeeb6c779b217a` · 2019 `aaa20bfc8c30bc04` · 2020 `f82966c8965d5465` · 2021 `6cf746129913c1fb` · 2022 `a013eadc63c547ac` · 2023 `a162588175e5958d` · 2024 `65b82081ea38f1ea` · 2025 `3c686658c711097d` · 2026 `88660e24fe4a7b00` |
| `pull_manifest.json` (Statcast) | pull log | `ec86e2d6982f182f` |
| `sites.py` | public anatomic-site taxonomy (regex + text-keyed hand readings) | `bd025c4f2a307bba` |

No adjudicated case table, curated registry, or third-party injury list is read by any twin script.
Scripts assert every sha above at load and exit non-zero on mismatch.

## 3. Labels (public)

**Placements.** Records whose description contains "placed" and "injured list" (or "disabled list"),
parsed with the inherited defect-repair rules for the Stats API feed (year typos repaired by id
sequence; retroactive dates parsed from the text; misdated duplicate placements removed; 60-day
records that continue an open stint without an intervening activation treated as continuations, not new
events). Kept: placements by one of the 30 MLB clubs; excluded: paternity, bereavement, restricted,
suspension, military and COVID-labelled moves. The IL start is the effective (retroactive) date.

**Site and family.** `sites.classify_site()` on the diagnosis text of the placement, plus the file's
`HAND_CORRECTIONS` (text-keyed, public). No adjudicated override of any kind. Families:

| twin label | sites.py site(s) | sealed analogue |
|---|---|---|
| batter oblique | `core_oblique` | WS-010 oblique (adjudicated, strict oblique) |
| batter hamstring | `hamstring_quad` | WS-010 hamstring (adjudicated) |
| batter hand/wrist | `hand_wrist_finger` | WS-010 hand/wrist (adjudicated) |
| pitcher all-cause | any site | WS-002/005A/008 y30 (feed-derived) |
| pitcher shoulder | `shoulder` | WS-003 y30_shoulder (research-database tagged) |
| pitcher elbow | `elbow_other`, `forearm_flexor`, `ulnar_nerve`, `ucl_graft` | WS-003 y30_elbow |

The twin families are broader than the adjudicated ones (e.g., `core_oblique` includes rib and
intercostal placements). **Declared sensitivity (strict text):** oblique = `core_oblique` AND text
matches `obliqu|oliqu|intercostal`; hamstring = text matches `hamstring`. Reported beside the primary,
never substituted for it.

**Label.** y30_family = an IL placement of that family with start in (g, g + 30 days] after the scored
game/outing g (strictly after, so a stint already open is not a future event).

## 4. Windows

- **Primary (matched to the sealed windows, to isolate the label source):**
  - batters: tracked coverage 2023-07-14+ / 2024-04-03+ / 2025-03-27+ / 2026-03-26+; labels counted
    through 2026-06-30; scored rows ≤ 2026-05-31 (identical to WS-010);
  - pitchers: outings 2015-2026; scored rows ≤ 2026-07-12 (identical to WS-005A/WS-008).
- **Extended (declared secondary, new data):** labels through 2026-09-29 (the feed pull date); scored
  rows ≤ 2026-08-30. Reported beside the primary, never substituted for it.

## 5. Batter arm — exact replica of WS-010 (sealed prereg `881bb2f8e35038fd`) with public labels

Swing rows: regular-season (`game_type == 'R'`) pitches whose `description` is one of swinging_strike,
swinging_strike_blocked, foul, foul_tip, hit_into_play, foul_bunt, missed_bunt, bunt_foul_tip,
swinging_pitchout, foul_pitchout (and the pre-2019 `hit_into_play_*` variants). Tracked swing =
`bat_speed` non-null and description not containing "bunt". Batter-game grain; features exactly as
sealed: `z_bs_dev` (game mean vs mean/SD of ≤ 25 prior in-season tracked-game means, ≥ 10 prior tracked
games, ≥ 5 tracked swings in g), `streak_low` (0.5 SD), `var_dev` (SD of last 10 incl. g minus SD of
games g−19..g−10, ≥ 20 prior, else NaN → training-median fill). Workload null: [`n_swings_g`,
`days_since_prev`, `cum_swings_season`, `games_so_far`]. W1.1: rows on/after the batter's first
same-season **musculoskeletal** placement (any sites.py site except `illness_other` and `unspecified`,
the public analogue of the 12-family Arm B case list) are dropped. Clean control batter-seasons (F-DESC)
have no such placement in that season.

**F-DESC first (family of 6, BH):** case index game (last scored game ≤ 30 d before placement) vs one
month-matched pseudo-index game per clean control batter-season; `z_bs_dev` and `var_dev`; cluster bootstrap by batter, B = 2000.
**F-DISC (family of 3, BH):** ΔAUC of (null + [`z_bs_dev`, `streak_low`, `var_dev`]) vs null on
y30_family; season-forward out-of-fold L2 logistic (C = 1.0, unpenalized intercept, standardized,
median-filled), folds need ≥ 500 training rows and ≥ 20 training positives; cluster bootstrap by batter,
B = 2000. **Mandatory index-game ablation.** Cluster gate: < 30 injured batters on the scored grid →
gate-closed, reported, not relaxed.

## 6. Pitcher arm — replicas of WS-002 / WS-003 / WS-008 with public labels

**Outing panel.** Per pitcher-game (regular season): total pitches (all types) and four-seam (`FF`)
pitches; FF means of release_speed, release_spin_rate, release_extension, arm_angle, release_pos_x
(absolute value used), release_pos_z; FF share. Keep outings with ≥ 20 total and ≥ 5 FF pitches.
Features exactly as WS-002 D-LEAN POST (decision point immediately after outing t; baselines exclude
outing t; ≥ 5 prior in-season outings): `z_` of the six metrics, level, 3-outing delta, 5-outing slope,
release drift, `z_ff_share`, prior-season means/ΔS1; workload null [pitches in outing, days rest,
cumulative season pitches, outings so far]. M3 as WS-004: `m3_streak_low_velo` (consecutive outings with
own-baseline velocity z ≤ −0.5), `m3_rest_dev`, `m3_z_velo_x_rest`.

Season-forward out-of-fold L2 logistic (as §5; first two seasons never scored), cluster bootstrap by
pitcher, B = 2000.

**Confirmatory family P (BH across the four):**

| test | contrast | label | sealed reference |
|---|---|---|---|
| P1 | W0-only (z of six metrics + z_ff_share + release drift) vs workload null | y30 all-cause | WS-005A +0.0059 |
| P2 | W0 + M3 vs workload null, **full and index-outing ablated** (WS-008 escalation bar) | y30 all-cause | WS-008 +0.0120 / +0.0043 |
| P3 | W0-only vs workload null | y30 shoulder | WS-003 +0.0379 (censoring-fixed +0.0315) |
| P4 | full means ladder vs workload null | y30 all-cause | WS-005A +0.0028 |

**New test, declared here (not in any sealed prereg):** P3-ablated — P3 with each case's index outing
(last outing ≤ 30 d before a shoulder placement) removed. The sealed shoulder cell was never ablation-
tested; this closes that gap. Reported with the label "twin-new".
**Declared sensitivities:** PRE timing (features through outing t−1) for P1; y30 elbow for P3's design.

## 7. Decision rules (inherited, restated)

- Escalation bar (WS-008 standard): full ΔAUC ≥ +0.01 with 95% CI > 0 **and** index-ablated ΔAUC ≥
  +0.005 with CI > 0. Anything else is described as cohort-level information, not a warning sign.
- The twin **agrees** with the sealed verdict for a test if both fail the escalation bar (or both pass).
  Point estimates are compared, and the difference (twin − sealed) is reported with its bootstrap CI
  where rows can be matched; no significance test of the difference is used to "confirm" anything.
- No added features, windows, families or model classes after results are seen. Logistic only.

## 8. Reconciliation R1 (declared, descriptive)

On one set of hitter-games — 2024-03-20 → labels through 2026-07-21, tracked regular-season games with
≥ 10 prior tracked games and ≥ 5 tracked swings — the oblique label computed by one rule (§3 public,
and separately by the adjudicated rule, which is run only inside the private library and reported as a
number), report side by side: (a) WS-010's ΔAUC contrast, (b) the logistic OR per −1 SD of `z_bs_dev`,
and (c) the logistic OR per −1 SD of the 2026-09-30 exploratory feature (recent 50-PA bat speed minus
regressed prior history, park- and week-adjusted, from `hot_streak_study.py`), hitter-clustered
bootstrap CIs. This answers whether the two published-in-draft numbers disagree because of the feature,
the window, or the label — before either is quoted.

## 9. What the twin does not do

It does not reproduce WS-004 M1/M2/M4/M5/M6, WS-006, WS-007 or WS-009 (the paper reports those as
sealed). It does not use any adjudicated diagnosis, operative status, or return date. It makes no
claim about training, bullpen, warm-up, wearable or laboratory data.

## 10. Seeds and software

Seed 20260930 for every bootstrap and sampling step. Python 3, numpy, pandas, pyarrow. Estimator: a
numpy IRLS logistic solving sklearn's `LogisticRegression(penalty="l2", C=1.0)` objective (verified
against sklearn in `tests/`).
