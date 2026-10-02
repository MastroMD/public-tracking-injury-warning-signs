# Public twin — within-season injury warning signs from public MLB data

Open code that rebuilds the preregistered within-season warning-sign tests for pitchers and batters from
two public sources only: the MLB Stats API transaction feed and Baseball Savant pitch-level Statcast data.
It exists so that anyone can check the claim that, at game-aggregate grain, a within-season change in a
player's public tracking metrics does not identify the player about to be placed on the injured list.

**Preregistered before any feature was computed:** `PREREGISTRATION_PUBLIC_TWIN_2026-09-30.md`
(sha256[:16] `48619f924703fe98`) and `PREREGISTRATION_FRAMING_2026-09-30.md` (`1898c1f6e60769bd`);
`SEAL.txt` records both. Addendum A1 (OpenCommand, `../opencommand/`) and addendum A2
(`PREREGISTRATION_ADDENDUM_A2_2026-10-02.md`, `SEAL_A2.txt`, sha16 `5a1f00b5420b4e29`) were sealed before their outputs.
Every departure is in `DEVIATIONS.md`.

## Inputs (not redistributed; pull them yourself)

| file | source | sha256[:16] |
|---|---|---|
| `data/txns_live.jsonl` | `statsapi.mlb.com/api/v1/transactions?sportId=1`, 2010-01-01 → 2026-09-29, one JSON record per line | `f65ea265e77b5735` |
| `data/statcast_season/statcast_YYYY.parquet` (2015–2026) | Baseball Savant `statcast_search/csv`, `type=details`, all game types | see `twin/common.py::PINS` |

Every script asserts these hashes and the preregistration hashes at load. A fresh pull from the APIs will
not match the pinned hashes byte for byte (Savant and the transaction feed are revised); set the pins to
your pull and expect estimates within bootstrap error.

## Run order

```
export TWIN_DATA=/path/to/data TWIN_OUT=/path/to/out
python3 -m twin.census                      # public injured-list census -> out/placements_public.csv
python3 -m twin.batter primary public       # batter arm (WS-010 replica), primary window
python3 -m twin.batter extended public      #   declared secondary window
python3 -m twin.batter primary strict       #   declared strict-text label sensitivity
python3 -m twin.batter extended strict
python3 -m twin.pitcher primary             # pitcher arm (WS-002/003/008 replicas)
python3 -m twin.pitcher extended
python3 -m twin.r1 public                   # reconciliation R1 (also: strict, or a case-list CSV)
python3 -m twin.framing_f1 FG_WAR.csv games_flat.csv standings.csv [third_party_team_season.csv]
python3 -m twin.framing_f2 sealed_arms_ci.csv
python3 -m twin.descriptive                 # post hoc trajectories + simple-rule PPV -> figures/Figure1
OPENCOMMAND_DATA=../opencommand/data python3 -m twin.command   # addendum A1: pitcher command (OpenCommand), 2024-2026
python3 -m tests.test_estimator             # IRLS logistic == sklearn L2 logistic (needs scikit-learn)
python3 -m twin.a2 reproduce|horizon|calibration|command|anatomy|power   # addendum A2 (sealed 2026-10-02), needs the
                                            #   feature panels from twin.pitcher primary/extended and twin.batter primary public
```

Python 3.11, numpy, pandas, pyarrow, statsmodels, scipy. Runtime about 25 minutes on two cores.

## What each script replicates

| script | replicates | label source here | notes |
|---|---|---|---|
| `batter.py` | WS-010 (prereg `881bb2f8e35038fd`): F-DESC family of 6, F-DISC family of 3, index-game ablation | sites.py `core_oblique`, `hamstring_quad`, `hand_wrist_finger` | features, eligibility, folds and bootstrap copied from the sealed script |
| `pitcher.py` | WS-002 D-LEAN (W0, ladder, PRE timing), WS-003 shoulder W0, WS-008 composite + ablation | all-cause, sites.py `shoulder`, elbow family | adds one test the sealed programme never ran: the shoulder cell with the index outing removed |
| `r1.py` | — | one label rule on one set of hitter-games | batter-arm ΔAUC vs the 2026-09-30 exploratory cold-swing odds ratio |
| `framing_f1.py` | WAR lost to the IL per team-season | public episodes | brackets: Marcel-with-zeros (lower), same-season pace (upper) |
| `framing_f2.py` | minimum detectable effects; decision threshold for a rest decision | — | |
| `command.py` | — (addendum A1, sealed 2026-10-01) | public | pitcher command from OpenCommand inferred targets vs the workload model, with ablation |
| `descriptive.py` | — | public | post hoc, descriptive: z-score trajectory before placement; PPV of a z ≤ −1 rule (DEVIATIONS #6) |
| `a2.py` | — (addendum A2, sealed 2026-10-02) | public | reproduction check; horizon sweep; calibration and decision curve; command on the extended window and by pitch type; index-outing anatomy; MDE table |

## OpenCommand

`twin/command.py` uses OpenCommand (Kim T., v1.2.0, open-command.com, CC BY-NC-SA 4.0), per-pitch inferred
catcher targets from broadcast video, 2024–2026. The data is not redistributed here; `../opencommand/
!RUN_OPENCOMMAND_PULL.command` pulls it from Hugging Face. Cite OpenCommand when you use it.

## What it does not cover

In-game public data only. Nothing here speaks to training, bullpen or side sessions, warm-ups, wearables,
or laboratory biomechanics. No adjudicated diagnosis, operative status or return date is used.

## Layout

```
twin/        common.py (pins, estimator, AUC, bootstrap), census.py, batter.py, pitcher.py, r1.py,
             framing_f1.py, framing_f2.py, sites.py (public site taxonomy), hot_streak_study.py (vendored)
tests/       test_estimator.py
out/         results CSV/JSON (feature parquets are regenerated, not kept)
figures/     Figure1_trajectory.png/.pdf
sealed_arms_ci.csv   the sealed arms' ΔAUC and 95% CI, used by framing_f2.py
```
