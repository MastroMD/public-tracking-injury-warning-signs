# Preregistration — the two framing numbers (Prompt 20 §5)

**Written and sealed 2026-09-30, before either number is computed from the public feed.** sha256[:16]
recorded in `SEAL.txt`. Exploratory predecessors exist (a first pass and a v1 built on a third-party IL
list); their values are known to the author and are listed in §4 so that the reader can see what was
known before sealing. The public rebuild may move them in either direction and is reported as it falls.

## F1 — WAR lost to the injured list per team-season, from the public transaction feed

**Episodes.** Placements from the Stats API feed exactly as in the twin prereg §3 (MLB clubs only,
non-IL and COVID moves excluded, defect repairs). End date = first activation/reinstatement after the
start, validated against appearances: if the player appears in a Statcast regular-season game (as batter
or pitcher) before the paired activation, the end is that first game back; with no activation, the end
is the first game back, else the season's last scheduled date (censored). Consecutive stints of one
player with the same club separated by ≤ 1 day, or linked by a 60-day transfer, are one episode.

**Games missed** = the club's regular-season games (schedule from the Stats API) dated in
[start, end) and inside the season. **Projected rate** (primary) = Marcel-style fWAR per team game:
Σ_k w_k·WAR_{S−k} / (Σ_k w_k·Exposure_{S−k} + K), k = 1..3, w = 5/4/3, Exposure = club games minus
games missed that season, K = 100; negative rates floored at 0; nothing from season S.
**Bracketing versions (declared):** lower = Marcel with missing seasons counted as zero WAR and no
exposure denominator (the first-pass method); upper = same-season pace (season-S fWAR per game played ×
games missed). **WAR lost** = rate × games missed, summed by club-season. Seasons 2015–2025, 2020
excluded as an outcome.

**Numbers reported (sealed list):** (i) mean WAR lost per team-season; (ii) SD across the 30 clubs within
season, averaged over seasons; (iii) year-to-year R² of a club's WAR lost (season S on S−1, pooled);
(iv) wins per WAR lost from OLS of wins on WAR lost + the roster's summed projected WAR + prior-season
wins + season fixed effects, SEs clustered by club. fWAR: FanGraphs season exports (hitters + pitchers).
Standings: Stats API. A club-season that cannot be matched to the schedule is dropped and counted.

**Agreement check (declared):** club-season placement counts from the feed vs the third-party list
(Pearson r; the v1 build reported 0.945).

## F2 — "what it would take"

**(a) Minimum detectable effect of each sealed within-season contrast**, 80% power, two-sided α = .05:
MDE_ΔAUC = 2.80 × SE, SE = (CI_upper − CI_lower) / 3.92 from the sealed bootstrap CI. Computed for every
row of the arms table. For the batter oblique arm also as an odds ratio per −1 SD of a game-aggregate
feature, MDE_OR = exp(2.80 × SE_logOR), SE from the twin R1 logistic CI.

**(b) Decision threshold for acting on a warning sign.** A club rests a flagged player for r days. Resting
a healthy player costs r days of his availability; resting a player who would otherwise have been
injured avoids e × D days of IL time, where D is the median IL days for that label and e is the fraction
of such injuries that rest prevents. The player's WAR per day appears on both sides and cancels, so
acting beats not acting when **PPV > PPV\* = r / (e × D)**. Grid (declared): r ∈ {5, 15} days;
e ∈ {0.25, 0.5, 1.0}; D = median IL days of public-feed episodes for (pitcher all-cause), (pitcher
shoulder), (batter oblique). PPV\* is converted to the positive likelihood ratio needed at the sealed base
rate π, LR\* = [PPV\*/(1 − PPV\*)] / [π/(1 − π)], and to the AUC an equal-variance binormal score would need
to reach LR\* at a 10% alert fraction (solve for d with FPR ≈ 0.10, TPR = 1 − Φ(Φ⁻¹(0.90) − d),
AUC = Φ(d/√2)). Compared with the best observed operating point (sealed WS-002 top-10%: PPV lift 1.27×).

The claim F2 licenses, if the observed LR and AUC fall below every grid cell: "no within-season signal
large enough to change a rest decision exists at this grain." If any cell is met, it is reported as
such.

## 4. Values known before sealing (exploratory, third-party IL list)

First pass: mean 5.1, SD 2.7, YoY R² 0.18, 0.88 wins per WAR lost. v1: mean 6.1, SD 3.1, 0.75 wins per
WAR lost, preseason out-of-sample R² 0.36. Neither is quoted in the paper; the public rebuild replaces
both.
