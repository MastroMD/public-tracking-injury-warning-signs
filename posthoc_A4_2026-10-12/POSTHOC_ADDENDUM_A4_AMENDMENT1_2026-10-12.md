# Amendment 1 to post hoc addendum A4 (2026-10-12)

Written after A4.1–A4.7 and A4.9–A4.11 were seen, so these are post hoc of post hoc and are labelled so. Hashed into
`SEAL_A4_AMENDMENT1.txt` before any of them runs.

A4.1 showed that a workload model built from every appearance (NA) discriminates better than the paper's workload model
(N4), and that the composite's gain over NA is smaller than over N4. Three A4 results that suggested a residual signal were
computed against N4. Each is repeated against NA, with nothing else changed (same labels, masks, folds, bootstrap):

* **A4.3b** nonspecific-diagnosis placements (A4.3 c): NA + W0 + M3 vs NA, all outings and last outing removed.
* **A4.5b** role (A4.5): the NA composite predictions, AUCs within starters and within relievers, all outings and last outing
  removed.
* **A4.6b** season-stratified gain (A4.6): the NA composite predictions, all outings and last outing removed, weighted mean
  of within-season gains with the cluster bootstrap.
