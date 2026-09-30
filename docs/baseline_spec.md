# Baseline specification

Written and committed before any scoring of the baseline. After scoring starts, these two files
(`configs/baseline.yaml` and this page) must not change; the scoring code refuses to run if they differ
from the commit stamped into the results. The numbers live in `configs/baseline.yaml`; this page explains them.

## 1. Question and headline numbers
What is the probability that a new fire starts in a forest cell (5 km) on day D+1, using data up to day D?
Every table reports both ranking views: within-day spatial AUC (which cells are riskier on the same day) and
within-cell temporal AUC (which days are riskier for the same cell). Also reported: average precision (AP), log loss and
Brier score after correcting the sampling rate (0.0476) to the true rate (0.0147), top-k capture and lift, and a
reliability table by decile.

## 2. Data periods
| Period | Use | Scored |
|---|---|---|
| 2019-2024 | fit years | many times |
| 2025 | last development fold. Already scored in earlier work, so it is not untouched | many times |
| 2026 Jan-Jul | sealed test | once |
| states outside the fit domain | fresh geographic test: UK always; AP, KA, TN, KL only if their fire data passes the completeness check | once, every access recorded in a ledger |
| five reserve cells | descriptive only, never called held-out | - |

## 3. Validation
Rolling origin by year: train on earlier years, validate on the next. Validation years are 2021-2025.
No random splits. Tuning uses 2021-2024 only (stricter reading: 2025 never drives a choice).

## 4. Arms
Same rows, labels, folds, metrics and calibration for all.
- Ladder: climatology_cell, landcover_season, fwi_season, logistic_weather_static, logistic_with_burn, boosting, tabular_network
  (MLP 3 layers of 256, dropout 0.1, AdamW, weight decay, cross-entropy with the design weights, early stopping on validation log loss
  inside the training years, 10 seeds).
- Diagnostics (measure the "which cell" shortcut): coords_only (cell centroid), cell_id_only (cell climatology rate),
  static_only (boosting on static layers only). They are reported next to every headline number.

## 5. Feature tiers (all use data at or before day D)
- tabular: the 42 existing inputs.
- weather_windows: for the 11 weather variables and 6 fire-weather codes: mean and max over the last 3, 7, 14, 30, 60 days;
  7-day change for the codes; days since rain above 2 mm (cap 60); cumulative rain over 7, 30, 60 days; anomaly against the
  cell's day-of-year climate computed from the training years of that fold only.
- neighbourhood: for the 8 neighbours and the 24-cell ring: means of tree cover, burn history, vegetation index, fire-weather index,
  and the count of onsets in the previous 28 days (never includes day D+1). Edge cells use the neighbours that exist.
- imagery_summary: declared now, built when imagery exists, with the same code and protocol (a pre-declared extension).

Logistic arms, boosting and the network run at two information levels: tabular, and enriched (tabular + weather_windows + neighbourhood).
"Enriched boosting vs network on enriched inputs" is the architecture comparison; "tabular vs enriched" is the information comparison.

## 6. Tuning
Development years only, same budget for boosting and network: 12 random-search trials each, rolling-origin validation, selection by
validation log loss (AP breaks ties). Search spaces are in `configs/baseline.yaml`. Every trial is recorded.

## 7. Decision rule
PASS: AP gain >= 0.010, 99% paired ISO-week cluster-bootstrap lower bound > 0, and no state worse than -0.03.
FAIL: the interval's upper bound is below 0.010. Otherwise INCONCLUSIVE. The minimum detectable effect (MDE) is computed before
any verdict; if it exceeds 0.010 the verdict is INCONCLUSIVE.
Contrasts, all reported including failures: (a) enriched boosting vs tabular boosting, (b) enriched boosting vs enriched logistic,
(c) network vs enriched boosting, (d) every arm vs climatology_cell.

## 8. Frozen baseline
The arm set, tuned settings and feature lists, tagged `baseline-final` with the code hash, config snapshot, data manifest and artifact manifest.

## 9. Guards before any score is written
Feature timestamps at or before day D (a deliberately shifted feature must fail); label shuffle (AUC 0.50 +- 0.02); coordinate and grid
unit test; identity gate; no validation-fold information in tuning or climatology; provenance (clean tree + code hash); power gate;
geographic-test ledger.

## Compute limits declared in advance
- Every fit uses at most 2,000,000 training rows (all onsets kept, negatives thinned uniformly with a fixed seed); tuning fits use at most 600,000.
  The same thinned rows are used for every arm and level within a fold.
- Every validation year uses the rows of January to July (the 2025 data stops at the end of July).

## Clarifications taken (stricter reading)
- Tuning excludes 2025 (see section 3).
- The spec hash stamped in results is the last commit that touched the two spec files.
