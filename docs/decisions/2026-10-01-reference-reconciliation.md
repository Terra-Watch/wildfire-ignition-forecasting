# Reconciliation with the earlier reference results
Status: documentation only. Nothing was re-scored, no code was changed, the freeze and the ledger are untouched. The earlier reference runs are the pre-baseline pipeline in the earlier project folder under `~/Desktop/Projects` (its name ends in `-<ref>`, where `<ref>` is the label that this repository's name check forbids; its scripts are written `<ref>_run.py`, `<ref>_geo_score.py` and so on; results `research/<ref>/results/geo_score_report.txt`, plan `METRICS_PLAN.md`, hyperparameters `hyperparameters_chosen.json`; holdout predictions `~/wildfire-data/processed/holdout_predictions_main.pkl`). Only differences that were found are listed; none is explained beyond what the files show.

## A. Uttarakhand (UK): earlier boosting AP 0.059 and FWI+season AP 0.107, now 0.1075 and 0.0869
Which earlier arms the numbers belong to (`geo_score_report.txt`, block "UK 2019-2025 Jan-Jul", 509,056 rows, 22,970 onsets):
- Boosting 0.059 = arm `M3`: AP 0.05850, log loss 0.28605 (reported before any prior shift, on the model's own sample scale; the constant fit-prevalence reference is 0.18397).
- FWI+season 0.107 = arm **`B1_l0`**: AP 0.10733. `B1_l0` is defined in `<ref>_lib.py` as the six FWI codes with the `_l0` weather columns plus season, which the script `<ref>_models.py` describes as "weather of day D: same-day, NOT a forecast". The earlier next-day FWI+season arm `B1` has AP 0.08616 in the same block. The present `fwi_season` (next-day inputs) is the counterpart of `B1`: 0.0869 against 0.0862 on different rows.

Differences found, earlier run against the baseline fresh scoring:
| Item | Earlier run | Baseline fresh scoring |
|---|---|---|
| Fit years | six fit states, 2019-2024 (`T = T[T.year <= 2024]`) | all fit-state years before 2026, so 2019 to 2025 (through 2025-07-31) |
| Row cap | `<ref>_geo_score.py` fits on all rows of the arm (it calls no thinning); the holdout run of `<ref>_run.py` used `thin(y, 2_000_000, seed=3)` | at most 2,000,000 fit rows, all onsets kept, thinning seed 1 |
| Label and onset rule | onset = a detection with none in the 3 days before; 20 matched negatives per onset (`<ref>_labels.py`) | the same rule and ratio (`domain.yaml`); negatives were drawn again: of the earlier UK rows (31,593 onsets, 644,753 negatives, band A, through 2025-07-31) all 31,593 onsets are present in the new UK rows (32,738 onsets, 654,760 negatives) but only 212,399 negatives are; labels agree on every shared row |
| Forest mask and buffer | band A: tree fraction at least 0.20 (`wc_tree >= 0.20`); UK = grid role `heldout_UK`; reserve cells scored separately; AP not scored | the same threshold (`forest.tree_fraction_min: 0.20`), role `heldout_UK`; AP (role `unused`) scored as well; reserve cells not scored |
| Hyperparameters | boosting `M3`: depth 6, learning rate 0.1, 300 iterations (`hyperparameters_chosen.json`); logistic C 0.01 (as `fwi_season`) | boosting from `tuned.json`: depth 8, learning rate 0.05, 500 iterations, leaf 100, no L2; logistic `fwi_season` C 0.01 |
| Feature set | 42 inputs of the M2 set for `M3`; same-day variants for `*_l0` | `boosting@tabular` reads the 42 tabular inputs; the feature tables were built by different code and were not compared for the UK rows (the reproduction gate compared the fit-state tables only) |
| Test row set | UK rows 2019-2025, January to July only, common rows of all arms: 509,056 rows, 22,970 onsets | UK rows 2020-2025 (no 2019 rows scored), every fire-season month through 2025-07-31, complete rows of all arms including the enriched inputs: 590,870 rows, 27,511 onsets |
| Log loss definition | raw probabilities, unweighted, model's own sample scale (the 0.286) | prior-shift corrected to the rate 0.0147 with unit weights (the 0.2208) |

Measured from the saved baseline UK predictions (no new scoring):
| Arm | Rows | AP | Log loss, raw and unweighted |
|---|---|---|---|
| `boosting@tabular`, all scored rows (590,870) | 2020-2025, all months | 0.1075 | 0.2667 |
| `boosting@tabular`, January-July rows only (508,703 rows, 22,970 onsets) | | 0.0801 | 0.2800 |
| earlier `M3` (509,056 rows, 22,970 onsets) | 2019-2025, January-July | 0.0585 | 0.2861 |
| `fwi_season`, all scored rows | | 0.0869 | 0.2056 |
| `fwi_season`, January-July rows only | | 0.0947 | 0.2036 |
| earlier `B1` (next-day), `B1_l0` (same-day) | 2019-2025, January-July | 0.0862, 0.1073 | 0.1795, 0.1752 |
The January-July subset of the new rows has the same number of onsets as the earlier test set (22,970) and 353 fewer rows. The remaining difference between 0.0801 and 0.0585 for boosting was not traced further.

## B. 2025 holdout: earlier boosting AP 0.200 and climatology 0.117, now 0.2000 and 0.1167
Compared on the saved predictions (`holdout_predictions_main.pkl` against `predictions_2025.parquet`):
- Test rows: the earlier holdout has 982,750 rows; the 2025 fold has 980,554, all of them in the earlier set (2,196 rows only in the earlier one); labels agree on all shared rows.
- Fit years: 2019-2024 in both. Row cap: 2,000,000 in both; thinning seed 3 (earlier) against 1 (now); the earlier fit used the rows with complete 42 inputs, the present fit additionally requires the enriched arrays to be complete.
- Hyperparameters: earlier `M3` depth 6, learning rate 0.1, 300 iterations; present `boosting@tabular` depth 8, learning rate 0.05, 500 iterations. Inputs: 42 in both.
- Climatology: the two sets of predictions are identical on shared rows (maximum difference 0.0).
- Boosting: the predictions are not identical: on the 980,554 shared rows the maximum absolute difference is 0.687, the mean 0.086 and the correlation 0.953; AP on the shared rows is 0.2001 (earlier) and 0.2000 (now), 0.20004 and 0.20002 on each set's full rows. The agreement in AP is therefore not agreement of predictions. The logistic arms differ as well (mean absolute differences 0.099 to 0.102 on shared rows; correlations 0.95 to 0.998).
