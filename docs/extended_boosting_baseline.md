# Extended boosting baseline

## Purpose
Name the baseline that every later model is compared with, say exactly what it adds over the plain boosting version, and record its results.

## What it is
Histogram gradient boosting (scikit-learn) that predicts, for each 5 km forest cell and day, whether a new fire starts on the next day. It is called the **extended boosting baseline**. In the code and in older records the same model is called `boosting@enriched`; the plain version is `boosting@tabular`.

Settings (frozen, chosen by random search on validation years 2021 to 2024, never on 2025): depth 8, learning rate 0.05, 500 iterations, minimum 100 rows per leaf, no L2 penalty. Each fit uses at most 2,000,000 rows (all onsets kept, negatives thinned with a fixed seed). Rolling-origin validation: each validation year is predicted by a model fitted on the years before it (fit years start in 2019).

## What the extended version adds over plain boosting
Plain boosting reads 42 inputs. The extended boosting baseline reads the same 42 plus 209 more (251 in all):

| Block | Columns | What it holds |
|---|---|---|
| Plain inputs (both versions) | 42 | weather of the day before (11), fire-weather codes (6), terrain and land cover (16), human layers (5), season (2), burn history (2) |
| Weather windows (extended only) | 180 | mean and maximum of each of the 17 weather variables and fire-weather codes over the last 3, 7, 14, 30 and 60 days (170); 7-day change of the 6 fire-weather codes; rain sums over 7, 30 and 60 days; days since 2 mm of rain |
| Anomalies (extended only) | 17 | each weather variable and code minus its day-of-year average over the fit years |
| Neighbourhood (extended only) | 12 | for the 8 nearest and for the 24 nearest cells: mean tree cover, burn count, years since burn, vegetation index, fire-weather index, and the number of fire onsets in the last 28 days |

Every input uses data observed at or before the day before the target day (see `docs/features.md`).

## Recorded results (rows of the fire season only)
The baseline of record uses in-season rows only: rows whose month is one of the state's fire-season months. The reason: negative rows exist only in the season months, so on the original all-year rows the day of the year alone marked out-of-season rows as fires (a sampling artifact). The code is in `src/wildfire/pipelines/in_season.py`. 21,266 out-of-season onset rows are dropped; nothing else changes and no setting is tuned again.

Average precision (AP), same rows and same folds for both versions:

| Validation fold | Plain boosting | Extended boosting baseline | Difference |
|---|---|---|---|
| 2021 | 0.1608 | 0.2035 | +0.0427 |
| 2022 | 0.2085 | 0.2264 | +0.0179 |
| 2023 | 0.1528 | 0.1797 | +0.0269 |
| 2024 | 0.1807 | 0.2267 | +0.0460 |
| 2025 | 0.1863 | **0.2315** | +0.0452 |
| All five folds pooled | 0.1626 | **0.1977** | +0.0350 |

Paired difference on 2025 (exact bootstrap over 27 week blocks, 2,000 draws): +0.0452, 95% interval +0.0281 to +0.0586. Pooled: +0.0350, 95% interval +0.0264 to +0.0425. Fold 2025 was used in earlier work, so it is a development fold, not an untouched test.

Reference lines on the same 2025 rows: cell climatology 0.1155, fire weather index with season 0.1194. The extended network (mean of 10 seeds) matches the extended boosting baseline within noise (difference +0.0006 on 2025, interval -0.0142 to +0.0110).

For the earlier all-year row set, kept only as a reference line, the extended boosting scored 0.2461 on 2025 and 0.2261 pooled; those numbers are superseded.

### Metric table for 2024 and 2023
`results/extended_boosting_baseline/metric_table_2024_2023.csv` holds the full table for the two folds, with every metric defined in `deep_multimodal/docs/RESULTS.md`: AP and AUROC in three views, precision, recall and lift at the top 1% and 5% of cells per day, Brier score and skill against cell climatology, log loss, calibration error, probability of detection, false alarm ratio and critical success index (one threshold fixed on the other fold), per-state values and the quiet stratum (no neighbouring fire detection in the previous 7 days). The calibration curves are in `calibration_curve_2024_2023.csv`.

| Fold | AP (95% interval) | AUROC | Precision top 1% | Recall top 5% | Brier skill score | Calibration error |
|---|---|---|---|---|---|---|
| 2024 | 0.2267 (0.1704 to 0.2679) | 0.8346 | 0.1033 | 0.2092 | 0.0266 | 0.00048 |
| 2023 | 0.1797 (0.1419 to 0.2192) | 0.8059 | 0.1092 | 0.1871 | 0.0237 | 0.00079 |

Only AP has an interval; every other number is a point estimate.

## Files
- `results/extended_boosting_baseline/all_arms_by_fold.csv`: every arm, five folds and pooled, plain labels.
- `average_precision_by_state.csv` (2025 and pooled), `paired_differences.csv`, `metric_table_2024_2023.csv`, `calibration_curve_2024_2023.csv`.
- Notebook `notebooks/08_extended_boosting_results.ipynb` reads these files only.

## How to re-run
    python scripts/build_in_season_table.py "$ARTIFACTS_ROOT/in_season"
    ARTIFACTS_ROOT="$ARTIFACTS_ROOT/in_season" python scripts/rerun_in_season_baseline.py 2021 2022 2023 2024 2025

The first needs the original model table, windows and neighbourhood arrays from `wildfire build-features`, `build-model-table` and `build-enriched`. The second writes `predictions_<year>.parquet` and `fit_prevalence_<year>.json` under the in-season folder; the committed tables were produced this way and then converted to plain labels.

## Known limits
- The recorded tables come from a run on one machine; the raw data is not in the repository.
- The two soil-moisture anomaly inputs use a 2017-2023 climate that includes some validation years (see the decision record).
- Per-state values are saved for 2025 and the pooled folds only; the 2024 and 2023 per-state values are in the metric table.
