# Deviation: soil-moisture anomaly inputs use a norm that includes the validation years
Status: deviation from the "past information only" principle, logged for approval. Extends the earlier record `2026-10-01-soil-moisture-anomaly-look-ahead.md`; nothing is changed in code or in the stamped specification.

What is affected: two of the 42 tabular inputs, `sw1_anom` and `sw2_anom` (soil moisture minus a norm).
- The norm is `soil_climatology` in `features/tabular/weather.py`: the same-ISO-week mean of noon soil moisture over 2017-2023 (`soil_climatology_years` in `configs/features.yaml`), computed once per weather-grid pixel from weather only. No label enters it.
- For a row in 2021-2023 the norm therefore contains the row's own year and later years. For rows in 2024 and 2025 it is entirely in the past. The look-ahead is in weather statistics, not in labels, and only for validation years 2021-2023 (which are also the tuning years).
- UK and AP rows (the fresh geographic test) use the same function, the same 2017-2023 window and the pixel of their own cell; the computation does not depend on the cell's grid role. So their anomalies are inside the norm for 2019-2023 and entirely past for 2024-2025.
- The weather-window anomalies (17 columns of the enriched level) are not affected: their norm is rebuilt per fold from the fit years only (`dataset.anomaly_climatology`).

What the guard `no_validation_information_in_tuning` tests, and why it passes: it compares two numbers in `configs/baseline.yaml` (`max(tuning_years)` = 2024 is below `last_development_fold` = 2025). It does not read any data or any climatology, although its docstring says it also covers the climatology. It passes because 2024 < 2025; it cannot detect the 2017-2023 norm. `validity.climatology_first_year_is_nan` (the label-based cell-week climatology must be empty in the first year) exists but no pipeline calls it.

Effect on results: measured for one model only. Ablation A (`2026-10-01-ablations.md`: `boosting@enriched` without `sw1_anom` and `sw2_anom`, same folds, rows and settings) gives ΔAP +0.0010 (99% CI -0.0003 to 0.0025) against `boosting@enriched`. The logistic arms and the network were not tested. The deviation stands.

Why it is not fixed now: the 42 inputs are the reproduced inputs and the specification is stamped; changing them would change the baseline after scoring started.

Needs approval: keep the 42 inputs as they are (recommended), or recompute the two anomalies per fold from fit years and re-run everything.
