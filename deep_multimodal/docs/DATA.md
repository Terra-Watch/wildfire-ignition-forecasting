# Data

## What the model predicts
For each 5 km forest cell and each day, the probability that a new fire starts in that cell on that day, given only what was observed up to the day before. It is a probability per cell and day, not an ignition point and not a fire-spread forecast. A fire start ("onset") is a satellite fire detection in a cell with no detection in the 3 days before. Days are UTC calendar days.

## Raw data is not in this repository
No raw data, no model input packages and no trained weights are stored here. This repository holds code, settings, small result tables and the saved predictions of the two scored folds (`results/saved_predictions/`). Everything that needs the raw sources has to be rebuilt from them (see "How to obtain the data").

**The grid files were reconstructed after a deletion.** The original grid folder (`grid.csv` and `grid_meta.json`, the cell table of the 5 km analysis grid) was deleted by mistake and could not be recovered. A replacement was rebuilt from the cell coordinates and states that survived in saved tables. It carries the cell centres (longitude, latitude, projected coordinates) of the study cells, but not the lattice indices, and it is not byte-identical to the original, so the checksum recorded for the original grid in the data manifest of the baseline repository **cannot match**. Two checks of the baseline repository that need the lattice indices are skipped for that reason. The saved predictions and the cell coordinates used for maps here (`results/saved_predictions/cells.csv`) do not depend on the lost indices; the coordinates agree with the saved coordinates of the 4,780 cells for which both exist.

## Rows
One row is one cell on one target day D. Inputs use only days up to D-1. Study area: forest cells (tree fraction at least 0.20) of six states: Chhattisgarh (CG), Jharkhand (JH), Maharashtra (MH), Madhya Pradesh (MP), Odisha (OD) and Telangana (TS); 17,764 cells appear in the model table. Only days in each state's fire season are used (the months that hold 90% of the state's onsets up to 2023). Rows with a missing input are dropped.

Every onset is kept as a positive row. Negative rows are a sample: for each onset about 20 random cell-days of the same state, year and season with no detection on days D-3 to D. Fit rows are thinned to at most 2,000,000. Row weights turn the sample back into the real population (onsets weigh 1; negatives weigh the number of unsampled cell-days they stand for), so Brier score, log loss and calibration are weighted, and raw model probabilities are corrected to the true per-state fire rate (see MODEL.md). Rank metrics (average precision, AUROC, top shares) use the raw scores on the sample, as the baseline does, so absolute average precision is not forecast skill; compare models on the same rows.

## Folds
Models are always trained on years before the one they are scored on (rolling origin).

| Scored year | Fit years | Fit rows | Of which stage-1 training | Stopping rows (last fit year) | Scored rows | Onsets among scored rows |
|---|---|---|---|---|---|---|
| 2024 | 2019 to 2023 | 1,977,409 | 1,518,464 | 458,945 | 725,441 | 36,841 |
| 2023 | 2019 to 2022 | 1,970,004 | 1,436,314 | 533,690 | 936,380 | 47,168 |

Stage 1 trains on the fit years before the last fit year and stops on the last fit year; stage 2 retrains on all fit rows (see MODEL.md). Scored rows are the January-to-July rows of the scored year (the in-season rows). Fold 2024 was also the year on which the settings were chosen; fold 2023 replicates them without change. **No separate test fold is reported for this model**: neither fold is untouched, and the numbers are development-fold numbers. The baseline is scored on the same rows.

## Inputs
All inputs are aligned to the 5 km cell and the UTC day. "Window" means the days before the target day D, never D itself.

| Input | Source | Native resolution | Time window | Units and form | Alignment to the cell-day |
|---|---|---|---|---|---|
| Fire history (neighbourhood by days) | Satellite active-fire detections from the two VIIRS instruments (confidence nominal or high), 2018-2025 | 375 m points, daily | days D-14 to D-1 | per neighbour cell and day: detected (0/1), onset (0/1), log of count, log of radiative power | points binned to cells; a 7 by 7 window (radius 3 cells) around the target cell; only cells of the fit grid role count as neighbours; the 32 occupied (day, neighbour) entries nearest in space, then most recent, are kept as tokens |
| Recent fire summaries | same detections | cell | days D-7 to D-1 and D-14 to D-1 | detection records in the 8 cells around the target cell and in the 48 cells of the 7 by 7 window around it over 7 days; number of days with a detection in the own cell over 14 days; days since the last own-cell detection (15 if none) | counted from the cell table |
| Weather sequence | ERA5-Land hourly reanalysis (Copernicus), 2017-2025, and the fire-weather codes computed from it | about 9 km grid (5,255 weather points hold study cells), noon values per day | days D-60 to D-1 | 20 daily channels: air temperature (degrees C), vapour pressure deficit, wind speed (km/h), relative humidity (%), soil moisture of two layers, 7-day rain, days since rain, six fire-weather codes (FFMC, DMC, DC, ISI, BUI, FWI), maximum temperature at 09 UTC, two soil-moisture anomalies, daily rain (mm), wind components u and v (m/s); plus 17 anomalies against the day-of-year average of the fit years | each cell takes the series of its weather point (several 5 km cells share one point) |
| Day-ahead forecast | NOAA GFS 0.25 degree forecasts, the 06 UTC run of day D-1, valid on day D (forecast hours 18 to 41) | 0.25 degree, hourly | valid day D, hourly steps 06 to 17 UTC | 5 variables (2 m temperature, relative humidity, 10 m wind u and v, precipitation): 12 hourly values and the daily minimum, maximum, mean and sum of each (20 values); stored as integers (value times 10 for humidity, times 100 for the rest) | mean of the 5 by 5 forecast pixels around the cell centre; the forecast is issued before day D, never an analysis field; two days (2022-04-20 and 2022-04-21) have no complete run and stay missing (masked, never filled) |
| Static features, 21 columns | terrain from an elevation model (mean, standard deviation, slope, sine and cosine of aspect, relative position); land cover shares (WorldCover classes 10, 20, 30, 40, 50, 60, 80, 90), edge density and log distance to built-up land; human layers (log population, two components of the global human modification index, night-time lights percentile rank, road length per area) | 10 m to 1 km, summarised per 5 km cell | one epoch per layer (time-invariant) | as named; standardised with fit-row statistics | summarised per cell |
| Burn history, 2 columns | monthly burned-area product, 2000 to 2025-07 | per cell and month | years before the target day | log of one plus the number of burns in the 3 by 3 neighbourhood over the look-back window (missing where the burned-area months were not observed); years since the last observed burn (capped) | per cell |
| Calendar, 2 columns | the date | | | sine and cosine of the day of the year | |
| Auxiliary target (training only) | detections on the target day | cell | day D | log(1 + detections) and log(1 + radiative power) | never an input; used only as a small extra training loss |

Every input uses information available before day D. The weather window ends on D-1, fire history on D-1, the forecast is an issued forecast, and a truncation test (changing the data of day D must not move any input) is part of the baseline repository's guards.

## Known gaps in the data
- Burned-area months missing upstream (2023-09, 2024-07, and others) stay missing; rows needing them are dropped.
- Detections of both instruments are missing on some days (2022-07-24 to 2022-08-08 and 2024-05-29 to 2024-06-02); rows near them are dropped.
- The night-lights product has a level shift in 2022; one frozen epoch is used.
- Weather is about 9 km and the forecast 0.25 degree; several 5 km cells share one value.
- Detections are a proxy for fires, not a record of causes, and a fire no satellite saw is not a label.

## Model input package (what the code reads)
The code reads a package of plain numpy arrays per validation year (written by a package builder that is not part of this repository). `src/deep_multimodal/synthetic.py` writes a small random package with the same layout, used by the tests and notebooks.

| Folder | Files |
|---|---|
| `fold_<year>/` | `meta.json` (year, fit prevalence, stopping-year pool counts), `fit_` and `val_` `token_features.npy` (rows, 32, 4), `token_positions.npy` (rows, 32; negative = empty slot; position = lag index times 49 plus neighbour cell index), `key.npy` (cell, day), `y.npy`; `fit_year.npy` |
| `dm_<year>/` | `meta.json` (names the folders above and below), `fit_extra.npy`, `val_extra.npy` (rows, 9; column order in `constants.ROW_COLUMNS`), `fit_aux.npy` (rows, 3; detections, radiative power, next-day detection) |
| `dm_common/` | `series.npy` (days, 20, weather points; half precision, NaN = missing), `cell_point.npy`, `static.npy` (cells, 21), `meta.json` |
| `fc_lookup/` | `gfs_i16.npy` (pixels, days, 80 integer codes), `days.npy`, `cell_pixel.npy`, `cell_state.npy` |

## How to obtain the data
1. Fire detections: NASA FIRMS archive of VIIRS active fires (both instruments), downloaded as daily files; requires a free NASA FIRMS key.
2. Weather: ERA5-Land hourly data from the Copernicus Climate Data Store (free account); the baseline repository's `scripts` and `docs/data.md` list the variables and months, and compute the fire-weather codes.
3. Forecast: the NOAA GFS 0.25 degree collection in Google Earth Engine (free research account); export per month the 06 UTC run values for the 5 by 5 pixel patches with the integer scaling above.
4. Static layers: ESA WorldCover, an elevation model, human-modification, population, road and night-lights layers, summarised per cell; burned-area product per month.
5. Build the grid, labels and model table with the baseline repository (`wildfire build-labels`, `build-features`, `build-model-table`), then write the packages. Row counts of the packages must equal the table above; a mismatch means the build differs from the one behind the saved predictions.
