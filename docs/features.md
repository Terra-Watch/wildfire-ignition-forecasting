# Features

## Purpose
Describe every input the models see and how each one avoids looking at the future.

## Inputs and tiers
| Tier | Columns | What it holds |
|---|---|---|
| tabular | 42 | weather of the day before (11), fire-weather codes (6), terrain and land cover (16), human layers (5), season (2), burn history (2) |
| weather windows | 180 + 17 | mean and max of each weather variable and code over the last 3, 7, 14, 30, 60 days; 7-day change of the codes; rain sums; days since rain; anomalies against the fit-year climate |
| neighbourhood | 12 | means over the 8 nearest and the 24 nearest cells of tree cover, burn history, vegetation index, fire-weather index; number of onsets in the last 28 days |
| imagery summary | - | declared, not built (no imagery in the data) |

Every predictor uses data at or before the day before the target day. The rule is checked in code (`assert_before_cutoff`), by a test that changes the target day's values and requires the features not to move, and by a deliberately shifted feature that must be detected.

## Outputs
`tabular.pkl`, `model_table.pkl`, `windows.npy`, `neighbourhood.npy`, `daily_weather/` under the artifacts folder.

## How to re-run
    wildfire build-features
    wildfire build-model-table
    wildfire build-weather-series
    wildfire build-enriched

## Known limits
- Two soil-moisture anomaly inputs use a 2017-2023 climate that includes some validation years (see the decision record).
- Burn history is missing where the burned-area product is missing; those rows have no burn inputs and are left out of arms that need them.
- Weather comes at about 9 km; several 5 km cells share one weather cell.
- Enriched features are computed for rows with no missing value in any block; the number of rows lost is shown in the features notebook.
