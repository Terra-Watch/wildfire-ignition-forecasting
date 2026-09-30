# Models

## Purpose
List the arms compared, how each is fitted and tuned.

## Arms
| Arm | What it is |
|---|---|
| climatology_cell | rate of onsets in the same cell and ISO week in earlier years, calibrated |
| landcover_season | logistic model on land cover and season |
| fwi_season | logistic model on the six fire-weather codes and season |
| logistic_weather_static | logistic model on the 42 inputs without burn history |
| logistic_with_burn | the same plus two burn-history inputs |
| boosting | histogram gradient boosting on the 42 inputs |
| tabular_network | multilayer perceptron (3 layers of 256, dropout 0.1, AdamW), trained with design weights, early stopping, 10 seeds |
| coords_only, cell_id_only, static_only | diagnostics: what a model knows from the cell's place alone |

Logistic, boosting and network arms run at two information levels: tabular (42 inputs) and enriched (tabular + weather windows + neighbourhood).

## Tuning
Boosting and network get the same budget: 12 random-search trials, rolling-origin validation on 2021-2024, choice by validation log loss (AP as tie-break). Logistic regularisation is fixed at the value chosen earlier. Every trial is saved.

## Inputs / Outputs
Inputs: model table and enriched arrays. Outputs: `tuning_trials.csv`, `tuned.json`, per-year prediction files under the results folder.

## How to re-run
    wildfire tune
    wildfire rolling-origin

## Known limits
- Each fit uses at most 2,000,000 rows (all onsets kept), tuning fits at most 600,000.
- The network stops early on the last fit year, so it trains on one year fewer than the other arms.
- Tuning was done at the enriched level and reused at the tabular level.
