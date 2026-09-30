# Labels

## Purpose
Define what the model predicts: a new fire start (onset) in a 5 km forest cell on a day, against matched days without one.

## Inputs
Fire detections, the grid (`grid.csv`), land cover, configuration in `configs/domain.yaml`.

## Rules
| Rule | Setting |
|---|---|
| Onset | a cell-day with a detection and none in the 3 days before |
| Forest domain | tree fraction of at least 0.20 |
| Season | months holding 90% of a state's onsets up to 2023 |
| Negatives | 20 per onset: random cell-day in the same state, year and season, with no detection on days D-3 to D |
| Predictors | observed before the target day (the day before, or earlier) |

The negatives are a sample, so the sampled onset share (0.0476) is much higher than the true rate. Probabilities are corrected to the true per-state rate (about 0.0147 overall) before log loss and Brier are computed.

## Outputs
`rows.csv` (labelled rows), `cell_days.csv` (all detection cell-days with the onset flag), `season.json`, `day_coverage.csv` (per-day satellite counts and gap flags).

## How to re-run
    wildfire build-labels

## Known limits
- Labels come from detections: a fire that no satellite saw is not a label, and a detection can be a burn that is not a wildfire.
- There are no cause labels.
- The negative sample is uniform in the season, so absolute AP is not forecast skill; compare arms, not the raw number.
- One random generator is shared across groups in a fixed order; changing group handling changes the negatives.
