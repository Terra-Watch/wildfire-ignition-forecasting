# Constants moved into configuration
Decision: days per year, the temperature offset and unit conversions of the weather code, numeric tolerances, row-chunk sizes, the early-stopping margin, the guard year and seed count, the series settling time and the presentation sizes now live in configs/ (features.yaml, models.yaml, guards.yaml, reporting.yaml). Reference-equation coefficients of the fire-weather codes stay in their module as part of the published equations. The weather code was re-run against the saved table after the change.
Why: everything configurable comes from configs/.
