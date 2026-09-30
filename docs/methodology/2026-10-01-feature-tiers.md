# Feature tiers

Features come in two tiers and every set is scored by the same harness.

1. Summary tier (run first): `tabular`, `weather_windows`, `neighbourhood`.
2. Imagery tier (only if the summary tier leaves a gap): `imagery_summary`, currently a placeholder.

Rule for all sets: a predictor must be observed strictly before the target day (`wildfire.guards.temporal`).
Window lengths, neighbourhood radii, metrics and splits are undecided and will be set in `configs/` only.
