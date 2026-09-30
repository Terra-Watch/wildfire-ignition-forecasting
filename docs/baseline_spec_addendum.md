# Baseline specification addendum

Written after the specification was committed. It does not change the specification; it records facts found while preparing the data.

## Fire-data completeness
Checked from the raw files (script and numbers: `wildfire-research/studies/fresh_states/`).
| Area | Result |
|---|---|
| Southern strip (8-15N, 72-90E), both satellites | 512 five-day chunks each, 2019-01-01 to 2026-01-03, no gaps between chunks, no duplicate detections |
| Main box (15-32N, 72-88.5E), both satellites | 567 chunks each, 2018-04-01 to 2025-12-31, spacing 5 days (one 2-day step at the turn of 2024/2025), no gaps |
| Eastern band 88.5-90E | almost no detections (3 in seven years): open sea and border land, so it adds no test rows |
Days with zero detections exist in the southern strip (129 for NOAA-20, 162 for SNPP in seven years); they cannot be told apart from quiet days in the raw files alone. They fall mainly in the monsoon months, outside the fire season used for sampling.
States passing the check: UK, AP (main box); KA, TN, KL (southern strip) pass but have only reserve and buffer cells, so they add no test rows (see the decision on scope).

## Sealed season
Unavailable (see `docs/decisions/2026-10-01-sealed-season-unavailable.md`). 2025 is not used in its place.
