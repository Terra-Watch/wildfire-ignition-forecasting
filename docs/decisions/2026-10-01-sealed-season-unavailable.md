# Sealed season (2026 Jan-Jul) unavailable
Decision: the sealed test is declared unavailable. 2025 is not substituted.
Why: the burned-area product has no items after 2025-07 (searched on 2026-10-01 for 2025-08 to 2026-07). With the 30-day lag and the 95% observation rule, every 2026 row would have missing burn inputs, so the burn arms (and every arm that includes them) cannot be scored. The other sources were not pulled because the test cannot be completed without burn.
Revisit: if the product catches up, pull the 2026 sources with the same checks as the development years and score once.

## Note: partial sealed-season pulls on disk (added 2026-10-01 after the decision)
Under `DATA_ROOT/sealed/` (times are local file times, 2026-10-01 IST; this decision was committed at 02:51 that day, so these pulls were made after it, with the download code committed at 04:01 under the burn-free-variants decision):
| Source | What is on disk | Pulled |
|---|---|---|
| FIRMS fire detections | 172 files: VIIRS SNPP and NOAA-20, main and south boxes, 43 five-day chunks each, 2026-01-01 to 2026-07-30 | 08:23-08:32 |
| ERA5-Land weather | `era5land_primary_2026-01.nc` and `era5land_primary_2026-02.nc` only; March to July were never received | 08:35 and 11:26 |
| Vegetation index | `ndvi_cells_2026.npz`, 16 composites (days 20441-20669) | 09:08 |
- The text above that "the other sources were not pulled" is therefore not accurate for these files; the conclusion (the season cannot be completed) stands, because ERA5 stops at February and the burned-area product has no items after 2025-07.
- None of these files is used for scoring. No scoring code for the sealed season exists, and no sealed score has been written.
- The March ERA5 request is abandoned and no further sealed pulls will be made.
- `wildfire sealed-coverage` was not run on these files and no coverage report exists.
