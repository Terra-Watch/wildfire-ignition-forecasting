# Data

## Purpose
Say what data the project uses, where it lives, and how complete it is.

## Inputs
| Source | What it is | Coverage |
|---|---|---|
| Fire detections | Satellite active-fire points (two VIIRS satellites), confidence "nominal" or "high" only | 2019-2025 used |
| Hourly weather | ERA5-Land reanalysis: temperature, dew point, wind, rain, soil moisture | 2017-2025 |
| Land cover | Tree fraction and land-cover classes per 5 km cell | one epoch |
| Terrain | Height, slope, aspect, relative position per cell | static |
| Human layers | Population, human-modification index, roads, night lights (frozen at one epoch each) | static |
| Burned area | Monthly burned-area product, per cell | 2000-2025-07 |
| Vegetation index | 16-day composites, per cell | to 2025 |

Data stays outside the repository. `DATA_ROOT` points at it; `configs/data.yaml` lists every dataset; `data/MANIFEST.json` records file counts, sizes and content hashes.

## Outputs
Nothing is written to the data folders. Everything derived goes under `ARTIFACTS_ROOT`.

## How to re-run
    wildfire data-root          # print where the data is expected
    wildfire verify-data        # compare the folders with data/MANIFEST.json
    make manifest               # rebuild the manifest after a deliberate change

## Known limits
- Burned-area months missing upstream stay missing (2023-09, 2024-07, 2025-05 and from 2025-08 on).
- Two satellites' detections are missing on some days (2022-07-24 to 08-08, 2024-05-29 to 06-02); rows near those days are dropped.
- The night-lights product has an unexplained level shift in 2022; the model uses one frozen year.
- Detections are a proxy for fires, not a record of causes.
