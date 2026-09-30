# Data sources

Data stays outside the repository. `DATA_ROOT` (see `.env.example`) must contain `raw/` and `processed/`; `configs/data.yaml` defines every dataset and `data/MANIFEST.json` records its file count, size and content hash. Rebuild the manifest with `make manifest`, check it with `wildfire verify-data`.

| dataset | location | coverage | state |
|---|---|---|---|
| fire_detections | raw/firms, raw/firms_ext | 2018-04 to 2025-12 | complete |
| weather_hourly | raw/era5 | 2017-01 to 2025-12 | main and extra_a groups complete; extra_b partial (54 of 108 months), remaining items marked abandoned in `raw/era5/status.json` |
| night_lights | raw/viirs_nl | 2018 to 2024 | complete; raw product has an unexplained level shift in 2022 |
| admin_boundaries | raw/*.geojson | n/a | complete |
| grid, static_layers | processed | 52,750 cells | complete |
| vegetation_index | processed | 16-day composites | catalogue gaps |
| burned_area | processed | monthly | 8 months absent upstream; one tile missing for 2025-07 |
| derived_tables | processed | n/a | rebuildable, not authoritative |

Sources are the earlier working data folders, exposed through neutral links under `~/wildfire-data`.
