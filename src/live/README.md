# Live scoring job

Scheduled: `~/Library/LaunchAgents/com.terrawatch.dailyforecast.plist` runs `run_daily.sh` at 17:00 local (11:30 UTC); state, logs and run outputs are in `wildfire-ignition-forecasting/live_state/` (`daily.log`). Stop it with `launchctl bootout gui/$(id -u)/com.terrawatch.dailyforecast`. A full run takes about 15 minutes.

Daily sequence for forecast day d (the forecast is for d + 1), run after the 06 UTC GFS run is out (about 11:00 UTC):

1. `make_bundle.py` (once per model release): builds the bundle and its golden test from finished study runs.
2. Fire history: extend the store with FIRMS rows (`fire.fetch_firms` + `fire.daily_flags`), starting from `fire.store_from_dataset`. The store was back-filled from the dataset's last detections (3 Jan 2026) to 1 Nov 2026; `refresh_fire.py` extends it daily.
3. `refresh_weather.py --day d --cells BUNDLE/cells.parquet --store W.parquet` (Open-Meteo; see its header for rate limits).
4. `gfs_fetch.py --day YYYYMMDD ...` in the environment that has eccodes (`wildfire-dataset/.venv-gfs`).
5. `score_daily.py --day d --bundle B --weather W --fire F --gfs gfs_YYYYMMDD.npz --out RUN` writes run.json, scores.parquet and scores.csv.
6. `website/backend`: `python -m inference.import_live_run RUN` loads the run; the API serves it through `/api/risk/geojson` and `/api/risk/status`.

Tests: `RISK_TEST_BUNDLE=BUNDLE python -m unittest tests.test_bundle tests.test_live_fire` (bundle golden test, fire-history parity).
Known gaps are flagged in each run (`degraded`): NDVI is a climatology, a missing GFS run scores with the GFS channels at zero.
