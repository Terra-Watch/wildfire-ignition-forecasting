# wildfire-ignition-forecasting

Next-day forest fire detection forecasting for Indian forest cells: models, evaluation protocol and live scoring.
The data comes from the sibling folder `../terrawatch-datasets/dataset/` (override with `WILDFIRE_DATA`).

## Layout

| Path | Content |
|---|---|
| `src/wildfire_forecast/` | the library: `data` (loaders), `metrics`, `harness` (train and score one model), `neural_nets`, `reference_models`, `train` (job queues), `evaluate`, `freeze`, `score_blind`, `confidence_intervals`, `paired_tests`, `permutation_importance`, `landcover_split`, `risk_map`, `rescore_check` |
| `scripts/` | numbered stages `01` to `15` (`12` to `15` build what the dashboard shows: `12_level_check` tests the levels, `13_forecast_levels` fits the calibration, areas and the week's usual, `14_named_places` builds the places table, `15_replay_area_forecast` writes replay days), each a thin launcher for the module of the same name; `train_one_model.py` is called by the queues |
| `scripts/cloud/` | shell jobs for the cloud machines (training, evaluation, blind scoring, extras) |
| `scripts/live/` | building the live bundle and stores for the dashboard (see its README) |
| `notebooks/` | numbered notebooks that explain the data, `00` to `08` |
| `results/` | the final study tables: `all_runs.csv`, `per_state.csv`, `paired.csv`, `seeds.csv`, `intervals.csv`, `freeze.json` (hash of the 131 runs), `blind/` (blind states, scored once), `extras/` (permutation importance, land-cover split), `run_manifests/` (manifest of every run) |
| `tests/` | unit tests |
| `runs/`, `runs_local/`, `live_state*/`, `bundle_inputs/` | run outputs and live state (not tracked) |

## Run

```
make check   # compile every file, no data needed
make test    # unit tests, needs the dataset
python scripts/02_train_models.py cpu --out runs      # one stage, arguments are passed to the module
```

The blind states are scored once, after `04_freeze_protocol.py`, with `WILDFIRE_FINAL=1 python scripts/05_score_blind_states.py`.
Exact package versions: `requirements.lock`.

## What the dashboard uses
The dashboard shows a calibrated chance per 0.5 degree area, not the model's scores. `scripts/13_forecast_levels.py` fits an isotonic calibration of the five-seed mean on the 2024 validation year (cells, then areas), the areas themselves, the usual fire days per area and week (2019 to 2023), the level cut-offs (10, 30 and 60 %), a default action table and the per-state reliability from 2025. It writes `results/forecast_levels.json`; `results/named_places.parquet` holds the villages and towns used to name areas and places to look. The daily job in `../website/forecast/` applies them (`area_forecast.py`). The label is a fire detection inside forest on the next day, new or continuing. Plan and status: `../docs/20-forecast-for-rangers-and-control-room.md`.
