# wildfire-ignition-forecasting

Next-day forest fire detection forecasting for Indian forest cells: models, evaluation protocol and live scoring.
The data comes from the sibling folder `../wildfire-dataset/dataset/` (override with `WILDFIRE_DATA`).

## Layout

| Path | Content |
|---|---|
| `src/wildfire_forecast/` | the library: `data` (loaders), `metrics`, `harness` (train and score one model), `neural_nets`, `reference_models`, `train` (job queues), `evaluate`, `freeze`, `score_blind`, `confidence_intervals`, `paired_tests`, `permutation_importance`, `landcover_split`, `risk_map`, `rescore_check` |
| `scripts/` | numbered stages `01` to `11`, each a thin launcher for the module of the same name; `train_one_model.py` is called by the queues |
| `scripts/cloud/` | shell jobs for the cloud machines (training, evaluation, blind scoring, extras) |
| `scripts/live/` | building the live bundle and stores for the dashboard (see its README) |
| `notebooks/` | numbered notebooks that explain the data, `00` to `08` |
| `results/` | the final tables: `all_runs.csv`, `per_state.csv`, `paired.csv`, `seeds.csv`, `refs_only/` |
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
