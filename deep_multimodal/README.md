# Deep multimodal model for next-day wildfire occurrence

A neural network that estimates, for each 5 km forest cell and day, the probability that a new fire starts in the cell the next day. It reads four streams: the fire history of the neighbourhood over the previous 14 days (a transformer), a 60-day weather sequence (a recurrent network), the day-ahead weather forecast, and static cell features, fused by cross-attention. The final model is an ensemble of 8 seeds trained with hard labels only. The output is a probability per cell and day, not an ignition point.

This folder is self-contained: it does not import the baseline library of the parent repository. It is compared with the **extended boosting baseline** (see `../docs/extended_boosting_baseline.md`).

## Contents
| Path | What it holds |
|---|---|
| `src/deep_multimodal/` | the model, data loading, training, metrics, evaluation, reports and hazard maps |
| `configs/final.yaml` | the final settings |
| `scripts/` | `regenerate_results.py` (tables and maps from saved predictions), `train_ensemble.py`, `make_synthetic_package.py`, `export_saved_predictions.py` |
| `tests/` | tests that run on a CPU with a synthetic package and the saved predictions |
| `docs/DATA.md`, `docs/MODEL.md`, `docs/RESULTS.md` | data, model and results descriptions (including limitations) |
| `results/` | result tables, `saved_predictions/` (the two scored folds), `hazard_maps/` |
| `notebooks/` | data, model, results and hazard maps; they run from the synthetic sample and the saved predictions |

Raw data, model input packages and trained weights are **not** in the repository (see `docs/DATA.md` and `docs/MODEL.md`).

## Install
    python3 -m venv .venv
    .venv/bin/python -m pip install -e ".[dev]"      # run from this folder

Python 3.10 or newer; PyTorch is installed by the command above.

## Run
    .venv/bin/python -m pytest -q                              # about 10 seconds on a CPU
    .venv/bin/python scripts/regenerate_results.py             # about 2 minutes: tables, calibration, maps; prints code and configuration versions
    .venv/bin/python scripts/make_synthetic_package.py /tmp/sample_package
    .venv/bin/python scripts/train_ensemble.py --package /tmp/sample_package --year 2022 --synthetic --output /tmp/sample_run --seeds 0 1
    .venv/bin/python scripts/train_ensemble.py --package <real package> --year 2024 --output <folder>     # needs a real package and a GPU

Notebooks (run from this folder after installing):

    .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_data.ipynb

## Results in one table (validation years, same rows for both models)
| Fold | Average precision, baseline | Average precision, deep model | Difference (95% interval) |
|---|---|---|---|
| 2024 | 0.2267 | 0.2913 | +0.0646 (+0.0543 to +0.0734) |
| 2023 (provisional) | 0.1797 | 0.2532 | +0.0734 (+0.0549 to +0.0872) |

The deep model is ahead on almost every metric, but not all (calibration on 2023 and the quiet stratum are worse; see `docs/RESULTS.md`), and its inputs are richer than the baseline's, so the gain is not attributed to the architecture alone. Both folds were development folds. **Every fold-2023 value, and every probability of detection, false alarm ratio and critical success index, is provisional** (fold-2023 soil-moisture anomalies use a norm that includes later 2023 weather; every model's settings were chosen on scored rows; the fold-2023 alert threshold was chosen on fold-2024 labels). See Limitations in `docs/MODEL.md`.
