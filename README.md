# wildfire-ignition-forecasting

Models for next-day forest-fire occurrence, on the v4 forest-boundary datasets in `../wildfire-dataset/dataset/`.
Protocol (frozen before any fit): `../wildfire-dataset/DECISIONS.md`, section "PRE-REGISTERED MODEL COMPARISON" and its amendment.

- `src/data.py`: loader for `forest_all` (training states) and the guarded loader for `forest_unseen` (blind states, step 6 only).
- `src/metrics.py`: AUPRC, AUROC, top-k, Brier, ECE, paired week-block bootstrap.
- `tests/`: `PYTHONPATH=. .venv/bin/python -m unittest discover -s tests`
- `results/`: one folder per run (config hash, metrics, predictions).

Groups for leave-one-group-out = the `set` column: six, D1 (AP), D2 (UK), south, himachal, northeast.
