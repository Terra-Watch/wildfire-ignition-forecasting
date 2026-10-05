"""Pack the saved predictions of one validation year into results/saved_predictions/predictions_<year>.npz.

Inputs (all produced by earlier runs; none is read from raw data):
  --baseline-table   parquet file with one row per validation row: cell, day, year, week, state, label, weight, pi_true, p_boosting@enriched, p_climatology_cell
  --logits-dir       folder with headline_seed0.npy ... headline_seed7.npy, the validation logits of the 8 seeds, in the row order of the baseline table
  --val-rows         the per-row columns file of the package (dm_<year>/val_extra.npy); column 4 is the neighbouring-detections count of the last 7 days (quiet stratum when 0)
The row weight and the true fire rate take one value per (state, label) and per state; they are stored as small tables and checked to rebuild every row exactly.
Usage: python scripts/export_saved_predictions.py --year 2024 --baseline-table ... --logits-dir ... --val-rows ... [--output results/saved_predictions]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True, choices=[2023, 2024])
parser.add_argument("--baseline-table", required=True)
parser.add_argument("--logits-dir", required=True)
parser.add_argument("--val-rows", required=True)
parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "results" / "saved_predictions"))
args = parser.parse_args()

columns = ["cell", "day", "year", "week", "state", "label", "weight", "pi_true", "p_boosting@enriched", "p_climatology_cell"]
table = pd.read_parquet(args.baseline_table, columns=columns)
logits = [np.load(Path(args.logits_dir) / f"headline_seed{k}.npy") for k in range(8)]
assert all(len(z) == len(table) for z in logits), "seed arrays and baseline rows differ in length"
deep = np.mean([1 / (1 + np.exp(-z)) for z in logits], 0)
quiet = np.load(args.val_rows)[:, 4] == 0
assert len(quiet) == len(table)
states = sorted(table.state.unique())
state_index = table.state.map({s: i for i, s in enumerate(states)}).values.astype(np.int8)
weight_table = np.zeros((len(states), 2))
true_rate = np.zeros(len(states))
for i, s in enumerate(states):
    for label in (0, 1):
        values = table.weight[(state_index == i) & (table.label.values == label)].unique()
        assert len(values) == 1, "row weight is not one value per state and label"
        weight_table[i, label] = values[0]
    values = table.pi_true[state_index == i].unique()
    assert len(values) == 1
    true_rate[i] = values[0]
assert np.array_equal(weight_table[state_index, table.label.values], table.weight.values)
assert np.array_equal(true_rate[state_index], table.pi_true.values)
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output / f"predictions_{args.year}.npz", cell=table.cell.values.astype(np.int32), day=table.day.values.astype(np.int32), week=table.week.values.astype(np.int8),
                    year=np.int16(args.year), state_index=state_index, state_names=np.array(states), label=table.label.values.astype(np.uint8), quiet=quiet,
                    weight_table=weight_table, true_rate=true_rate, deep_probability=deep.astype(np.float32), baseline_probability=table["p_boosting@enriched"].values.astype(np.float32),
                    climatology_probability=table["p_climatology_cell"].values.astype(np.float32))
print("wrote", output / f"predictions_{args.year}.npz", len(table), "rows")
