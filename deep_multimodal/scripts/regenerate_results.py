"""Regenerate the results tables and sample hazard maps from the saved predictions (no raw data, no training).
Prints the code and configuration versions it used. Usage: python scripts/regenerate_results.py [--predictions results/saved_predictions] [--output results] [--draws 2000]"""
import argparse
import hashlib
import json
from pathlib import Path

import deep_multimodal
from deep_multimodal import evaluation, hazard_map, report

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--predictions", default=str(ROOT / "results" / "saved_predictions"))
parser.add_argument("--output", default=str(ROOT / "results"))
parser.add_argument("--draws", type=int, default=2000)
args = parser.parse_args()

config_path = ROOT / "configs" / "final.yaml"
print("deep_multimodal version:", deep_multimodal.__version__)
print("model:", deep_multimodal.MODEL_NAME)
print("configuration:", config_path.name, "sha256", hashlib.sha256(config_path.read_bytes()).hexdigest()[:16])
for year in evaluation.FOLDS:
    print(f"predictions {year}: sha256", hashlib.sha256((Path(args.predictions) / f"predictions_{year}.npz").read_bytes()).hexdigest()[:16])
results = evaluation.evaluate(Path(args.predictions), draws=args.draws)
report.write_report(results, Path(args.output))
(Path(args.output) / "checks.json").write_text(json.dumps(results["checks"], indent=1, default=float))
print("wrote results table, per-state, quiet-stratum and calibration files under", args.output)

import pandas as pd
cells = pd.read_csv(Path(args.predictions) / "cells.csv")
maps = Path(args.output) / "hazard_maps"
maps.mkdir(parents=True, exist_ok=True)
for year in evaluation.FOLDS:
    fold = evaluation.load_fold_predictions(Path(args.predictions), year)
    day = hazard_map.busiest_day(fold)
    frame = hazard_map.day_table(fold, cells, day)
    hazard_map.draw_day(frame, f"Deep multimodal model, day {day} of the {year} fold (most onsets)", maps / f"hazard_map_{year}.png")
    print("hazard map", year, "day", day, "cells", len(frame), "onsets", int(frame.label.sum()))
