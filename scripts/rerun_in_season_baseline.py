"""Refit the whole ladder of arms (extended boosting baseline, plain boosting and every other arm) on the in-season table, one validation year per call.
The frozen tuned settings in <in-season folder>/results/tuned.json are used; nothing is tuned again.
Usage: ARTIFACTS_ROOT=<in-season folder> python scripts/rerun_in_season_baseline.py 2021 2022 2023 2024 2025"""
import json
import sys

from wildfire import config
from wildfire.pipelines import dataset, in_season

CODE_VERSION = "in-season-rerun-1"

if __name__ == "__main__":
    results = config.output_dir("results")
    tuned = json.loads((results / "tuned.json").read_text())
    data = dataset.load_dataset()
    dataset.clear_cache()
    print("rows", len(data.table), "complete", int(data.complete.sum()), flush=True)
    for year in [int(y) for y in sys.argv[1:]]:
        in_season.run_year(data, tuned, year, results, CODE_VERSION, log=lambda *a: print(*a, flush=True))
