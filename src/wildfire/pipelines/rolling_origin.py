"""Rolling-origin validation: fit every arm on the years before each validation year and save its predictions."""
from __future__ import annotations

import json

import pandas as pd

from wildfire import config
from wildfire.guards import provenance
from wildfire.pipelines import dataset, ladder_fit, tuning


def prediction_paths(year: int):
    out = config.output_dir("results")
    return out / f"predictions_{year}.parquet", out / f"fit_prevalence_{year}.json"


def fold_folder(year: int):
    """Models of a fold that is still running, one file each (removed when the fold's predictions are saved)."""
    return config.output_dir("results") / f"fold_{year}_models"


def load_year(year: int) -> tuple[pd.DataFrame, dict]:
    """Saved predictions and fit prevalences of one validation year."""
    pred, prev = prediction_paths(year)
    return pd.read_parquet(pred), json.loads(prev.read_text())


def run(log=print) -> None:
    """Fit and predict every validation year that has not been done yet. The run resumes at model granularity: every
    finished model of an unfinished year is kept and not fitted again."""
    from wildfire.guards import baseline_guards
    provenance.require_clean()
    baseline_guards.require_record()
    tuned = tuning.load_tuned()
    ds = dataset.load_dataset()
    dataset.clear_cache()
    for year in config.load("baseline")["validation"]["validation_years"]:
        pred, prev = prediction_paths(year)
        if pred.exists():
            continue
        store = ladder_fit.FoldStore(fold_folder(year), provenance.code_version())
        out, prevalence = ladder_fit.run_fold(ds, year, tuned, log, store)
        out.to_parquet(pred)
        prev.write_text(json.dumps(prevalence))
        store.clear()
        dataset.clear_cache()
        log("saved", year, out.shape)
