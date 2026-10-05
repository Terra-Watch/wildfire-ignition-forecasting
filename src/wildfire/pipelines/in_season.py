"""In-season row set and the baseline re-run on it.

The model table first built for the project holds rows of every calendar month of the validation years. Negative rows exist only in the fire-season months of
each state, so the day of the year alone tells a model that an out-of-season row is a fire. The baseline of record therefore uses in-season rows only: rows whose
calendar month is one of the state's fire-season months. This module filters the table and the arrays aligned with it into a second artifacts folder (the original
files are only read) and refits the whole ladder of arms on that folder with the frozen settings and no new tuning.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("1970-01-01")


def in_season_mask(day: np.ndarray, state: np.ndarray, season_months: dict) -> np.ndarray:
    """True for rows whose calendar month is a fire-season month of the row's state. `day` counts days since 1970-01-01."""
    month = pd.to_datetime(EPOCH + pd.to_timedelta(np.asarray(day), "D")).month
    return np.array([m in season_months[s] for m, s in zip(month, np.asarray(state))])


def file_hash(path: Path, block: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def _link(source: Path, target: Path) -> None:
    if not target.exists():
        target.symlink_to(source)


def build_in_season_folder(source: Path, target: Path) -> dict:
    """Write the in-season model table and its row-aligned arrays under `target` from the original artifacts folder `source`; returns the check record.

    Weather windows are stored as (columns, rows) and the neighbourhood block as (rows, columns). Files that do not depend on the rows are linked, not copied.
    """
    for name in ("features", "labels", "results", "cache"):
        (target / name).mkdir(parents=True, exist_ok=True)
    table = pd.read_pickle(source / "features" / "model_table.pkl")
    season = json.loads((source / "labels" / "season.json").read_text())
    keep_mask = in_season_mask(table.day.values, table.state.values, season)
    outside = ~keep_mask
    if not table.label.values[outside].all():
        raise ValueError("an out-of-season row is a negative; the support rule assumes only onsets lie outside the season")
    keep = np.flatnonzero(keep_mask)
    features = target / "features"
    kept = table.iloc[keep].reset_index(drop=True)
    kept.to_pickle(features / "model_table.pkl")
    np.save(features / "keep_index.npy", keep)
    windows = np.load(source / "features" / "windows.npy", mmap_mode="r")
    windows_out = np.lib.format.open_memmap(features / "windows.npy", mode="w+", dtype=np.float32, shape=(windows.shape[0], len(keep)))
    for j in range(windows.shape[0]):
        windows_out[j] = windows[j][keep]
    windows_out.flush()
    neighbourhood = np.load(source / "features" / "neighbourhood.npy", mmap_mode="r")
    np.save(features / "neighbourhood.npy", np.asarray(neighbourhood[keep]))
    for name in ("arm_columns.json", "day_coverage.csv"):
        _link(source / "features" / name, features / name)
    _link(source / "features" / "daily_weather", features / "daily_weather")
    for name in ("season.json", "cell_days.csv"):
        _link(source / "labels" / name, target / "labels" / name)
    for name in ("tuned.json",):
        if (source / "results" / name).exists() and not (target / "results" / name).exists():
            shutil.copy(source / "results" / name, target / "results" / name)
    identical = {c: bool(np.array_equal(table[c].values[keep], kept[c].values, equal_nan=True)) for c in ("cell", "day", "label")}
    return dict(rows_before=len(table), rows_after=len(kept), onsets_removed=int(table.label.sum() - kept.label.sum()),
                negatives_removed=int((table.label == 0).sum() - (kept.label == 0).sum()), columns_identical_after_filter=identical,
                sha256={"in_season_model_table": file_hash(features / "model_table.pkl"), "keep_index": file_hash(features / "keep_index.npy")})


def run_year(dataset_, tuned: dict, year: int, results: Path, code_version: str, log=print) -> None:
    """Fit the whole ladder for one validation year and save its predictions and fit prevalences (skipped when the files exist)."""
    from wildfire.pipelines import dataset as dataset_module, ladder_fit
    predictions, prevalence = results / f"predictions_{year}.parquet", results / f"fit_prevalence_{year}.json"
    if predictions.exists():
        log("exists", year)
        return
    store = ladder_fit.FoldStore(results / f"fold_{year}_models", code_version)
    frame, fit_prevalence = ladder_fit.run_fold(dataset_, year, tuned, log, store)
    frame.to_parquet(predictions)
    prevalence.write_text(json.dumps(fit_prevalence))
    store.clear()
    dataset_module.clear_cache()
    log("saved", year, frame.shape)
