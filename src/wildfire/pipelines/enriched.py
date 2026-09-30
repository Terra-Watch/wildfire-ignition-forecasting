"""Build the enriched feature arrays (weather windows, neighbourhood) aligned row by row with the model table."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.features import neighbourhood, weather_windows
from wildfire.labels.detections import load_grid
from wildfire.pipelines.features import load_model_table

WINDOWS_FILE = "windows.npy"
NEIGHBOURHOOD_FILE = "neighbourhood.npy"
SERIES_FOLDER = "daily_weather"


def build_series(log=print) -> None:
    """Store the daily weather series at the weather cells that hold study cells."""
    t0 = time.time()
    cells, _ = load_grid()
    series = weather_windows.build_series(cells)
    weather_windows.save_series(series, config.output_dir("features") / SERIES_FOLDER)
    log("daily series", series["values"].shape, "runtime_s", round(time.time() - t0))


def build_arrays(log=print, table_name: str = "model_table", prefix: str = "") -> None:
    """Window and neighbourhood features for every row of a model table, written as memory-mapped arrays."""
    t0 = time.time()
    feats = config.output_dir("features")
    table, _ = load_model_table(table_name)
    series = weather_windows.load_series(feats / SERIES_FOLDER)
    cell, day = table.cell.values, table.day.values
    windows = np.lib.format.open_memmap(feats / (prefix + WINDOWS_FILE), mode="w+", dtype=np.float32,
                                        shape=(len(table), len(weather_windows.window_names())))
    weather_windows.window_matrix(series, cell, day, out=windows)
    windows.flush()
    log("windows", windows.shape, "runtime_s", round(time.time() - t0))
    cd = pd.read_csv(config.output_dir("labels") / "cell_days.csv")
    onsets = cd[cd.onset][["cell", "day"]]
    np.save(feats / (prefix + NEIGHBOURHOOD_FILE), neighbourhood.neighbourhood_matrix(cell, day, series, onsets))
    log("neighbourhood done", "runtime_s", round(time.time() - t0))


def load_arrays(prefix: str = "") -> dict:
    """Memory-mapped enriched arrays and the weather series."""
    feats = config.output_dir("features")
    return dict(windows=np.load(feats / (prefix + WINDOWS_FILE), mmap_mode="r"),
                neighbourhood=np.load(feats / (prefix + NEIGHBOURHOOD_FILE), mmap_mode="r"),
                series=weather_windows.load_series(feats / SERIES_FOLDER))
