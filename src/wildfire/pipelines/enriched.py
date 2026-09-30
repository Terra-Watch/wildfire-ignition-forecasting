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


def build_windows(log=print, table_name: str = "model_table", prefix: str = "") -> None:
    """Window features for every row of a model table, one feature per row of a memory-mapped file."""
    t0 = time.time()
    feats = config.output_dir("features")
    table, _ = load_model_table(table_name)
    series = weather_windows.load_series(feats / SERIES_FOLDER)
    t = table.day.values - 1 - series["first_day"]
    p = np.asarray(series["cell_point"])[table.cell.values]
    shape = (len(weather_windows.window_names()), len(table))
    windows = np.lib.format.open_memmap(feats / (prefix + WINDOWS_FILE), mode="w+", dtype=np.float32, shape=shape)
    for j, (_, column) in enumerate(weather_windows.column_series(series)):
        windows[j] = column[t, p]
    windows.flush()
    log("windows", shape, "runtime_s", round(time.time() - t0))


def build_neighbourhood(log=print, table_name: str = "model_table", prefix: str = "") -> None:
    """Neighbourhood features for every row of a model table."""
    t0 = time.time()
    feats = config.output_dir("features")
    table, _ = load_model_table(table_name)
    series = weather_windows.load_series(feats / SERIES_FOLDER)
    cd = pd.read_csv(config.output_dir("labels") / "cell_days.csv")
    matrix = neighbourhood.neighbourhood_matrix(table.cell.values, table.day.values, series, cd[cd.onset][["cell", "day"]])
    np.save(feats / (prefix + NEIGHBOURHOOD_FILE), matrix)
    log("neighbourhood", matrix.shape, "runtime_s", round(time.time() - t0))


def build_arrays(log=print, table_name: str = "model_table", prefix: str = "") -> None:
    """Both enriched blocks for a model table."""
    build_windows(log, table_name, prefix)
    build_neighbourhood(log, table_name, prefix)


def load_arrays(prefix: str = "") -> dict:
    """Memory-mapped enriched arrays (rows x features views) and the weather series."""
    feats = config.output_dir("features")
    return dict(windows=np.load(feats / (prefix + WINDOWS_FILE), mmap_mode="r").T,
                neighbourhood=np.load(feats / (prefix + NEIGHBOURHOOD_FILE), mmap_mode="r"),
                series=weather_windows.load_series(feats / SERIES_FOLDER))
