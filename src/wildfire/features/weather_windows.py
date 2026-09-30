"""Weather-window features: summaries of the last days of weather before the target day.

A daily series of the weather inputs is stored once per weather-grid cell that holds a study cell. For a target
day D the latest day used is X = D-1, so every window ends before D. Anomalies depend on the fold: the day-of-year
climatology is computed from the training years of that fold only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from wildfire import config
from wildfire.features.base import TIER_SUMMARY, FeatureSet, register
from wildfire.features.tabular import weather
from wildfire.labels.detections import day_number

STATS = ("mean", "max")
CODES = ("ffmc", "dmc", "dc", "isi", "bui", "fwi")


def settings() -> dict:
    return config.load("baseline")["features"]["weather_windows"]


def series_first_day() -> int:
    """First stored day: a year before the first target day, so the rain counter is settled."""
    return day_number(config.load("domain")["study_window"]["start"]) - 365


def build_series(cells: pd.DataFrame) -> dict:
    """Daily weather (days, variables, points) and daily rain (days, points) at the weather cells holding study cells."""
    layout = weather.grid_layout()
    li, lj = weather.grid_indices(cells.lat.values, cells.lon.values, layout)
    points, cell_point = np.unique(li * layout["nlon"] + lj, return_inverse=True)
    rows, cols = points // layout["nlon"], points % layout["nlon"]
    first = series_first_day()
    last = day_number(config.load("features")["tabular"]["window_end"]) - 1
    values = np.full((last - first + 1, len(weather.NAMES), len(points)), np.nan, np.float32)
    rain = np.full((last - first + 1, len(points)), np.nan, np.float32)
    for dn, _, fields, tp in weather.daily_fields(layout, last):
        if dn >= first:
            values[dn - first] = fields[:, rows, cols]
            rain[dn - first] = tp[rows, cols]
    return dict(values=values, rain=rain, cell_point=cell_point, first_day=first)


def save_series(series: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for k in ("values", "rain", "cell_point"):
        np.save(folder / f"{k}.npy", series[k])
    (folder / "meta.json").write_text(json.dumps({"first_day": series["first_day"]}))


def load_series(folder: Path) -> dict:
    meta = json.loads((folder / "meta.json").read_text())
    arrays = {k: np.load(folder / f"{k}.npy", mmap_mode="r") for k in ("values", "rain", "cell_point")}
    return dict(arrays, first_day=meta["first_day"])


def trailing(series: np.ndarray, window: int, stat: str) -> np.ndarray:
    """Statistic over the `window` days ending at each day, shape of `series`; the first window-1 days are NaN."""
    view = sliding_window_view(series, window, axis=0)
    out = {"mean": view.mean, "max": view.max, "sum": view.sum}[stat](-1)
    pad = np.full((window - 1,) + series.shape[1:], np.nan, series.dtype)
    return np.concatenate([pad, out.astype(series.dtype)], axis=0)


def days_since_rain(rain: np.ndarray, threshold: float, cap: int) -> np.ndarray:
    """Days since the daily total last exceeded the threshold (0 on a rainy day), capped."""
    out = np.empty_like(rain)
    run = np.full(rain.shape[1], float(cap), np.float32)
    for t in range(len(rain)):
        run = np.where(rain[t] > threshold, 0.0, np.minimum(run + 1, cap))
        out[t] = run
    return out


def window_names() -> list[str]:
    s = settings()
    names = [f"{v}_{st}{w}" for v in weather.NAMES for w in s["window_days"] for st in STATS]
    names += [f"{c}_change{s['code_change_days']}" for c in CODES]
    names += [f"rain_sum{w}" for w in s["cumulative_precipitation_days"]] + ["days_since_rain_2mm"]
    return names


def column_series(series: dict):
    """Yield (name, array (days, points)) for every window feature, one at a time (memory)."""
    s = settings()
    values, rain = series["values"], series["rain"]
    names = iter(window_names())
    for k in range(len(weather.NAMES)):
        for w in s["window_days"]:
            for st in STATS:
                yield next(names), trailing(np.asarray(values[:, k]), w, st)
    d = s["code_change_days"]
    for c in CODES:
        x = np.asarray(values[:, weather.NAMES.index(c)])
        yield next(names), np.concatenate([np.full((d,) + x.shape[1:], np.nan, x.dtype), x[d:] - x[:-d]])
    for w in s["cumulative_precipitation_days"]:
        yield next(names), trailing(np.asarray(rain), w, "sum")
    yield next(names), days_since_rain(np.asarray(rain), s["days_since_rain_mm"], s["days_since_rain_cap"])


def window_matrix(series: dict, cell: np.ndarray, day: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Window features for (cell, target day) pairs: (rows, columns) float32. The window ends on day - 1.
    `out` may be a preallocated (or memory-mapped) array of the right shape."""
    t = np.asarray(day) - 1 - series["first_day"]
    p = np.asarray(series["cell_point"])[cell]
    if out is None:
        out = np.empty((len(t), len(window_names())), np.float32)
    for j, (_, arr) in enumerate(column_series(series)):
        out[:, j] = arr[t, p]
    return out


def anomaly_climatology(series: dict, training_years: list[int]) -> np.ndarray:
    """Mean by day of year (367, variables, points) over the training years only."""
    dates = pd.to_datetime(series["first_day"] + np.arange(series["values"].shape[0]), unit="D")
    use = np.isin(dates.year, training_years)
    doy = dates.dayofyear.values
    out = np.full((367,) + tuple(series["values"].shape[1:]), np.nan, np.float32)
    for d in range(1, 367):
        m = use & (doy == d)
        if m.any():
            out[d] = np.asarray(series["values"][m]).mean(0)
    return out


def anomalies(series: dict, climatology: np.ndarray, cell: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Value on the day before the target day minus the climatology for that day of year: (rows, variables)."""
    t = np.asarray(day) - 1 - series["first_day"]
    p = np.asarray(series["cell_point"])[cell]
    doy = pd.to_datetime(np.asarray(day) - 1, unit="D").dayofyear.values
    return np.asarray(series["values"])[t, :, p] - climatology[doy, :, p]


@register
class WeatherWindows(FeatureSet):
    name = "weather_windows"
    tier = TIER_SUMMARY

    def columns(self) -> list[str]:
        return window_names() + [f"{v}_anomaly" for v in weather.NAMES]

    def build(self, keys):
        """Window features for (cell, day) keys from the stored series (anomalies are added per fold)."""
        series = load_series(config.output_dir("features") / "daily_weather")
        m = window_matrix(series, keys.cell.values, keys.day.values)
        return pd.DataFrame(m, columns=window_names(), index=keys.index)
