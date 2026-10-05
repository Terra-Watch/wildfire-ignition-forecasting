"""Reading a model input package and turning it into batches.

A package is a set of plain numpy arrays written once per validation year (layout in docs/DATA.md):
    fold_<year>/     fire-history tokens, row keys (cell, day), labels and years of the fit and validation rows
    dm_<year>/       per-row columns (burn history, recent fire history, calendar) and the auxiliary targets of the fit rows
    dm_common/       the daily reanalysis series of the weather points, the cell-to-point map and the static table
    fc_lookup/       the day-ahead forecast lookup
Weather windows and forecast rows are built here from the (cell, day) keys, so the package stays small.
Standardising constants come from the fit rows only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from deep_multimodal import constants as c
from deep_multimodal.forecast import forecast_inputs


def read_split(package: Path, year: int, which: str, max_tokens: int = c.MAX_FIRE_TOKENS) -> tuple[dict, dict]:
    """Arrays of the fit or validation rows of one fold (nothing standardised yet) and the package metadata."""
    package = Path(package)
    model_dir = package / f"dm_{year}"
    meta = json.loads((model_dir / "meta.json").read_text())
    base = package / meta["base"]
    meta = {**json.loads((base / "meta.json").read_text()), **meta}
    load = lambda name: np.load(base / name, mmap_mode="r")
    labels = np.asarray(load(f"{which}_y.npy"))
    arrays = dict(fire_features=np.asarray(load(f"{which}_token_features.npy")[:, :max_tokens]), fire_positions=np.asarray(load(f"{which}_token_positions.npy")[:, :max_tokens]),
                  key=np.asarray(load(f"{which}_key.npy")).astype(np.int64), label=labels, row_values=np.load(model_dir / f"{which}_extra.npy").astype(np.float32))
    if arrays["row_values"].shape != (len(labels), len(c.ROW_COLUMNS)):
        raise ValueError(f"row values have shape {arrays['row_values'].shape}, expected ({len(labels)}, {len(c.ROW_COLUMNS)})")
    if which == "fit":
        arrays["year"] = np.asarray(load("fit_year.npy"))
        arrays["auxiliary"] = np.load(model_dir / "fit_aux.npy").astype(np.float32)
    hourly, daily, missing = [], [], []
    for start in range(0, len(labels), 400000):                       # forecast rows are made from the keys, in chunks
        h, d, m = forecast_inputs(arrays["key"][start:start + 400000], package / meta["lookup"])
        hourly.append(h), daily.append(d), missing.append(m)
    arrays.update(forecast_hours=np.concatenate(hourly), forecast_daily=np.concatenate(daily), forecast_missing=np.concatenate(missing))
    return arrays, meta


class WeatherSeries:
    """The daily reanalysis series of the study cells' weather points on the device, the per-cell static table and the fit-year day-of-year climatology."""

    def __init__(self, common_dir: Path, device: torch.device, climatology_years):
        common_dir = Path(common_dir)
        self.meta = json.loads((common_dir / "meta.json").read_text())
        self.device, self.first_day = device, int(self.meta["first_day"])
        self.series = torch.from_numpy(np.load(common_dir / "series.npy")).to(device)                # (days, 20, points) half precision; NaN = missing
        self.cell_point = torch.from_numpy(np.load(common_dir / "cell_point.npy").astype(np.int64)).to(device)
        self.static = torch.from_numpy(np.load(common_dir / "static.npy").astype(np.float32)).to(device)   # (cells, 21); NaN for cells outside the table
        days = self.series.shape[0]
        dates = np.datetime64("1970-01-01") + (self.first_day + np.arange(days)).astype("timedelta64[D]")
        year = dates.astype("datetime64[Y]").astype(int) + 1970
        day_of_year = np.minimum((dates - dates.astype("datetime64[Y]")).astype(int), 365)
        self.day_of_year = torch.from_numpy(day_of_year).long().to(device)
        use = torch.from_numpy(np.isin(year, list(climatology_years))).to(device)
        points = self.series.shape[2]
        climatology = torch.full((366, c.ANOMALY_CHANNELS, points), float("nan"), device=device)
        for k in range(366):
            selected = use & (self.day_of_year == k)
            if bool(selected.any()):
                climatology[k] = torch.nanmean(self.series[selected][:, :c.ANOMALY_CHANNELS].float(), 0)
        for k in range(1, 366):                                         # a day of year without a fit-year day takes the previous day's value
            if torch.isnan(climatology[k]).all():
                climatology[k] = climatology[k - 1]
        self.climatology = climatology
        self.offsets = torch.arange(c.WEATHER_DAYS - 1, -1, -1, device=device)

    def window(self, cell: torch.Tensor, day: torch.Tensor) -> torch.Tensor:
        """(batch, 60, 37): the 20 channels of the 60 days up to the day before `day` and 17 anomalies against the fit-year climatology. Missing stays NaN."""
        time = (day - 1 - self.first_day)[:, None] - self.offsets[None]
        point = self.cell_point[cell][:, None]
        raw = self.series[time, :, point].float()
        anomaly = raw[..., :c.ANOMALY_CHANNELS] - self.climatology[self.day_of_year[time], :, point]
        return torch.cat([raw, anomaly], -1)


class Standardiser:
    """Means and standard deviations of the fit rows, used for every split of the fold."""

    def __init__(self, fit: dict, series: WeatherSeries, weather_sample: int = 20000, seed: int = 0):
        key = fit["key"]
        static = series.static[torch.from_numpy(key[:, 0]).to(series.device)].cpu().numpy()
        self.static_mean, self.static_std = np.nanmean(static, 0), np.nanstd(static, 0) + 1e-6
        values = fit["row_values"]
        self.row_mean, self.row_std = np.nanmean(values, 0), np.nanstd(values, 0) + 1e-6
        self.row_mean[c.CALENDAR_COLUMNS], self.row_std[c.CALENDAR_COLUMNS] = 0.0, 1.0          # sine and cosine stay as they are
        present = ~fit["forecast_missing"]
        hours = fit["forecast_hours"][present].reshape(-1, len(c.FORECAST_VARIABLES))
        self.hour_mean, self.hour_std = np.nanmean(hours, 0), np.nanstd(hours, 0) + 1e-6
        daily = fit["forecast_daily"][present]
        self.daily_mean, self.daily_std = np.nanmean(daily, 0), np.nanstd(daily, 0) + 1e-6
        chosen = np.random.default_rng(seed).choice(len(key), min(weather_sample, len(key)), replace=False)
        window = series.window(torch.from_numpy(key[chosen, 0]).to(series.device), torch.from_numpy(key[chosen, 1]).to(series.device)).reshape(-1, c.WEATHER_INPUTS).cpu().numpy()
        self.weather_mean, self.weather_std = np.nanmean(window, 0), np.nanstd(window, 0) + 1e-6


class SplitTensors:
    """All rows of one split as tensors on the device. `batch(index)` gathers a batch; weather windows are built here from the keys."""

    def __init__(self, raw: dict, standardiser: Standardiser, series: WeatherSeries, device: torch.device):
        self.size, self.device, self.series = len(raw["label"]), device, series
        tensor = lambda a, dtype=None: torch.from_numpy(np.array(a, copy=True)).to(device) if dtype is None else torch.from_numpy(np.array(a, copy=True)).to(device, dtype)
        s = standardiser
        self.fire_features, self.fire_positions, self.key, self.label = tensor(raw["fire_features"]), tensor(raw["fire_positions"]), tensor(raw["key"]), tensor(raw["label"], torch.float32)
        self.row_values = tensor(np.nan_to_num((raw["row_values"] - s.row_mean) / s.row_std).astype(np.float32))
        self.forecast_hours = tensor(np.nan_to_num((raw["forecast_hours"] - s.hour_mean) / s.hour_std).astype(np.float16))
        self.forecast_daily = tensor(np.nan_to_num((raw["forecast_daily"] - s.daily_mean) / s.daily_std).astype(np.float16))
        self.forecast_missing = tensor(raw["forecast_missing"])
        self.auxiliary = tensor(raw["auxiliary"]) if "auxiliary" in raw else None
        self.year = raw.get("year")
        self.static_mean, self.static_std = tensor(s.static_mean.astype(np.float32)), tensor(s.static_std.astype(np.float32))
        self.weather_mean, self.weather_std = tensor(s.weather_mean.astype(np.float32)), tensor(s.weather_std.astype(np.float32))

    def subset(self, mask: np.ndarray) -> "SplitTensors":
        selected = torch.from_numpy(np.flatnonzero(mask)).to(self.device)
        out = object.__new__(SplitTensors)
        out.__dict__.update(self.__dict__)
        for name in ("fire_features", "fire_positions", "key", "label", "row_values", "forecast_hours", "forecast_daily", "forecast_missing", "auxiliary"):
            value = getattr(self, name)
            setattr(out, name, None if value is None else value[selected])
        out.size = len(selected)
        out.year = None if self.year is None else self.year[mask]
        return out

    def batch(self, index: torch.Tensor) -> dict:
        key = self.key[index]
        cell, day = key[:, 0], key[:, 1]
        batch = dict(fire_features=self.fire_features[index].float(), fire_positions=self.fire_positions[index].long(),
                     static=torch.nan_to_num((self.series.static[cell] - self.static_mean) / self.static_std),
                     row_values=self.row_values[index], weather=torch.nan_to_num((self.series.window(cell, day) - self.weather_mean) / self.weather_std),
                     forecast_hours=self.forecast_hours[index].float(), forecast_daily=self.forecast_daily[index].float(), forecast_missing=self.forecast_missing[index], label=self.label[index])
        if self.auxiliary is not None:
            batch["auxiliary"] = self.auxiliary[index]
        return batch


@dataclass
class FoldData:
    """The rows of one validation year: `train` (years before the last fit year), `stop` (the last fit year, used to choose the number of epochs), `fit_all` (both) and `validation`."""
    train: SplitTensors
    stop: SplitTensors
    fit_all: SplitTensors
    validation: SplitTensors
    meta: dict
    series: WeatherSeries
    standardiser: Standardiser


def load_fold(package: Path, year: int, device: torch.device) -> FoldData:
    package = Path(package)
    fit_raw, meta = read_split(package, year, "fit")
    if "fit_prevalence" not in meta:
        meta["fit_prevalence"] = float(np.load(package / meta["base"] / "fit_y.npy", mmap_mode="r").mean())
    common_meta = json.loads((package / meta["common"] / "meta.json").read_text())
    series = WeatherSeries(package / meta["common"], device, range(int(common_meta["clim_first_year"]), int(meta["year"])))
    standardiser = Standardiser(fit_raw, series)
    fit_all = SplitTensors(fit_raw, standardiser, series, device)
    del fit_raw
    validation_raw, _ = read_split(package, year, "val")
    validation = SplitTensors(validation_raw, standardiser, series, device)
    stop_rows = fit_all.year == int(meta["year"]) - 1
    return FoldData(fit_all.subset(~stop_rows), fit_all.subset(stop_rows), fit_all, validation, meta, series, standardiser)
