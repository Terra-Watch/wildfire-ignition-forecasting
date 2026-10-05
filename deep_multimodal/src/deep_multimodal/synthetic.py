"""A small synthetic model input package with the same layout as a real one, for tests, notebooks and a quick check of the code on a CPU. The values are random;
the label depends weakly on a few inputs so that training has something to find. Nothing here is real data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from deep_multimodal import constants as c
from deep_multimodal.forecast import COLUMN_NAMES, COLUMN_SCALES

EPOCH_DAY_OF_2020 = int((np.datetime64("2020-01-01") - np.datetime64("1970-01-01")).astype(int))


def _rows(rng, count, first, last, cells):
    day = rng.integers(first, last, count)
    cell = rng.integers(0, cells, count)
    return np.stack([cell, day], 1).astype(np.int32)


def write_synthetic_package(root: Path | str, year: int = 2022, cells: int = 40, fit_rows: int = 1200, validation_rows: int = 300, seed: int = 0) -> Path:
    """Write fold_<year>, dm_<year>, dm_common and fc_lookup under `root`; return `root`. Fit years are 2020 and 2021, the validation year is `year` (default 2022)."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    first_day, days, points, pixels = EPOCH_DAY_OF_2020, 3 * 366, 6, 5
    year_start = {y: int((np.datetime64(f"{y}-01-01") - np.datetime64("1970-01-01")).astype(int)) for y in (2020, 2021, 2022, 2023)}
    for name in (f"fold_{year}", f"dm_{year}", "dm_common", "fc_lookup"):
        (root / name).mkdir(parents=True, exist_ok=True)
    # shared weather series, cell-to-point map, static table
    series = rng.normal(0, 1, (days, len(c.WEATHER_CHANNELS), points)).astype(np.float16)
    series[rng.random(series.shape) < 0.01] = np.nan
    np.save(root / "dm_common" / "series.npy", series)
    np.save(root / "dm_common" / "cell_point.npy", rng.integers(0, points, cells).astype(np.int32))
    np.save(root / "dm_common" / "static.npy", rng.normal(0, 1, (cells, len(c.STATIC_COLUMNS))).astype(np.float32))
    (root / "dm_common" / "meta.json").write_text(json.dumps(dict(first_day=first_day, days=days, points=points, clim_first_year=2020, cells=cells, channels=c.WEATHER_CHANNELS)))
    # forecast lookup (integers; one code marks a missing day)
    lookup_days = np.arange(first_day, first_day + days)
    table = rng.integers(-500, 500, (pixels, days, len(COLUMN_NAMES))).astype(np.int16)
    table[:, rng.random(days) < 0.01, :] = -32768
    np.save(root / "fc_lookup" / "gfs_i16.npy", table)
    np.save(root / "fc_lookup" / "days.npy", lookup_days)
    np.save(root / "fc_lookup" / "cell_pixel.npy", rng.integers(0, pixels, cells).astype(np.int32))
    np.save(root / "fc_lookup" / "cell_state.npy", rng.integers(0, 6, cells).astype(np.int32))
    window_start = first_day + c.WEATHER_DAYS + 2

    def split(which, count, first, last):
        key = _rows(rng, count, first, last, cells)
        tokens = rng.integers(0, c.HISTORY_DAYS * c.NEIGHBOURHOOD_SIDE ** 2, (count, c.MAX_FIRE_TOKENS)).astype(np.int16)
        empty = rng.random((count, c.MAX_FIRE_TOKENS)) < 0.6
        tokens[empty] = -1
        features = rng.normal(0, 1, (count, c.MAX_FIRE_TOKENS, len(c.FIRE_CHANNELS))).astype(np.float16)
        values = rng.normal(0, 1, (count, len(c.ROW_COLUMNS))).astype(np.float32)
        angle = 2 * np.pi * ((key[:, 1] - first_day) % 365) / 365.25
        values[:, c.CALENDAR_COLUMNS[0]], values[:, c.CALENDAR_COLUMNS[1]] = np.sin(angle), np.cos(angle)
        signal = 1.5 * values[:, 3] + 0.5 * (~empty).sum(1) / c.MAX_FIRE_TOKENS - 1.0
        label = (rng.random(count) < 1 / (1 + np.exp(-2.5 * signal))).astype(np.uint8)
        if label.sum() < 5:
            label[:5] = 1
        return key, tokens, features, values, label

    for which, count, first, last in (("fit", fit_rows, window_start, year_start[year]), ("val", validation_rows, year_start[year], year_start[year] + 180)):
        key, tokens, features, values, label = split(which, count, first, last)
        base, model_dir = root / f"fold_{year}", root / f"dm_{year}"
        np.save(base / f"{which}_key.npy", key)
        np.save(base / f"{which}_token_positions.npy", tokens)
        np.save(base / f"{which}_token_features.npy", features)
        np.save(base / f"{which}_y.npy", label)
        np.save(model_dir / f"{which}_extra.npy", values)
        if which == "fit":
            year_of_day = np.where(key[:, 1] >= year_start[year - 1], year - 1, year - 2).astype(np.int16)
            np.save(base / "fit_year.npy", year_of_day)
            np.save(model_dir / "fit_aux.npy", np.stack([label * rng.integers(1, 5, count), label * rng.random(count) * 20, np.zeros(count)], 1).astype(np.float32))
            prevalence, stop = float(label.mean()), year_of_day == year - 1
            (base / "meta.json").write_text(json.dumps(dict(year=year, fit_prevalence=prevalence, stop_year=year - 1, stop_year_pool_rows=int(stop.sum()) * 3, stop_year_pool_onsets=int(label[stop].sum()))))
    (root / f"dm_{year}" / "meta.json").write_text(json.dumps(dict(year=year, base=f"fold_{year}", common="dm_common", lookup="fc_lookup", extra_names=c.ROW_COLUMNS)))
    return root
