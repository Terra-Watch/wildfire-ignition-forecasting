"""The modelling dataset: model table, enriched arrays, cell coordinates and the rows usable by every arm."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.features import weather_windows
from wildfire.features.tabular import weather
from wildfire.labels.detections import load_grid
from wildfire.models import arms
from wildfire.models.fit import thin
from wildfire.pipelines import enriched
from wildfire.pipelines.features import load_model_table

LEVELS = ("tabular", "enriched")


@dataclass
class Dataset:
    table: pd.DataFrame
    tabular_columns: dict
    arrays: dict
    cells: pd.DataFrame
    complete: np.ndarray
    climatology: dict = field(default_factory=dict)


def feature_names(dataset_columns: dict) -> list[str]:
    """Every tabular input used by a fitted arm (the climatology column is not an input)."""
    clim = arms.reference_arm()
    return sorted({c for a, v in dataset_columns.items() if a != clim for c in v})


def complete_rows(table: pd.DataFrame, columns: dict, arrays: dict, chunk: int = 500_000) -> np.ndarray:
    """Rows where every tabular input and every enriched feature is present."""
    names = feature_names(columns)
    ok = table[names].notna().all(axis=1).values.copy()
    for start in range(0, len(table), chunk):
        sl = slice(start, start + chunk)
        ok[sl] &= ~np.isnan(arrays["windows"][sl]).any(axis=1) & ~np.isnan(arrays["neighbourhood"][sl]).any(axis=1)
    return ok


def load_dataset(table_name: str = "model_table", prefix: str = "") -> Dataset:
    """The development dataset, or (with a table name and array prefix) another table built the same way."""
    table, columns = load_model_table(table_name)
    arrays = enriched.load_arrays(prefix)
    cells, _ = load_grid()
    return Dataset(table, columns, arrays, cells, complete_rows(table, columns, arrays))


def validation_rows(ds: Dataset, year: int) -> np.ndarray:
    """Indices of complete rows of `year` in the validation months that also have a climatology value."""
    cfg = config.load("baseline")["validation"]
    clim = ds.tabular_columns[arms.reference_arm()][0]
    month = pd.to_datetime(ds.table.day.values, unit="D").month
    m = (ds.table.year.values == year) & np.isin(month, cfg["validation_months"]) & ds.complete & ds.table[clim].notna().values
    return np.flatnonzero(m)


def fit_rows(ds: Dataset, year: int, cap: int | None = None) -> np.ndarray:
    """Indices of complete rows of years before `year`, thinned to the cap (all onsets kept)."""
    cfg = config.load("baseline")["validation"]
    cap = cfg["fit_row_cap"] if cap is None else cap
    idx = np.flatnonzero((ds.table.year.values < year) & ds.complete)
    keep = thin(ds.table.label.values[idx], cap, seed=cfg["fit_row_cap_seed"])
    return idx[keep]


def anomaly_climatology(ds: Dataset, year: int) -> np.ndarray:
    """Day-of-year climatology from the fit years of the fold validating `year` (cached per fold)."""
    if year not in ds.climatology:
        first = config.load("evaluation")["splits"]["first_year"]
        ds.climatology[year] = weather_windows.anomaly_climatology(ds.arrays["series"], list(range(first, year)))
    return ds.climatology[year]


def input_columns(ds: Dataset, arm: str) -> list[str]:
    return ds.tabular_columns[arm]


def take_columns(table: pd.DataFrame, idx: np.ndarray, columns: list[str]) -> np.ndarray:
    """Rows `idx` of the named columns as float32, without copying whole columns into one big array."""
    return np.column_stack([table[c].values[idx] for c in columns]).astype(np.float32)


def matrix(ds: Dataset, idx: np.ndarray, columns: list[str], level: str, year: int) -> np.ndarray:
    """Input matrix (rows, columns) float32: the tabular columns, plus the enriched blocks at the enriched level.
    `year` is the validation year of the fold (it fixes the anomaly climatology)."""
    parts = [take_columns(ds.table, idx, columns)]
    if level == "enriched":
        t = ds.table
        anom = weather_windows.anomalies(ds.arrays["series"], anomaly_climatology(ds, year), t.cell.values[idx], t.day.values[idx])
        parts += [ds.arrays["windows"][idx], anom.astype(np.float32), ds.arrays["neighbourhood"][idx]]
    return np.hstack(parts)


def enriched_names() -> list[str]:
    from wildfire.features import neighbourhood
    return weather_windows.window_names() + [f"{v}_anomaly" for v in weather.NAMES] + neighbourhood.column_names()


def feature_groups_static(ds: Dataset) -> list[str]:
    """Columns of the static group (terrain and land cover)."""
    return arms.feature_groups(ds.table)["static"]


def matrix_file(ds: Dataset, idx: np.ndarray, columns: list[str], level: str, year: int, path, chunk: int = 200_000) -> np.ndarray:
    """Write the input matrix to a file in chunks and return it memory-mapped (large validation matrices stay off memory)."""
    width = len(columns) + (len(enriched_names()) if level == "enriched" else 0)
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(idx), width))
    for start in range(0, len(idx), chunk):
        out[start:start + chunk] = matrix(ds, idx[start:start + chunk], columns, level, year)
    out.flush()
    return np.load(path, mmap_mode="r")


def predict_chunks(predict, x: np.ndarray, chunk: int = 200_000, transform=None) -> np.ndarray:
    """Apply `predict` to row chunks of a (memory-mapped) matrix; `transform` (optional) prepares each chunk."""
    parts = []
    for start in range(0, len(x), chunk):
        block = np.array(x[start:start + chunk], dtype=np.float32)
        parts.append(predict(transform(block) if transform else block))
    return np.concatenate(parts)


def clear_cache() -> None:
    """Delete the temporary validation matrices."""
    for f in config.output_dir("cache").glob("*.npy"):
        f.unlink()
