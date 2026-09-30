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


def matrix(ds: Dataset, idx: np.ndarray, columns: list[str], level: str, year: int) -> np.ndarray:
    """Input matrix (rows, columns) float32: the tabular columns, plus the enriched blocks at the enriched level.
    `year` is the validation year of the fold (it fixes the anomaly climatology)."""
    parts = [ds.table[columns].values[idx].astype(np.float32)]
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
