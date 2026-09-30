"""Fire detections -> grid cells -> cell-day table with onset flags."""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd
import pyproj

from wildfire import config


def epoch() -> pd.Timestamp:
    return pd.Timestamp(config.load("domain")["epoch_day"])


def day_number(date) -> int:
    return (pd.Timestamp(date) - epoch()).days


def read_detections() -> pd.DataFrame:
    """Detections passing the confidence and satellite filters, duplicates removed."""
    cfg = config.load("domain")["detections"]
    files = sorted(f for p in cfg["file_patterns"] for f in glob.glob(str(config.data_root() / p)))
    parts = [pd.read_csv(f, usecols=cfg["columns"], dtype={"confidence": str}) for f in files]
    d = pd.concat(parts, ignore_index=True)
    d = d[d.confidence.isin(cfg["confidence_keep"]) & d.satellite.isin(cfg["satellites_keep"])]
    return d.drop_duplicates(cfg["duplicate_key"])


def load_grid() -> tuple[pd.DataFrame, dict]:
    cells = pd.read_csv(config.data_file("grid"))
    return cells, json.load(open(config.data_file("grid_meta")))


def assign_cells(d: pd.DataFrame, cells: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Add `cell` (grid index) and `day` (days since epoch) and keep detections inside the grid and study window."""
    dom = config.load("domain")
    size = dom["grid"]["cell_size_m"]
    nx, ny = meta["nx"], meta["ny"]
    lut = np.full(nx * ny, -1, np.int64)
    lut[cells.ix.values * ny + cells.iy.values] = np.arange(len(cells))
    tf = pyproj.Transformer.from_crs(dom["grid"]["source_crs"], dom["grid"]["projected_crs"], always_xy=True)
    x, y = tf.transform(d.longitude.values, d.latitude.values)
    i = np.floor((x - meta["x0"]) / size).astype(np.int64)
    j = np.floor((y - meta["y0"]) / size).astype(np.int64)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    ci = np.full(len(d), -1)
    ci[ok] = lut[i[ok] * ny + j[ok]]
    d = d.assign(cell=ci)
    d = d[d.cell >= 0]
    d["day"] = (pd.to_datetime(d.acq_date) - epoch()).dt.days
    lo, hi = day_number(dom["study_window"]["start"]), day_number(dom["study_window"]["end"])
    return d[(d.day >= lo) & (d.day <= hi)]


def cell_day_table(d: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, day) with a detection; `onset` = no detection in the quiet days before."""
    quiet = config.load("domain")["onset"]["quiet_days_before"]
    cd = d.groupby(["cell", "day"]).agg(n=("frp", "size"), frp=("frp", "sum")).reset_index().sort_values(["cell", "day"])
    cd["state"] = cells.state.values[cd.cell.values]
    cd["year"] = (epoch() + pd.to_timedelta(cd.day, "D")).dt.year
    prev = cd.groupby("cell").day.shift(1)
    gap = cd.day - prev
    cd["onset"] = prev.isna() | (gap >= quiet + 1)
    return cd
