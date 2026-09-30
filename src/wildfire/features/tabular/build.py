"""Build the tabular feature table: one row per labelled cell-day, every predictor observed before the day."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.features.tabular import burn as burn_mod
from wildfire.features.tabular import climatology, masks, static, weather
from wildfire.features.tabular.gap_days import touches_flagged
from wildfire.guards import inputs
from wildfire.labels.detections import day_number

_DTYPES = dict(cell="int32", day="int32", label="int8", year="int16", week="int8", kind="category",
               state="category", group="category", band="category", cluster="category", date="category")


def _burn_features(rows: pd.DataFrame, cells: pd.DataFrame, cfg: dict, log) -> None:
    """Add the burn-history columns to `rows` (in place)."""
    meta = json.load(open(config.data_file("grid_meta")))
    ny, nx = meta["ny"], meta["nx"]
    b = burn_mod.load_burn(config.data_file("burn"))
    if b["legacy"]:
        raise SystemExit("burn file has no per-cell coverage (legacy format); rebuild it before building features")
    burn, covered = b["burn_ord"], b["covered"]
    bad = burn_mod.check_burn_dates_in_month(burn, covered, b["month_start"], b["month_end"])
    assert bad == 0, f"{bad} burn dates fall outside their stated month"
    ix, iy = cells.ix.values, cells.iy.values
    ncell = len(cells)
    lut = np.full(nx * ny, -1, np.int64)
    lut[ix * ny + iy] = np.arange(ncell)
    ords: dict[int, np.ndarray] = {}

    def cell_ords(c):
        if c not in ords:
            ords[c] = np.sort(burn[:, c][burn[:, c] > 0]).astype(np.int64)
        return ords[c]

    cols = ("bf_log_n10_nbhd", "burn_obs_frac", "bf_years_since", "bf_ys_obs_frac", "bf_ys_censored")
    out = {k: np.full(len(rows), np.nan) for k in cols}
    for c, g in rows.groupby("cell"):
        nb = [c] + [int(lut[(ix[c] + a) * ny + iy[c] + bb]) for a in (-1, 0, 1) for bb in (-1, 0, 1)
                    if (a or bb) and 0 <= ix[c] + a < nx and 0 <= iy[c] + bb < ny and lut[(ix[c] + a) * ny + iy[c] + bb] >= 0]
        o9 = np.sort(np.concatenate([cell_ords(q) for q in nb]))
        o1 = cell_ords(c)
        f = burn_mod.burn_features(
            g.day.values, o9, o1, covered[:, nb].mean(1), covered[:, c].astype(float), b["month_start"], b["month_end"],
            min_frac=cfg["min_observed_share"], lookback=cfg["lookback_days"], cap=cfg["years_since_cap"],
            archive_start=cfg["archive_start_day"], censor=cfg["censor_years_since"], lag=cfg["lag_days"])
        for k, v in f.items():
            out[k][g.index.values] = v
    for k, v in out.items():
        rows[k] = v
    rows["bf_ys_censored"] = rows.bf_ys_censored.astype(bool)
    if cfg["censor_years_since"]:
        assert rows.bf_years_since.notna().all(), "bf_years_since must never be NaN when censored (a dropna on burn inputs would drop rows)"


def build_features(rows_path, day_coverage_path, climatology_onsets: pd.DataFrame, log=print) -> pd.DataFrame:
    """Return the feature table for the rows in `rows_path`.

    rows_path: labelled rows (csv); day_coverage_path: per-day satellite gap flags (csv);
    climatology_onsets: cell-days with onset=True (columns cell, day).
    """
    t0 = time.time()
    cfg = config.load("features")["tabular"]
    end_day = day_number(cfg["window_end"])
    rows = pd.read_csv(rows_path, dtype=_DTYPES)
    cells = pd.read_csv(config.data_file("grid"))
    ncell = len(cells)
    inputs.assert_consistent_inputs()
    rows = rows[rows.day.values <= end_day].reset_index(drop=True)
    cov = pd.read_csv(day_coverage_path)
    flag_col = "snpp_gap | snpp_thin_candidate" if cfg["gap_days"]["include_data_driven"] else "snpp_gap"
    flagged = cov.query(flag_col).day.values.astype(np.int64)
    lat = cells.lat.values[rows.cell.values]
    lon = cells.lon.values[rows.cell.values]
    day = rows.day.values.astype(np.int64)
    rows["firms_cov_flag"] = touches_flagged(day, flagged, cfg["gap_days"]["lookback_days"])

    feat, report = weather.weather_features(lat, lon, day)
    for k, nm in enumerate(weather.NAMES):
        rows[nm] = feat[:, k]
    del feat
    masked = masks.mask_dependent_columns(rows, "t2m", weather.NAMES)
    masks.assert_same_nan_rows(rows, weather.NAMES)
    log("weather done", round(time.time() - t0), "masked rows", masked, report)
    dts = pd.to_datetime(rows.date)
    year_days = cfg["constants"]["days_per_year"]
    rows["sin_doy"] = np.sin(2 * np.pi * dts.dt.dayofyear / year_days)
    rows["cos_doy"] = np.cos(2 * np.pi * dts.dt.dayofyear / year_days)
    static.attach_static(rows, ncell)
    _burn_features(rows, cells, cfg["burn"], log)
    log("burn done", round(time.time() - t0))
    clim, _ = climatology.cell_week_climatology(
        climatology_onsets.cell.values, climatology_onsets.day.values, flagged, rows.cell.values, day, ncell,
        first_year=cfg["climatology"]["first_year"], years=cfg["climatology"]["years"], week_cap=cfg["weather"]["iso_week_cap"])
    rows[cfg["climatology_column"]] = clim
    if cfg["gap_days"]["policy"] == "drop":
        rows = rows[~rows.firms_cov_flag.values].reset_index(drop=True)
    for c in rows.columns:
        if rows[c].dtype == np.float64:
            rows[c] = rows[c].astype(np.float32)
    return rows
