"""Weather inputs for every cell-day, taken from the day before (local noon) of hourly land reanalysis.

Noon weather of day X = D-1 is the mean of the 06 and 07 UTC hourly fields. Accumulated variables (rain) are running
totals since 00 UTC, so the 00 UTC value of day X is the total of day X-1; the 24 h total ending 06:30 UTC of X is
A(X, 06:30) + [A(X, 00) - A(X-1, 06:30)]. Fire-weather codes come from a daily recursion started on the configured
first day; soil-moisture anomaly is noon soil moisture minus the same-ISO-week mean over the climatology years.
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd
import xarray as xr

from wildfire import config
from wildfire.features.tabular import fwi
from wildfire.guards.temporal import assert_before_cutoff
from wildfire.labels.detections import epoch

NAMES = ["t2m", "vpd", "ws", "rh", "sw1", "sw2", "rain7", "dsr", "ffmc", "dmc", "dc", "isi", "bui", "fwi",
         "tmax09", "sw1_anom", "sw2_anom"]


def _saturation_vapour_pressure(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def _weather_path(group: str, year: int, month: int):
    data = config.load("data", expand=False)["inputs"]
    return config.data_root() / data["weather_dir"] / data["weather_file"].format(group=group, year=year, month=month)


def _load_month(group, year, month, names):
    p = _weather_path(group, year, month)
    if not p.exists():
        return None, None
    ds = xr.open_dataset(p)
    tv = pd.to_datetime(ds.valid_time.values)
    v = {k: ds[k].values.astype(np.float64) for k in names if k in ds}
    ds.close()
    return tv, v


def _hour_indices(tv, d, hours):
    return [int(np.where(tv == d + pd.Timedelta(hours=h))[0][0]) for h in hours]


def weather_features(lat: np.ndarray, lon: np.ndarray, day: np.ndarray, log=print) -> tuple[np.ndarray, dict]:
    """Features (rows, len(NAMES)) float32 for cell-days at (lat, lon) on `day`; also returns a report dict."""
    w = config.load("features")["tabular"]["weather"]
    hours = w["hours_utc"]
    e0 = epoch()
    day = np.asarray(day, np.int64)
    sample = xr.open_dataset(_weather_path("primary", 2019, 1))
    nlat, nlon = sample.sizes["latitude"], sample.sizes["longitude"]
    latg = sample.latitude.values
    lat0, lon0 = float(latg[0]), float(sample.longitude.values[0])
    sample.close()
    step = w["grid_step_deg"]
    li = np.clip(np.round((lat0 - lat) / step).astype(int), 0, nlat - 1)
    lj = np.clip(np.round((lon - lon0) / step).astype(int), 0, nlon - 1)
    bands = [np.where((latg > lo) & (latg <= hi))[0] for lo, hi in w["lat_bands"]]
    bands = [b for b in bands if len(b)]
    band_lat = [float(latg[b].mean()) for b in bands]
    start = (pd.Timestamp(w["first_day"]) - e0).days
    x = day - 1
    assert x.min() >= start + w["first_row_margin_days"]
    byday = {int(k): np.where(x == k)[0] for k in np.unique(x)}
    feat = np.full((len(day), len(NAMES)), np.nan, np.float32)
    cap = w["iso_week_cap"]

    # soil-moisture climatology (training years only)
    cs = np.zeros((cap + 1, 2, nlat, nlon))
    cc = np.zeros(cap + 1)
    y0, y1 = w["soil_climatology_years"]
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            tv, v = _load_month("primary", y, m, ("swvl1", "swvl2"))
            if tv is None:
                log("missing primary file", y, m)
                continue
            for d in pd.date_range(f"{y}-{m:02d}-01", periods=pd.Period(f"{y}-{m:02d}").days_in_month):
                _, i6, i7, _ = _hour_indices(tv, d, hours)
                wk = min(int(d.isocalendar()[1]), cap)
                cs[wk, 0] += (v["swvl1"][i6] + v["swvl1"][i7]) / 2
                cs[wk, 1] += (v["swvl2"][i6] + v["swvl2"][i7]) / 2
                cc[wk] += 1
    clim = cs / np.maximum(cc, 1)[:, None, None, None]

    ffm = np.full((nlat, nlon), w["initial_codes"]["ffmc"])
    dm = np.full((nlat, nlon), w["initial_codes"]["dmc"])
    dcc = np.full((nlat, nlon), w["initial_codes"]["dc"])
    hist = collections.deque(maxlen=w["rain_window_days"])
    dsr = np.zeros((nlat, nlon))
    prev06 = np.zeros((nlat, nlon))
    lo_rh, hi_rh = w["humidity_bounds"]
    last_needed = max(byday)
    latest_source, cutoff = [], []
    missing = []
    for y in range(pd.Timestamp(w["first_day"]).year, int((e0 + pd.Timedelta(days=last_needed)).year) + 1):
        for m in range(1, 13):
            tv, v = _load_month("primary", y, m, ("t2m", "d2m", "u10", "v10", "tp", "swvl1", "swvl2"))
            if tv is None:
                missing.append(f"{y}-{m:02d}")
                continue
            for d in pd.date_range(f"{y}-{m:02d}-01", periods=pd.Period(f"{y}-{m:02d}").days_in_month):
                dn = (d - e0).days
                if dn > last_needed:
                    break
                i0, i6, i7, i9 = _hour_indices(tv, d, hours)
                t = (v["t2m"][i6] + v["t2m"][i7]) / 2 - 273.15
                td = (v["d2m"][i6] + v["d2m"][i7]) / 2 - 273.15
                ws = np.hypot((v["u10"][i6] + v["u10"][i7]) / 2, (v["v10"][i6] + v["v10"][i7]) / 2) * 3.6
                rh = np.clip(100 * _saturation_vapour_pressure(td) / _saturation_vapour_pressure(t), lo_rh, hi_rh)
                tp00 = np.nan_to_num(v["tp"][i0]) * 1000.0
                tp0630 = (np.nan_to_num(v["tp"][i6]) + np.nan_to_num(v["tp"][i7])) / 2 * 1000.0
                rain24 = np.maximum(tp0630 + tp00 - prev06, 0.0)
                prev06 = tp0630
                hist.append(tp00)
                dsr = np.where(tp00 > w["rain_threshold_mm"], 0.0, np.minimum(dsr + 1, w["dry_spell_cap_days"]))
                rain7 = np.sum(hist, axis=0)
                for b, lm in zip(bands, band_lat):
                    ffm[b] = fwi.ffmc(ffm[b], t[b], rh[b], ws[b], rain24[b])
                    dm[b] = fwi.dmc(dm[b], t[b], rh[b], rain24[b], lm, d.month)
                    dcc[b] = fwi.dc(dcc[b], t[b], rh[b], rain24[b], lm, d.month)
                ii = fwi.isi(ffm, ws)
                bu = fwi.bui(dm, dcc)
                fw = fwi.fwi(ii, bu)
                if dn in byday:
                    wk = min(int(d.isocalendar()[1]), cap)
                    s1 = (v["swvl1"][i6] + v["swvl1"][i7]) / 2
                    s2 = (v["swvl2"][i6] + v["swvl2"][i7]) / 2
                    g = np.stack([t, np.maximum(_saturation_vapour_pressure(t) - _saturation_vapour_pressure(td), 0), ws, rh,
                                  s1, s2, rain7, dsr, ffm, dm, dcc, ii, bu, fw, v["t2m"][i9] - 273.15,
                                  s1 - clim[wk, 0], s2 - clim[wk, 1]], 0).astype(np.float32)
                    ix = byday[dn]
                    feat[ix] = g[:, li[ix], lj[ix]].T
                    latest_source.append(d + pd.Timedelta(hours=hours[-1]))
                    cutoff.append(e0 + pd.Timedelta(days=int(dn + 1)))
    assert_before_cutoff(latest_source, cutoff)
    return feat, dict(missing_months=missing, day_groups=len(latest_source), all_strictly_before=True)
