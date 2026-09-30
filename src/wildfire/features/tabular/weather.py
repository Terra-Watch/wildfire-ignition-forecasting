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
PRIMARY_VARIABLES = ("t2m", "d2m", "u10", "v10", "tp", "swvl1", "swvl2")


def _saturation_vapour_pressure(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def _weather_path(group: str, year: int, month: int):
    data = config.load("data", expand=False)["inputs"]
    return config.data_root() / data["weather_dir"] / data["weather_file"].format(group=group, year=year, month=month)


def _load_month(group, year, month, names):
    """(times, {variable: array}) of one monthly file, or (None, None) when the file is missing."""
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


def grid_layout() -> dict:
    """Size and origin of the weather grid, and the latitude bands used by the code recursion."""
    w = config.load("features")["tabular"]["weather"]
    sample = xr.open_dataset(_weather_path("primary", 2019, 1))
    latg = sample.latitude.values
    layout = dict(nlat=sample.sizes["latitude"], nlon=sample.sizes["longitude"], latg=latg,
                  lat0=float(latg[0]), lon0=float(sample.longitude.values[0]))
    sample.close()
    bands = [np.where((latg > lo) & (latg <= hi))[0] for lo, hi in w["lat_bands"]]
    layout["bands"] = [b for b in bands if len(b)]
    layout["band_lat"] = [float(latg[b].mean()) for b in layout["bands"]]
    return layout


def grid_indices(lat, lon, layout):
    """Weather-grid row and column of each (lat, lon)."""
    step = config.load("features")["tabular"]["weather"]["grid_step_deg"]
    li = np.clip(np.round((layout["lat0"] - lat) / step).astype(int), 0, layout["nlat"] - 1)
    lj = np.clip(np.round((lon - layout["lon0"]) / step).astype(int), 0, layout["nlon"] - 1)
    return li, lj


def soil_climatology(layout: dict) -> np.ndarray:
    """Same-ISO-week mean of noon soil moisture (layers 1 and 2) over the climatology years."""
    w = config.load("features")["tabular"]["weather"]
    cap = w["iso_week_cap"]
    cs = np.zeros((cap + 1, 2, layout["nlat"], layout["nlon"]))
    cc = np.zeros(cap + 1)
    y0, y1 = w["soil_climatology_years"]
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            tv, v = _load_month("primary", y, m, ("swvl1", "swvl2"))
            if tv is None:
                continue
            for d in pd.date_range(f"{y}-{m:02d}-01", periods=pd.Period(f"{y}-{m:02d}").days_in_month):
                _, i6, i7, _ = _hour_indices(tv, d, w["hours_utc"])
                wk = min(int(d.isocalendar()[1]), cap)
                cs[wk, 0] += (v["swvl1"][i6] + v["swvl1"][i7]) / 2
                cs[wk, 1] += (v["swvl2"][i6] + v["swvl2"][i7]) / 2
                cc[wk] += 1
    return cs / np.maximum(cc, 1)[:, None, None, None]


class CodeState:
    """Running state of the fire-weather recursion and the rain counters on the weather grid."""

    def __init__(self, layout: dict, clim: np.ndarray):
        self.w = config.load("features")["tabular"]["weather"]
        shape = (layout["nlat"], layout["nlon"])
        init = self.w["initial_codes"]
        self.layout, self.clim = layout, clim
        self.ffmc, self.dmc, self.dc = (np.full(shape, init[k]) for k in ("ffmc", "dmc", "dc"))
        self.rain_hist = collections.deque(maxlen=self.w["rain_window_days"])
        self.dry_spell = np.zeros(shape)
        self.prev_rain_0630 = np.zeros(shape)

    def _update_codes(self, t, rh, ws, rain24, month):
        for b, lm in zip(self.layout["bands"], self.layout["band_lat"]):
            self.ffmc[b] = fwi.ffmc(self.ffmc[b], t[b], rh[b], ws[b], rain24[b])
            self.dmc[b] = fwi.dmc(self.dmc[b], t[b], rh[b], rain24[b], lm, month)
            self.dc[b] = fwi.dc(self.dc[b], t[b], rh[b], rain24[b], lm, month)

    def update(self, v: dict, idx: list, date: pd.Timestamp):
        """Advance one day. Returns (fields (len(NAMES), nlat, nlon) float32, daily rain total in mm)."""
        w = self.w
        i0, i6, i7, i9 = idx
        noon = lambda name: (v[name][i6] + v[name][i7]) / 2
        t, td = noon("t2m") - 273.15, noon("d2m") - 273.15
        ws = np.hypot(noon("u10"), noon("v10")) * 3.6
        rh = np.clip(100 * _saturation_vapour_pressure(td) / _saturation_vapour_pressure(t), *w["humidity_bounds"])
        tp00 = np.nan_to_num(v["tp"][i0]) * 1000.0
        tp0630 = (np.nan_to_num(v["tp"][i6]) + np.nan_to_num(v["tp"][i7])) / 2 * 1000.0
        rain24 = np.maximum(tp0630 + tp00 - self.prev_rain_0630, 0.0)
        self.prev_rain_0630 = tp0630
        self.rain_hist.append(tp00)
        self.dry_spell = np.where(tp00 > w["rain_threshold_mm"], 0.0, np.minimum(self.dry_spell + 1, w["dry_spell_cap_days"]))
        self._update_codes(t, rh, ws, rain24, date.month)
        isi = fwi.isi(self.ffmc, ws)
        bui = fwi.bui(self.dmc, self.dc)
        wk = min(int(date.isocalendar()[1]), w["iso_week_cap"])
        s1, s2 = noon("swvl1"), noon("swvl2")
        fields = np.stack([t, np.maximum(_saturation_vapour_pressure(t) - _saturation_vapour_pressure(td), 0), ws, rh, s1, s2,
                           np.sum(self.rain_hist, axis=0), self.dry_spell, self.ffmc, self.dmc, self.dc, isi, bui,
                           fwi.fwi(isi, bui), v["t2m"][i9] - 273.15, s1 - self.clim[wk, 0], s2 - self.clim[wk, 1]], 0)
        return fields.astype(np.float32), tp00


def daily_fields(layout: dict, last_day: int):
    """Yield (day number, date, fields, daily rain total) for every day from the first day through last_day.
    Months with no primary file are skipped (the recursion state is not advanced)."""
    w = config.load("features")["tabular"]["weather"]
    state = CodeState(layout, soil_climatology(layout))
    e0 = epoch()
    for y in range(pd.Timestamp(w["first_day"]).year, (e0 + pd.Timedelta(days=last_day)).year + 1):
        for m in range(1, 13):
            tv, v = _load_month("primary", y, m, PRIMARY_VARIABLES)
            if tv is None:
                continue
            for d in pd.date_range(f"{y}-{m:02d}-01", periods=pd.Period(f"{y}-{m:02d}").days_in_month):
                dn = (d - e0).days
                if dn > last_day:
                    return
                fields, rain = state.update(v, _hour_indices(tv, d, w["hours_utc"]), d)
                yield dn, d, fields, rain


def weather_features(lat: np.ndarray, lon: np.ndarray, day: np.ndarray) -> tuple[np.ndarray, dict]:
    """Features (rows, len(NAMES)) float32 for cell-days at (lat, lon) on `day`; also returns a report dict."""
    w = config.load("features")["tabular"]["weather"]
    day = np.asarray(day, np.int64)
    layout = grid_layout()
    li, lj = grid_indices(lat, lon, layout)
    x = day - 1
    assert x.min() >= (pd.Timestamp(w["first_day"]) - epoch()).days + w["first_row_margin_days"]
    byday = {int(k): np.where(x == k)[0] for k in np.unique(x)}
    feat = np.full((len(day), len(NAMES)), np.nan, np.float32)
    source_time, cutoff = [], []
    for dn, d, fields, _ in daily_fields(layout, max(byday)):
        if dn in byday:
            ix = byday[dn]
            feat[ix] = fields[:, li[ix], lj[ix]].T
            source_time.append(d + pd.Timedelta(hours=w["hours_utc"][-1]))
            cutoff.append(epoch() + pd.Timedelta(days=int(dn + 1)))
    assert_before_cutoff(source_time, cutoff)
    return feat, dict(day_groups=len(source_time), all_strictly_before=True)
