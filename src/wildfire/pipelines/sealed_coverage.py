"""Coverage checks on the sealed-season sources (read-only): fire detections, hourly weather, vegetation index.
Thresholds come from configs/sealed.yaml; the report is written next to the data."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.data import downloads
from wildfire.labels import detections


def _firms_2026() -> pd.DataFrame:
    cfg = config.load("domain")["detections"]
    folder = downloads.sealed_folder("firms")
    parts = []
    for f in sorted(folder.glob("*.csv")):
        d = pd.read_csv(f, usecols=cfg["columns"], dtype={"confidence": str})
        d["box"] = "south" if "_south_" in f.name else "main"
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d = d[d.confidence.isin(cfg["confidence_keep"]) & d.satellite.isin(cfg["satellites_keep"])]
    return d.drop_duplicates(cfg["duplicate_key"])


def check_firms() -> dict:
    s, c = config.load("sealed"), config.load("sealed")["coverage"]
    period, fcfg = s["period"], s["firms"]
    folder = downloads.sealed_folder("firms")
    missing = [downloads.firms_file(p, b, d) for p in fcfg["products"] for b in fcfg["boxes"]
               for d in downloads.chunk_starts(period["start"], period["end"], fcfg["chunk_days"])
               if not (folder / downloads.firms_file(p, b, d)).exists()]
    d = _firms_2026()
    days = pd.date_range(period["start"], period["end"])
    d = d[d.acq_date.isin(days.strftime("%Y-%m-%d"))]
    per_day = d.groupby("acq_date").size().reindex(days.strftime("%Y-%m-%d"), fill_value=0)
    south = d[d.box == "south"] if "box" in d else d
    south_days = south.groupby("acq_date").size().reindex(days.strftime("%Y-%m-%d"), fill_value=0)
    prev = detections.read_detections()
    prev = prev[prev.acq_date.between(f"{int(period['start'][:4]) - 1}{period['start'][4:]}", f"{int(period['end'][:4]) - 1}{period['end'][4:]}")]
    prev_days = pd.date_range(f"{int(period['start'][:4]) - 1}{period['start'][4:]}", f"{int(period['end'][:4]) - 1}{period['end'][4:]}")
    prev_med = prev.groupby("acq_date").size().reindex(prev_days.strftime("%Y-%m-%d"), fill_value=0).median()
    ratio = float(per_day.median() / prev_med) if prev_med else float("nan")
    south_zero = float((south_days == 0).mean())
    lo, hi = c["daily_count_ratio_range"]
    return dict(missing_chunks=len(missing), rows=int(len(d)), days_with_none=int((per_day == 0).sum()),
                median_daily=float(per_day.median()), median_daily_previous_year=float(prev_med), ratio=ratio,
                ratio_ok=bool(lo <= ratio <= hi), south_zero_day_share=south_zero,
                south_ok=bool(south_zero <= c["south_zero_day_share_max"]),
                passed=bool(not missing and lo <= ratio <= hi and south_zero <= c["south_zero_day_share_max"]),
                last_day_with_data=str(per_day[per_day > 0].index.max()))


def check_weather() -> dict:
    import xarray as xr
    s = config.load("sealed")
    cfg, period = s["weather"], s["period"]
    folder = downloads.sealed_folder("weather")
    out, ok = {}, True
    for y, m in downloads.month_list(period["start"], period["end"]):
        f = folder / cfg["file_name"].format(year=y, month=m)
        key = f"{y}-{m:02d}"
        if not f.exists():
            out[key] = "missing"
            ok = False
            continue
        ds = xr.open_dataset(f)
        tname = "valid_time" if "valid_time" in ds else "time"
        n_days = pd.Timestamp(y, m, 1).days_in_month
        want = n_days * len(cfg["hours"])
        got = ds.sizes[tname]
        problems = [] if got == want else [f"time steps {got} != {want}"]
        for var in cfg["short_names"]:
            if var not in ds:
                problems.append(f"{var} missing")
                continue
            a = ds[var].values
            if np.isnan(a).all(axis=tuple(range(1, a.ndim))).any():
                problems.append(f"{var}: empty time step")
            lo_hi = cfg["ranges"].get(var)
            if lo_hi and (np.nanmin(a) < lo_hi[0] or np.nanmax(a) > lo_hi[1]):
                problems.append(f"{var}: outside physical range")
        out[key] = problems or "ok"
        ok &= not problems
    out["passed"] = bool(ok)
    return out


def check_vegetation() -> dict:
    s = config.load("sealed")
    cfg = s["vegetation"]
    f = downloads.sealed_folder("vegetation") / cfg["output"]
    if not f.exists():
        return dict(passed=False, problem="file missing")
    z = np.load(f)
    share = np.mean(~np.isnan(z["ndvi"]), axis=1)
    start = (dt.date.fromisoformat(s["period"]["start"]) - detections.epoch().date()).days
    end = (dt.date.fromisoformat(s["period"]["end"]) - detections.epoch().date()).days
    covered = bool(z["start"].min() <= start and z["end"].max() >= end)
    return dict(composites=int(len(share)), min_cell_share=float(share.min()), period_covered=covered,
                passed=bool(covered and share.min() >= cfg["min_composite_coverage"]))


def run(log=print) -> dict:
    result = dict(firms=check_firms(), weather=check_weather(), vegetation=check_vegetation())
    result["all_passed"] = all(v["passed"] for v in result.values())
    path = downloads.sealed_folder("firms").parent / "coverage.json"
    path.write_text(json.dumps(result, indent=1))
    log(json.dumps(result, indent=1))
    return result
