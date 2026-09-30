"""Downloads for the sealed season. Credentials come from the environment and are never printed or logged."""
from __future__ import annotations

import datetime as dt
import os
import time
from pathlib import Path

import requests

from wildfire import config


def sealed_folder(kind: str) -> Path:
    path = config.data_root() / config.load("sealed")["folders"][kind]
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunk_starts(start: str, end: str, step: int) -> list[dt.date]:
    """Start dates of consecutive chunks covering start..end (the last chunk may run past end)."""
    d, stop, out = dt.date.fromisoformat(start), dt.date.fromisoformat(end), []
    while d <= stop:
        out.append(d)
        d += dt.timedelta(days=step)
    return out


def firms_file(product: str, box: str, day: dt.date) -> str:
    return f"{product}_{box}_{day}.csv"


def fetch_chunk(product: str, box: str, day: dt.date, log=print) -> str | None:
    """One FIRMS area request; returns the csv text, or None after the attempts are used up. The key is never logged."""
    cfg = config.load("sealed")["firms"]
    key = os.environ["FIRMS_MAP_KEY"].strip().strip('"')
    url = f"{cfg['api']}/{key}/{product}/{cfg['boxes'][box]}/{cfg['chunk_days']}/{day}"
    for attempt in range(cfg["max_attempts"]):
        try:
            r = requests.get(url, timeout=300)
            if r.status_code == 200 and r.text.startswith("latitude"):
                return r.text
            log("firms response", product, box, day, r.status_code, r.text[:80].replace(key, "<key>"))
        except requests.RequestException as e:
            log("firms error", product, box, day, type(e).__name__)
        time.sleep(cfg["retry_wait_seconds"] * (attempt + 1))
    return None


def fetch_firms(log=print) -> dict:
    """Download every missing chunk for every product and box; returns the number of chunks fetched and failed."""
    cfg, period = config.load("sealed")["firms"], config.load("sealed")["period"]
    folder = sealed_folder("firms")
    counts = dict(fetched=0, failed=0)
    for product in cfg["products"]:
        for box in cfg["boxes"]:
            for day in chunk_starts(period["start"], period["end"], cfg["chunk_days"]):
                f = folder / firms_file(product, box, day)
                if f.exists():
                    continue
                text = fetch_chunk(product, box, day, log)
                if text is None:
                    counts["failed"] += 1
                    continue
                f.write_text(text)
                counts["fetched"] += 1
    log("firms", counts)
    return counts


def month_list(start: str, end: str) -> list[tuple[int, int]]:
    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    out, y, m = [], s.year, s.month
    while (y, m) <= (e.year, e.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def fetch_weather(log=print) -> dict:
    """One monthly ERA5-Land file per month of the period (same variables, hours and area as the development files)."""
    import cdsapi
    cfg, period = config.load("sealed")["weather"], config.load("sealed")["period"]
    folder = sealed_folder("weather")
    client = cdsapi.Client(url=os.environ["CDSAPI_URL"].strip().strip('"'), key=os.environ["CDSAPI_KEY"].strip().strip('"'), quiet=True)
    counts = dict(fetched=0, failed=0)
    for y, m in month_list(period["start"], period["end"]):
        target = folder / cfg["file_name"].format(year=y, month=m)
        if target.exists():
            continue
        request = dict(variable=cfg["variables"], year=str(y), month=f"{m:02d}", day=[f"{d:02d}" for d in range(1, 32)],
                       time=cfg["hours"], area=cfg["area"], data_format="netcdf", download_format="unarchived")
        try:
            client.retrieve(cfg["dataset"], request, str(target) + ".part")
            os.replace(str(target) + ".part", target)
            counts["fetched"] += 1
            log("weather", y, m, "done")
        except Exception as e:  # noqa: BLE001 - any download failure is recorded, not hidden
            counts["failed"] += 1
            log("weather failed", y, m, type(e).__name__)
    return counts
