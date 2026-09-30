"""Vegetation-index composites of the sealed season, aggregated to the 5 km study cells."""
from __future__ import annotations

import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyproj
import rasterio
import requests
from rasterio.enums import Resampling

from wildfire import config
from wildfire.data.downloads import sealed_folder
from wildfire.labels.detections import load_grid

EPOCH = dt.date(1970, 1, 1)


def search(start: str, end: str) -> list[dict]:
    """Composite items of the collection over the study box for the period."""
    cfg = config.load("sealed")["vegetation"]
    box = config.load("domain")["bounding_box"]
    body = {"collections": [cfg["collection"]], "bbox": [box["west"], box["south"], box["east"], box["north"]],
            "datetime": f"{start}/{end}", "limit": 1000}
    return requests.post(cfg["stac"].format(api=cfg["api_number"]), json=body, timeout=120).json().get("features", [])


def signed_href(href: str) -> str:
    cfg = config.load("sealed")["vegetation"]
    for attempt in range(6):
        try:
            return requests.get(cfg["sign"].format(api=cfg["api_number"]), params={"href": href}, timeout=60).json()["href"]
        except Exception:  # noqa: BLE001 - retried, then raised below
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("could not sign the asset address")


def cell_lookup():
    """Function mapping projected x, y to grid cell indices (-1 outside the study grid)."""
    cells, meta = load_grid()
    size = config.load("domain")["grid"]["cell_size_m"]
    lut = np.full(meta["nx"] * meta["ny"], -1, np.int64)
    lut[cells.ix.values * meta["ny"] + cells.iy.values] = np.arange(len(cells))

    def lookup(x, y):
        i = np.floor((x - meta["x0"]) / size).astype(np.int64)
        j = np.floor((y - meta["y0"]) / size).astype(np.int64)
        ok = (i >= 0) & (i < meta["nx"]) & (j >= 0) & (j < meta["ny"])
        out = np.full(len(x), -1, np.int64)
        out[ok] = lut[i[ok] * meta["ny"] + j[ok]]
        return out
    return lookup, len(cells)


def aggregate_item(item: dict, lookup, ncell: int):
    """Sum and count of valid vegetation-index pixels per study cell for one tile; None if the tile cannot be read."""
    cfg = config.load("sealed")["vegetation"]
    grid = config.load("domain")["grid"]
    try:
        with rasterio.open(signed_href(item["assets"][cfg["asset"]]["href"])) as ds:
            step = cfg["decimate"]
            a = ds.read(1, out_shape=(ds.height // step, ds.width // step), resampling=Resampling.nearest).astype(np.float64)
            tr = ds.transform * ds.transform.scale(step, step)
            to_ll = pyproj.Transformer.from_crs(ds.crs, grid["source_crs"], always_xy=True)
        to_grid = pyproj.Transformer.from_crs(grid["source_crs"], grid["projected_crs"], always_xy=True)
        r, c = np.meshgrid(np.arange(a.shape[0]), np.arange(a.shape[1]), indexing="ij")
        lon, lat = to_ll.transform((tr.c + (c + .5) * tr.a).ravel(), (tr.f + (r + .5) * tr.e).ravel())
        x, y = to_grid.transform(lon, lat)
        cell, v = lookup(x, y), a.ravel()
        lo, hi = cfg["valid_range"]
        ok = (cell >= 0) & (v > lo) & (v <= hi)
        return np.bincount(cell[ok], v[ok] / cfg["scale"], ncell), np.bincount(cell[ok], minlength=ncell)
    except Exception:  # noqa: BLE001 - an unreadable tile is counted and reported by the caller
        return None


def composite_end(start: dt.date, days: int) -> int:
    """Day number of the composite end: start + composite_days, capped at 31 December of its year."""
    end = min(start + dt.timedelta(days=days), dt.date(start.year, 12, 31))
    return (end - EPOCH).days


def fetch_vegetation(log=print) -> dict:
    """Composites of the period as per-cell means (composites, cells), with start and end day numbers; saved to a file."""
    cfg, period = config.load("sealed")["vegetation"], config.load("sealed")["period"]
    lookup, ncell = cell_lookup()
    items = search(period["start"], period["end"])
    sums: dict[str, list] = {}
    errors = 0
    with ThreadPoolExecutor(4) as pool:
        for item, result in zip(items, pool.map(lambda i: aggregate_item(i, lookup, ncell), items)):
            if result is None:
                errors += 1
                continue
            start = (item["properties"].get("start_datetime") or item["properties"]["datetime"])[:10]
            acc = sums.setdefault(start, [np.zeros(ncell), np.zeros(ncell)])
            acc[0] += result[0]
            acc[1] += result[1]
    keys = sorted(sums)
    start_days = np.array([(dt.date.fromisoformat(k) - EPOCH).days for k in keys], np.int32)
    end_days = np.array([composite_end(dt.date.fromisoformat(k), cfg["composite_days"]) for k in keys], np.int32)
    mean = np.array([np.where(sums[k][1] > 0, sums[k][0] / np.maximum(sums[k][1], 1), np.nan) for k in keys], np.float32)
    np.savez_compressed(sealed_folder("vegetation") / cfg["output"], ndvi=mean, start=start_days, end=end_days)
    log("vegetation composites", len(keys), "items", len(items), "unreadable tiles", errors)
    return dict(composites=len(keys), items=len(items), unreadable=errors)
