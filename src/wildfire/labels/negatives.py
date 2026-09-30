"""Matched negatives: random (cell, day) in the domain and season with no recent detection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.labels.detections import epoch

KEY_STRIDE = 100000   # cell * KEY_STRIDE + day identifies a cell-day


def excluded_keys(cd: pd.DataFrame) -> np.ndarray:
    """Sorted keys of every cell-day within the exclusion window after a detection."""
    after = config.load("domain")["negatives"]["excluded_days_after_detection"]
    c = cd.cell.values.astype(np.int64)
    d = cd.day.values.astype(np.int64)
    return np.unique(np.concatenate([c * KEY_STRIDE + d + k for k in range(after + 1)]))


def draw_negatives(rng, excluded, season: list[int], year: int, candidate_cells, n_neg: int,
                   group: str, state: str, band: str) -> pd.DataFrame:
    """Draw up to n_neg distinct (cell, day) pairs; the generator is shared across calls, in a fixed order."""
    cfg = config.load("domain")["negatives"]
    e0 = epoch()
    days = np.arange((pd.Timestamp(f"{year}-01-01") - e0).days, (pd.Timestamp(f"{year}-12-31") - e0).days + 1)
    days = days[np.isin((e0 + pd.to_timedelta(days, "D")).month, season)]
    got: set[int] = set()
    res: list[tuple[int, int]] = []
    tries = 0
    while len(res) < n_neg and tries < cfg["max_tries"] and len(candidate_cells):
        m = int((n_neg - len(res)) * cfg["oversample_factor"]) + cfg["oversample_extra"]
        cc = rng.choice(candidate_cells, m)
        dd = rng.choice(days, m)
        keys = cc.astype(np.int64) * KEY_STRIDE + dd
        blocked = np.isin(keys, excluded)
        for k, c, d, b in zip(keys.tolist(), cc.tolist(), dd.tolist(), blocked.tolist()):
            if b or k in got:
                continue
            got.add(k)
            res.append((c, d))
            if len(res) >= n_neg:
                break
        tries += 1
    return pd.DataFrame(res, columns=["cell", "day"]).assign(
        label=0, kind="negative", state=state, year=year, group=group, band=band)
