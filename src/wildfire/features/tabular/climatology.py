"""Same-cell, same-ISO-week onset rate over strictly earlier years."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wildfire.labels.detections import epoch


def cell_week_climatology(cell_on, day_on, flagged_days, row_cells, row_days, ncell, *, first_year, years, week_cap):
    """Rate per observed day, over calendar years first_year .. row year - 1.

    Onsets on flagged days are excluded from the numerator and those days from the denominator, only in the earlier
    (look-up) years; a row's own year is never used. Rows in first_year (no earlier year) or whose earlier weeks are all
    flagged get NaN. Returns (rate, year index of each row).
    """
    e0 = epoch()
    fl = set(int(x) for x in np.asarray(flagged_days))
    cell_on = np.asarray(cell_on)
    day_on = np.asarray(day_on, np.int64)
    keep = np.array([int(d) not in fl for d in day_on], bool)
    d_on = e0 + pd.to_timedelta(day_on[keep], "D")
    wk = np.minimum(d_on.isocalendar().week.values.astype(int), week_cap)
    yi = d_on.year.values - first_year
    counts = np.zeros((ncell, years, week_cap + 1), np.float32)
    okk = (yi >= 0) & (yi < years)
    np.add.at(counts, (cell_on[keep][okk], yi[okk], wk[okk]), 1.0)
    observed = np.full((years, week_cap + 1), 7.0)
    for d in fl:
        t = e0 + pd.Timedelta(days=d)
        if 0 <= t.year - first_year < years:
            observed[t.year - first_year, min(int(t.isocalendar()[1]), week_cap)] -= 1.0
    r = e0 + pd.to_timedelta(np.asarray(row_days, np.int64), "D")
    rwk = np.minimum(r.isocalendar().week.values.astype(int), week_cap)
    ryi = r.year.values - first_year
    rc = np.asarray(row_cells)
    clim = np.full(len(rc), np.nan)
    for k in range(1, years):
        m = ryi == k
        if m.any():
            den = observed[:k][:, rwk[m]].sum(0)
            clim[m] = counts[rc[m], :k, rwk[m]].sum(1) / np.where(den > 0, den, np.nan)
    return clim, ryi
