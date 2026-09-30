"""Burned-area history features with explicit observation coverage."""
from __future__ import annotations

import datetime as dt

import numpy as np


def month_end_days(month_start) -> np.ndarray:
    """Exclusive end day of each month, from the array of month-start days."""
    ms = np.asarray(month_start, np.int64)
    out = np.empty_like(ms)
    for i, d in enumerate(ms):
        x = dt.date(1970, 1, 1) + dt.timedelta(days=int(d))
        nx = dt.date(x.year + (x.month == 12), x.month % 12 + 1, 1)
        out[i] = (nx - dt.date(1970, 1, 1)).days
    return out


def load_burn(path) -> dict:
    """burn_ord (months, cells), month_start, month_end, covered (months, cells) and legacy flag.

    covered[m, c] is True only if that cell-month was read; burn_ord == 0 there means 'no burn', elsewhere 'unknown'.
    """
    z = np.load(path, allow_pickle=False)
    b = z["burn_ord"]
    ms = z["month_start"].astype(np.int64)
    if "cell_month_covered" in z.files:
        cov = z["cell_month_covered"].astype(bool)
        legacy = False
    else:
        cov = np.repeat(z["fetched"].astype(bool)[:, None], b.shape[1], 1)
        legacy = True
    return dict(burn_ord=b, month_start=ms, month_end=month_end_days(ms), covered=cov, legacy=legacy,
                month_status=(z["month_status"] if "month_status" in z.files else None))


def window_coverage(cov_month, month_start, month_end, lo, hi) -> np.ndarray:
    """Day-weighted fraction of the window [lo, hi) that lies in observed months (cov_month may be fractional).
    An empty window counts as fully observed."""
    lo = np.asarray(lo, np.int64)[:, None]
    hi = np.asarray(hi, np.int64)[:, None]
    w = np.clip(np.minimum(month_end[None, :], hi) - np.maximum(month_start[None, :], lo), 0, None).astype(np.float64)
    tot = w.sum(1)
    return np.where(tot > 0, (w @ np.asarray(cov_month, np.float64)) / np.maximum(tot, 1), 1.0)


def burn_features(day, o9, o1, cov_nb, cov_c, month_start, month_end, *, min_frac, lookback, cap, archive_start,
                  censor, lag) -> dict:
    """Burn-history features for the cell-days of one cell.

    o9: sorted burn dates in the 3x3 neighbourhood; o1: sorted burn dates of the cell; cov_nb / cov_c: per-month
    share of the neighbourhood / the cell that was observed. Burns count only if dated before day - lag; the window is
    [day - lookback, day - lag) and the lag days are not counted as unobserved.
    bf_log_n10_nbhd is NaN unless the window was observed at least min_frac. bf_years_since is the time since the last
    observed burn, capped (a lower bound when censored); never NaN when censor is true.
    """
    day = np.asarray(day, np.int64)
    lo = day - lookback
    hi = day - lag
    n = np.searchsorted(o9, hi, "left") - np.searchsorted(o9, lo, "left")
    obs = window_coverage(cov_nb, month_start, month_end, lo, hi)
    last = np.searchsorted(o1, hi, "left") - 1
    has = (last >= 0) & (len(o1) > 0)
    ref = np.where(has, o1[np.clip(last, 0, max(len(o1) - 1, 0))] if len(o1) else 0, archive_start)
    ys = np.minimum((day - ref) / 365.25, cap)
    ys_obs = window_coverage(cov_c, month_start, month_end,
                             np.maximum(ref, np.floor(day - cap * 365.25).astype(np.int64)), hi)
    tol = 1e-9
    return dict(
        bf_log_n10_nbhd=np.where(obs >= min_frac - tol, np.log1p(n), np.nan),
        burn_obs_frac=obs,
        bf_years_since=(ys if censor else np.where(ys_obs >= min_frac - tol, ys, np.nan)),
        bf_ys_obs_frac=ys_obs,
        bf_ys_censored=(ys_obs < 1 - tol),
    )


def check_burn_dates_in_month(burn_ord, covered, month_start, month_end) -> int:
    """Burn dates outside their own month, or non-zero in an uncovered cell-month. Must be 0."""
    b = np.asarray(burn_ord, np.int64)
    ms = np.asarray(month_start, np.int64)[:, None]
    me = np.asarray(month_end, np.int64)[:, None]
    nz = b > 0
    return int((nz & ((b < ms) | (b >= me))).sum() + (nz & ~np.asarray(covered, bool)).sum())
