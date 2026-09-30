"""Days on which one satellite's detections are missing upstream, and the rows they touch."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.labels import detections


def flag_gap_days(days, snpp_n, noaa20_n, ratio_thr: float, min_n20: int):
    """A day is flagged when NOAA-20 saw >= min_n20 detections but SNPP saw < ratio_thr x NOAA-20.
    Days where both are thin are not flagged."""
    s = np.asarray(snpp_n, float)
    n = np.asarray(noaa20_n, float)
    return np.asarray(days)[(n >= min_n20) & (s < ratio_thr * n)]


def touches_flagged(day, flagged_sorted, lookback: int):
    """True where any flagged day lies in [day - lookback, day]."""
    f = np.sort(np.asarray(flagged_sorted, np.int64))
    day = np.asarray(day, np.int64)
    if len(f) == 0:
        return np.zeros(len(day), bool)
    return (np.searchsorted(f, day, "right") - np.searchsorted(f, day - lookback, "left")) > 0


def build_day_coverage(det: pd.DataFrame) -> pd.DataFrame:
    """Per-day detection counts of the two satellites with the gap flags (det: filtered, deduplicated detections)."""
    cfg = config.load("features")["tabular"]["gap_days"]
    win = config.load("domain")["study_window"]
    e0 = detections.epoch()
    d = det[["acq_date", "satellite"]].copy()
    d["day"] = (pd.to_datetime(d.acq_date) - e0).dt.days
    d0, d1 = detections.day_number(win["start"]), detections.day_number(win["end"])
    d = d[(d.day >= d0) & (d.day <= d1)]
    c = d.groupby(["day", "satellite"]).size().unstack(fill_value=0).reindex(np.arange(d0, d1 + 1), fill_value=0)
    c = c.rename(columns={"N": "snpp_n", "N20": "n20_n"})
    c["ratio"] = c.snpp_n / c.n20_n.clip(lower=1)
    flag = flag_gap_days(c.index.values, c.snpp_n.values, c.n20_n.values, cfg["ratio_threshold"], cfg["min_noaa20_detections"])
    c["snpp_thin_candidate"] = c.index.isin(flag)
    named = np.zeros(len(c), bool)
    for a, b in cfg["named_spans"]:
        named |= (c.index.values >= detections.day_number(a)) & (c.index.values <= detections.day_number(b))
    c["snpp_named_gap"] = named
    c["snpp_gap"] = named | (c.snpp_thin_candidate if cfg["include_data_driven"] else False)
    c["date"] = (e0 + pd.to_timedelta(c.index, "D")).strftime("%Y-%m-%d")
    return c.reset_index()[["day", "date", "snpp_n", "n20_n", "ratio", "snpp_named_gap", "snpp_thin_candidate", "snpp_gap"]]
