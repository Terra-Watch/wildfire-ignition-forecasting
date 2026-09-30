"""Sampling frame, per-state base rates and strata for the label design."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.evaluation import prior_shift
from wildfire.features.tabular.gap_days import touches_flagged
from wildfire.labels.detections import epoch


def cell_frame() -> pd.DataFrame:
    """Training-role cells in the forest domain: columns cell, state."""
    f = config.load("domain")["forest"]
    cells = pd.read_csv(config.data_file("grid"))
    tree = pd.read_csv(config.data_file("land_cover"), usecols=["cell", "wc_tree"])
    c = cells.merge(tree, on="cell")
    return c[(c.role == "train") & (c.wc_tree >= f["tree_fraction_min"])][["cell", "state"]].reset_index(drop=True)


def season_months() -> dict[str, list[int]]:
    return json.loads((config.output_dir("labels") / "season.json").read_text())


def flagged_days() -> np.ndarray:
    f = pd.read_csv(config.output_dir("features") / "day_coverage.csv")
    return f[f.snpp_gap].day.values


def frame_sizes(years, exclude_cells=(), gap_days=None, lookback=None) -> dict:
    """{state|year: eligible cell-days}: domain cells of the state (minus exclude_cells) x season days of that year up to
    the modelling window end (minus days whose look-back touches a flagged gap day)."""
    cf = cell_frame()
    ex = set(int(x) for x in exclude_cells)
    sm = season_months()
    end = pd.Timestamp(config.load("features")["tabular"]["window_end"])
    e0 = epoch()
    out = {}
    gap = None if gap_days is None else np.sort(np.asarray(gap_days, np.int64))
    for st, g in cf.groupby("state"):
        if st not in sm:
            continue
        nc = int((~g.cell.isin(ex)).sum())
        for y in years:
            days = pd.date_range(f"{y}-01-01", min(pd.Timestamp(f"{y}-12-31"), end))
            dn = np.array([(d - e0).days for d in days])
            ok = np.isin(days.month, sm[st])
            if gap is not None:
                ok &= ~touches_flagged(dn, gap, lookback)
            out[f"{st}|{y}"] = nc * int(ok.sum())
    return out


def gap_lookback() -> int:
    return config.load("features")["tabular"]["gap_days"]["lookback_days"]


def base_rates(table: pd.DataFrame) -> dict:
    """Per-state true onset rate per eligible cell-day from training years only."""
    last = config.load("evaluation")["splits"]["fit_last_year"]
    tr = table[table.year <= last]
    fr = frame_sizes(sorted(tr.year.unique()), gap_days=flagged_days(), lookback=gap_lookback())
    rate = {}
    for st in sorted(tr.state.unique()):
        rate[st] = float(tr[(tr.state == st) & (tr.label == 1)].shape[0] / sum(v for k, v in fr.items() if k.startswith(st + "|")))
    return rate


def strata(rows: pd.DataFrame):
    return (rows.state.astype(str) + "|" + rows.year.astype(int).astype(str)).values


def iso_clusters(rows: pd.DataFrame):
    scheme = config.load("evaluation")["clusters"]["scheme"]
    return prior_shift.cluster_labels(rows.year.values, rows.week.values, None, scheme)


def rates_before(table: pd.DataFrame, year: int) -> dict:
    """Per-state onset rate per eligible cell-day from the years before `year`."""
    tr = table[table.year < year]
    fr = frame_sizes(sorted(tr.year.unique()), gap_days=flagged_days(), lookback=gap_lookback())
    return {st: float(tr[(tr.state == st) & (tr.label == 1)].shape[0] / sum(v for k, v in fr.items() if k.startswith(st + "|")))
            for st in tr.state.unique()}


def january_to_july_frame(year: int, masked_cells) -> dict:
    """{state|year: eligible cell-days} for January-July of `year`, minus masked cells and days touching a gap day."""
    cf = cell_frame()
    season = season_months()
    flagged, lookback = flagged_days(), gap_lookback()
    e0 = epoch()
    days = pd.date_range(f"{year}-01-01", f"{year}-07-31")
    dn = np.array([(d - e0).days for d in days])
    frame = {}
    for st, g in cf.groupby("state"):
        if st not in season:
            continue
        ok = np.isin(days.month, season[st]) & ~touches_flagged(dn, flagged, lookback)
        frame[f"{st}|{year}"] = int((~g.cell.isin(set(masked_cells))).sum()) * int(ok.sum())
    return frame
