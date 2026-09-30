"""Assemble the labelled rows: onsets and their matched negatives."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.labels import detections, domain, negatives

ROW_COLUMNS = ["cell", "day", "label", "kind", "state", "year", "group", "band"]


def build_rows(cd: pd.DataFrame, cells: pd.DataFrame, roles: dict | None = None,
               bands: tuple = ("A", "B")) -> tuple[pd.DataFrame, dict]:
    """Return (rows, season months per state). `roles` maps grid roles to group names (default: the configured
    roles); `bands` selects the tree-cover bands that get rows."""
    cfg = config.load("domain")
    cells = domain.cells_with_domain(cells, roles)
    cd2 = cd.merge(cells[["cell", "group", "domain20", "domain10", "domain40", "bandB"]], on="cell")
    cd2 = cd2[(cd2.group != "none") & cd2.domain10]
    cd2 = domain.add_month(cd2)
    season = domain.season_months(cd2)
    excluded = negatives.excluded_keys(cd)
    rng = np.random.default_rng(cfg["negatives"]["seed"])
    k = cfg["negatives"]["per_onset"]
    out = []
    for (group, state, year), g in cd2[cd2.onset].groupby(["group", "state", "year"]):
        if state not in season:
            continue
        for band, in_band, onsets in (("A", cells.domain20.values, g[g.domain20]),
                                      ("B", cells.bandB.values, g[g.bandB])):
            if band not in bands or not len(onsets):
                continue
            pos = onsets.assign(label=1, kind="onset", band=band)
            cand = cells.cell.values[(cells.group.values == group) & (cells.state.values == state) & in_band]
            out.append(pos[ROW_COLUMNS])
            out.append(negatives.draw_negatives(rng, excluded, season[state], year, cand, k * len(pos), group, state, band))
    rows = pd.concat(out, ignore_index=True)
    rows["date"] = detections.epoch() + pd.to_timedelta(rows.day, "D")
    rows["week"] = rows.date.dt.isocalendar().week.astype(int)
    rows["cluster"] = rows.state + "|" + rows.year.astype(str) + "|" + rows.week.astype(str).str.zfill(2)
    rows["holdout_year"] = rows.year == config.load("evaluation")["splits"]["holdout_year"]
    rows = rows.merge(cells[["cell", "domain10", "domain20", "domain40"]], on="cell", how="left")
    return rows, season
