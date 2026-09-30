"""Static, human and vegetation-independent layers attached to each row by cell."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.features.tabular.masks import percentile_rank

LAYER_COLUMNS = {    # input key -> columns read as they are
    "land_cover": ["wc_10", "wc_20", "wc_30", "wc_40", "wc_50", "wc_60", "wc_80", "wc_90", "wc_edge_density", "wc_dist_built_m"],
    "terrain": ["dem_mean", "dem_sd", "dem_slope", "dem_sin_aspect", "dem_cos_aspect", "dem_tpi"],
}


def attach_static(rows: pd.DataFrame, ncell: int) -> None:
    """Add land-cover, terrain, log distance to built area and the human layers to `rows` (in place).

    Human layers are frozen at one epoch each (no year signal): population and human-modification index at their
    epoch, roads at their epoch, and night lights as the percentile rank of one year's composite over all grid cells.
    """
    epoch = config.load("features")["tabular"]["human_frozen_epoch"]
    cidx = rows.cell.values

    def read(key, cols):
        s = pd.read_csv(config.data_file(key), usecols=["cell"] + cols)
        assert (s.cell.values == np.arange(ncell)).all(), f"{key}: cell order != grid"
        return s

    for key, cols in LAYER_COLUMNS.items():
        s = read(key, cols)
        for c in cols:
            rows[c] = s[c].values.astype(np.float32)[cidx]
    rows["log_dist_built"] = np.log1p(rows.wc_dist_built_m)
    rows.drop(columns="wc_dist_built_m", inplace=True)

    human = read("human", [f"pop_{epoch['pop']}", f"ghm_aa_{epoch['ghm']}", f"ghm_bu_{epoch['ghm']}"])
    rows["hb_log_pop"] = human[f"pop_{epoch['pop']}"].values.astype(np.float32)[cidx]
    rows["hb_ghm_aa"] = human[f"ghm_aa_{epoch['ghm']}"].values.astype(np.float32)[cidx]
    rows["hb_ghm_bu"] = human[f"ghm_bu_{epoch['ghm']}"].values.astype(np.float32)[cidx]
    roads = read("roads", [f"road_km_per_km2_{epoch['roads']}"])
    rows["hb_road_km"] = roads[f"road_km_per_km2_{epoch['roads']}"].values.astype(np.float32)[cidx]
    nl = read("night_lights", [f"nl_{epoch['night_lights']}"])
    rows["hb_nl_rank"] = percentile_rank(nl[f"nl_{epoch['night_lights']}"].values)[cidx]
