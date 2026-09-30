"""Forest domain, roles and fire season per state."""
from __future__ import annotations

import pandas as pd

from wildfire import config
from wildfire.labels.detections import epoch


def cells_with_domain(cells: pd.DataFrame, roles: dict | None = None) -> pd.DataFrame:
    dom = config.load("domain")
    tree = pd.read_csv(config.data_file("land_cover"), usecols=["cell", "wc_tree"])
    cells = cells.merge(tree, on="cell", how="left")
    f = dom["forest"]
    cells["domain20"] = cells.wc_tree >= f["tree_fraction_min"]
    cells["domain10"] = cells.wc_tree >= f["secondary_band_min"]
    cells["domain40"] = cells.wc_tree >= f["dense_subset_min"]
    cells["bandB"] = cells.domain10 & ~cells.domain20
    roles = dom["roles"] if roles is None else roles
    cells["group"] = cells.role.map(roles).fillna("none")
    return cells


def season_months(cd2: pd.DataFrame) -> dict[str, list[int]]:
    """Months holding the configured share of each state's onsets in the reference years."""
    s = config.load("domain")["season"]
    ref = cd2[cd2.onset & cd2.domain20 & (cd2.year <= s["last_reference_year"])]
    out = {}
    for state, g in ref.groupby("state"):
        c = g.month.value_counts().sort_values(ascending=False)
        cum = c.cumsum() / c.sum()
        out[state] = sorted(c.index[: int((cum < s["share_of_onsets"]).sum() + 1)].tolist())
    return out


def add_month(cd2: pd.DataFrame) -> pd.DataFrame:
    cd2["month"] = (epoch() + pd.to_timedelta(cd2.day, "D")).dt.month.values
    return cd2
