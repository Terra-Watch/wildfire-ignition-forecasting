"""Tables and plots for the labels notebook."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.labels.detections import load_grid
from wildfire.pipelines.features import load_model_table
from wildfire.reporting.style import COLOURS, new_figure, plt

ROLE_COLOURS = {"train": COLOURS["good"], "heldout_UK": COLOURS["primary"], "reserve": COLOURS["warn"], "unused": "#d1d5db", "buffer": "#9ca3af"}


def rule_table() -> pd.DataFrame:
    """The label rules and their settings, read from the configuration."""
    d = config.load("domain")
    return pd.DataFrame([
        ("grid cell", f"{d['grid']['cell_size_m']} m square, projection {d['grid']['projected_crs']}"),
        ("forest domain", f"tree fraction >= {d['forest']['tree_fraction_min']}"),
        ("onset", f"a detection with none in the {d['onset']['quiet_days_before']} days before"),
        ("negatives", f"{d['negatives']['per_onset']} per onset: random cell-day in the domain and season, no detection on days D-{d['negatives']['excluded_days_after_detection']}..D"),
        ("season", f"months holding {int(d['season']['share_of_onsets'] * 100)}% of a state's onsets up to {d['season']['last_reference_year']}"),
        ("predictors", f"observed {d['target']['horizon_days']} day(s) before the target day or earlier")],
        columns=["rule", "setting"])


def season_table() -> pd.DataFrame:
    s = json.loads((config.output_dir("labels") / "season.json").read_text())
    return pd.DataFrame(dict(state=list(s), months=[", ".join(map(str, m)) for m in s.values()]))


def onset_example(cell_rows: int = 12) -> pd.DataFrame:
    """Consecutive detection days of one busy cell, with the onset flag the rule gives."""
    cd = pd.read_csv(config.output_dir("labels") / "cell_days.csv", usecols=["cell", "day", "n", "onset"])
    busiest = cd.groupby("cell").size().idxmax()
    return cd[cd.cell == busiest].head(cell_rows).reset_index(drop=True)


def rows_by_year() -> pd.DataFrame:
    """Onsets and negatives per year in the modelling table."""
    t, _ = load_model_table()
    return t.pivot_table(index="year", columns="label", values="cell", aggfunc="size").rename(columns={0: "negatives", 1: "onsets"})


def plot_rows_by_year():
    r = rows_by_year()
    fig, ax = new_figure()
    ax.bar(r.index.astype(str), r["onsets"] / 1e3, color=COLOURS["secondary"], label="onsets")
    ax.set_xlabel("Year")
    ax.set_ylabel("Onsets (thousands)")
    ax.set_title("Onsets per year in the modelling table (20 negatives are drawn for each)")
    return fig


def cell_block(cells: pd.DataFrame, cell: int) -> pd.DataFrame:
    """The cell and its neighbours that are in the grid, with their offsets in cells (dx, dy from the cell)."""
    c = cells.set_index("cell").loc[cell]
    near = cells[(cells.ix - c.ix).abs().le(1) & (cells.iy - c.iy).abs().le(1)].copy()
    near["dx"], near["dy"] = near.ix - c.ix, near.iy - c.iy
    return near[["cell", "role", "state", "dx", "dy"]].reset_index(drop=True)


@lru_cache(maxsize=1)
def _label_rows() -> pd.DataFrame:
    """Labelled rows the models are built from: training cells of the main forest band (A), up to the end of the modelling window."""
    rows = pd.read_csv(config.output_dir("labels") / "rows.csv", usecols=["cell", "day", "label", "kind", "state", "year", "group", "band", "date"],
                       dtype={"kind": "category", "state": "category", "group": "category", "band": "category"})
    keep = (rows.group == "train") & (rows.band == "A") & (rows.date <= config.load("domain")["modelling_window_end"])
    return rows[keep].reset_index(drop=True)


def example_cell() -> tuple[int, int]:
    """The training cell with the most onsets, and the year in which it has the most."""
    rows = _label_rows()
    onsets = rows[rows.label == 1]
    cell = int(onsets.groupby("cell").size().idxmax())
    return cell, int(onsets[onsets.cell == cell].groupby("year").size().idxmax())


def plot_cell_block():
    """One 5 km cell and its eight neighbours, to scale."""
    size_km = config.load("domain")["grid"]["cell_size_m"] / 1000
    cells, _ = load_grid()
    cell, _ = example_cell()
    info = cells.set_index("cell").loc[cell]
    block = cell_block(cells, cell).set_index(["dx", "dy"])
    fig, ax = plt.subplots(figsize=(5, 5))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            here = (dx, dy) in block.index
            ax.add_patch(plt.Rectangle(((dx - 0.5) * size_km, (dy - 0.5) * size_km), size_km, size_km,
                                       facecolor=ROLE_COLOURS.get(block.loc[(dx, dy), "role"], "k") if here else "white",
                                       edgecolor="k", lw=2.5 if (dx, dy) == (0, 0) else 0.6, hatch=None if here else "//", alpha=0.85))
            ax.text(dx * size_km, dy * size_km, str(int(block.loc[(dx, dy), "cell"])) if here else "outside\ndomain", ha="center", va="center", fontsize=8)
    ax.set_xlim(-1.5 * size_km, 1.5 * size_km)
    ax.set_ylim(-1.5 * size_km, 1.5 * size_km)
    ax.set_aspect("equal")
    ax.set_xlabel("km east of the cell")
    ax.set_ylabel("km north of the cell")
    ax.set_title(f"One cell: #{cell}, {size_km:g} x {size_km:g} km = {size_km ** 2:g} km2\n{info.state}, {info.lat:.2f} N {info.lon:.2f} E, role {info.role}", fontsize=9)
    fig.tight_layout()
    return fig


def plot_cell_days():
    """The example cell over its busiest year: detections per day, and the rows (one per cell-day) built from them."""
    cell, year = example_cell()
    cd = pd.read_csv(config.output_dir("labels") / "cell_days.csv", usecols=["cell", "day", "n", "onset", "year"])
    mine = cd[(cd.cell == cell) & (cd.year == year)]
    rows = _label_rows()
    r = rows[(rows.cell == cell) & (rows.year == year)]
    fig, ax = new_figure(9, 3.6)
    ax.bar(pd.to_datetime(mine.day, unit="D"), mine.n, width=1.0, color=[COLOURS["secondary"] if o else COLOURS["warn"] for o in mine.onset],
           label="detections that day (red = onset)")
    for label, colour, marker, size, name in ((0, COLOURS["neutral"], "o", 14, "row with label 0 (no fire)"),
                                              (1, COLOURS["secondary"], "v", 40, "row with label 1 (onset)")):
        d = r[r.label == label]
        ax.scatter(pd.to_datetime(d.day, unit="D"), np.full(len(d), -0.35), s=size, color=colour, marker=marker, label=name)
    ax.set_xlabel(f"Day in {year}")
    ax.set_ylabel("detections in the cell")
    ax.set_title(f"Cell #{cell} in {year}: each marker on the bottom line is one row = this cell on one day ({len(r)} rows)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    return ax.figure


def cell_rows_table(n: int = 14) -> pd.DataFrame:
    """The first rows of the example cell in its busiest year: one row per cell and day."""
    cell, year = example_cell()
    cd = pd.read_csv(config.output_dir("labels") / "cell_days.csv", usecols=["cell", "day", "n", "frp"])
    rows = _label_rows()
    r = rows[(rows.cell == cell) & (rows.year == year)].merge(cd, on=["cell", "day"], how="left").sort_values("day").head(n)
    r["detections"] = r.n.fillna(0).astype(int)
    return r[["cell", "date", "label", "kind", "state", "detections", "frp"]].rename(columns={"frp": "frp_mw"}).reset_index(drop=True)


def row_counts_table() -> pd.DataFrame:
    """From possible cell-days to the labelled rows kept, and how many rows a cell has."""
    rows = _label_rows()
    n_cells = int(rows.cell.nunique())
    days = int(rows.day.max() - rows.day.min() + 1)
    per_cell = rows.groupby("cell").size()
    return pd.DataFrame([
        ("training cells with rows (5 x 5 km each)", f"{n_cells:,}"),
        ("days in the label window", f"{days:,}"),
        ("possible cell-days", f"{n_cells * days:,}"),
        ("labelled rows kept (main forest band)", f"{len(rows):,}"),
        ("  onsets (label 1)", f"{int(rows.label.sum()):,}"),
        ("  matched non-fire days (label 0)", f"{int((rows.label == 0).sum()):,}"),
        ("rows per cell: median", f"{per_cell.median():.0f}"),
        ("rows per cell: largest", f"{int(per_cell.max()):,}")], columns=["quantity", "value"])
