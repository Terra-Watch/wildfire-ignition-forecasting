"""Tables and plots for the data notebook."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.labels.detections import epoch, load_grid
from wildfire.reporting.style import COLOURS, new_figure


def manifest_table() -> pd.DataFrame:
    """One row per dataset from data/MANIFEST.json: files, size and completeness."""
    m = json.loads((config.REPO_ROOT / "data" / "MANIFEST.json").read_text())["datasets"]
    return pd.DataFrame([dict(dataset=k, files=v["file_count"], size_gb=round(v["bytes"] / 1e9, 2), complete=v["complete"],
                              coverage=(v["coverage"] or {}).get("start", "") + " to " + (v["coverage"] or {}).get("end", ""))
                         for k, v in m.items()])


def detections_by_year() -> pd.DataFrame:
    """Fire detections per year and state, from the cell-day table."""
    cd = pd.read_csv(config.output_dir("labels") / "cell_days.csv", usecols=["state", "year", "n"])
    return cd.pivot_table(index="year", columns="state", values="n", aggfunc="sum", fill_value=0)


def plot_detections_by_year():
    t = detections_by_year().sum(axis=1)
    fig, ax = new_figure()
    ax.bar(t.index.astype(str), t.values / 1e6, color=COLOURS["primary"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Fire detections (millions)")
    ax.set_title("Satellite fire detections kept after filtering")
    return fig


def gap_day_table() -> pd.DataFrame:
    """Days on which one satellite's detections are missing upstream, grouped into runs."""
    c = pd.read_csv(config.output_dir("features") / "day_coverage.csv")
    flagged = c[c.snpp_gap]
    runs = (flagged.day.diff().fillna(2) > 1).cumsum()
    return flagged.groupby(runs).agg(first=("date", "min"), last=("date", "max"), days=("day", "size")).reset_index(drop=True)


def plot_gap_days():
    c = pd.read_csv(config.output_dir("features") / "day_coverage.csv")
    dates = epoch() + pd.to_timedelta(c.day, "D")
    fig, ax = new_figure(8, 3.4)
    ax.plot(dates, c.snpp_n, color=COLOURS["primary"], lw=0.6, label="SNPP")
    ax.plot(dates, c.n20_n, color=COLOURS["secondary"], lw=0.6, label="NOAA-20")
    ax.fill_between(dates, 0, c.n20_n.max(), where=c.snpp_gap.values, color=COLOURS["warn"], alpha=0.4, label="flagged gap")
    ax.set_xlabel("Date")
    ax.set_ylabel("Detections per day")
    ax.set_title("Daily detections of the two satellites")
    ax.legend()
    return fig


def cell_roles_table() -> pd.DataFrame:
    cells, _ = load_grid()
    return cells.groupby(["role", "state"]).size().unstack(0, fill_value=0)


def plot_cell_map():
    cells, _ = load_grid()
    fig, ax = new_figure(6, 5.5)
    palette = {"train": COLOURS["primary"], "heldout_UK": COLOURS["secondary"], "reserve": COLOURS["warn"],
               "unused": "#d1d5db", "buffer": "#9ca3af"}
    for role, g in cells.groupby("role"):
        ax.scatter(g.lon, g.lat, s=1, color=palette.get(role, "k"), label=role)
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.set_title("5 km grid cells by role")
    ax.legend(markerscale=6)
    return fig


def burn_gap_table() -> pd.DataFrame:
    """Months with no burned-area observation, from the burn file."""
    z = np.load(config.data_file("burn"))
    month = epoch() + pd.to_timedelta(z["month_start"].astype(int), "D")
    status = z["month_status"]
    return pd.DataFrame(dict(month=month.strftime("%Y-%m"), status=status))[status != "ok"].reset_index(drop=True)
