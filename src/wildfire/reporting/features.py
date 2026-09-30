"""Tables and plots for the features notebook."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.features import neighbourhood, weather_windows
from wildfire.features.tabular import weather
from wildfire.models import arms
from wildfire.pipelines import dataset
from wildfire.reporting.style import COLOURS, new_figure


def group_table() -> pd.DataFrame:
    """Feature groups and how many columns each has."""
    g = arms.feature_groups()
    rows = [(k, len(v), ", ".join(v[:4]) + (" ..." if len(v) > 4 else "")) for k, v in g.items()]
    rows += [("weather windows", len(weather_windows.window_names()), "mean and max over 3-60 days, changes, rain sums"),
             ("weather anomalies", len(weather.NAMES), "value minus day-of-year climate of the fit years"),
             ("neighbourhood", len(neighbourhood.column_names()), ", ".join(neighbourhood.column_names()[:3]) + " ...")]
    return pd.DataFrame(rows, columns=["group", "columns", "examples"])


def missing_table() -> pd.DataFrame:
    """Share of missing values in each feature block, on a random sample of modelling rows."""
    ds = dataset.load_dataset()
    sample = config.load("reporting")["missing_table_sample"]
    idx = np.sort(np.random.default_rng(0).choice(len(ds.table), sample, replace=False))
    tab = ds.table[dataset.feature_names(ds.tabular_columns)].iloc[idx]
    return pd.DataFrame(dict(block=["tabular", "weather windows", "neighbourhood"],
                             missing_share=[float(tab.isna().values.mean()), float(np.isnan(ds.arrays["windows"][idx]).mean()),
                                            float(np.isnan(ds.arrays["neighbourhood"][idx]).mean())]))


def plot_window_example():
    """The fire-weather index in the 70 days before one onset, with the 7- and 30-day means the windows use."""
    ds = dataset.load_dataset()
    row = ds.table[ds.table.label == 1].iloc[config.load("reporting")["feature_example_row"]]
    s = ds.arrays["series"]
    p = int(s["cell_point"][int(row.cell)])
    k = weather.NAMES.index("fwi")
    t_end = int(row.day) - 1 - s["first_day"]
    x = np.arange(-69, 1)
    y = np.asarray(s["values"][t_end - 69:t_end + 1, k, p])
    fig, ax = new_figure()
    ax.plot(x, y, color=COLOURS["primary"], label="daily value")
    for w, c in ((7, COLOURS["secondary"]), (30, COLOURS["good"])):
        ax.hlines(y[-w:].mean(), -w + 1, 0, color=c, lw=3, label=f"{w}-day mean")
    ax.set_xlabel("Days before the target day (0 = the day before it)")
    ax.set_ylabel("Fire-weather index")
    ax.set_title("What a window feature summarises")
    ax.legend()
    return fig


def plot_feature_by_label(column: str = "fwi"):
    """Distribution of one input for onsets and for negatives."""
    ds = dataset.load_dataset()
    t = ds.table
    hist = config.load("reporting")["histogram"]
    fig, ax = new_figure()
    for lab, name, c in ((0, "negatives", COLOURS["neutral"]), (1, "onsets", COLOURS["secondary"])):
        ax.hist(t[column][t.label == lab].dropna().sample(hist["sample"], random_state=0), bins=hist["bins"], density=True, alpha=0.6, color=c, label=name)
    ax.set_xlabel(column)
    ax.set_ylabel("Density")
    ax.set_title(f"{column} for onsets and negatives")
    ax.legend()
    return fig
