"""Tables and plots for the labels notebook."""
from __future__ import annotations

import json

import pandas as pd

from wildfire import config
from wildfire.pipelines.features import load_model_table
from wildfire.reporting.style import COLOURS, new_figure


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
