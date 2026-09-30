"""Tables and plots for the validation notebook."""
from __future__ import annotations

import json

import pandas as pd

from wildfire import config
from wildfire.pipelines import rolling_origin
from wildfire.reporting.style import COLOURS, new_figure


def fold_table() -> pd.DataFrame:
    """Fit years and validation rows of every rolling-origin fold."""
    years = config.load("baseline")["validation"]["validation_years"]
    first = config.load("evaluation")["splits"]["first_year"]
    rows = []
    for y in years:
        p, _ = rolling_origin.load_year(y)
        rows.append(dict(validation_year=y, fit_years=f"{first}-{y - 1}", rows=len(p), onsets=int(p.label.sum())))
    return pd.DataFrame(rows)


def plot_folds():
    t = fold_table()
    first = config.load("evaluation")["splits"]["first_year"]
    fig, ax = new_figure(7, 3)
    for i, r in t.iterrows():
        y = int(r.validation_year)
        ax.barh(i, y - first, left=first, color=COLOURS["primary"], height=0.6, label="fit years" if i == 0 else None)
        ax.barh(i, 1, left=y, color=COLOURS["secondary"], height=0.6, label="validation year" if i == 0 else None)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([str(y) for y in t.validation_year])
    ax.set_xlabel("Calendar year")
    ax.set_ylabel("Fold (validation year)")
    ax.set_title("Rolling-origin folds")
    ax.legend()
    return fig


def guards_table() -> pd.DataFrame:
    g = json.loads((config.output_dir("results") / "guards.json").read_text())
    return pd.DataFrame(dict(check=list(g), result=[str(v) for v in g.values()]))


def metrics_by_year() -> pd.DataFrame:
    """Within-day spatial AUC, within-cell temporal AUC and AP per arm and year (network seeds averaged)."""
    t = pd.read_csv(config.output_dir("results") / "baseline_metrics_by_year.csv")
    t["arm"] = t.arm.str.replace(r"#\d+$", "", regex=True)
    cols = ["auc_spatial_within_day", "auc_temporal_within_cell", "ap"]
    return t.groupby(["arm", "year"])[cols].mean().reset_index()


def plot_by_year(metric: str = "auc_spatial_within_day", arms_shown=("climatology_cell", "logistic_with_burn@tabular", "boosting@tabular", "boosting@enriched")):
    t = metrics_by_year()
    from wildfire.reporting.style import colour
    fig, ax = new_figure()
    for a in arms_shown:
        g = t[t.arm == a]
        ax.plot(g.year, g[metric], marker="o", color=colour(a), label=a)
    ax.set_xlabel("Validation year")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Stability over validation years")
    ax.set_xticks(sorted(t.year.unique()))
    ax.legend(fontsize=8)
    return fig
