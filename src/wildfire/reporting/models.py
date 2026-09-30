"""Tables and plots for the models notebook."""
from __future__ import annotations

import json

import pandas as pd

from wildfire import config
from wildfire.models import arms
from wildfire.pipelines import ladder_fit
from wildfire.reporting.style import colour, new_figure


def ladder_table() -> pd.DataFrame:
    """Every arm: model type, information levels and number of tabular inputs."""
    cfg = config.load("models")["arms"]
    cols = arms.arm_columns()
    cols["tabular_network"] = cols["logistic_with_burn"]
    two = config.load("baseline")["arms"]["two_levels"]
    rows = []
    for arm in config.load("baseline")["arms"]["ladder"]:
        kind = cfg.get(arm, {}).get("kind", "network")
        rows.append(dict(arm=arm, model=kind, levels="tabular, enriched" if arm in two else "tabular",
                         tabular_inputs=len(cols[arm])))
    return pd.DataFrame(rows)


def diagnostics_table() -> pd.DataFrame:
    return pd.DataFrame(dict(arm=config.load("baseline")["arms"]["diagnostics"],
                             inputs=["cell centroid (x, y)", "onset share of the cell in the fit years", "terrain and land cover only"]))


def tuning_table() -> pd.DataFrame:
    """Mean validation log loss and AP of every trial, best first."""
    t = pd.read_csv(config.output_dir("results") / "tuning_trials.csv")
    g = t.groupby(["kind", "trial", "params"])[["log_loss", "ap"]].mean().reset_index()
    return g.sort_values(["kind", "log_loss"]).reset_index(drop=True)


def chosen_settings() -> pd.DataFrame:
    tuned = json.loads((config.output_dir("results") / "tuned.json").read_text())
    return pd.DataFrame([dict(model=k, **v) for k, v in tuned.items()])


def plot_tuning():
    t = tuning_table()
    fig, ax = new_figure()
    for kind, c in (("boosting", colour("boosting")), ("network", colour("tabular_network"))):
        g = t[t.kind == kind]
        ax.scatter(g.log_loss, g.ap, color=c, label=kind)
    ax.set_xlabel("Mean validation log loss (lower is better)")
    ax.set_ylabel("Mean validation average precision")
    ax.set_title("Tuning trials (12 per model)")
    ax.legend()
    return fig


def arm_levels_table() -> pd.DataFrame:
    return pd.DataFrame(ladder_fit.arm_levels(), columns=["arm", "level"])
