"""Tables and plots for the transfer and limits notebook."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.evaluation import ladder
from wildfire.pipelines import baseline_tables
from wildfire.reporting.style import colour, new_figure


def fresh_metrics() -> pd.DataFrame:
    """Headline metrics of every arm on the fresh states, per state (network seeds averaged)."""
    folder = config.output_dir("results")
    rows = pd.read_parquet(folder / "fresh_predictions.parquet")
    prev = pd.read_json(folder / "fresh_fit_prevalence.json", typ="series").to_dict()
    out = []
    for state, g in rows.groupby("state"):
        g = g.reset_index(drop=True)
        for arm in baseline_tables.prediction_columns(g):
            score = g["score_" + arm].values if "score_" + arm in g else g["p_" + arm].values
            corrected = ladder.corrected_probabilities(g["p_" + arm].values, g.pi_true.values, prev[arm])
            out.append(dict(state=state, arm=arm, **ladder.arm_row(g, score, corrected, g.weight.values)))
    t = pd.DataFrame(out)
    t["arm"] = t.arm.str.replace(r"#\d+$", "", regex=True)
    return t.groupby(["state", "arm"], sort=False).mean(numeric_only=True).reset_index()


def transfer_table() -> pd.DataFrame:
    """Within-day spatial AUC and AP in the fit states (validation) and in the fresh states, side by side."""
    dev = pd.read_csv(config.output_dir("results") / "baseline_metrics.csv")
    dev["arm"] = dev.arm.str.replace(r"#\d+$", "", regex=True)
    dev = dev.groupby("arm").mean(numeric_only=True)[["auc_spatial_within_day", "auc_temporal_within_cell", "ap"]].add_prefix("fit_states_")
    fresh = fresh_metrics().pivot(index="arm", columns="state", values=["auc_spatial_within_day", "ap"])
    fresh.columns = [f"{m}_{s}" for m, s in fresh.columns]
    return dev.join(fresh).round(4).reset_index()


def plot_transfer(metric: str = "auc_spatial_within_day", arms_shown=("climatology_cell", "logistic_with_burn@enriched", "boosting@tabular", "boosting@enriched")):
    t = fresh_metrics()
    dev = pd.read_csv(config.output_dir("results") / "baseline_metrics.csv")
    dev["arm"] = dev.arm.str.replace(r"#\d+$", "", regex=True)
    dev = dev.groupby("arm").mean(numeric_only=True)
    states = sorted(t.state.unique())
    fig, ax = new_figure(7, 3.8)
    width = 0.8 / len(arms_shown)
    for i, a in enumerate(arms_shown):
        vals = [dev.loc[a, metric]] + [t[(t.state == s) & (t.arm == a)][metric].mean() for s in states]
        ax.bar(np.arange(len(vals)) + i * width, vals, width, color=colour(a), label=a)
    ax.set_xticks(np.arange(len(states) + 1) + 0.4 - width / 2)
    ax.set_xticklabels(["fit states (validation)"] + [f"{s} (fresh)" for s in states])
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Skill inside the fit states and in fresh states")
    ax.legend(fontsize=7)
    return fig


def sealed_status() -> pd.DataFrame:
    return pd.DataFrame([
        ("burned-area product", "no items after 2025-07 (searched 2026-10-01)"),
        ("sealed season 2026 Jan-Jul", "unavailable; not replaced by 2025"),
        ("arms needing burn inputs", "all arms that include them (logistic_with_burn, boosting, network and their enriched versions)"),
        ("other 2026 sources", "not pulled, because the test cannot be completed without burn")],
        columns=["item", "status"])
