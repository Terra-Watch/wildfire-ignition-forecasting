"""Tables and plots for the results notebook."""
from __future__ import annotations

import pandas as pd

from wildfire import config
from wildfire.reporting.style import colour, new_figure

HEADLINE = ["arm", "auc_spatial_within_day", "auc_temporal_within_cell", "ap", "log_loss", "brier",
            "capture_1pct", "lift_1pct", "capture_5pct", "lift_5pct"]
DIAGNOSTICS = ("coords_only", "cell_id_only", "static_only")


def _results(name: str) -> pd.DataFrame:
    return pd.read_csv(config.output_dir("results") / name)


def metrics() -> pd.DataFrame:
    """Pooled validation metrics; network seeds collapsed to mean over seeds."""
    t = _results("baseline_metrics.csv")
    t["arm"] = t.arm.str.replace(r"#\d+$", "", regex=True)
    return t.groupby("arm", sort=False).mean(numeric_only=True).reset_index()


def headline_table() -> pd.DataFrame:
    m = metrics()
    return m[[c for c in HEADLINE if c in m.columns]].round(4)


def identity_table() -> pd.DataFrame:
    """The diagnostics arms beside the best arm: how much of the skill needs only the cell, not the day."""
    m = metrics().set_index("arm")
    shown = [a for a in DIAGNOSTICS if a in m.index] + ["boosting@enriched", "climatology_cell"]
    return m.loc[shown, ["auc_spatial_within_day", "auc_temporal_within_cell", "ap"]].round(4).reset_index()


def plot_two_views():
    m = metrics()
    fig, ax = new_figure(6.5, 5)
    for _, r in m.iterrows():
        ax.scatter(r.auc_temporal_within_cell, r.auc_spatial_within_day, color=colour(r.arm), s=40)
        ax.annotate(r.arm, (r.auc_temporal_within_cell, r.auc_spatial_within_day), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Within-cell temporal AUC (which days are riskier)")
    ax.set_ylabel("Within-day spatial AUC (which cells are riskier)")
    ax.set_title("The two ranking views of every arm")
    return fig


def plot_reliability(arms_shown=("climatology_cell", "logistic_with_burn@enriched", "boosting@enriched")):
    t = _results("baseline_reliability.csv")
    fig, ax = new_figure(5.5, 5)
    top = t[["mean_predicted", "observed_rate"]].max().max()
    ax.plot([0, top], [0, top], color="k", lw=0.8, ls="--", label="perfect")
    for a in arms_shown:
        g = t[t.arm == a]
        ax.plot(g.mean_predicted, g.observed_rate, marker="o", color=colour(a), label=a)
    ax.set_xlabel("Mean predicted probability (corrected to the true rate)")
    ax.set_ylabel("Observed onset rate (weighted)")
    ax.set_title("Reliability by decile")
    ax.legend(fontsize=8)
    return fig


def contrast_table() -> pd.DataFrame:
    """The pre-declared contrasts (a)-(c); network seeds are summarised by their median and the verdict count."""
    c = _results("baseline_contrasts.csv")
    c = c[~c.contrast.str.endswith("_vs_climatology")].copy()
    c["seed"] = c.arm.str.extract(r"#(\d+)$")[0]
    keep = ["contrast", "arm", "reference", "delta_ap", "ap_lo99", "ap_hi99", "mde", "worst_state", "worst_state_delta_ap", "verdict"]
    return c[c.seed.isna()][keep].round(4).reset_index(drop=True)


def network_seed_contrasts() -> pd.DataFrame:
    c = _results("baseline_contrasts.csv")
    n = c[c.contrast.str.startswith("network_vs_enriched_boosting")]
    return n.groupby("arm")[["delta_ap", "mde"]].median().round(4).join(n.groupby("arm").verdict.agg(lambda v: ", ".join(sorted(set(v)))))


def versus_climatology() -> pd.DataFrame:
    c = _results("baseline_contrasts.csv")
    v = c[c.contrast.str.endswith("_vs_climatology") & ~c.arm.str.contains("#")]
    return v[["arm", "delta_ap", "ap_lo99", "ap_hi99", "mde", "verdict"]].round(4).reset_index(drop=True)
