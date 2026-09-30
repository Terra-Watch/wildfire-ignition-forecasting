"""Tables of the baseline: per-arm metrics, diagnostics beside them, reliability and the pre-declared contrasts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.evaluation import ladder
from wildfire.guards import provenance
from wildfire.pipelines import rolling_origin

NETWORK = "tabular_network"


def prediction_columns(frame: pd.DataFrame) -> list[str]:
    return [c[2:] for c in frame.columns if c.startswith("p_")]


def load_pooled() -> tuple[pd.DataFrame, dict, dict]:
    """All validation years stacked: rows, corrected probabilities per arm, and raw scores per arm."""
    years = config.load("baseline")["validation"]["validation_years"]
    frames, prevs = zip(*[rolling_origin.load_year(y) for y in years])
    rows = pd.concat(frames, ignore_index=True)
    corrected, scores = {}, {}
    for arm in prediction_columns(rows):
        parts = []
        for f, prev in zip(frames, prevs):
            parts.append(ladder.corrected_probabilities(f["p_" + arm].values, f.pi_true.values, prev[arm]))
        corrected[arm] = np.concatenate(parts)
        scores[arm] = rows["score_" + arm].values if "score_" + arm in rows else rows["p_" + arm].values
    return rows, corrected, scores


def metrics_table(rows: pd.DataFrame, corrected: dict, scores: dict) -> pd.DataFrame:
    """One row per arm (every network seed is its own row)."""
    w = rows.weight.values
    return pd.DataFrame([dict(arm=a, **ladder.arm_row(rows, scores[a], corrected[a], w)) for a in corrected])


def seed_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over the network seeds, per information level."""
    net = table[table.arm.str.startswith(NETWORK)].copy()
    net["level"] = net.arm.str.extract(r"@(\w+)#")[0]
    cols = [c for c in net.columns if c not in ("arm", "level")]
    return net.groupby("level")[cols].agg(["mean", "std"])


def contrast_pairs(arms_present: list[str]) -> list[tuple[str, str, str]]:
    """(contrast name, arm, reference) for every pre-declared contrast that can be computed."""
    pairs = [("enriched_boosting_vs_tabular_boosting", "boosting@enriched", "boosting@tabular"),
             ("enriched_boosting_vs_enriched_logistic", "boosting@enriched", "logistic_with_burn@enriched")]
    pairs += [("network_vs_enriched_boosting", a, "boosting@enriched") for a in arms_present if a.startswith(NETWORK + "@enriched")]
    ref = config.load("models")["arms"]
    clim = next(a for a, s in ref.items() if s["kind"] == "climatology")
    pairs += [(f"{a}_vs_climatology", a, clim) for a in arms_present if a != clim]
    return pairs


def contrast_table(rows: pd.DataFrame, corrected: dict, scores: dict) -> pd.DataFrame:
    """Every contrast with its bootstrap interval, minimum detectable effect and verdict."""
    clusters = ladder.cluster_labels(rows)
    out = [dict(contrast=name, **ladder.contrast(rows, scores, corrected, rows.weight.values, arm, ref, clusters))
           for name, arm, ref in contrast_pairs(list(corrected))]
    return pd.DataFrame(out)


def run(log=print) -> None:
    """Write every baseline table to the results folder."""
    provenance.require_clean()
    out = config.output_dir("results")
    rows, corrected, scores = load_pooled()
    table = metrics_table(rows, corrected, scores)
    provenance.stamp_df(table, out / "baseline_metrics.csv")
    seed_summary(table).to_csv(out / "baseline_network_seeds.csv")
    bins = config.load("baseline")["reporting"]["reliability_bins"]
    rel = [ladder.reliability(rows.label.values, corrected[a], rows.weight.values, bins).assign(arm=a) for a in corrected if "#" not in a]
    provenance.stamp_df(pd.concat(rel), out / "baseline_reliability.csv")
    provenance.stamp_df(contrast_table(rows, corrected, scores), out / "baseline_contrasts.csv")
    log("tables written", out)
