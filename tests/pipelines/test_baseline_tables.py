import numpy as np
import pandas as pd

from wildfire.pipelines import baseline_tables as bt


def test_prediction_columns_and_contrast_pairs():
    frame = pd.DataFrame(columns=["p_climatology_cell", "p_boosting@tabular", "p_boosting@enriched", "p_logistic_with_burn@enriched",
                                  "p_tabular_network@enriched#0", "p_tabular_network@enriched#1", "score_climatology_cell"])
    arms = bt.prediction_columns(frame)
    assert "climatology_cell" in arms and len(arms) == 6
    pairs = bt.contrast_pairs(arms)
    names = [p[0] for p in pairs]
    assert names.count("network_vs_enriched_boosting") == 2 and "enriched_boosting_vs_tabular_boosting" in names
    assert all(p[2] == "climatology_cell" for p in pairs if p[0].endswith("_vs_climatology")) and len(pairs) == 2 + 2 + 5


def test_seed_summary_averages_over_seeds_per_level():
    t = pd.DataFrame(dict(arm=["tabular_network@enriched#0", "tabular_network@enriched#1", "tabular_network@tabular#0"], ap=[0.2, 0.4, 0.1]))
    s = bt.seed_summary(t)
    assert abs(s.loc["enriched", ("ap", "mean")] - 0.3) < 1e-12 and s.loc["tabular", ("ap", "mean")] == 0.1
