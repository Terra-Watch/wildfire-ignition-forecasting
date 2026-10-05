"""The committed result tables of the extended boosting baseline match the numbers quoted in the documents."""
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results" / "extended_boosting_baseline"
EXTENDED, PLAIN = "Extended boosting baseline", "Plain boosting"


def average_precision(fold: str, model: str) -> float:
    table = pd.read_csv(RESULTS / "all_arms_by_fold.csv", dtype={"fold": str})
    row = table[(table.fold == fold) & (table.model == model)]
    assert len(row) == 1
    return float(row.average_precision.iloc[0])


def test_extended_and_plain_boosting_average_precision_on_the_same_folds():
    assert round(average_precision("2025", EXTENDED), 4) == 0.2315
    assert round(average_precision("pooled over the five folds", EXTENDED), 4) == 0.1977
    assert round(average_precision("2025", PLAIN), 4) == 0.1863
    assert round(average_precision("pooled over the five folds", PLAIN), 4) == 0.1626
    for fold in ("2021", "2022", "2023", "2024", "2025"):
        assert average_precision(fold, EXTENDED) > average_precision(fold, PLAIN)


def test_all_five_folds_and_the_pooled_row_are_present_for_every_model():
    table = pd.read_csv(RESULTS / "all_arms_by_fold.csv", dtype={"fold": str})
    assert set(table.fold) == {"2021", "2022", "2023", "2024", "2025", "pooled over the five folds"}
    assert (table.groupby("model").size() == 6).all()


def test_metric_table_for_2024_and_2023_matches_the_recorded_values():
    table = pd.read_csv(RESULTS / "metric_table_2024_2023.csv")
    whole = table[table.stratum == "all rows"].set_index("fold")
    assert abs(whole.loc[2024, "average_precision"] - 0.2267) < 5e-5 and abs(whole.loc[2023, "average_precision"] - 0.1797) < 1e-4
    assert set(whole.index) == {2023, 2024}
    assert not (table.fold == 2025).any()
