from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deep_multimodal import evaluation, hazard_map, metrics

PREDICTIONS = Path(__file__).resolve().parents[1] / "results" / "saved_predictions"


@pytest.fixture(scope="module")
def folds():
    return {year: evaluation.load_fold_predictions(PREDICTIONS, year) for year in evaluation.FOLDS}


def test_only_folds_2024_and_2023_exist_and_no_later_year_is_stored():
    assert sorted(p.name for p in PREDICTIONS.glob("predictions_*.npz")) == ["predictions_2023.npz", "predictions_2024.npz"]
    assert evaluation.FOLDS == (2024, 2023)


def test_saved_predictions_reproduce_the_recorded_average_precision(folds):
    for (model, year), recorded in evaluation.RECORDED_AVERAGE_PRECISION.items():
        assert metrics.average_precision(folds[year].label, folds[year].raw[model]) == pytest.approx(recorded, abs=evaluation.RECORDED_TOLERANCE)


def test_both_models_are_scored_on_the_same_rows_with_the_documented_counts(folds):
    assert len(folds[2024].label) == 725441 and len(folds[2023].label) == 936380
    assert int(folds[2024].label.sum()) == 36841 and int(folds[2023].label.sum()) == 47168
    for fold in folds.values():
        assert len(fold.raw["deep"]) == len(fold.raw["baseline"]) == len(fold.label) and set(fold.state) == {"CG", "JH", "MH", "MP", "OD", "TS"}


def test_the_alert_threshold_of_a_fold_is_never_fitted_on_that_fold(folds):
    limits = evaluation.thresholds(folds)
    for year, other in ((2024, 2023), (2023, 2024)):
        expected = metrics.best_threshold(folds[other].label, folds[other].weight, folds[other].probability["deep"])
        assert limits[("deep", year)] == pytest.approx(expected)


def test_a_hazard_map_table_has_one_row_per_cell_of_the_day_with_coordinates(folds):
    cells = pd.read_csv(PREDICTIONS / "cells.csv")
    day = hazard_map.busiest_day(folds[2024])
    frame = hazard_map.day_table(folds[2024], cells, day)
    assert frame.cell.is_unique and frame.lon.between(60, 100).all() and frame.lat.between(5, 40).all() and frame.label.sum() > 0
    figure = hazard_map.draw_day(frame, "test")
    assert len(figure.axes) >= 1


def test_committed_results_table_matches_the_saved_predictions_for_the_headline_metric():
    table = pd.read_csv(Path(__file__).resolve().parents[1] / "results" / "results_table.csv")
    whole = table[(table.rows_scored == "All rows") & (table.model == "Deep multimodal model")].set_index("fold")
    assert whole.loc[2024, "Average precision (pooled)"] == pytest.approx(0.2913, abs=5e-5) and whole.loc[2023, "Average precision (pooled)"] == pytest.approx(0.2532, abs=5e-5)
    assert set(table.fold) == {2023, 2024}
