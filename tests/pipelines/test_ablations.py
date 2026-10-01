import numpy as np

from wildfire import config
from wildfire.pipelines import ablations, dataset
from tests.pipelines.test_dataset import tiny_dataset


def test_ablations_are_defined_in_the_configuration_and_drop_only_known_inputs():
    spec = config.load("ablations")
    assert set(spec["ablations"]) == {"A_no_soil_anomaly", "B_windows_only", "B_neighbourhood_only"} and spec["reference"] == "boosting@enriched"
    assert spec["ablations"]["A_no_soil_anomaly"]["drop"] == ["sw1_anom", "sw2_anom"]
    assert set(spec["blocks"]) == {"windows", "anomalies", "neighbourhood"}


def test_block_matrix_with_every_block_equals_the_enriched_matrix_and_subsets_are_slices_of_it():
    ds = tiny_dataset()
    idx = np.arange(20)
    full = dataset.matrix(ds, idx, ["x1", "x2"], "enriched", 2022)
    assert np.array_equal(ablations.block_matrix(ds, idx, ["x1", "x2"], ["windows", "anomalies", "neighbourhood"], 2022), full)
    windows = ablations.block_matrix(ds, idx, ["x1", "x2"], ["windows", "anomalies"], 2022)
    neighbours = ablations.block_matrix(ds, idx, ["x1", "x2"], ["neighbourhood"], 2022)
    assert windows.shape[1] == 2 + 180 + 17 and neighbours.shape[1] == 2 + 12
    assert np.array_equal(windows, full[:, :199]) and np.array_equal(neighbours[:, 2:], full[:, -12:])
    assert ablations.block_width(["x1", "x2"], ["windows", "anomalies", "neighbourhood"]) == full.shape[1]


def test_the_soil_anomaly_ablation_drops_exactly_those_two_columns(monkeypatch):
    ds = tiny_dataset()
    monkeypatch.setattr(ablations.ladder_fit, "columns_for", lambda d, arm: ["a", "sw1_anom", "b", "sw2_anom", "c"])
    spec = config.load("ablations")["ablations"]["A_no_soil_anomaly"]
    assert ablations.ablation_columns(ds, spec) == ["a", "b", "c"]
