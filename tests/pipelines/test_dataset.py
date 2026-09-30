import numpy as np
import pandas as pd

from wildfire.features import weather_windows
from wildfire.features.tabular import weather
from wildfire.pipelines import dataset, ladder_fit

FIRST = 18000


def tiny_dataset(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    day = FIRST + 400 + rng.integers(0, 1500, n)
    dates = pd.to_datetime(day, unit="D")
    table = pd.DataFrame(dict(cell=rng.integers(0, 3, n), day=day, year=dates.year, week=dates.isocalendar().week.astype(int).values,
                              state="OD", label=(rng.random(n) < 0.1).astype(np.int8), climatology_cell=rng.random(n)))
    names = dataset.feature_names({"a": ["x1", "x2"], "climatology_cell": ["climatology_cell"]})
    for c in names:
        table[c] = rng.normal(size=n).astype(np.float32)
    series = dict(values=rng.normal(size=(2200, len(weather.NAMES), 3)).astype(np.float32), rain=np.zeros((2200, 3), np.float32),
                  cell_point=np.array([0, 1, 2]), first_day=FIRST)
    arrays = dict(windows=rng.normal(size=(n, len(weather_windows.window_names()))).astype(np.float32),
                  neighbourhood=rng.normal(size=(n, 12)).astype(np.float32), series=series)
    return dataset.Dataset(table, {"a": ["x1", "x2"], "climatology_cell": ["climatology_cell"]}, arrays, None,
                           np.ones(n, bool))


def test_fit_rows_come_only_from_earlier_years_and_are_capped_with_every_onset_kept():
    ds = tiny_dataset()
    idx = dataset.fit_rows(ds, 2021, cap=1500)
    assert (ds.table.year.values[idx] < 2021).all() and len(idx) <= 1500
    earlier = ds.table.year.values < 2021
    assert ds.table.label.values[idx].sum() == ds.table.label.values[earlier].sum()


def test_validation_rows_are_one_year_in_the_validation_months_with_a_climatology_value():
    ds = tiny_dataset()
    ds.table.loc[:50, "climatology_cell"] = np.nan
    idx = dataset.validation_rows(ds, 2022)
    months = pd.to_datetime(ds.table.day.values[idx], unit="D").month
    assert (ds.table.year.values[idx] == 2022).all() and (months <= 7).all()
    assert ds.table.climatology_cell.notna().values[idx].all()


def test_incomplete_rows_are_excluded_from_both_sets():
    ds = tiny_dataset()
    ds.complete[:3000] = False
    assert (dataset.fit_rows(ds, 2023) >= 3000).all() and (dataset.validation_rows(ds, 2022) >= 3000).all()


def test_matrix_shapes_at_both_levels_and_anomalies_depend_only_on_fit_years():
    ds = tiny_dataset()
    idx = np.arange(10)
    tab = dataset.matrix(ds, idx, ["x1", "x2"], "tabular", 2022)
    full = dataset.matrix(ds, idx, ["x1", "x2"], "enriched", 2022)
    assert tab.shape == (10, 2) and full.shape == (10, 2 + 180 + 17 + 12)
    assert np.array_equal(full[:, :2], tab)
    c1 = dataset.anomaly_climatology(ds, 2022)
    assert c1 is dataset.anomaly_climatology(ds, 2022)                # cached per fold


def test_arm_levels_and_names():
    levels = ladder_fit.arm_levels()
    assert ("boosting", "tabular") in levels and ("boosting", "enriched") in levels and ("landcover_season", "tabular") in levels
    assert ("landcover_season", "enriched") not in levels
    assert ladder_fit.arm_name("boosting", "enriched") == "boosting@enriched" and ladder_fit.arm_name("fwi_season", "tabular") == "fwi_season"


def test_a_standard_fit_returns_probabilities_and_the_fit_prevalence():
    ds = tiny_dataset()
    fit, val = dataset.fit_rows(ds, 2023), dataset.validation_rows(ds, 2023)
    p, prev = ladder_fit.fit_standard(ds, fit, ds, val, ["x1", "x2"], "enriched", 2023, "logistic", {"C": 0.1})
    assert len(p) == len(val) and ((p > 0) & (p < 1)).all() and abs(prev - ds.table.label.values[fit].mean()) < 1e-12


def test_climatology_arm_is_calibrated_on_earlier_years_only():
    ds = tiny_dataset()
    val = dataset.validation_rows(ds, 2022)
    score, prob, prevalence = ladder_fit.climatology_arm(ds, 2022, ds, val)
    assert np.array_equal(score, ds.table.climatology_cell.values[val]) and ((prob > 0) & (prob < 1)).all()
    earlier = ds.table.year.values < 2022
    assert abs(prevalence - ds.table.label.values[earlier].mean()) < 1e-12


def test_diagnostic_arms_use_only_place_and_static_inputs(monkeypatch):
    ds = tiny_dataset()
    ds.cells = pd.DataFrame(dict(cx=[0.0, 1.0, 2.0], cy=[5.0, 6.0, 7.0]))
    ds.table["wc_10"] = ds.table.x1
    fit, val = dataset.fit_rows(ds, 2023), dataset.validation_rows(ds, 2023)
    monkeypatch.setattr(dataset.arms, "feature_groups", lambda table=None: {"static": ["wc_10"]})
    out = ladder_fit.diagnostic_arms(ds, fit, ds, val, {"max_depth": 3, "learning_rate": 0.1, "max_iter": 20})
    assert set(out) == {"coords_only", "cell_id_only", "static_only"}
    assert all(len(p) == len(val) for p, _ in out.values())
