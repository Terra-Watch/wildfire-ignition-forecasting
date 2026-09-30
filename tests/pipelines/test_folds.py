from wildfire import config
from wildfire.pipelines.fit import forward_folds


def test_forward_chaining_never_trains_on_the_validation_year_or_later():
    for year, years in forward_folds():
        assert all(y < year for y in years) and min(years) == 2019
    assert [y for y, _ in forward_folds()] == [2021, 2022, 2023, 2024]


def test_holdout_year_is_never_used_to_fit():
    holdout = config.load("evaluation")["splits"]["holdout_year"]
    used = [y for _, years in forward_folds() for y in years] + [y for y, _ in forward_folds()]
    assert holdout not in used
    assert config.load("evaluation")["splits"]["fit_last_year"] < holdout


def test_protocol_constants():
    cfg = config.load("features")["tabular"]
    assert cfg["window_end"] == "2025-07-31" and cfg["burn"]["lag_days"] == 30 and cfg["burn"]["years_since_cap"] == 10.0
    assert cfg["burn"]["min_observed_share"] == 0.95 and cfg["gap_days"]["lookback_days"] >= 7
    assert cfg["human_frozen_epoch"] == {"pop": 2020, "ghm": 2020, "roads": 2021, "night_lights": 2018}
    assert config.load("domain")["negatives"]["per_onset"] == 20
