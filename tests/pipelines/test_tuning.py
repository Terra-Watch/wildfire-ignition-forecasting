import json

import pandas as pd

from wildfire.pipelines import tuning


def test_trials_are_distinct_reproducible_and_inside_the_space():
    space = dict(a=[1, 2, 3], b=[0.1, 0.2])
    t1, t2 = tuning.sample_trials(space, 4, 0), tuning.sample_trials(space, 4, 0)
    assert t1 == t2 and len({json.dumps(t, sort_keys=True) for t in t1}) == 4
    assert all(t["a"] in space["a"] and t["b"] in space["b"] for t in t1)


def test_choice_uses_mean_log_loss_then_average_precision():
    rec = pd.DataFrame([
        dict(kind="boosting", trial=0, params='{"x": 0}', year=2021, log_loss=0.30, ap=0.10),
        dict(kind="boosting", trial=0, params='{"x": 0}', year=2022, log_loss=0.30, ap=0.10),
        dict(kind="boosting", trial=1, params='{"x": 1}', year=2021, log_loss=0.29, ap=0.05),
        dict(kind="boosting", trial=1, params='{"x": 1}', year=2022, log_loss=0.31, ap=0.05),
        dict(kind="boosting", trial=2, params='{"x": 2}', year=2021, log_loss=0.30, ap=0.20),
        dict(kind="boosting", trial=2, params='{"x": 2}', year=2022, log_loss=0.30, ap=0.20)])
    assert tuning.choose(rec, "boosting") == {"x": 2}      # trials 0 and 2 tie on log loss; higher AP wins


def test_search_budget_and_years_come_from_the_spec():
    from wildfire import config
    cfg = config.load("baseline")
    assert cfg["tuning"]["trials"] == 12 and cfg["validation"]["tuning_years"] == [2021, 2022, 2023, 2024]
    assert len(tuning.sample_trials(cfg["tuning"]["boosting_space"], 12, 0)) == 12
    assert len(tuning.sample_trials(cfg["tuning"]["network_space"], 12, 0)) == 12
