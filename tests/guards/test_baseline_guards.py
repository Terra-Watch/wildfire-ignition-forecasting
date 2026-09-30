import os

import pandas as pd
import pytest

from wildfire.guards import baseline_guards as g


def test_window_features_do_not_see_the_target_day():
    assert g.timestamp_before_day()


def test_a_deliberately_shifted_feature_is_detected():
    assert g.shifted_feature_fails()


@pytest.mark.skipif(not os.environ.get("DATA_ROOT"), reason="needs DATA_ROOT")
def test_grid_coordinates_agree():
    assert g.coordinates_consistent()


def test_tuning_never_touches_the_last_development_year():
    assert g.no_validation_information_in_tuning()


def test_geo_ledger_allows_one_scoring_only(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path))
    g.record_geo_access("unit_test_state")
    with pytest.raises(SystemExit):
        g.record_geo_access("unit_test_state")
    g.record_geo_access("another_state")
    assert len(pd.read_csv(tmp_path / "results" / g.LEDGER_FILE)) == 2


def test_no_score_without_a_passing_guard_record_for_this_code(tmp_path, monkeypatch):
    import json
    from wildfire.guards import provenance
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path))
    (tmp_path / "results").mkdir()
    path = tmp_path / "results" / g.GUARDS_FILE
    with pytest.raises(SystemExit):
        g.require_record()                                             # no record
    path.write_text(json.dumps({"timestamp_before_day": True, "label_shuffle": False, "code_version": provenance.code_version()}))
    with pytest.raises(SystemExit):
        g.require_record()                                             # a guard failed
    path.write_text(json.dumps({"timestamp_before_day": True, "code_version": "0" * 40}))
    with pytest.raises(SystemExit):
        g.require_record()                                             # record from other code
    path.write_text(json.dumps({"timestamp_before_day": True, "code_version": provenance.code_version()}))
    assert g.require_record()["timestamp_before_day"] is True


def test_every_scoring_entry_point_requires_the_guard_record():
    from wildfire import config
    root = config.REPO_ROOT / "src" / "wildfire" / "pipelines"
    for name in ("tuning", "rolling_origin", "baseline_tables", "fresh_states"):
        assert "require_record()" in (root / f"{name}.py").read_text(), name
