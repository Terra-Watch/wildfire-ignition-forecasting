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
    from wildfire.labels.detections import load_grid
    if load_grid()[1].get("reconstructed"):
        pytest.skip("the grid files were reconstructed after a deletion and carry no lattice indices")
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
    monkeypatch.setattr(provenance, "code_state", lambda: dict(commit="x", dirty=False, dirty_files=[]))
    (tmp_path / "results").mkdir()
    (tmp_path / "features").mkdir()
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
    with pytest.raises(SystemExit):
        g.require_record()                                             # no array fingerprints
    (tmp_path / "features" / "a.npy").write_bytes(b"one")
    st = (tmp_path / "features" / "a.npy").stat()
    arrays = {"a.npy": dict(size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=g._sha256(tmp_path / "features" / "a.npy"))}
    record = {"timestamp_before_day": True, "code_version": provenance.code_version(), "arrays": arrays}
    path.write_text(json.dumps(record))
    assert g.require_record()["timestamp_before_day"] is True
    (tmp_path / "features" / "a.npy").write_bytes(b"two")             # same size, rebuilt: hash differs
    with pytest.raises(SystemExit):
        g.require_record()                                             # array changed after the guards
    monkeypatch.setattr(provenance, "code_state", lambda: dict(commit="x", dirty=True, dirty_files=["src/a.py"]))
    path.write_text(json.dumps({**record, "arrays": {}}))
    with pytest.raises(SystemExit):
        g.require_record()                                             # dirty tree


def test_every_scoring_entry_point_requires_the_guard_record():
    from wildfire import config
    root = config.REPO_ROOT / "src" / "wildfire" / "pipelines"
    for name in ("tuning", "rolling_origin", "baseline_tables", "fresh_states"):
        assert "require_record()" in (root / f"{name}.py").read_text(), name


def test_power_gate_holds_back_a_verdict_when_the_detectable_effect_is_too_large(monkeypatch):
    from wildfire.evaluation import verdict
    assert g.power_gate()
    clear_gain = dict(point=0.05, se=0.0102, lo99=0.04, hi99=0.06)         # mde = (2.575829 + 0.841621) x 0.0102 = 0.0349
    assert verdict.baseline_verdict(clear_gain, 0.0)[0] == "INCONCLUSIVE"
    assert verdict.baseline_verdict(dict(clear_gain, se=0.002), 0.0)[0] == "PASS"
    monkeypatch.setattr(verdict, "minimum_detectable_effect", lambda se: 0.0)       # a gate that never holds anything back must be caught
    assert not g.power_gate()


def test_identity_gate_means_no_cell_day_is_both_fit_and_validation(monkeypatch):
    from tests.pipelines.test_dataset import tiny_dataset
    from wildfire.pipelines import dataset
    ds = tiny_dataset()
    assert g.identity_gate(ds)
    monkeypatch.setattr(dataset, "fit_rows", lambda d, year, cap=None: dataset.validation_rows(d, year))     # fit rows that are the validation rows
    assert not g.identity_gate(ds)
