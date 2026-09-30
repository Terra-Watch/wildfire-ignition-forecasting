import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wildfire import config
from wildfire.evaluation import harness
from wildfire.guards import inputs, provenance, sanity, shuffle, temporal


def test_shuffle_keeps_group_label_counts_and_actually_shuffles():
    rng = np.random.default_rng(0)
    n = 20000
    key = rng.integers(0, 500, n)
    y = (rng.random(n) < 0.1).astype(np.int8)
    ys = shuffle.shuffle_within(y, key, 1)
    assert int(ys.sum()) == int(y.sum())
    assert (pd.Series(y).groupby(key).sum().values == pd.Series(ys).groupby(key).sum().values).all()
    assert int((ys != y).sum()) > 0.05 * n


def test_shuffle_breaks_feature_label_link_inside_groups():
    rng = np.random.default_rng(1)
    n = 40000
    key = rng.integers(0, 200, n)
    x = rng.normal(size=n)
    y = (x + rng.normal(size=n) > 1.5).astype(np.int8)
    ys = shuffle.shuffle_within(y, key, 2)
    assert abs(np.corrcoef(x, y)[0, 1]) > 0.3 and abs(np.corrcoef(x, ys)[0, 1]) < 0.03


def test_temporal_guard():
    temporal.assert_before_cutoff(["2020-01-01T09:00"], ["2020-01-02T00:00"])
    with pytest.raises(AssertionError):
        temporal.assert_before_cutoff(["2020-01-02T00:00"], ["2020-01-02T00:00"])


def test_sanity_decisions():
    prev = 0.05
    ok = sanity.sanity_decisions([0.051, 0.049], prev, [0.06, 0.07], 0.11, [0.0505, 0.0495])
    assert ok["a"] and ok["b"] and ok["c"] and not ok["stop"]
    bad = sanity.sanity_decisions([0.08, 0.08], prev, [0.06, 0.12], 0.11, [0.0505])
    assert not bad["a"] and not bad["b"] and bad["stop"]


def test_stamp_carries_a_commit_hash():
    with tempfile.TemporaryDirectory() as t:
        provenance.stamp_df(pd.DataFrame(dict(a=[1, 2])), Path(t) / "x.csv")
        back = pd.read_csv(Path(t) / "x.csv")
        assert back.columns[0] == "code_commit" and len(back.code_commit.iloc[0].split("+")[0]) == 40
        provenance.stamp_text(Path(t) / "x.txt", "hello")
        assert (Path(t) / "x.txt").read_text().startswith("code_commit: ")


def test_held_out_geographic_split_is_refused():
    for split in ("geo_heldout", "test", "anything_else"):
        with pytest.raises(ValueError):
            harness.check_split(split)
    harness.check_split("holdout_year")


def test_scoring_entry_points_require_a_clean_tree():
    root = config.REPO_ROOT / "src" / "wildfire" / "pipelines"
    for name in ("holdout", "leave_year_out", "out_of_fold", "sanity"):
        assert "provenance.require_clean()" in (root / f"{name}.py").read_text()


class TestInputGuard:
    def make(self, d, n=4, static_rows=None, legacy=False, burn=True):
        import time
        (d / "processed").mkdir(exist_ok=True)
        pd.DataFrame(dict(cell=np.arange(n))).to_csv(d / "processed" / "grid.csv", index=False)
        time.sleep(0.02)
        pd.DataFrame(dict(cell=np.arange(static_rows if static_rows is not None else n), x=1.0)).to_csv(d / "processed" / "static_x.csv", index=False)
        if burn:
            kw = dict(burn_ord=np.zeros((3, n), np.int32), fetched=np.ones(3, bool), month_start=np.zeros(3, np.int32))
            if not legacy:
                kw["cell_month_covered"] = np.ones((3, n), bool)
            np.savez_compressed(d / "processed" / "burn_cells.npz", **kw)

    def test_ok_and_refusals(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATA_ROOT", str(tmp_path))
        self.make(tmp_path)
        inputs.assert_consistent_inputs()
        for kw in (dict(legacy=True), dict(static_rows=3), dict(burn=False)):
            d = tmp_path / ("case_" + "_".join(kw))
            d.mkdir()
            self.make(d, **kw)
            monkeypatch.setenv("DATA_ROOT", str(d))
            with pytest.raises(SystemExit):
                inputs.assert_consistent_inputs()
