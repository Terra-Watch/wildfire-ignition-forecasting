import numpy as np
import pytest

from wildfire.pipelines import dataset, ladder_fit
from wildfire.pipelines.ladder_fit import FoldStore
from tests.pipelines.test_dataset import tiny_dataset

PARAMS = {"max_depth": 3, "learning_rate": 0.1, "max_iter": 5}


class Stopped(Exception):
    pass


def fake_models(monkeypatch, fail_after=None):
    """Replace every fit by a cheap deterministic one; return the list of fits made. Optionally stop after n standard fits."""
    fitted = []

    def fit_standard(ds, fit_idx, dv, val_idx, cols, level, year, kind, params):
        if fail_after is not None and len(fitted) == fail_after:
            raise Stopped
        fitted.append((kind, level))
        return np.full(len(val_idx), 0.1 + 0.01 * len(fitted)), 0.05 + 0.001 * len(fitted)

    monkeypatch.setattr(ladder_fit, "fit_standard", fit_standard)
    monkeypatch.setattr(ladder_fit, "columns_for", lambda ds, arm: ["x1"])
    monkeypatch.setattr(ladder_fit, "diagnostic_arms", lambda ds, fit, dv, val, params: {n: (np.full(len(val), 0.2), 0.1) for n in ladder_fit.DIAGNOSTIC_ARMS})
    monkeypatch.setattr(ladder_fit, "fit_network", lambda ds, fit, dv, val, cols, level, year, params, log, store=None, name="":
                        ({"#0": np.full(len(val), 0.3)}, {"#0": 0.04}))
    return fitted


def run_predict(ds, store=None):
    fit, val = dataset.fit_rows(ds, 2023), dataset.validation_rows(ds, 2023)
    return ladder_fit.predict_all(ds, fit, ds, val, 2023, {"boosting": PARAMS, "network": {}}, log=lambda *a: None, store=store)


def test_a_stopped_fold_resumes_without_refitting_finished_models_and_gives_the_same_predictions(monkeypatch, tmp_path):
    ds = tiny_dataset()
    fitted = fake_models(monkeypatch)
    expected, expected_prev = run_predict(ds)
    n_models = len(fitted)
    assert n_models > 4

    store = FoldStore(tmp_path / "fold", "abc")
    stopped = fake_models(monkeypatch, fail_after=3)
    with pytest.raises(Stopped):
        run_predict(ds, store)
    assert len(stopped) == 3 and len(list((tmp_path / "fold").glob("*.npz"))) == 3

    again = fake_models(monkeypatch)
    resumed, resumed_prev = run_predict(ds, store)
    assert len(again) == n_models - 3
    assert list(resumed.columns) == list(expected.columns) and resumed_prev.keys() == expected_prev.keys()
    kept = [c for c in expected.columns if c.startswith("p_")][:3]
    assert all(np.array_equal(resumed[c], expected[c]) for c in kept)


def test_a_finished_fold_is_read_back_without_any_fit(monkeypatch, tmp_path):
    ds = tiny_dataset()
    fake_models(monkeypatch)
    store = FoldStore(tmp_path / "fold", "abc")
    first, prev = run_predict(ds, store)
    fitted = fake_models(monkeypatch)
    monkeypatch.setattr(ladder_fit, "diagnostic_arms", lambda *a: pytest.fail("diagnostics refitted"))
    again, prev_again = run_predict(ds, store)
    assert fitted == [] and again.equals(first) and prev_again == prev


def test_network_seeds_resume_one_seed_at_a_time(monkeypatch, tmp_path):
    ds = tiny_dataset()
    fit, val = dataset.fit_rows(ds, 2023), dataset.validation_rows(ds, 2023)
    fitted = []

    class Stub:
        def __init__(self, params, seed):
            self.seed = seed

        def fit(self, *a):
            fitted.append(self.seed)
            self.fit_prevalence, self.best_validation_loss = 0.01 * self.seed, 0.5
            return self

        def predict(self, x):
            return np.full(len(x), 0.1 * (self.seed + 1))

    monkeypatch.setattr(ladder_fit, "Network", Stub)
    monkeypatch.setattr(ladder_fit, "validation_file", lambda *a: np.zeros((len(val), 2), np.float32))
    monkeypatch.setattr(ladder_fit, "training_weights", lambda t, idx: np.ones(len(idx)))
    monkeypatch.setattr(dataset, "matrix", lambda ds, idx, cols, level, year: np.random.default_rng(0).normal(size=(len(idx), 2)).astype(np.float32))
    real = ladder_fit.config.load
    monkeypatch.setattr(ladder_fit.config, "load", lambda name, **k: {**real(name, **k), "network": {"seeds": 4}} if name == "models" else real(name, **k))
    store = FoldStore(tmp_path / "fold", "abc")
    for seed in (0, 1):
        store.put(f"network@enriched#{seed}", np.full(len(val), 0.1 * (seed + 1)), 0.01 * seed)
    preds, prev = ladder_fit.fit_network(ds, fit, ds, val, ["x1"], "enriched", 2023, {}, lambda *a: None, store, "network@enriched")
    assert fitted == [2, 3] and list(preds) == ["#0", "#1", "#2", "#3"]
    assert all(np.allclose(preds[f"#{s}"], 0.1 * (s + 1)) and abs(prev[f"#{s}"] - 0.01 * s) < 1e-12 for s in range(4))
    assert len(list((tmp_path / "fold").glob("*.npz"))) == 4


def test_a_model_written_by_other_code_is_refused_and_a_partial_file_is_never_read(tmp_path):
    FoldStore(tmp_path / "f", "old").put("boosting", np.ones(3), 0.2)
    with pytest.raises(SystemExit):
        FoldStore(tmp_path / "f", "new").get("boosting")
    (tmp_path / "f" / "half.npz.tmp").write_bytes(b"x")
    assert FoldStore(tmp_path / "f", "old").get("half") is None
    store = FoldStore(tmp_path / "f", "old")
    store.clear()
    assert not (tmp_path / "f").exists()
