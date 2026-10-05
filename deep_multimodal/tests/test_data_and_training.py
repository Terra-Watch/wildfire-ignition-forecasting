import numpy as np
import torch

from deep_multimodal import constants as c
from deep_multimodal.data import load_fold
from deep_multimodal.forecast import forecast_inputs
from deep_multimodal.model import ModelSettings
from deep_multimodal.training import TrainingSettings, predict_logits, train_ensemble, train_two_stage

SMALL = ModelSettings(width=16, layers=1, heads=2, dropout=0.0)
FAST = TrainingSettings(max_epochs=3, batch_size=256, seeds=(0, 1))


def test_forecast_rows_have_the_documented_shapes_and_missing_days_are_nan_and_flagged(package):
    keys = np.load(package / "fold_2022" / "val_key.npy")
    hourly, daily, missing = forecast_inputs(keys, package / "fc_lookup")
    assert hourly.shape == (len(keys), len(c.FORECAST_HOURS), len(c.FORECAST_VARIABLES)) and daily.shape == (len(keys), c.FORECAST_DAILY_VALUES)
    assert np.isnan(hourly[missing]).all() and not np.isnan(hourly[~missing]).any()


def test_fold_loading_splits_the_fit_rows_by_year_and_builds_weather_windows(package):
    fold = load_fold(package, 2022, torch.device("cpu"))
    assert fold.fit_all.size == fold.train.size + fold.stop.size and fold.stop.size > 0 and fold.validation.size == 300
    assert set(np.unique(fold.stop.year)) == {2021} and 2021 not in set(np.unique(fold.train.year))
    batch = fold.validation.batch(torch.arange(8))
    assert batch["weather"].shape == (8, c.WEATHER_DAYS, c.WEATHER_INPUTS) and torch.isfinite(batch["weather"]).all()
    assert batch["static"].shape == (8, len(c.STATIC_COLUMNS)) and batch["row_values"].shape == (8, len(c.ROW_COLUMNS))


def test_standardising_constants_come_from_the_fit_rows_only(package):
    fold = load_fold(package, 2022, torch.device("cpu"))
    assert fold.standardiser.row_mean.shape == (len(c.ROW_COLUMNS),)
    assert np.all(fold.standardiser.row_mean[c.CALENDAR_COLUMNS] == 0) and np.all(fold.standardiser.row_std[c.CALENDAR_COLUMNS] == 1)


def test_two_stage_training_returns_a_model_and_valid_logits(package):
    fold = load_fold(package, 2022, torch.device("cpu"))
    model, epochs = train_two_stage(fold, SMALL, FAST, 0, torch.device("cpu"))
    logits = predict_logits(model, fold.validation)
    assert 1 <= epochs <= FAST.max_epochs and logits.shape == (fold.validation.size,) and np.isfinite(logits).all()


def test_training_is_reproducible_for_a_seed_and_the_ensemble_is_a_mean_of_probabilities(package, tmp_path):
    fold = load_fold(package, 2022, torch.device("cpu"))
    first = predict_logits(train_two_stage(fold, SMALL, FAST, 1, torch.device("cpu"))[0], fold.validation)
    second = predict_logits(train_two_stage(fold, SMALL, FAST, 1, torch.device("cpu"))[0], fold.validation)
    assert np.allclose(first, second)
    ensemble = train_ensemble(fold, SMALL, FAST, torch.device("cpu"), tmp_path, log=lambda *a: None)
    seeds = [1 / (1 + np.exp(-np.load(tmp_path / f"seed{s}_validation_logits.npy"))) for s in FAST.seeds]
    assert np.allclose(ensemble, np.mean(seeds, 0)) and ((ensemble > 0) & (ensemble < 1)).all()
    assert (tmp_path / "seed0_weights.pt").exists()


def test_saved_weights_load_back_into_the_model_and_give_the_same_logits(package, tmp_path):
    from deep_multimodal.weights import load_weights
    fold = load_fold(package, 2022, torch.device("cpu"))
    model, _ = train_two_stage(fold, SMALL, FAST, 0, torch.device("cpu"))
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    loaded = load_weights(tmp_path / "weights.pt", SMALL)
    assert np.allclose(predict_logits(model, fold.validation), predict_logits(loaded, fold.validation))
