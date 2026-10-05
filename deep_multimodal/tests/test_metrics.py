import numpy as np
import pytest

from deep_multimodal import bootstrap, metrics


def test_average_precision_is_one_for_a_perfect_ranking_and_handles_ties():
    labels = np.array([1, 0, 1, 0, 0])
    assert metrics.average_precision(labels, np.array([0.9, 0.1, 0.8, 0.2, 0.3])) == pytest.approx(1.0)
    assert metrics.average_precision(labels, np.full(5, 0.5)) == pytest.approx(labels.mean())      # all tied: precision of the whole set
    assert np.isnan(metrics.average_precision(np.zeros(4), np.arange(4)))


def test_prior_shift_correction_lowers_probabilities_when_the_fit_rows_were_oversampled_for_fires():
    corrected = metrics.correct_prior_shift(np.array([0.5]), 0.01, 0.1)
    assert corrected[0] == pytest.approx(0.01 / 0.99 / (0.1 / 0.9) / (1 + 0.01 / 0.99 / (0.1 / 0.9)))
    assert metrics.correct_prior_shift(np.array([0.3]), 0.1, 0.1)[0] == pytest.approx(0.3)


def test_expected_calibration_error_is_zero_for_probabilities_equal_to_the_observed_rates_and_positive_otherwise():
    probability = np.repeat([0.1, 0.5], 500)
    labels = np.concatenate([np.r_[np.ones(50), np.zeros(450)], np.r_[np.ones(250), np.zeros(250)]])
    weights = np.ones(1000)
    assert metrics.calibration_curve(labels, weights, probability, bins=2)[1] == pytest.approx(0.0)
    assert metrics.calibration_curve(labels, weights, np.clip(probability + 0.2, 0, 1), bins=2)[1] == pytest.approx(0.2)


def test_best_threshold_maximises_the_critical_success_index():
    labels = np.array([1, 1, 0, 0, 1, 0])
    probability = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    weights = np.ones(6)
    threshold = metrics.best_threshold(labels, weights, probability)
    assert threshold == 0.8                           # two hits, no false alarm, one miss: index 2/3, better than any other cut


def test_top_share_reports_precision_recall_and_lift_per_day():
    labels = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    scores = np.array([0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4])
    days = [np.arange(5), np.arange(5, 10)]
    top = metrics.top_share(labels, np.ones(10), scores, days, 0.2)
    assert top["precision"] == pytest.approx(0.5) and top["recall"] == pytest.approx(0.5) and top["lift"] == pytest.approx(0.5 / 0.2)


def test_stopping_score_prefers_the_better_calibrated_probabilities():
    labels = np.r_[np.ones(10), np.zeros(90)]
    good, bad = np.where(labels == 1, 0.8, 0.1), np.full(100, 0.5)
    assert metrics.stopping_score(good, labels, 0.1)[0] < metrics.stopping_score(bad, labels, 0.1)[0]


def test_paired_bootstrap_gives_a_zero_difference_for_identical_scores_and_a_positive_one_for_a_better_model():
    generator = np.random.default_rng(0)
    labels = (generator.random(4000) < 0.1).astype(int)
    good = labels * 0.5 + generator.random(4000)
    noisy = generator.random(4000)
    blocks = np.repeat([f"2024-W{i:02d}" for i in range(20)], 200)
    same = bootstrap.paired_average_precision(labels, blocks, {"a": good, "b": good}, "b", draws=200)
    assert same["a"]["difference_to_reference"]["point"] == 0 and same["a"]["difference_to_reference"]["lower"] == 0
    better = bootstrap.paired_average_precision(labels, blocks, {"a": good, "b": noisy}, "b", draws=200)
    assert better["a"]["difference_to_reference"]["lower"] > 0 and better["blocks"] == 20
    interval = better["a"]["average_precision"]
    assert interval["lower"] <= interval["point"] <= interval["upper"]
