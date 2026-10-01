import numpy as np

from wildfire.evaluation import classification as cl, importance as imp


def test_best_threshold_separates_a_clean_signal():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.15, 0.3, 0.8, 0.9])
    t = cl.best_threshold(y, p, np.ones(6))
    m = cl.threshold_metrics(y, p, np.ones(6), t)
    assert 0.3 < t <= 0.8 and m["f1"] == 1.0 and m["accuracy"] == 1.0


def test_weights_change_precision_but_not_recall():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.9, 0.1, 0.9, 0.1])
    plain = cl.threshold_metrics(y, p, np.ones(4), 0.5)
    heavy = cl.threshold_metrics(y, p, np.array([10.0, 10.0, 1.0, 1.0]), 0.5)
    assert plain["recall"] == heavy["recall"] == 0.5 and heavy["precision"] < plain["precision"]


def test_always_no_accuracy_uses_weights():
    assert cl.always_no_accuracy(np.array([0, 1]), np.array([9.0, 1.0])) == 0.9


def test_block_permutation_flags_the_informative_block_only():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 3))
    y = (x[:, 0] + 0.3 * rng.normal(size=4000) > 1).astype(int)
    base, drops = imp.block_permutation_drops(lambda a: a[:, 0], x, y, {"signal": [0], "noise": [1, 2]})
    assert drops["signal"] > 0.2 and abs(drops["noise"]) < 1e-9 and base > 0.5
