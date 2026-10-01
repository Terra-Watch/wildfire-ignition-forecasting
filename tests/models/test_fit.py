import numpy as np

from wildfire import config
from wildfire.models.fit import Model, thin


def test_thin_keeps_every_onset_and_caps_rows():
    rng = np.random.default_rng(2)
    y = (rng.random(100000) < 0.05).astype(int)
    k = thin(y, 30000, seed=1)
    assert int(y[k].sum()) == int(y.sum()) and int(k.sum()) == 30000
    assert thin(y, 10**6).all()


def test_logistic_and_boosting_learn_a_simple_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 3)).astype(np.float32)
    y = (x[:, 0] + 0.3 * rng.normal(size=4000) > 1).astype(int)
    for kind, params in (("logistic", {"C": 1.0}), ("boosting", {"max_depth": 3, "learning_rate": 0.1, "max_iter": 50})):
        m = Model(kind, params).fit(x, y)
        p = m.predict(x)
        assert p[y == 1].mean() > p[y == 0].mean() + 0.2
        assert abs(m.fit_prevalence - y.mean()) < 1e-12


def test_boosting_is_deterministic():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3000, 4)).astype(np.float32)
    y = (x[:, 1] + rng.normal(size=3000) > 1).astype(int)
    p = {"max_depth": 4, "learning_rate": 0.1, "max_iter": 40}
    assert (Model("boosting", p).fit(x, y).predict(x) == Model("boosting", p).fit(x, y).predict(x)).all()


def test_float64_standardisation_gives_the_float32_fit_and_leaves_the_input_alone(tmp_path):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(5)
    x = rng.normal(3.0, 2.0, size=(3000, 4)).astype(np.float32)
    y = (x[:, 0] + rng.normal(size=3000) > 4).astype(int)
    before = x.copy()
    a = Model("logistic", {"C": 0.1}).fit(x, y)
    np.save(tmp_path / "x.npy", x)
    b = Model("logistic", {"C": 0.1}).fit(np.load(tmp_path / "x.npy", mmap_mode="r"), y)
    cfg = config.load("models")["logistic"]
    legacy = LogisticRegression(C=0.1, solver=cfg["solver"], max_iter=cfg["max_iter"], tol=cfg["tol"]).fit((x - a.mu) / a.sd, y)
    assert np.array_equal(x, before) and np.array_equal(a.predict(x), b.predict(x))
    assert np.array_equal(a.m.coef_, legacy.coef_) and np.array_equal(a.m.intercept_, legacy.intercept_)
