import numpy as np
from sklearn.metrics import average_precision_score
from wildfire_forecast.evaluate import topk_day, ap_from, binned


def test_topk_day_ranks_within_each_day():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]); p = np.r_[np.arange(10, 0, -1), np.arange(10, 0, -1)] / 10.0; day = np.r_[np.zeros(10), np.ones(10)]
    r = topk_day(y, p, day, qs=(0.1,))                              # top 1 cell of each day: day 0's positive is first, day 1's is last
    assert r["top10_prec"] == 0.5 and r["top10_rec"] == 0.5


def test_binned_ap_matches_sklearn():
    rng = np.random.default_rng(0); y = (rng.random(5000) < 0.1).astype(int); p = np.clip(y * 0.3 + rng.random(5000) * 0.7, 0, 1); wk = rng.integers(0, 10, 5000)
    P, N = binned(y, p, wk, np.unique(np.quantile(p, np.linspace(0, 1, 2001)[1:-1])), 10)
    assert abs(ap_from(P.sum(0), N.sum(0)) - average_precision_score(y, p)) < 2e-3
