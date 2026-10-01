"""Model fitting: standardised L2 logistic regression and histogram gradient boosting."""
from __future__ import annotations

import numpy as np

from wildfire import config


def thin(y, cap: int, seed: int = 0):
    """Boolean mask keeping every onset and a uniform subsample of negatives so that at most `cap` rows remain."""
    rng = np.random.default_rng(seed)
    n = len(y)
    if n <= cap:
        return np.ones(n, bool)
    pos = y == 1
    keep = pos.copy()
    neg = np.flatnonzero(~pos)
    k = max(cap - int(pos.sum()), 0)
    keep[rng.choice(neg, min(k, len(neg)), replace=False)] = True
    return keep


class Model:
    def __init__(self, kind: str, params: dict):
        self.kind, self.params = kind, params

    def fit(self, x, y):
        """Fit on (x, y); `x` may be memory-mapped and is never modified.
        The logistic arm standardises `x` in float32, in row chunks, straight into a float64 matrix. scikit-learn's
        lbfgs multiplies by float64 coefficients, which turns a float32 input into a full float64 copy on every call
        (about 2.2 times the input in extra memory); handing it float64 gives the same numbers without those copies."""
        cfg = config.load("models")
        if self.kind == "logistic":
            from sklearn.linear_model import LogisticRegression
            lg = cfg["logistic"]
            self.mu = x.mean(0, dtype=np.float64).astype(np.float32)
            self.sd = (x.std(0, dtype=np.float64) + lg["scale_epsilon"]).astype(np.float32)
            xs = np.empty(x.shape, np.float64)
            chunk = cfg["chunks"]["matrix_file"]
            for start in range(0, len(x), chunk):
                block = np.array(x[start:start + chunk], dtype=np.float32)
                block -= self.mu
                block /= self.sd
                xs[start:start + chunk] = block
            self.m = LogisticRegression(C=self.params["C"], solver=lg["solver"], max_iter=lg["max_iter"], tol=lg["tol"]).fit(xs, y)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            bs = cfg["boosting"]
            self.m = HistGradientBoostingClassifier(random_state=bs["random_state"], early_stopping=bs["early_stopping"],
                                                    **self.params).fit(x, y)
        self.fit_prevalence = float(np.mean(y))
        return self

    def predict(self, x):
        xs = (x - self.mu) / self.sd if self.kind == "logistic" else x
        return self.m.predict_proba(xs)[:, 1]
