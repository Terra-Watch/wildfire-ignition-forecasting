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

    def fit(self, x, y, inplace: bool = False):
        """Fit on (x, y). With inplace=True the logistic arm standardises `x` itself instead of a copy (memory)."""
        cfg = config.load("models")
        if self.kind == "logistic":
            from sklearn.linear_model import LogisticRegression
            lg = cfg["logistic"]
            self.mu = x.mean(0, dtype=np.float64).astype(np.float32)
            self.sd = (x.std(0, dtype=np.float64) + lg["scale_epsilon"]).astype(np.float32)
            if inplace:
                x -= self.mu
                x /= self.sd
                xs = x
            else:
                xs = (x - self.mu) / self.sd
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
