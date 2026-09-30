"""Isotonic calibration of the climatology rate, fitted on training years only."""
from __future__ import annotations

import numpy as np

from wildfire import config


class Isotonic:
    """Monotone non-decreasing map score -> probability. Refuses to fit on rows after `max_train_year`."""

    def __init__(self, max_train_year: int | None = None, eps: float | None = None):
        cfg = config.load("evaluation")["calibration"]
        self.max_train_year = cfg["max_train_year"] if max_train_year is None else max_train_year
        self.eps = cfg["clip"] if eps is None else eps
        self.ir = None

    def fit(self, score, y, year):
        from sklearn.isotonic import IsotonicRegression
        year = np.asarray(year)
        if len(year) and int(year.max()) > self.max_train_year:
            raise ValueError(f"calibration rows include year {int(year.max())} > {self.max_train_year}: "
                             "the temporal holdout must not be used to calibrate")
        score = np.asarray(score, float)
        ok = ~np.isnan(score)
        self.ir = IsotonicRegression(y_min=self.eps, y_max=1 - self.eps, increasing=True,
                                     out_of_bounds="clip").fit(score[ok], np.asarray(y, float)[ok])
        return self

    def transform(self, score):
        score = np.asarray(score, float)
        out = np.full(len(score), np.nan)
        ok = ~np.isnan(score)
        out[ok] = self.ir.predict(score[ok])
        return out


def calibrate_climatology(score_train, y_train, year_train, score_test, max_train_year: int | None = None):
    """Map the climatology rate to a probability with an isotonic fit on training years; apply to the test rows."""
    m = Isotonic(max_train_year).fit(score_train, y_train, year_train)
    return m.transform(score_test), m
