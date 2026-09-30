"""Point metrics: average precision, log loss, Brier score."""
from __future__ import annotations

import numpy as np

from wildfire import config


def clip_eps() -> float:
    return config.load("evaluation")["probability_clip"]


def average_precision(y, s) -> float:
    """Exact AP with tied scores grouped into one threshold: sum_k (R_k - R_{k-1}) * P_k. NaN if there is no positive."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    p = y.sum()
    if p == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    ys = y[o]
    ss = s[o]
    thr = np.r_[np.flatnonzero(np.diff(ss)), len(ys) - 1]
    tp = np.cumsum(ys)[thr]
    prec = tp / (thr + 1.0)
    rec = tp / p
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def log_loss(y, p, eps: float | None = None, w=None) -> float:
    eps = clip_eps() if eps is None else eps
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    loss = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    return float(np.average(loss, weights=w))


def brier(y, p, w=None) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.average((p - y) ** 2, weights=w))
