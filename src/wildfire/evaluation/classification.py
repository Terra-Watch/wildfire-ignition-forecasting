"""NOT OFFICIAL: thresholded metrics (accuracy, precision, recall, F1) used only by the walk-through notebook.
The official tables report ranking and probability metrics; they do not report these. All take design weights, so they
describe the real population of cell-days, not the sampled rows."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score


def best_threshold(y, p, w) -> float:
    """Probability threshold that maximises the weighted F1."""
    pr, rc, th = precision_recall_curve(y, p, sample_weight=w)
    f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    return float(th[np.argmax(f1)])


def threshold_metrics(y, p, w, threshold: float) -> dict:
    """Weighted accuracy, balanced accuracy, precision, recall and F1 of the rule 'predict fire when p >= threshold'."""
    yhat = np.asarray(p) >= threshold
    return dict(accuracy=accuracy_score(y, yhat, sample_weight=w), balanced_accuracy=balanced_accuracy_score(y, yhat, sample_weight=w),
                precision=precision_score(y, yhat, sample_weight=w, zero_division=0), recall=recall_score(y, yhat, sample_weight=w),
                f1=f1_score(y, yhat, sample_weight=w))


def always_no_accuracy(y, w) -> float:
    """Weighted accuracy of never predicting a fire."""
    return float(1 - np.average(np.asarray(y), weights=w))
