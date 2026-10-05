"""Scoring functions. Rank metrics use raw scores; probability metrics use probabilities corrected to the true fire rate and the row weights."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROBABILITY_CLIP = 1e-15


def average_precision(labels, scores) -> float:
    """Average precision with tied scores grouped into one threshold: sum over thresholds of (recall step) times precision. NaN without a positive."""
    labels, scores = np.asarray(labels, float), np.asarray(scores, float)
    positives = labels.sum()
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels, sorted_scores = labels[order], scores[order]
    last = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_labels) - 1]
    hits = np.cumsum(sorted_labels)[last]
    return float(np.sum(np.diff(np.r_[0.0, hits / positives]) * hits / (last + 1.0)))


def correct_prior_shift(probability, true_rate, sample_rate):
    """Multiply the odds by the ratio of the true odds to the odds of the rows the model was fitted on (negatives are thinned, so raw probabilities are too high)."""
    probability = np.clip(np.asarray(probability, float), PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    true_rate, sample_rate = np.asarray(true_rate, float), np.asarray(sample_rate, float)
    odds = probability / (1 - probability) * (true_rate / (1 - true_rate)) / (sample_rate / (1 - sample_rate))
    return odds / (1 + odds)


def brier_score(labels, probability, weights=None) -> float:
    return float(np.average((np.asarray(probability, float) - np.asarray(labels, float)) ** 2, weights=weights))


def log_loss(labels, probability, weights=None) -> float:
    labels = np.asarray(labels, float)
    probability = np.clip(np.asarray(probability, float), PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    return float(np.average(-(labels * np.log(probability) + (1 - labels) * np.log1p(-probability)), weights=weights))


def stopping_score(probability, labels, fit_rate, true_rate=None, negative_weight=1.0) -> tuple[float, float]:
    """The score that chooses the number of training epochs: log loss after the prior-shift correction, with thinned negatives weighted back up; and average precision as tie-break."""
    labels = np.asarray(labels, float)
    true_rate = float(labels.mean() if true_rate is None else true_rate)
    corrected = np.clip(correct_prior_shift(probability, true_rate, fit_rate), PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    weights = np.where(labels == 1, 1.0, negative_weight)
    return log_loss(labels, corrected, weights), average_precision(labels, probability)


def group_mean(labels, scores, groups, function) -> tuple[float, int]:
    """Mean of `function` over the groups (index arrays) that hold at least one positive and one negative row; also returns how many groups were used."""
    values = [function(labels[i], scores[i]) for i in groups if labels[i].sum() > 0 and (1 - labels[i]).sum() > 0 and len(i) > 1]
    return float(np.mean(values)), len(values)


def top_share(labels, weights, scores, day_groups, share: float) -> dict:
    """Per day, take the top `share` of rows by score (at least one). Precision is the mean over days of the weighted precision;
    recall is weighted positives found over all weighted positives; lift is the pooled weighted precision over the weighted fire rate."""
    precision_by_day, found, taken = [], 0.0, 0.0
    total_positive = float((weights * labels).sum())
    for index in day_groups:
        top = index[np.argsort(-scores[index], kind="mergesort")[:max(1, int(np.ceil(share * len(index))))]]
        precision_by_day.append(np.average(labels[top], weights=weights[top]))
        found += float((weights[top] * labels[top]).sum())
        taken += float(weights[top].sum())
    rate = total_positive / float(weights.sum())
    return dict(precision=float(np.mean(precision_by_day)), recall=found / total_positive, lift=(found / taken) / rate)


def calibration_curve(labels, weights, probability, bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Equal-count groups of the probability: mean prediction and weighted observed rate; and the expected calibration error (weighted mean absolute gap)."""
    order = np.argsort(probability, kind="mergesort")
    rows = [dict(group=k + 1, rows=len(i), weight=float(weights[i].sum()), mean_prediction=float(np.average(probability[i], weights=weights[i])),
                 observed_rate=float(np.average(labels[i], weights=weights[i]))) for k, i in enumerate(np.array_split(order, bins))]
    curve = pd.DataFrame(rows)
    return curve, float(((curve.weight / curve.weight.sum()) * (curve.mean_prediction - curve.observed_rate).abs()).sum())


def best_threshold(labels, weights, probability) -> float:
    """The probability threshold that maximises the weighted critical success index, hits / (positives + false alarms)."""
    order = np.argsort(-probability, kind="mergesort")
    sorted_p = probability[order]
    hits, false_alarms = np.cumsum((weights * labels)[order]), np.cumsum((weights * (1 - labels))[order])
    last = np.r_[np.flatnonzero(np.diff(sorted_p)), len(sorted_p) - 1]
    index = hits[last] / (hits[-1] + false_alarms[last])
    return float(sorted_p[last][np.argmax(index)])


def score_rows(labels, weights, scores, probability, climatology, day, cell, threshold) -> tuple[dict, pd.DataFrame]:
    """Every metric of the results table for one set of rows. `day` and `cell` give the group of each row for the within-day and within-cell views."""
    position = pd.Series(np.arange(len(labels)))
    day_groups, cell_groups = list(position.groupby(day).indices.values()), list(position.groupby(cell).indices.values())
    out = {"average_precision": average_precision(labels, scores), "auroc": roc_auc_score(labels, scores)}
    out["average_precision_within_day"], _ = group_mean(labels, scores, day_groups, average_precision)
    out["auroc_within_day"], out["days_used"] = group_mean(labels, scores, day_groups, roc_auc_score)
    out["average_precision_within_cell"], _ = group_mean(labels, scores, cell_groups, average_precision)
    out["auroc_within_cell"], out["cells_used"] = group_mean(labels, scores, cell_groups, roc_auc_score)
    for share, name in ((0.01, "top1pct"), (0.05, "top5pct"), (0.10, "top10pct")):
        top = top_share(labels, weights, scores, day_groups, share)
        out[f"precision_{name}"] = top["precision"]
        if name != "top10pct":
            out[f"recall_{name}"], out[f"lift_{name}"] = top["recall"], top["lift"]
    brier, brier_reference = brier_score(labels, probability, weights), brier_score(labels, climatology, weights)
    out.update(brier=brier, brier_cell_climatology=brier_reference, brier_skill_score=1 - brier / brier_reference, log_loss=log_loss(labels, probability, weights),
               fire_rate_weighted=float((weights * labels).sum() / weights.sum()), rows=len(labels), onsets=int(labels.sum()))
    curve, out["calibration_error_ece"] = calibration_curve(labels, weights, probability)
    alert = probability >= threshold
    hits, false_alarms, positives = float((weights * labels * alert).sum()), float((weights * (1 - labels) * alert).sum()), float((weights * labels).sum())
    out.update(alert_threshold=threshold, alert_rate=float((weights * alert).sum() / weights.sum()), probability_of_detection=hits / positives,
               false_alarm_ratio=false_alarms / (hits + false_alarms) if hits + false_alarms else float("nan"), critical_success_index=hits / (positives + false_alarms))
    return out, curve
