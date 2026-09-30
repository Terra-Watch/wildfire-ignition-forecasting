"""Tables for the model ladder: per-arm metrics, top-k capture, reliability, and the pre-declared contrasts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.evaluation import bootstrap, metrics, prior_shift, verdict, views


def top_k_capture(y, score, shares) -> dict:
    """For each share k: fraction of onsets among the top-k share of rows by score, and that fraction divided by k."""
    order = np.argsort(-np.asarray(score), kind="mergesort")
    hit = np.cumsum(np.asarray(y)[order])
    out = {}
    for k in shares:
        n = max(int(round(k * len(y))), 1)
        cap = float(hit[n - 1] / max(hit[-1], 1))
        out[f"capture_{int(k * 100)}pct"] = cap
        out[f"lift_{int(k * 100)}pct"] = cap / k
    return out


def reliability(y, p, w, bins: int) -> pd.DataFrame:
    """Decile table: mean predicted probability and weighted observed rate per bin of the predicted probability."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)[1:-1]))
    b = np.searchsorted(edges, p, side="right")
    rows = []
    for k in np.unique(b):
        m = b == k
        rows.append(dict(bin=int(k), rows=int(m.sum()), mean_predicted=float(np.average(p[m], weights=w[m])),
                         observed_rate=float(np.average(np.asarray(y)[m], weights=w[m]))))
    return pd.DataFrame(rows)


def corrected_probabilities(raw: np.ndarray, pi_true: np.ndarray, pi_fit: float) -> np.ndarray:
    """Prior-shift correction from the sampling rate of the fit rows to the true per-state rate."""
    return prior_shift.prior_shift_probs(raw, pi_true, pi_fit)


def arm_row(rows: pd.DataFrame, score, corrected, w) -> dict:
    """Headline metrics of one arm on the validation rows: both views, AP, corrected log loss and Brier, top-k."""
    cfg = config.load("baseline")["reporting"]
    y = rows.label.values
    v = views.views(rows, score)
    out = dict(auc_spatial_within_day=v["auc_spatial_within_day"], auc_temporal_within_cell=v["auc_temporal_within_cell"],
               auc_pooled=v["auc_pooled"], ap=v["ap"], log_loss=metrics.log_loss(y, corrected, w=w),
               brier=metrics.brier(y, corrected, w=w), rows=len(rows), onsets=int(y.sum()))
    return {**out, **top_k_capture(y, score, cfg["top_k_share"])}


def per_state_delta_ap(rows: pd.DataFrame, score, ref_score) -> dict:
    """AP difference (arm minus reference) within each state."""
    out = {}
    for st, g in rows.groupby("state"):
        ix = g.index.values
        out[st] = metrics.average_precision(g.label.values, score[ix]) - metrics.average_precision(g.label.values, ref_score[ix])
    return out


def contrast(rows: pd.DataFrame, scores: dict, probs: dict, weights_, arm: str, ref: str, clusters) -> dict:
    """One contrast: paired bootstrap of arm vs ref, per-state AP differences, minimum detectable effect and verdict.
    `rows` must have a 0..n-1 index."""
    arms = {a: dict(score=scores[a], prob=probs[a]) for a in (arm, ref)}
    o = bootstrap.paired_week_blocked_bootstrap(rows.label.values, clusters, arms, ref, weights=weights_)
    d = o[arm]["delta_ap"]
    states = per_state_delta_ap(rows, scores[arm], scores[ref])
    worst = min(states.values())
    v, mde = verdict.baseline_verdict(d, worst)
    return dict(arm=arm, reference=ref, delta_ap=d["point"], ap_lo99=d["lo99"], ap_hi99=d["hi99"], ap_se=d["se"], mde=mde,
                worst_state=min(states, key=states.get), worst_state_delta_ap=worst,
                delta_log_loss=o[arm]["delta_logloss"]["point"], delta_brier=o[arm]["delta_brier"]["point"], verdict=v)


def cluster_labels(rows: pd.DataFrame):
    return prior_shift.cluster_labels(rows.year.values, rows.week.values, None, config.load("evaluation")["clusters"]["scheme"])
