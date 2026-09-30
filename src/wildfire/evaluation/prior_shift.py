"""Correct probabilities for the difference between the sampled and the true base rate; design weights."""
from __future__ import annotations

import numpy as np

from wildfire.evaluation.metrics import clip_eps


def prior_shift_probs(p, pi_true, pi_sample):
    """Odds are multiplied by [pi_t/(1-pi_t)] / [pi_s/(1-pi_s)]. pi_true may be an array (per row, from training years only);
    pi_sample = onsets / (onsets + sampled negatives)."""
    eps = clip_eps()
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    pt = np.asarray(pi_true, float)
    ps = np.asarray(pi_sample, float)
    k = (pt / (1 - pt)) / (ps / (1 - ps))
    o = p / (1 - p) * k
    return o / (1 + o)


def design_weights(y, stratum, frame_size: dict):
    """Importance weights turning the sampled rows into the full frame: onsets weight 1, negatives
    (frame_size[stratum] - onsets[stratum]) / sampled negatives[stratum]."""
    y = np.asarray(y).astype(int)
    st = np.asarray(stratum).astype(str)
    w = np.ones(len(y))
    for k in np.unique(st):
        m = st == k
        npos = int(y[m].sum())
        nneg = int((1 - y[m]).sum())
        if nneg:
            w[m & (y == 0)] = (frame_size[k] - npos) / nneg
    return w


def cluster_labels(year, week, state=None, scheme: str = "week"):
    """'week': one cluster per ISO week over all states; 'state_week': one per state x week. Year is part of the label."""
    year = np.asarray(year).astype(int).astype(str)
    week = np.asarray(week).astype(int)
    lab = np.char.add(np.char.add(year, "-W"), np.char.zfill(week.astype(str), 2))
    return lab if scheme == "week" else np.char.add(np.char.add(np.asarray(state).astype(str), "|"), lab)
