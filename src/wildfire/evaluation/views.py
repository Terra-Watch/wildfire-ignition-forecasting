"""Ranking views: pooled AUC, within-day spatial AUC, within-cell temporal AUC, and AP."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score


def auc(y, s) -> float:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def group_auc(y, s, g):
    """Mean over groups holding both classes of the within-group AUC, and the number of groups used."""
    df = pd.DataFrame(dict(y=y, s=s, g=g))
    a = [auc(q.y.values, q.s.values) for _, q in df.groupby("g") if 0 < q.y.sum() < len(q)]
    a = [x for x in a if np.isfinite(x)]
    return float(np.mean(a)) if a else np.nan, len(a)


def views(d: pd.DataFrame, s) -> dict:
    """d: rows with label, cell, day; s: scores."""
    y = d.label.values
    sp, nsp = group_auc(y, s, d.day.values)
    tm, ntm = group_auc(y, s, d.cell.values)
    return dict(auc_pooled=auc(y, s), auc_spatial_within_day=sp, n_days_used=nsp, auc_temporal_within_cell=tm,
                n_cells_used=ntm, ap=float(average_precision_score(y, s)), n=len(d), n_pos=int(y.sum()))
