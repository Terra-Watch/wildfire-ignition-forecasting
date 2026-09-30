"""Pass rules and screens for the null-model sanity checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.evaluation import metrics


def sanity_decisions(ap_global, prevalence, ap_cell_week, ap_climatology, ap_permuted) -> dict:
    """(a) mean AP of the global-shuffle null within a band of the prevalence; (b) every cell x week shuffle AP below
    the climatology AP; (c) mean AP of permuted holdout labels within a tighter band of the prevalence."""
    s = config.load("evaluation")["sanity"]
    a = bool(abs(np.mean(ap_global) / prevalence - 1) <= s["band_global"])
    b = bool(np.max(ap_cell_week) < ap_climatology)
    c = bool(abs(np.mean(ap_permuted) / prevalence - 1) <= s["band_permuted"])
    return dict(a=a, b=b, c=c, stop=not (a and b))


def univariate_screen(x, y, names) -> pd.DataFrame:
    """AP and AUC of each input alone (rows where it is not NaN); flag AUC outside the configured band."""
    from sklearn.metrics import roc_auc_score
    lo, hi = config.load("evaluation")["sanity"]["univariate_auc_flag"]
    rows = []
    for k, n in enumerate(names):
        ok = ~np.isnan(x[:, k])
        yy = y[ok]
        s = x[ok, k].astype(np.float64)
        if yy.min() == yy.max() or np.ptp(s) == 0:
            rows.append(dict(input=n, n=int(ok.sum()), AP=np.nan, AP_negated=np.nan, AUC=np.nan, flag=False))
            continue
        a = float(roc_auc_score(yy, s))
        rows.append(dict(input=n, n=int(ok.sum()), AP=metrics.average_precision(yy, s),
                         AP_negated=metrics.average_precision(yy, -s), AUC=a, flag=bool(a > hi or a < lo)))
    return pd.DataFrame(rows)
