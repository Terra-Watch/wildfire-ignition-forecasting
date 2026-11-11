import numpy as np
from sklearn.metrics import average_precision_score as auprc, roc_auc_score as auroc

def topk(y, p, frac):
    """precision and recall among the top `frac` of scores."""
    k = max(1, int(len(y) * frac)); i = np.argsort(-p, kind="stable")[:k]; return y[i].mean(), y[i].sum() / max(1, y.sum())

def brier(y, p): return float(np.mean((p - y) ** 2))

def ece(y, p, bins=15):
    b = np.minimum((p * bins).astype(int), bins - 1); return float(sum(abs(y[b == k].mean() - p[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any()))

def week_bootstrap(y, pa, pb, weeks, draws=2000, seed=0, metric=auprc):
    """Paired calendar-week block bootstrap of metric(pa) - metric(pb); returns (mean difference, 2.5%, 97.5%)."""
    rng = np.random.default_rng(seed); u, inv = np.unique(weeks, return_inverse=True); idx = [np.flatnonzero(inv == k) for k in range(len(u))]; d = []
    for _ in range(draws):
        s = np.concatenate([idx[k] for k in rng.integers(0, len(u), len(u))]); d.append(metric(y[s], pa[s]) - metric(y[s], pb[s]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
