"""Paired cluster bootstraps: the same cluster draws for every arm."""
from __future__ import annotations

import numpy as np

from wildfire import config
from wildfire.evaluation import metrics
from wildfire.evaluation.views import auc


def _settings() -> dict:
    return config.load("evaluation")["bootstrap"]


# ---- week-blocked bootstrap of AP, log loss, Brier
def hist(score, y, ci, ncl, nbins):
    """Cluster x score-bin positive and negative counts. nbins=None -> one bin per distinct score (for tests)."""
    q = np.unique(np.quantile(score, np.linspace(0, 1, nbins + 1)[1:-1])) if nbins else np.unique(score)[1:]
    b = np.searchsorted(q, score, side="right")
    nb = len(q) + 1
    hp = np.zeros((ncl, nb))
    hn = np.zeros((ncl, nb))
    np.add.at(hp, (ci[y == 1], b[y == 1]), 1)
    np.add.at(hn, (ci[y == 0], b[y == 0]), 1)
    return hp, hn


def ap_from_hist(p, n):
    """p, n: (draws, bins) counts with bins in ascending score order -> AP per draw (ties inside a bin grouped)."""
    pd_ = p[:, ::-1]
    nd = n[:, ::-1]
    tp = np.cumsum(pd_, 1)
    fp = np.cumsum(nd, 1)
    tot = np.maximum(pd_.sum(1, keepdims=True), 1)
    prec = tp / np.maximum(tp + fp, 1)
    return ((pd_ / tot) * prec).sum(1)


def paired_week_blocked_bootstrap(y, cluster, arms: dict, ref: str, nboot: int | None = None, seed: int | None = None,
                                  nbins="config", weights=None):
    """arms: {name: dict(score=..., prob=...)} (prob defaults to score); ref: name of the reference arm.

    Clusters are resampled with replacement, the same draws for every arm. weights (n,) are per-row importance weights
    applied to log loss and Brier only (AP is always computed on the rows as sampled). Each metric and each delta versus
    ref reports point, 95% (lo, hi) and 99% (lo99, hi99) percentile bounds.
    """
    cfg = _settings()
    nboot = cfg["draws"] if nboot is None else nboot
    seed = cfg["seed"] if seed is None else seed
    nbins = cfg["bins"] if nbins == "config" else nbins
    eps = metrics.clip_eps()
    y = np.asarray(y).astype(int)
    cl, ci = np.unique(np.asarray(cluster).astype(str), return_inverse=True)
    ncl = len(cl)
    rng = np.random.default_rng(seed)
    draws = np.stack([np.bincount(rng.integers(0, ncl, ncl), minlength=ncl) for _ in range(nboot)]).astype(np.float64)
    wv = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    n_c = np.bincount(ci, weights=wv, minlength=ncl)
    stats = {}
    for a, d in arms.items():
        s = np.asarray(d["score"], float)
        p = np.asarray(d.get("prob", d["score"]), float)
        hp, hn = hist(s, y, ci, ncl, nbins)
        pc = np.clip(p, eps, 1 - eps)
        ll = np.bincount(ci, weights=wv * -(y * np.log(pc) + (1 - y) * np.log1p(-pc)), minlength=ncl)
        br = np.bincount(ci, weights=wv * (p - y) ** 2, minlength=ncl)
        stats[a] = dict(ap=ap_from_hist(draws @ hp, draws @ hn), logloss=(draws @ ll) / (draws @ n_c),
                        brier=(draws @ br) / (draws @ n_c),
                        point=dict(ap=metrics.average_precision(y, s), logloss=metrics.log_loss(y, p, w=wv),
                                   brier=metrics.brier(y, p, w=wv)))

    def interval(v, pt):
        return dict(point=float(pt), se=float(np.std(v)), lo=float(np.quantile(v, cfg["alpha"] / 2)), hi=float(np.quantile(v, 1 - cfg["alpha"] / 2)),
                    lo99=float(np.quantile(v, cfg["alpha_verdict"] / 2)), hi99=float(np.quantile(v, 1 - cfg["alpha_verdict"] / 2)))

    out = {}
    for a in arms:
        out[a] = {}
        for m in ("ap", "logloss", "brier"):
            out[a][m] = interval(stats[a][m], stats[a]["point"][m])
            if a != ref:
                out[a]["delta_" + m] = interval(stats[a][m] - stats[ref][m], stats[a]["point"][m] - stats[ref]["point"][m])
    out["_n_clusters"] = ncl
    out["_nboot"] = nboot
    return out


# ---- bootstrap of the within-state AUC difference
def binned(d, s, nbins: int):
    """Cluster x bin histograms for the fast bootstrap: (Hpos, Hneg)."""
    q = np.unique(np.quantile(s, np.linspace(0, 1, nbins + 1)[1:-1]))
    b = np.searchsorted(q, s, side="right")
    nb = len(q) + 1
    cl, ci = np.unique(d.cluster.astype(str).values, return_inverse=True)
    y = d.label.values
    hp = np.zeros((len(cl), nb))
    hn = np.zeros((len(cl), nb))
    np.add.at(hp, (ci[y == 1], b[y == 1]), 1)
    np.add.at(hn, (ci[y == 0], b[y == 0]), 1)
    return hp, hn


def auc_from_hist(p, n):
    """p, n: (draws, bins) counts -> AUC per draw."""
    cn = np.cumsum(n, 1) - n
    return ((p * (cn + 0.5 * n)).sum(1)) / np.maximum(p.sum(1) * n.sum(1), 1)


def paired_cluster_bootstrap(states_data: dict, arms: list, ref: str, seed: int | None = None, nboot: int | None = None,
                             weights=None):
    """states_data: {state: (df, {arm: scores})}. Per-state and onset-weighted pooled AUC differences against `ref`
    with percentile intervals, the same cluster draws for every arm."""
    cfg = _settings()
    seed = cfg["seed"] if seed is None else seed
    nboot = cfg["draws"] if nboot is None else nboot
    nbins = cfg["leave_year_out_bins"]
    z = sum(cfg["power_z"])
    rng = np.random.default_rng(seed)
    out = {}
    wts = weights or {s: int(dd.label.sum()) for s, (dd, _) in states_data.items()}
    tot = sum(wts.values())
    draws = {}
    h = {}
    for st, (dd, sc) in states_data.items():
        h[st] = {a: binned(dd, sc[a], nbins) for a in [ref] + arms}
        nc = h[st][ref][0].shape[0]
        draws[st] = np.stack([np.bincount(rng.integers(0, nc, nc), minlength=nc) for _ in range(nboot)]).astype(np.float32)

    def arm_auc(st, a):
        hp, hn = h[st][a]
        return auc_from_hist(draws[st] @ hp, draws[st] @ hn)

    base = {st: arm_auc(st, ref) for st in states_data}
    pt = {st: {a: auc(states_data[st][0].label.values, states_data[st][1][a]) for a in [ref] + arms} for st in states_data}
    for a in arms:
        per = {}
        pool = np.zeros(nboot)
        for st in states_data:
            dd_ = arm_auc(st, a) - base[st]
            per[st] = dict(d=pt[st][a] - pt[st][ref], ci_lo=float(np.percentile(dd_, 2.5)), ci_hi=float(np.percentile(dd_, 97.5)),
                           auc_arm=pt[st][a], auc_ref=pt[st][ref], n_pos=wts[st])
            pool += wts[st] * dd_ / tot
        d = sum(wts[st] * per[st]["d"] for st in per) / tot
        se = float(pool.std())
        out[a] = dict(per=per, d=float(d), ci_lo=float(np.percentile(pool, 2.5)), ci_hi=float(np.percentile(pool, 97.5)),
                      se=se, mde=z * se, auc_arm=sum(wts[st] * per[st]["auc_arm"] for st in per) / tot,
                      auc_ref=sum(wts[st] * per[st]["auc_ref"] for st in per) / tot, n_pos=tot)
    return out
