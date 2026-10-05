"""Exact paired bootstrap of average precision over blocks of weeks (no binning of scores). The same block draws are used for every model, so differences are paired."""
from __future__ import annotations

import numpy as np


def weighted_average_precision(sorted_labels, weights, last_of_tie_group) -> float:
    hits = np.cumsum(weights * sorted_labels)[last_of_tie_group]
    all_rows = np.cumsum(weights * (1 - sorted_labels))[last_of_tie_group]
    total = hits[-1]
    if total == 0:
        return float("nan")
    step = np.diff(np.r_[0.0, hits])
    denominator = hits + all_rows
    precision = np.divide(hits, denominator, out=np.zeros_like(hits), where=denominator > 0)
    return float(np.sum(step / total * precision))


def interval(draws, point) -> dict:
    if np.isnan(draws).any():
        raise ValueError("a bootstrap draw had no positives")
    return dict(point=float(point), lower=float(np.quantile(draws, 0.025)), upper=float(np.quantile(draws, 0.975)), standard_error=float(np.std(draws)))


def paired_average_precision(labels, blocks, scores: dict, reference: str, draws: int = 2000, seed: int = 0) -> dict:
    """labels: 0/1; blocks: one label per row (year and ISO week); scores: model name -> scores. Returns, per model, the average precision with its 95% interval and,
    for every model except `reference`, the difference to the reference with its interval; plus the number of blocks."""
    labels = np.asarray(labels).astype(float)
    names, block_index = np.unique(np.asarray(blocks).astype(str), return_inverse=True)
    generator = np.random.default_rng(seed)
    counts = np.stack([np.bincount(generator.integers(0, len(names), len(names)), minlength=len(names)) for _ in range(draws)]).astype(np.float64)
    prepared = {}
    for model, score in scores.items():
        score = np.asarray(score, float)
        order = np.argsort(-score, kind="mergesort")
        sorted_score = score[order]
        prepared[model] = (labels[order], block_index[order], np.r_[np.flatnonzero(np.diff(sorted_score)), len(sorted_score) - 1])
    values = {model: np.array([weighted_average_precision(y, counts[d][b], last) for d in range(draws)]) for model, (y, b, last) in prepared.items()}
    points = {model: weighted_average_precision(y, np.ones(len(y)), last) for model, (y, b, last) in prepared.items()}
    result = {model: {"average_precision": interval(values[model], points[model])} for model in scores}
    for model in scores:
        if model != reference:
            result[model]["difference_to_reference"] = interval(values[model] - values[reference], points[model] - points[reference])
    result["blocks"], result["draws"] = len(names), draws
    return result
