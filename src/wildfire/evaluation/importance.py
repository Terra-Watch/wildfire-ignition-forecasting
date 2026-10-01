"""NOT OFFICIAL: block permutation importance used only by the walk-through notebook."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def block_permutation_drops(predict, x: np.ndarray, y: np.ndarray, blocks: dict, seed: int = 0) -> tuple[float, dict]:
    """Average precision of `predict` on (x, y) and, for each named block of columns, the drop in AP when the rows of that
    block are shuffled together (the rest of x is untouched). Returns (base AP, {block: drop})."""
    base = average_precision_score(y, predict(x))
    rng = np.random.default_rng(seed)
    drops = {}
    for name, idx in blocks.items():
        xp = x.copy()
        xp[:, idx] = xp[rng.permutation(len(xp))][:, idx]
        drops[name] = base - average_precision_score(y, predict(xp))
    return base, drops
