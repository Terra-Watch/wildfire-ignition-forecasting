"""Label-shuffle null: permute labels and check that skill disappears."""
from __future__ import annotations

import numpy as np


def shuffle_within(y, key, seed: int):
    """Permute labels within groups defined by `key` (every group keeps its exact label count)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    order = np.argsort(key, kind="stable")
    r = np.lexsort((rng.random(n), key))
    out = np.empty_like(y)
    out[order] = y[r]
    return out
