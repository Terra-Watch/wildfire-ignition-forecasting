"""Guard: every predictor must be observed strictly before the forecast cutoff."""
from __future__ import annotations

import numpy as np


def assert_before_cutoff(observed_at, cutoff) -> None:
    """Raise if any observation time is at or after its cutoff (arrays or lists of timestamps, same length)."""
    obs = np.asarray(observed_at, dtype="datetime64[ns]")
    cut = np.asarray(cutoff, dtype="datetime64[ns]")
    late = obs >= cut
    if late.any():
        raise AssertionError(f"{int(late.sum())} observations are not strictly before their cutoff")
