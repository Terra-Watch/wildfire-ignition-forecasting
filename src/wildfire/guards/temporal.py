"""Guard: every predictor must be observed strictly before the forecast cutoff."""
from __future__ import annotations


def assert_before_cutoff(observed_at, cutoff) -> None:
    """Raise if any observation time is at or after its cutoff."""
    raise NotImplementedError
