import pytest

from wildfire.guards.temporal import assert_before_cutoff


def test_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        assert_before_cutoff(None, None)
