import pytest

from wildfire.evaluation.harness import evaluate


def test_held_out_split_is_refused():
    with pytest.raises(ValueError):
        evaluate([], None, split="test")
