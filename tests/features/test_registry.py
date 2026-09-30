import pytest

from wildfire.features import REGISTRY, TIER_IMAGERY, TIER_SUMMARY


def test_two_tiers():
    tiers = {n: c.tier for n, c in REGISTRY.items()}
    assert tiers == {
        "tabular": TIER_SUMMARY,
        "weather_windows": TIER_SUMMARY,
        "neighbourhood": TIER_SUMMARY,
        "imagery_summary": TIER_IMAGERY,
    }


def test_config_matches_registry():
    from wildfire import config
    cfg = config.load("features")["tiers"]
    assert {n for names in cfg.values() for n in names} == set(REGISTRY)


@pytest.mark.parametrize("name", ["imagery_summary"])
def test_stubs_not_implemented(name):
    with pytest.raises(NotImplementedError):
        REGISTRY[name]().columns()
