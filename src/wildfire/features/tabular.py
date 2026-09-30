"""Static per-cell attributes and simple per-cell history."""
from wildfire.features.base import TIER_SUMMARY, FeatureSet, register


@register
class Tabular(FeatureSet):
    name = "tabular"
    tier = TIER_SUMMARY

    def columns(self) -> list[str]:
        raise NotImplementedError

    def build(self, keys):
        raise NotImplementedError
