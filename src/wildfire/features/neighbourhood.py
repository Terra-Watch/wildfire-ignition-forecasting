"""Attributes and recent fire activity in surrounding cells."""
from wildfire.features.base import TIER_SUMMARY, FeatureSet, register


@register
class Neighbourhood(FeatureSet):
    name = "neighbourhood"
    tier = TIER_SUMMARY

    def columns(self) -> list[str]:
        raise NotImplementedError

    def build(self, keys):
        raise NotImplementedError
