"""Placeholder. Summaries derived from satellite imagery; not implemented."""
from wildfire.features.base import TIER_IMAGERY, FeatureSet, register


@register
class ImagerySummary(FeatureSet):
    name = "imagery_summary"
    tier = TIER_IMAGERY

    def columns(self) -> list[str]:
        raise NotImplementedError

    def build(self, keys):
        raise NotImplementedError
