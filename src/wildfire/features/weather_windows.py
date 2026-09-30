"""Weather summarised over trailing windows before the target day."""
from wildfire.features.base import TIER_SUMMARY, FeatureSet, register


@register
class WeatherWindows(FeatureSet):
    name = "weather_windows"
    tier = TIER_SUMMARY

    def columns(self) -> list[str]:
        raise NotImplementedError

    def build(self, keys):
        raise NotImplementedError
