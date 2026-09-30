"""Common interface and registry for feature sets."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

TIER_SUMMARY = "summary"
TIER_IMAGERY = "imagery"

REGISTRY: dict[str, type["FeatureSet"]] = {}


class FeatureSet(ABC):
    """One family of features. Every set is scored by the same harness."""

    name: ClassVar[str]
    tier: ClassVar[str]

    @abstractmethod
    def columns(self) -> list[str]:
        """Names of the columns this set produces."""

    @abstractmethod
    def build(self, keys):
        """Return features for the given (cell, day) keys using only data
        observed before each key's day."""


def register(cls: type[FeatureSet]) -> type[FeatureSet]:
    REGISTRY[cls.name] = cls
    return cls
