"""Shared plot style so every figure looks the same."""
from __future__ import annotations

import sys

import matplotlib

if "ipykernel" not in sys.modules:       # scripts draw off screen; in a notebook the inline backend shows a returned figure
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOURS = {"primary": "#1f5f8b", "secondary": "#d1495b", "neutral": "#6b7280", "good": "#2f855a", "warn": "#c77c02"}
ARM_COLOURS = {"climatology_cell": "#6b7280", "landcover_season": "#9ca3af", "fwi_season": "#c4b5fd",
               "logistic_weather_static": "#60a5fa", "logistic_with_burn": "#1f5f8b", "boosting": "#d1495b",
               "tabular_network": "#2f855a", "coords_only": "#f59e0b", "cell_id_only": "#b45309", "static_only": "#92400e"}


def base_arm(name: str) -> str:
    """Arm name without the information level and seed suffix."""
    return name.split("@")[0].split("#")[0]


def colour(name: str) -> str:
    return ARM_COLOURS.get(base_arm(name), COLOURS["neutral"])


def new_figure(width: float = 7.0, height: float = 3.6):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.grid(alpha=0.25)
    return fig, ax
