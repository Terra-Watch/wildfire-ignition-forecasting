"""Hazard maps: the model's probability that a fire starts in each forest cell on one day, drawn at the cell centres. This is a probability per 5 km cell, not an ignition point."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def day_table(predictions, cells: pd.DataFrame, day: int) -> pd.DataFrame:
    """Rows of one day with the deep model's corrected probability, the baseline's, the label and the cell centre."""
    mask = predictions.day == day
    frame = pd.DataFrame(dict(cell=predictions.cell[mask], label=predictions.label[mask], deep=predictions.probability["deep"][mask], baseline=predictions.probability["baseline"][mask]))
    return frame.merge(cells[["cell", "lon", "lat"]], on="cell", how="inner")


def busiest_day(predictions) -> int:
    """The day with the most fire onsets among the rows (the most informative day to draw)."""
    days, counts = np.unique(predictions.day[predictions.label == 1], return_counts=True)
    return int(days[np.argmax(counts)])


def draw_day(frame: pd.DataFrame, title: str, path: Path | None = None, column: str = "deep"):
    """Scatter of cell centres coloured by probability; observed fire onsets are ringed. Returns the matplotlib figure."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    shown = axis.scatter(frame.lon, frame.lat, c=frame[column], s=9, cmap="YlOrRd", vmin=0, vmax=max(0.05, float(frame[column].quantile(0.99))))
    onsets = frame[frame.label == 1]
    axis.scatter(onsets.lon, onsets.lat, s=40, facecolors="none", edgecolors="black", linewidths=0.8, label=f"observed fire onsets ({len(onsets)})")
    axis.set_xlabel("longitude"), axis.set_ylabel("latitude"), axis.set_title(title), axis.legend(loc="lower left")
    figure.colorbar(shown, ax=axis, label="probability of a fire starting in the cell")
    figure.tight_layout()
    if path is not None:
        figure.savefig(path, dpi=130)
    return figure
