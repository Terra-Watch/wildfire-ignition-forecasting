"""Refuse to run on stale or mismatched inputs."""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from wildfire import config


def assert_consistent_inputs(require_burn: bool = True) -> None:
    """Every static layer has one row per grid cell in grid order and is newer than the grid; the burn file has the
    grid's cell count and per-cell coverage (the older format has a date bug and month-level coverage only)."""
    grid_path = config.data_file("grid")
    folder = grid_path.parent
    grid = pd.read_csv(grid_path, usecols=["cell"])
    n = len(grid)
    errs = []
    for f in sorted(glob.glob(str(folder / "static_*.csv"))):
        c = pd.read_csv(f, usecols=["cell"]).cell.values
        if len(c) != n or not (c == grid.cell.values).all():
            errs.append(f"{os.path.basename(f)}: {len(c)} rows / cell order does not match the grid ({n} cells)")
        if os.path.getmtime(f) < os.path.getmtime(grid_path):
            errs.append(f"{os.path.basename(f)} is older than the grid (stale layer?)")
    bp = config.data_file("burn")
    if bp.exists():
        z = np.load(bp)
        if z["burn_ord"].shape[1] != n:
            errs.append(f"burn file has {z['burn_ord'].shape[1]} cells, grid has {n}")
        if "cell_month_covered" not in z.files:
            errs.append("burn file is in the older format (no per-cell coverage)")
    elif require_burn:
        errs.append("burn file missing")
    if errs:
        raise SystemExit("INPUT GUARD FAILED:\n  " + "\n  ".join(errs))
