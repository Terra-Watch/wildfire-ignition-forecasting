"""Loaders for the merged forest datasets. Read-only on the dataset folder."""
import os
import numpy as np, pandas as pd

DATA = os.environ.get("WILDFIRE_DATA") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "terrawatch-datasets", "dataset")   # a VM sets WILDFIRE_DATA
GROUPS = ("six", "D1", "D2", "south", "himachal", "northeast")      # leave-one-group-out groups = the `set` column
COLS = ["cell_idx", "t", "date_t", "label_date", "state", "set", "split_t", "y_forest", "y_any", "ambiguous", "valid_primary", "valid_alt", "season_flag", "new_ignition_forest", "had_det_t_forest"]

class Dataset:
    def __init__(self, name):
        self.rows = pd.read_parquet(f"{DATA}/{name}/rows.parquet", columns=COLS); self.cells = pd.read_parquet(f"{DATA}/{name}/cells.parquet")
        self.cube = np.load(f"{DATA}/{name}/cube.npy", mmap_mode="r")      # [days, cells, 33] float32; 41 for the *_gfs datasets (33 + 8 GFS forecast channels)
        z = np.load(f"{DATA}/scaler_forest_{'AG' if name.endswith('_gfs') else 'A'}_train.npz", allow_pickle=True); self.mean = z["mean"].astype(np.float32); self.sd = z["sd"].astype(np.float32)
        assert len(self.mean) == self.cube.shape[2], "scaler and cube have different channel counts"

    def select(self, split=None, groups=None, valid="valid_primary"):
        r = self.rows; m = r[valid].values.copy()
        if split is not None: m &= (r.split_t == split).values
        if groups is not None: m &= r["set"].isin(groups).values
        return r[m]

    def tabular(self, r, chunk=500_000):
        """Scaled features of one cell-day each: [len(r), 33] float32 (NaN filled with 0 = the training mean after scaling)."""
        out = np.empty((len(r), self.cube.shape[2]), np.float32)
        for a in range(0, len(r), chunk):
            s = r.iloc[a:a + chunk]; out[a:a + chunk] = (self.cube[s.t.values, s.cell_idx.values] - self.mean) / self.sd
        return np.nan_to_num(out)

def load_train(gfs=True): return Dataset("forest_all_gfs" if gfs else "forest_all")

def load_unseen(gfs=True):
    """Blind states (Uttar Pradesh, West Bengal, Kerala). Step 6 only: refuses unless WILDFIRE_FINAL=1 is set deliberately."""
    if os.environ.get("WILDFIRE_FINAL") != "1": raise PermissionError("forest_unseen is scored once, after every model is frozen (DECISIONS.md step 6)")
    return Dataset("forest_unseen_gfs" if gfs else "forest_unseen")
