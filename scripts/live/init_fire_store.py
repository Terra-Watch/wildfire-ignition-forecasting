"""One-off: build the fire store (cell_id, date, forest) from the dataset's cleaned detections, and the compressed forest mask the live job needs.
Usage: python init_fire_store.py --cells BUNDLE/cells.parquet --out STATE_DIR
"""
import argparse, os
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
DATA = f"{ROOT}/terrawatch-datasets/dataset"; MIN_BND = 0.5
ap = argparse.ArgumentParser(); ap.add_argument("--cells", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
ids = set(pd.read_parquet(a.cells).cell_id)
det = pd.concat([pd.read_parquet(f"{DATA}/detections_forest.parquet"), pd.read_parquet(f"{DATA}/detections_new.parquet")]); det = det[det.cell_id.isin(ids)]
store = pd.DataFrame({"cell_id": det.cell_id.values, "date": det.date_utc.dt.normalize().values, "forest": det.bnd333.values >= MIN_BND}).groupby(["cell_id", "date"], as_index=False).forest.max()
os.makedirs(a.out, exist_ok=True); store.to_parquet(f"{a.out}/fire.parquet", index=False)
np.savez_compressed(f"{a.out}/forest_mask.npz", mask=np.load(f"{DATA}/forest_union_0p001.npy"))
print(f"fire store rows={len(store)}, mask saved ({os.path.getsize(f'{a.out}/forest_mask.npz') / 1e6:.1f} MB)")
