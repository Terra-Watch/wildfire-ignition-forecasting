"""One-off: the detection store of the live job. One row per detection in the bundle's cells since 2019, with the raw measurements the quality classifier needs, its bnd333/bnd777 forest shares,
the industrial flags and p_scar (the classifier score of the dataset build). The fire store (cell_id, date, forest) is then derived from the detections with p_scar >= hide, the rule of the clean dataset.
Usage: python init_detection_store.py --cells BUNDLE/cells.parquet --out STATE_DIR     (writes detections.parquet, fire.parquet; forest_mask.npz is kept)"""
import argparse, json, os
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
V4 = f"{ROOT}/wildfire-dataset"; DATA = f"{V4}/dataset"; A = f"{V4}/audit/false_detections"; MIN_BND = 0.5
ap = argparse.ArgumentParser(); ap.add_argument("--cells", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
ids = set(pd.read_parquet(a.cells).cell_id)
old = pd.read_parquet(f"{DATA}/detections_forest.parquet"); new = pd.read_parquet(f"{DATA}/detections_new.parquet")
att = pd.concat([pd.read_parquet(f"{A}/detection_attributes.parquet"), pd.read_parquet(f"{A}/new/detection_attributes.parquet").assign(rid=lambda x: x.rid + len(old))], ignore_index=True)
ind = pd.concat([pd.read_parquet(f"{A}/industrial_flags.parquet"), pd.read_parquet(f"{A}/new/industrial_flags.parquet").assign(rid=lambda x: x.rid + len(old))], ignore_index=True)
d = pd.concat([old, new], ignore_index=True); d["rid"] = np.arange(len(d))
d["p_scar"] = pd.read_parquet(f"{A}/detection_scores_all.parquet").sort_values("rid").p_scar.values
d = d.merge(att, on="rid", how="left").merge(ind, on="rid", how="left")
for k in ("0.5", "1", "2"): d[f"ind_{k}km"] = d[f"near_industrial_{k}km"].astype(bool)
d["date"] = pd.to_datetime(d.date_utc).dt.tz_localize(None).dt.normalize(); d = d[d.cell_id.isin(ids)]
cols = ["latitude", "longitude", "satellite", "date", "acq_time", "bright_ti4", "bright_ti5", "frp", "scan", "track", "daynight", "confidence", "bnd333", "bnd777", "ind_0.5km", "ind_1km", "ind_2km", "cell_id", "p_scar"]
d = d[cols].reset_index(drop=True); os.makedirs(a.out, exist_ok=True); d.to_parquet(f"{a.out}/detections.parquet", index=False)
hide = json.load(open(f"{A}/classifier_report_v2.json"))["thresholds_from_2019_2023"]["hide_keeps_98pct_of_confirmed"]
k = d[(d.p_scar >= hide) | d.p_scar.isna()]          # a detection without a score (no raw attributes in the dataset) is kept
store = pd.DataFrame({"cell_id": k.cell_id.values, "date": k.date.values, "forest": k.bnd333.values >= MIN_BND}).groupby(["cell_id", "date"], as_index=False).forest.max()
store.to_parquet(f"{a.out}/fire.parquet", index=False)
print(f"detections={len(d)} (scored {int(d.p_scar.notna().sum())}); kept above hide={hide:.4f}: {len(k)}; fire store rows={len(store)}; detections file {os.path.getsize(f'{a.out}/detections.parquet') / 1e6:.0f} MB")
