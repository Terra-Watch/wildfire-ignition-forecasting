"""Write the replay bundle that the dashboard's map risk layer reads (website/backend/models/risk) from the final study: the mean of the five xgboost seeds
(the deployed score) on the 2025 test year, per forest cell and forecast day, with the grid and states of the live model bundle.
Only ranks and bands are ever served from it (app/risk_model.py); the scores stay in the file.
Run:  .venv/bin/python scripts/17_risk_replay_bundle.py --out ../website/backend/models/risk"""
import argparse, json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
bundle = json.load(open(f"{ROOT}/live_state_clean/bundle/manifest.json")); cells = pd.read_parquet(f"{ROOT}/live_state_clean/bundle/cells.parquet", columns=["cell_id", "state", "lat", "lon", "forest_land_frac"])
frames = []
for k in range(5):
    f = pd.read_parquet(f"{ROOT}/runs/xgboost_A_s{k}/predictions/test.parquet", columns=["cell_id", "label_date", "p"]); f["p"] = f.p.astype(np.float32); frames.append(f.reset_index(drop=True))
base = frames[0].copy(); base["score"] = np.mean([f.p.values for f in frames], axis=0)
base = base.merge(cells[["cell_id", "state"]], on="cell_id", how="inner")
day = ((pd.to_datetime(base.label_date) - pd.Timestamp("1970-01-01")).dt.days).astype("int32").values
np.savez_compressed(f"{a.out}/replay_2025.npz", cell=base.cell_id.values.astype("int64"), day=day, state=base.state.values.astype("U2"), score=base.score.values.astype("float32"))
cells.rename(columns={"cell_id": "cell"})[["cell", "state", "lon", "lat", "forest_land_frac"]].to_csv(f"{a.out}/cell_centers.csv", index=False)
json.dump({"model_id": bundle["model_id"], "model_version": bundle["model_version"], "mode": "replay", "states": bundle["states"], "bands": bundle["bands"],
           "replay_files": ["replay_2025.npz"], "score_column": "score",
           "note": "the final study's five-seed mean on the 2025 test year, replayed; forecast day = the day the fire is detected"}, open(f"{a.out}/manifest.json", "w"), indent=1)
print(len(base), "rows;", base.label_date.min(), "to", base.label_date.max(), ";", base.state.nunique(), "states;", os.path.getsize(f"{a.out}/replay_2025.npz") // 1024, "KB")
