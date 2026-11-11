"""Does swapping the 11 ERA5-Land weather channels for Open-Meteo values change the XGBoost ranking? (plan section 4.1 gate)
Also the golden check: scoring the unchanged cube rows reproduces the stored test predictions.
Usage: python weather_swap_check.py --probe DIR/openmeteo_probe.npz --model DIR_s0 --start 2025-03-01
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../website/forecast/live'))  # runtime code lives with the dashboard
import argparse, json, os
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
DATA = f"{ROOT}/terrawatch-datasets/dataset"
ap = argparse.ArgumentParser(); ap.add_argument("--probe", required=True); ap.add_argument("--model", required=True); ap.add_argument("--start", required=True); a = ap.parse_args()
z = np.load(a.probe); truth_w, arch, fcst, cid = z["truth"], z["archive"], z["forecast"], z["cell_id"]
nd = truth_w.shape[0]
sc = np.load(f"{DATA}/scaler_forest_AG_train.npz", allow_pickle=True); mean, sd = sc["mean"].astype(np.float64), sc["sd"].astype(np.float64)
cells = pd.read_parquet(f"{DATA}/forest_all_gfs/cells.parquet"); pos = pd.Series(np.arange(len(cells)), index=cells.cell_id.values)[cid].values
cube = np.load(f"{DATA}/forest_all_gfs/cube.npy", mmap_mode="r"); i0 = (pd.Timestamp(a.start) - pd.Timestamp("2019-01-01")).days
raw = np.asarray(cube[i0:i0 + nd][:, pos, :], dtype=np.float64)
bst = xgb.Booster(); bst.load_model(f"{a.model}/model.json"); n_trees = json.load(open(f"{a.model}/manifest.json"))["epochs_or_trees"]
def score(x):
    s = (x - mean) / sd; s[~np.isfinite(s)] = 0
    return bst.predict(xgb.DMatrix(s.reshape(-1, s.shape[-1]).astype(np.float32)), iteration_range=(0, n_trees)).reshape(x.shape[:2])
variants = {"truth": raw, "archive": raw.copy(), "forecast": raw.copy()}
variants["archive"][..., :11] = arch; variants["forecast"][..., :11] = fcst
P = {k: score(v) for k, v in variants.items()}
# golden: stored predictions (rows keyed by day t = date, cell_id)
st = pd.read_parquet(f"{a.model}/test.parquet", columns=["cell_id", "date", "p"]); st["date"] = pd.to_datetime(st["date"])
days = pd.date_range(a.start, periods=nd); m = st.merge(pd.DataFrame({"cell_id": np.repeat(cid, nd), "date": np.tile(days, len(cid)), "mine": P["truth"].T.reshape(-1)}), on=["cell_id", "date"])
print(f"golden: {len(m)} cell-days matched, max |stored - mine| = {np.abs(m.p - m.mine).max():.2e}")
for k in ("archive", "forecast"):
    rho = [spearmanr(P["truth"][d], P[k][d])[0] for d in range(nd)]
    k1 = max(1, int(round(0.05 * len(cid))))
    ov = [len(set(np.argsort(-P["truth"][d])[:k1]) & set(np.argsort(-P[k][d])[:k1])) / k1 for d in range(nd)]
    print(f"{k:9s} per-day Spearman vs ERA5-Land inputs: mean {np.nanmean(rho):.3f} min {np.nanmin(rho):.3f} | top-5% overlap mean {np.mean(ov):.2f} ({len(cid)} cells/day)")
