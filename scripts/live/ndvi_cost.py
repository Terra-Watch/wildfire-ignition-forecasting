"""What does the NDVI fallback cost? Score the 2025 test rows with the real NDVI and with the live job's fallback
(each cell's mean NDVI for that day of year over 2019-2023, median NDVI age) and compare against the same labels.
Usage: python ndvi_cost.py --bundle B --out CSV
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../website/forecast/live'))  # runtime code lives with the dashboard
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bundle import Bundle  # noqa: E402
ROOT = os.path.abspath(os.path.join(HERE, "../../..")); D = f"{ROOT}/wildfire-dataset/dataset/forest_all_gfs"

ap = argparse.ArgumentParser(); ap.add_argument("--bundle", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
b = Bundle(a.bundle); names = b.names; iN, iA = names.index("ndvi"), names.index("ndvi_age_days")
clim = np.load(f"{a.bundle}/ndvi_clim.npz"); cl, age = clim["ndvi"], float(clim["age"])
cube = np.load(f"{D}/cube.npy", mmap_mode="r"); rows = pd.read_parquet(f"{D}/rows.parquet", columns=["cell_id", "t", "date_t", "y_forest", "valid_primary", "split_t"])
rows = rows[(rows.split_t == "test") & rows.valid_primary].reset_index(drop=True)
ps, py, pf, doy, state = [], [], [], [], []
cells = pd.read_parquet(f"{D}/cells.parquet"); hw = np.where(cells.has_weather.values)[0]; cmap = {c: i for i, c in enumerate(hw)}
cpos = pd.Series(np.arange(len(cells)), index=cells.cell_id.values); rows["cp"] = cpos[rows.cell_id.values].values
for t0 in range(int(rows.t.min()), int(rows.t.max()) + 1, 20):               # chunks of 20 days
    r = rows[(rows.t >= t0) & (rows.t < t0 + 20)]
    if r.empty: continue
    x = np.stack([np.asarray(cube[int(t), int(c), :], dtype=np.float64) for t, c in zip(r.t.values, r.cp.values)])
    y = r.y_forest.values; x2 = x.copy(); d = pd.to_datetime(r.date_t).dt.dayofyear.values.clip(max=366) - 1
    x2[:, iN] = cl[[cmap[int(c)] for c in r.cp.values], d]; x2[:, iA] = age
    keep = np.isfinite(x2[:, iN]); ps.append(b.score(x[keep])); pf.append(b.score(x2[keep])); py.append(y[keep]); print(t0, flush=True, end=" ")
ps, pf, py = np.concatenate(ps), np.concatenate(pf), np.concatenate(py)
def topk(p, y, q): k = int(q * len(p)); return float(y[np.argsort(-p)[:k]].sum() / y.sum())
res = pd.DataFrame([{"input": n, "rows": len(py), "auprc": average_precision_score(py, p), "auroc": roc_auc_score(py, p), "recall_top1pct": topk(p, py, .01), "recall_top5pct": topk(p, py, .05), "recall_top10pct": topk(p, py, .10)} for n, p in (("real NDVI", ps), ("seasonal-average NDVI", pf))])
from scipy.stats import spearmanr
print("\n", res.round(4).to_string(index=False), "\nSpearman between the two score sets:", round(spearmanr(ps[::50], pf[::50])[0], 4))
res.to_csv(a.out, index=False)
