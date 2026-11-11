"""Risk map of one test day: predicted probability (isotonic-calibrated on 2024, seed mean) and the forest detections observed the next day, for the main XGBoost model.
The day is the 2025 label day with the most detections unless --date is given. usage: python scripts/10_risk_map.py --runs DIR --out results_v4/figures [--date 2025-03-15]"""
import argparse, glob, os
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--date", default=""); ap.add_argument("--group", default="xgboost_A"); a = ap.parse_args()
D = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "..", "terrawatch-datasets", "dataset", "forest_all_gfs"); cells = pd.read_parquet(f"{D}/cells.parquet")
dirs = sorted(glob.glob(f"{a.runs}/{a.group}_s[0-9]")); tp, vp = [], []
for d in dirs:
    v = pd.read_parquet(f"{d}/predictions/validation.parquet", columns=["y", "p"]); t = pd.read_parquet(f"{d}/predictions/test.parquet")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(v.p.values, v.y.values); t["pc"] = iso.predict(t.p.values); tp.append(t)
T = tp[0][["cell_idx", "label_date", "y"]].copy(); T["p"] = np.mean([t.pc.values for t in tp], 0)
day = a.date or T.groupby("label_date").y.sum().idxmax(); X = T[T.label_date == day].merge(cells[["cell_idx", "lat", "lon", "state"]], on="cell_idx")
os.makedirs(a.out, exist_ok=True); X.to_csv(f"{a.out}/risk_map_{str(day)[:10]}.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True)
s = ax[0].scatter(X.lon, X.lat, c=X.p, s=7, cmap="YlOrRd", vmin=0, vmax=max(0.3, X.p.quantile(0.995))); ax[0].set_title(f"Predicted probability of a forest detection, {str(day)[:10]}"); plt.colorbar(s, ax=ax[0], shrink=0.7, label="calibrated probability")
ax[1].scatter(X.lon, X.lat, c="#d9d9d9", s=7); h = X[X.y == 1]; ax[1].scatter(h.lon, h.lat, c="#b2182b", s=14, label=f"detections ({len(h)})"); ax[1].legend(loc="lower left"); ax[1].set_title("Observed forest detections the next day")
for x_ in ax: x_.set_xlabel("longitude"); x_.set_aspect("equal")
ax[0].set_ylabel("latitude"); plt.tight_layout(); plt.savefig(f"{a.out}/risk_map_{str(day)[:10]}.png", dpi=150)
top = X.sort_values("p", ascending=False); k = max(1, int(0.05 * len(X))); print(day, "cells", len(X), "detections", int(X.y.sum()), "share of detections in top 5% of cells:", round(float(top.y.iloc[:k].sum() / max(1, X.y.sum())), 3))
