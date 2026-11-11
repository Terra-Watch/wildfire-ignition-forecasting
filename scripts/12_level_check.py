"""Check for the forecast page design (docs/20): how the deployed relative bands (top 1, 5, 15 % of all cells each day) and a calibrated chance with fixed levels behave by month and by state.
Calibration: isotonic regression fitted on the 2024 validation predictions, applied to the 2025 test predictions of one run (the test year has been looked at before).
Run: .venv/bin/python scripts/12_level_check.py [--run runs/xgboost_A_s0]"""
import argparse
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

ap = argparse.ArgumentParser(); ap.add_argument("--run", default="runs/xgboost_A_s0"); a = ap.parse_args()
cols = ["date", "state", "y", "p", "had_det_t"]
v = pd.read_parquet(f"{a.run}/predictions/validation.parquet", columns=cols); t = pd.read_parquet(f"{a.run}/predictions/test.parquet", columns=cols)
t["q"] = IsotonicRegression(out_of_bounds="clip").fit(v.p, v.y).predict(t.p)
t["r"] = t.groupby("date").p.rank(ascending=False, pct=True)
t["band"] = np.select([t.r <= 0.01, t.r <= 0.05, t.r <= 0.15], ["Very high", "High", "Moderate"], "Low")
t["m"] = pd.to_datetime(t.date).dt.month
print("Deployed relative bands, by month: fires, hit rate of Very high, share of fires in Very high + High, share of fires in Low")
print(t.groupby("m").apply(lambda d: pd.Series({"fires": int(d.y.sum()), "hit_VH": d[d.band == "Very high"].y.mean(), "fires_in_VH_H": d[d.band.isin(["Very high", "High"])].y.sum() / max(d.y.sum(), 1), "fires_in_Low": d[d.band == "Low"].y.sum() / max(d.y.sum(), 1)})).round(3).to_string())
print("\nCalibrated chance with fixed levels, by month: cells flagged per day at >= 10 % and >= 30 %, their hit rates, share of fires caught at >= 10 %")
print(t.groupby("m").apply(lambda d: pd.Series({"cells_ge10": (d.q >= 0.10).sum() / d.date.nunique(), "hit_ge10": d[d.q >= 0.10].y.mean(), "cells_ge30": (d.q >= 0.30).sum() / d.date.nunique(), "hit_ge30": d[d.q >= 0.30].y.mean(), "fires_ge10": d[d.q >= 0.10].y.sum() / max(d.y.sum(), 1)})).round(3).to_string())
print("\nCalibration by state: predicted over observed fire cell-days")
print(t.groupby("state").apply(lambda d: pd.Series({"fires": int(d.y.sum()), "predicted": d.q.sum(), "ratio": d.q.sum() / max(d.y.sum(), 1)})).round(3).sort_values("ratio").to_string())
dd = t.groupby(["state", "date"]).agg(obs=("y", "sum"), exp=("q", "sum")).reset_index()
print("\nDaily expected vs observed fire cells per state, median correlation over states:", round(dd.groupby("state").apply(lambda d: d.obs.corr(d.exp)).median(), 3))
print("New-fire cell-days (no detection in the cell on day t):", int(t[t.had_det_t == 0].y.sum()), "of", int(t.y.sum()))
