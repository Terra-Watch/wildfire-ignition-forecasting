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

# Relative economic value (cost-loss model): 0 = no better than always or never acting, 1 = perfect.
def rev(y, s, a):
    act = s >= a; H = (act & (y == 1)).sum(); F = (act & (y == 0)).sum(); M = ((~act) & (y == 1)).sum(); o = y.mean()
    e_f = (a * (H + F) + M) / len(y); e_c = min(a, o); return (e_c - e_f) / (e_c - a * o)
print("\nRelative value at cost-loss ratios (calibrated chance):", {a: round(rev(t.y.values, t.q.values, a), 3) for a in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)})
# Area level: blocks of 0.5 degree (about a forest range or a small division). Chance of at least one fire tomorrow, assuming independent cells.
from sklearn.metrics import average_precision_score, roc_auc_score
cid = pd.read_parquet(f"{a.run}/predictions/test.parquet", columns=["cell_id"]).cell_id.values
t["blk"] = (np.floor((cid // 1000) / 10.0 / 0.5)).astype(int) * 1000 + (np.floor((cid % 1000) / 10.0 / 0.5)).astype(int)
A = t.groupby(["blk", "date"]).agg(obs=("y", "sum"), pany=("q", lambda s: 1 - np.prod(1 - s.values))).reset_index(); A["any"] = (A.obs > 0).astype(int)
print(f"Area level (0.5 degree blocks): at least one fire tomorrow, prevalence {A['any'].mean():.3f}, AUROC {roc_auc_score(A['any'], A.pany):.3f}, AP {average_precision_score(A['any'], A.pany):.3f}")
