"""Fit what the area forecast pages need and write results/forecast_levels.json:
  - cell calibration: isotonic map from the deployed score (mean of the five XGBoost seeds) to a chance, fitted on the 2024 validation predictions;
  - area calibration: isotonic map from "chance that at least one cell of the area has a fire detection tomorrow, if cells were independent" to the observed rate, fitted on 2024;
  - the areas: blocks of 0.5 degree (about 55 km, about 10 cells) with their cells, state and centre;
  - "usual for this week" per area: mean daily fire cells and share of days with at least one fire, 2019 to 2023;
  - level cut-offs and a default action table;
  - per-state reliability from the 2025 test year (reported, never used for fitting).
The label is a fire detection inside forest on the next day (new or continuing). Run: .venv/bin/python scripts/13_forecast_levels.py"""
import json, os
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DS = os.path.join(os.path.dirname(ROOT), "terrawatch-datasets", "dataset", "forest_all_gfs")
SEEDS = [f"{ROOT}/runs/xgboost_A_s{k}/predictions" for k in range(5)]
CELLS = pd.read_parquet(f"{ROOT}/live_state_clean/bundle/cells.parquet", columns=["cell_id", "state", "lat", "lon", "forest_land_frac"])
BLOCK_DEG = 0.5
LEVELS = [("none", 0.0), ("moderate", 0.10), ("high", 0.30), ("extreme", 0.60)]           # cut-offs on the calibrated chance that a fire is detected in the area tomorrow
ACTIONS = [  # default action table; a division edits and signs its own. min_chance = the area chance at which the action is suggested.
    {"id": "warn_villages", "min_chance": 0.10, "text": "Warn villages and forest committees: no burning of leaves or fields, no fire in the forest."},
    {"id": "extra_patrol", "min_chance": 0.30, "text": "Add a patrol to the places listed, in the morning."},
    {"id": "crew_ready", "min_chance": 0.60, "text": "Keep a crew and a vehicle ready in the area all day."}]


def mean_seed_predictions(split):
    frames = [pd.read_parquet(f"{p}/{split}.parquet", columns=["cell_id", "date", "y", "p"]) for p in SEEDS]
    base = frames[0].copy()
    for f in frames[1:]:
        assert (f.cell_id.values == base.cell_id.values).all() and (f.date.values == base.date.values).all(), "seed predictions are not aligned"
    base["p"] = np.mean([f.p.values for f in frames], axis=0).astype(np.float32)
    return base


def knots(iso, n=400):
    x, y = iso.f_.x, iso.f_.y                                      # fitted step function: piecewise linear between its points
    if len(x) > n:
        idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int)); x, y = x[idx], y[idx]
    return {"x": [round(float(v), 6) for v in x], "y": [round(float(v), 6) for v in y]}


def with_blocks(df):
    df = df.merge(CELLS[["cell_id", "block", "state_code"]], on="cell_id", how="inner"); return df


CELLS["block"] = (np.floor(CELLS.lat / BLOCK_DEG).astype(int)) * 1000 + np.floor(CELLS.lon / BLOCK_DEG).astype(int); CELLS = CELLS.rename(columns={"state": "state_code"})
val = with_blocks(mean_seed_predictions("validation")); test = with_blocks(mean_seed_predictions("test"))
cell_iso = IsotonicRegression(out_of_bounds="clip").fit(val.p, val.y)
for d in (val, test): d["q"] = cell_iso.predict(d.p)


def area_days(d):
    a = d.groupby(["block", "date"]).agg(any=("y", "max"), fires=("y", "sum"), n=("y", "size"), pany=("q", lambda s: 1 - np.prod(1 - s.values)), expected=("q", "sum")).reset_index(); return a


av, at = area_days(val), area_days(test)
area_iso = IsotonicRegression(out_of_bounds="clip").fit(av.pany, av["any"])
for a in (av, at): a["chance"] = area_iso.predict(a.pany)

# usual for the week, 2019 to 2023 (label dates of the training years)
rows = pd.read_parquet(f"{DS}/rows.parquet", columns=["cell_id", "label_date", "y_forest", "split_t", "valid_primary"]); rows = rows[(rows.split_t == "train") & rows.valid_primary]
rows = rows.merge(CELLS[["cell_id", "block"]], on="cell_id"); rows["week"] = rows.label_date.dt.isocalendar().week.clip(upper=52).astype(int)
daily = rows.groupby(["block", "week", "label_date"]).y_forest.sum().reset_index()
usual = daily.groupby(["block", "week"]).y_forest.agg(fires="mean", any=lambda s: float((s > 0).mean())).reset_index()

blocks = CELLS.groupby("block").agg(lat=("lat", "mean"), lon=("lon", "mean"), cells=("cell_id", "size"), state=("state_code", lambda s: s.value_counts().index[0]), forest=("forest_land_frac", "mean")).reset_index()
blocks["lat0"] = (blocks.block // 1000) * BLOCK_DEG; blocks["lon0"] = (blocks.block % 1000) * BLOCK_DEG
usual_json = {str(b): {str(int(w)): [round(float(f), 3), round(float(a), 3)] for w, f, a in zip(g.week, g.fires, g["any"])} for b, g in usual.groupby("block")}

# how the levels behave
def level_of(c): return np.select([c >= LEVELS[3][1], c >= LEVELS[2][1], c >= LEVELS[1][1]], ["extreme", "high", "moderate"], "none")
report = {"fitted_on": "2024 validation predictions of the five-seed mean", "scored_on": "2025 test predictions (looked at before; reported only)"}
for tag, a in (("validation_2024", av), ("test_2025", at)):
    a = a.assign(level=level_of(a.chance)); tot = a["any"].sum()
    report[tag] = {lv: {"block_days": int((a.level == lv).sum()), "hit_rate": round(float(a[a.level == lv]["any"].mean()), 3) if (a.level == lv).any() else None, "share_of_fire_block_days": round(float(a[a.level == lv]["any"].sum() / tot), 3)} for lv in ("none", "moderate", "high", "extreme")}
    report[tag]["area_ap"] = round(float(average_precision_score(a["any"], a.chance)), 3); report[tag]["base_rate"] = round(float(a["any"].mean()), 3)
state_rows = []
for st, g in test.groupby("state_code"):
    ex, ob = g.q.sum(), g.y.sum(); ap = float(average_precision_score(g.y, g.q)) if g.y.sum() > 0 else None
    state_rows.append({"state": st, "fire_cell_days": int(ob), "predicted": round(float(ex), 1), "ratio": round(float(ex / max(ob, 1)), 2), "cell_ap": round(ap, 3) if ap is not None else None})
for r in state_rows: r["reliable"] = bool(r["cell_ap"] is not None and r["cell_ap"] >= 0.25 and 0.75 <= r["ratio"] <= 1.25 and r["fire_cell_days"] >= 300)
report["states"] = state_rows

out = {"version": "2026-11-a", "label": "a fire detection inside forest on the next day (new or continuing)", "block_deg": BLOCK_DEG, "levels": [{"id": i, "min_chance": c} for i, c in LEVELS], "actions": ACTIONS,
       "cell_calibration": knots(cell_iso), "area_calibration": knots(area_iso), "blocks": [{"id": int(r.block), "lat": round(float(r.lat), 3), "lon": round(float(r.lon), 3), "lat0": float(r.lat0), "lon0": float(r.lon0), "cells": int(r.cells), "state": r.state, "forest": round(float(r.forest), 2)} for r in blocks.itertuples()],
       "usual": usual_json, "state_reliability": {r["state"]: r["reliable"] for r in state_rows}, "report": report}
os.makedirs(f"{ROOT}/results", exist_ok=True); json.dump(out, open(f"{ROOT}/results/forecast_levels.json", "w"), separators=(",", ":"))
print(json.dumps(report, indent=1)); print("blocks", len(blocks), "size KB", os.path.getsize(f"{ROOT}/results/forecast_levels.json") // 1024)
