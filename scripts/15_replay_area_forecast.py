"""Replay the area forecast for days of the 2025 test year from the stored predictions of the five XGBoost seeds (mean = the deployed score), as the live job would have issued it,
and write one file per forecast date: OUT/area_forecast/<date>.json (+ index.json). The areas, calibration and levels are fitted on 2024 (scripts/13_forecast_levels.py), so 2025 is a fair replay.
Run: .venv/bin/python scripts/15_replay_area_forecast.py --out ../website/backend/runtime_data [--dates 2025-03-10:2025-03-20 2025-02-20 2025-07-15 ...]"""
import argparse, json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "website", "forecast", "live"))
import area_forecast as AF  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--dates", nargs="*", default=["2025-02-24", "2025-03-04:2025-03-20", "2025-04-12", "2025-05-20", "2025-07-15", "2025-08-20", "2025-10-20", "2025-12-10"]); a = ap.parse_args()
days = []
for tok in a.dates:
    if ":" in tok: days += [str(d.date()) for d in pd.date_range(*tok.split(":"))]
    else: days.append(tok)
days = sorted(set(days)); levels = AF.load_levels(f"{ROOT}/results/forecast_levels.json"); places = pd.read_parquet(f"{ROOT}/results/named_places.parquet")
cells = pd.read_parquet(f"{ROOT}/live_state_clean/bundle/cells.parquet", columns=["cell_id", "state", "lat", "lon", "forest_land_frac"])
seeds = [f"{ROOT}/runs/xgboost_A_s{k}/predictions/test.parquet" for k in range(5)]
want = sorted({str((pd.Timestamp(d) - pd.Timedelta(days=1)).date()) for d in days} | set(days))          # issue days = forecast date - 1; keep the day before too, for "yesterday"
issue_days = sorted({(pd.Timestamp(d) - pd.Timedelta(days=1)) for d in days} | {(pd.Timestamp(d) - pd.Timedelta(days=2)) for d in days})
frames = []
for k, p in enumerate(seeds):
    f = pd.read_parquet(p, columns=["cell_id", "date", "label_date", "y", "p"]); f = f[pd.to_datetime(f.date).isin(issue_days)]; f["p"] = f.p.astype(np.float32); frames.append(f.reset_index(drop=True))
base = frames[0]; base["score"] = np.mean([f.p.values for f in frames], axis=0); base["date"] = pd.to_datetime(base.date)
outdir = f"{a.out}/area_forecast"; os.makedirs(outdir, exist_ok=True); done = {}; index = []
for fd in sorted({str((d + pd.Timedelta(days=1)).date()) for d in issue_days}):                          # every forecast date we can build, in order, so "yesterday" is available
    d0 = pd.Timestamp(fd) - pd.Timedelta(days=1); g = base[base.date == d0]
    if g.empty: continue
    sc = g.merge(cells, on="cell_id", how="inner")[["cell_id", "state", "lat", "lon", "score", "forest_land_frac"]]; det = g[g.y == 1][["cell_id"]]
    prev = done.get(str((pd.Timestamp(fd) - pd.Timedelta(days=1)).date()))
    out = AF.build(fd, sc, levels, places=places, previous=prev, detected=det, mode="replay", model={"id": "xgboost-nextday-forest", "version": "2026-11", "seeds": 5}, issued_at=f"{d0.date()}T05:00:00+00:00")
    done[fd] = out
    if fd in days: json.dump(out, open(f"{outdir}/{fd}.json", "w"), separators=(",", ":")); index.append({"date": fd, "summary": out["summary"], "bytes": os.path.getsize(f"{outdir}/{fd}.json")})
json.dump({"mode": "replay", "levels": levels["levels"], "dates": index}, open(f"{outdir}/index.json", "w"), indent=1)
print(len(index), "dates;", sum(i["bytes"] for i in index) // 1024, "KB"); [print(i["date"], i["summary"], i["bytes"] // 1024, "KB") for i in index]
