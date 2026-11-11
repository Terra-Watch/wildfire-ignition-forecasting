"""Compare Open-Meteo weather with the dataset's ERA5-Land weather channels (the live weather gap, plan section 4.1).

Builds the 11 weather channels from Open-Meteo hourly data with the dataset's own feature code
(wildfire-datasets-v3/scripts/features_v3.py) and compares them with the stored cube on the same cells and days.
Sources: archive (era5_land; wind and precipitation from era5, as era5_land has none) = should reproduce the cube; historical-forecast (best_match) = live-style.
Usage: python openmeteo_probe.py --start 2025-02-01 --end 2025-03-31 --n-cells 40 --out DIR
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../website/forecast/live'))  # runtime code lives with the dashboard
import argparse, os, sys, time
import numpy as np, pandas as pd, httpx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, HERE)
import weather as W  # same code the daily job uses

DATA = f"{ROOT}/wildfire-dataset/dataset/forest_all_gfs"
D0 = pd.Timestamp("2019-01-01")


def channels(frames, days):
    """frames: hourly frames per cell over [start-30d, end]; array (days, cells, 11) for the last `days` days."""
    return np.stack([W.channels(W.daily(df))[-days:] for df in frames], axis=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--n-cells", type=int, default=40); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cells = pd.read_parquet(f"{DATA}/cells.parquet"); cells = cells[cells.has_weather].reset_index().rename(columns={"index": "pos"})
    pick = cells.groupby("state", group_keys=False).apply(lambda d: d.sample(min(len(d), max(2, a.n_cells // cells.state.nunique())), random_state=0)).reset_index(drop=True)
    cube = np.load(f"{DATA}/cube.npy", mmap_mode="r")
    d_start, d_end = pd.Timestamp(a.start), pd.Timestamp(a.end); ndays = (d_end - d_start).days + 1
    i0 = (d_start - D0).days; truth = np.asarray(cube[i0:i0 + ndays][:, pick.pos.values, :11]).astype(np.float64)
    pre = (d_start - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    V = ["temperature_2m", "dewpoint_2m", "wind_speed_10m", "wind_direction_10m", "precipitation", "soil_moisture_0_to_7cm"]
    res = {}
    for src in ("archive", "forecast"):
        frames = [None] * len(pick)
        for b in range(0, len(pick), 5):
            sl = slice(b, b + 5); lats, lons = pick.lat.values[sl].tolist(), pick.lon.values[sl].tolist()
            dfs = W.fetch_batch("archive" if src == "archive" else "historical_forecast", lats, lons, pre, a.end)
            for k, df in enumerate(dfs): frames[b + k] = df
            time.sleep(2)
        res[src] = channels(frames, ndays)
    rows = []
    for src, x in res.items():
        for j, n in enumerate(W.NAMES):
            m = np.isfinite(x[..., j]) & np.isfinite(truth[..., j]); d = x[..., j][m] - truth[..., j][m]
            rows.append(dict(source=src, channel=n, n=int(m.sum()), corr=float(np.corrcoef(x[..., j][m], truth[..., j][m])[0, 1]) if m.sum() > 2 else np.nan,
                             bias=float(d.mean()), mae=float(np.abs(d).mean()), truth_sd=float(truth[..., j][m].std())))
    out = pd.DataFrame(rows); out.to_csv(f"{a.out}/openmeteo_vs_cube.csv", index=False); print(out.round(3).to_string(index=False))
    np.savez_compressed(f"{a.out}/openmeteo_probe.npz", archive=res["archive"], forecast=res["forecast"], truth=truth, cell_id=pick.cell_id.values)


if __name__ == "__main__": main()
