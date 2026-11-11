"""Build a risk-model bundle from finished study runs (see bundle.py for the layout).
Usage: python make_bundle.py --runs DIR_WITH_s0..sN --out DIR --model-id ID --model-version V [--n-parity 200]
Each run folder needs model.json, manifest.json (study run manifest) and test.parquet; freeze.json supplies the freeze hash.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../website/forecast/live'))  # runtime code lives with the dashboard
import argparse, glob, json, os, shutil
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
DATA = os.environ.get("WILDFIRE_DATA") or f"{ROOT}/wildfire-dataset/dataset"      # the clean data root for the clean bundle
from bundle import sha256, Bundle  # noqa: E402

STATIC = ["forest_fraction", "shrub_fraction", "grass_fraction", "crop_fraction", "elevation", "slope", "population_density", "distance_to_roads_km", "forest_land_frac", "nonforest_land_ring8", "crop_ring8"]
NOTICES = [
    {"source": "NASA FIRMS VIIRS active fire (Suomi NPP, NOAA-20)", "use": "fire history", "terms": "NASA open data; acknowledge NASA FIRMS / LANCE"},
    {"source": "ERA5 and ERA5-Land reanalysis, served by Open-Meteo", "use": "weather", "terms": "Contains modified Copernicus Climate Change Service information (Copernicus licence); Open-Meteo data CC BY 4.0, free API for non-commercial use"},
    {"source": "NOAA GFS 0.25 degree forecast", "use": "forecast weather", "terms": "NOAA/NCEP open data"},
    {"source": "ESA WorldCover 2021", "use": "land cover (training)", "terms": "CC BY 4.0"},
    {"source": "Copernicus DEM GLO-30", "use": "elevation and slope (training)", "terms": "Copernicus licence: free with attribution"},
    {"source": "WorldPop", "use": "population (training)", "terms": "CC BY 4.0"},
    {"source": "OpenStreetMap", "use": "distance to roads (training)", "terms": "ODbL, (c) OpenStreetMap contributors"},
    {"source": "MODIS MOD13A2 NDVI", "use": "vegetation (training; live runs use a seasonal average)", "terms": "NASA LP DAAC open data"},
    {"source": "Forest-boundary layers (community compilation of Indian forest layers)", "use": "forest cells and fire labels", "terms": "published as CC0 by the compiler; not an official release, check before wider use"},
]
BANDS = [{"name": "Very high", "top_share": 0.01}, {"name": "High", "top_share": 0.05}, {"name": "Moderate", "top_share": 0.15}, {"name": "Low", "top_share": 1.0}]

ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model-id", required=True); ap.add_argument("--model-version", required=True); ap.add_argument("--n-parity", type=int, default=200)
ap.add_argument("--freeze", default=f"{ROOT}/wildfire-ignition-forecasting/results_v4/freeze.json"); ap.add_argument("--key", default=os.path.expanduser("~/.terrawatch/keys/risk-model-signing.pem")); ap.add_argument("--key-id", default="terrawatch-risk-2026-11"); a = ap.parse_args()
os.makedirs(f"{a.out}/models", exist_ok=True)
sc = np.load(f"{DATA}/scaler_forest_AG_train.npz", allow_pickle=True); names = [str(n) for n in sc["names"]]
np.savez(f"{a.out}/scaler.npz", mean=sc["mean"], sd=sc["sd"], names=np.array(names))
models, run_ids, val = [], [], []
for k, d in enumerate(sorted(glob.glob(f"{a.runs}/s[0-9]*"))):
    m = json.load(open(f"{d}/manifest.json")); shutil.copy(f"{d}/model.json", f"{a.out}/models/seed_{k}.json")
    models.append({"file": f"models/seed_{k}.json", "trees": int(m["epochs_or_trees"]), "seed": int(m["seed"])}); run_ids.append(m["run_id"]); val.append(m["best_val_auprc"])
CODES = {"Karnataka": "KA", "Tamil Nadu": "TN", "Arunachal Pradesh": "AR", "Assam": "AS", "Himachal Pradesh": "HP", "Tripura": "TR", "Mizoram": "MZ", "Manipur": "MN", "Sikkim": "SK"}  # the website scopes users by two-letter code
cells = pd.read_parquet(f"{DATA}/forest_all_gfs/cells.parquet"); cells = cells[cells.has_weather].copy(); cells["state"] = cells.state.replace(CODES)
assert cells.state.str.len().eq(2).all(), sorted(cells.state.unique())
cube = np.load(f"{DATA}/forest_all_gfs/cube.npy", mmap_mode="r"); last = cube.shape[0] - 1
pos = np.where(pd.read_parquet(f"{DATA}/forest_all_gfs/cells.parquet").has_weather.values)[0]
stat = np.asarray(cube[last][pos][:, [names.index(n) for n in STATIC]])  # static per cell (population: latest year)
out = cells[["cell_id", "state", "lat", "lon"]].reset_index(drop=True)
for j, n in enumerate(STATIC): out[n] = stat[:, j]
out.to_parquet(f"{a.out}/cells.parquet", index=False)
# NDVI has no free live source: the daily job falls back to each cell's mean NDVI by day of year over the training years (flagged in the run)
train = np.arange((pd.Timestamp("2023-12-31") - pd.Timestamp("2019-01-01")).days + 1); doy = pd.date_range("2019-01-01", periods=len(train)).dayofyear.values
nd = np.asarray(cube[train[0]:train[-1] + 1, :, names.index("ndvi")])[:, pos]; ag = np.asarray(cube[train[0]:train[-1] + 1, :, names.index("ndvi_age_days")])[:, pos]
clim = np.stack([np.nanmean(nd[doy == k], axis=0) if (doy == k).any() else np.full(len(pos), np.nan) for k in range(1, 367)], axis=1).astype(np.float32)
np.savez(f"{a.out}/ndvi_clim.npz", ndvi=clim, age=np.float32(np.nanmedian(ag)))
# parity cases: random test-year cell-days; expected = mean of the STORED study predictions of the seeds (an independent reference, not a round trip)
runs = sorted(glob.glob(f"{a.runs}/s[0-9]*"))
st = [pd.read_parquet(f"{d}/test.parquet", columns=["cell_id", "date", "p"]).set_index(["cell_id", "date"]).p.rename(f"p{k}") for k, d in enumerate(runs)]
stored = pd.concat(st, axis=1, join="inner"); idx = stored.sample(a.n_parity, random_state=0).index
cell_pos = pd.Series(np.arange(len(pd.read_parquet(f"{DATA}/forest_all_gfs/cells.parquet"))), index=pd.read_parquet(f"{DATA}/forest_all_gfs/cells.parquet").cell_id.values)
tt = np.array([(pd.Timestamp(d) - pd.Timestamp("2019-01-01")).days for _, d in idx]); cc = cell_pos[[c for c, _ in idx]].values
raw = np.stack([np.asarray(cube[t, c, :], dtype=np.float64) for t, c in zip(tt, cc)]); expected = stored.loc[idx].mean(axis=1).values
manifest = {"model_id": a.model_id, "model_version": a.model_version, "mode": "live", "label": "forest detection on the next UTC day",
            "study_ref": {"runs": run_ids, "freeze_sha256": json.load(open(a.freeze))["sha256"], "validation_auprc_per_seed": val},
            "grid_version": f"cells-{len(out)}", "states": sorted(out.state.unique().tolist()), "channel_names": names, "models": models,
            "calibration": {"method": "none", "note": "rank and band only; scores are not shown"}, "bands": BANDS,
            "min_input_coverage": 0.95, "max_input_age_hours": 36, "licence_review": "unreviewed: written from the sources' published terms, to be confirmed by the owner before any wider release", "licence_notices": NOTICES}
json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=2)
np.savez(f"{a.out}/parity_cases.npz", raw=raw, expected=expected, cells=cc, days=tt)
def write_sums():
    files = sorted(f for f in glob.glob(f"{a.out}/**/*", recursive=True) if os.path.isfile(f) and not f.endswith("SHA256SUMS"))
    open(f"{a.out}/SHA256SUMS", "w").write("".join(f"{sha256(f)}  {os.path.relpath(f, a.out)}\n" for f in files))
write_sums()
import base64
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
kp = os.path.expanduser(a.key)
if not os.path.exists(kp):  # integrity key for bundles built here; the private half never leaves this machine
    os.makedirs(os.path.dirname(kp), exist_ok=True, mode=0o700)
    open(kp, "wb").write(Ed25519PrivateKey.generate().private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())); os.chmod(kp, 0o600)
key = serialization.load_pem_private_key(open(kp, "rb").read(), password=None)
pub = base64.b64encode(key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
json.dump({"algorithm": "Ed25519", "key_id": a.key_id, "public_key": pub, "signature": base64.b64encode(key.sign(open(f"{a.out}/SHA256SUMS", "rb").read())).decode(), "meaning": "build integrity, not a human approval"}, open(f"{a.out}/SIGNATURE.json", "w"), indent=2)
print("signing public key (set RISK_BUNDLE_PUBLIC_KEY to this in the job's environment):", pub)
print("bundle written;", f"golden max diff vs stored study predictions {Bundle(a.out).golden():.2e}", f"| {len(models)} seeds, {len(out)} cells")
