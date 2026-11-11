"""Freeze record: one SHA-256 over the settings of every finished run (model, seed, split, hyperparameters, channels, label, validation score, code hash), written before the blind states are scored.
Usage: python scripts/04_freeze_protocol.py --out DIR --res RES"""
import argparse, glob, hashlib, json, os

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--res", required=True); a = ap.parse_args()
keys = ("model", "variant", "seed", "split", "hp", "channels", "label", "neighbour_features", "n_params", "epochs_or_trees", "best_val_auprc", "script_sha256", "train_rows", "val_rows")
runs = {}
for mp in sorted(glob.glob(f"{a.out}/*/manifest.json")):
    m = json.load(open(mp))
    if m.get("complete") and not m.get("no_training"): runs[m["run_id"]] = {k: m.get(k) for k in keys}
blob = json.dumps(runs, sort_keys=True, default=str).encode(); h = hashlib.sha256(blob).hexdigest()
os.makedirs(a.res, exist_ok=True); json.dump(dict(sha256=h, runs=len(runs), run_ids=sorted(runs)), open(f"{a.res}/freeze.json", "w"), indent=1)
print(h, len(runs))
