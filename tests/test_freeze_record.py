"""The frozen results can be reproduced: the SHA-256 in results/freeze.json is recomputed from the run manifests (same keys and rule as src/wildfire_forecast/freeze.py)."""
import glob, hashlib, json, os

RES = os.path.join(os.path.dirname(__file__), "..", "results")
KEYS = ("model", "variant", "seed", "split", "hp", "channels", "label", "neighbour_features", "n_params", "epochs_or_trees", "best_val_auprc", "script_sha256", "train_rows", "val_rows")


def recompute():
    runs = {}
    for mp in sorted(glob.glob(os.path.join(RES, "run_manifests", "*", "manifest.json"))):
        m = json.load(open(mp))
        if m.get("complete") and not m.get("no_training"): runs[m["run_id"]] = {k: m.get(k) for k in KEYS}
    return runs, hashlib.sha256(json.dumps(runs, sort_keys=True, default=str).encode()).hexdigest()


def test_freeze_hash_and_run_list_reproduce():
    frozen = json.load(open(os.path.join(RES, "freeze.json")))
    runs, h = recompute()
    assert sorted(runs) == frozen["run_ids"] and len(runs) == frozen["runs"]
    assert h == frozen["sha256"]


def test_a_changed_run_changes_the_hash():
    runs, h = recompute(); first = sorted(runs)[0]; runs[first]["best_val_auprc"] = (runs[first]["best_val_auprc"] or 0) + 1e-6
    assert hashlib.sha256(json.dumps(runs, sort_keys=True, default=str).encode()).hexdigest() != h
