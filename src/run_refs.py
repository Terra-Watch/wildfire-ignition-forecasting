"""The three forecasts that need no training (persistence, climatology, FWI daily-proxy) written as runs, so they go through the same evaluation as the learned models.
Usage: python src/run_refs.py --out DIR [--dataset forest_all_gfs]"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import harness as H, refs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--dataset", default=H.DATASET); a = ap.parse_args()
    cube = H.Cube(a.dataset, [], "cpu", use_torch=False); cube.cube = np.load(f"{H.DATA}/{a.dataset}/cube.npy", mmap_mode="r")
    rows = {s: H.pick_rows(cube, None, s, "y_forest") for s in ("train", "validation", "test")}
    t0 = time.time(); fwi = refs.fwi_for_dataset(cube); print("fwi done", round(time.time() - t0), "s", flush=True)
    score = {"persistence": lambda r: refs.persistence(r), "climatology": lambda r: refs.climatology(rows["train"], r), "fwi": lambda r: fwi[r.t.values.astype(np.int64), r.cell_idx.values.astype(np.int64)]}
    for name, f in score.items():
        d = f"{a.out}/ref_{name}"; os.makedirs(f"{d}/predictions", exist_ok=True)
        for s in ("validation", "test"): H.write_pred(f"{d}/predictions/{s}.parquet", cube, rows[s], f(rows[s]), s, name, 0)
        json.dump(dict(run_id=f"ref_{name}", model=name, variant=name, seed=0, split="A", label="y_forest", train_rows=len(rows["train"]), val_rows=len(rows["validation"]), complete=True, no_training=True,
                       note="FWI (daily-proxy implementation)" if name == "fwi" else name), open(f"{d}/manifest.json", "w"), indent=1)
    os.makedirs(f"{a.out}/refs", exist_ok=True); json.dump(dict(complete=True), open(f"{a.out}/refs/manifest.json", "w"))


if __name__ == "__main__": main()
