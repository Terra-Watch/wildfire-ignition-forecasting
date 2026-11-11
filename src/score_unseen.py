"""Blind scoring on the unseen states (Uttar Pradesh, West Bengal, Kerala), 2025. Run once, after every model is frozen (config hash in RUNLOG.md).
Refuses to run unless WILDFIRE_FINAL=1. Reads the saved model files of every top-level run with split A and writes predictions/blind.parquet next to them;
also scores the three no-training forecasts. Climatology on unseen cells has no cell history, so it is the training share of the calendar month of the label day.
Usage: WILDFIRE_FINAL=1 python src/score_unseen.py --out DIR [--runs id1,id2]"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import harness as H, refs

NEURAL = ("mlp", "lstm", "cnn", "convlstm", "c1", "c2")


def blind_rows(cube):
    r = cube.rows; r = r[(r.split_t == "blind_2025").values & r.valid_primary.values].copy(); r["y"] = r.y_forest.astype(np.int8); return r.reset_index(drop=True)


def load_predictor(d, man, cube_for):
    m = man["model"]; nb = man.get("neighbour_features", False)
    if m in NEURAL:
        import torch; from src import nets                      # torch only here: with XGBoost in one process it crashes on a Mac
        z = torch.load(f"{d}/model.pt", map_location="cpu"); cube = cube_for(z["channels"], True); cube.nbfeat = z.get("nbfeat", False); net = nets.build(m, z["F"], **{k: z["cfg"][k] for k in ("hid", "drop", "rounds", "cell") if k in z["cfg"]}); net.load_state_dict(z["state"])
        dev = "cuda" if torch.cuda.is_available() else "cpu"; net.to(dev); cube = cube_for(z["channels"], True, dev); cube.nbfeat = z.get("nbfeat", False)
        return cube, lambda rs: H.predict_net(net, cube, z["cfg"]["view"], rs, 8192, torch.device(dev), dev == "cuda")
    cube = cube_for(man["channels"], False)
    if m == "xgboost":
        import xgboost as xgb
        b = xgb.Booster(); b.load_model(f"{d}/model.json"); n = int(man["epochs_or_trees"]); return cube, lambda rs: b.predict(xgb.DMatrix(H.tab_matrix(cube, rs, nb)), iteration_range=(0, n))
    import joblib
    mdl = joblib.load(f"{d}/model.joblib"); return cube, lambda rs: mdl.predict_proba(H.tab_matrix(cube, rs, nb))[:, 1]


def main():
    assert os.environ.get("WILDFIRE_FINAL") == "1", "blind scoring only after the models are frozen: set WILDFIRE_FINAL=1"
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--runs", default=""); a = ap.parse_args()
    day = lambda s: (pd.Timestamp(s) - pd.Timestamp("2019-01-01")).days; cache = {}
    def cube_for(channels, torch_, dev="cpu"):
        k = (tuple(channels), torch_, dev)
        if k not in cache: cache.clear(); cache[k] = H.Cube("forest_unseen_gfs", channels, dev, t_range=(day("2024-12-15"), day("2025-12-31")), use_torch=torch_)
        return cache[k]
    # no-training forecasts
    cube = cube_for(["doy_sin"], False); rs = blind_rows(cube); raw = np.load(f"{H.DATA}/forest_unseen_gfs/cube.npy", mmap_mode="r")
    class D: pass
    ds = D(); ds.cube = raw; ds.scaler_path = cube.scaler_path
    tr = H.pick_rows(H.Cube(H.DATASET, [], "cpu", t_range=(0, 10), use_torch=False), None, "train", "y_forest")
    mo = lambda r: pd.to_datetime(r.label_date).dt.month.values; share = tr.assign(m=mo(tr)).groupby("m").y.mean()
    fw = refs.fwi_for_dataset(ds)
    for name, p in (("persistence", refs.persistence(rs)), ("climatology", share.reindex(mo(rs)).values.astype(np.float32)), ("fwi", fw[rs.t.values.astype(np.int64), rs.cell_idx.values.astype(np.int64)])):
        d = f"{a.out}/ref_{name}"; os.makedirs(f"{d}/predictions", exist_ok=True); H.write_pred(f"{d}/predictions/blind.parquet", cube, rs, p, "blind", name, 0)
    want = set(a.runs.split(",")) - {""}
    for mp in sorted(glob.glob(f"{a.out}/*/manifest.json")):
        d = os.path.dirname(mp); man = json.load(open(mp)); rid = man["run_id"]
        if man.get("no_training") or man.get("split") != "A" or (want and rid not in want) or os.path.exists(f"{d}/predictions/blind.parquet") or man.get("label") == "y_any": continue
        cube, pr = load_predictor(d, man, cube_for); rs = blind_rows(cube); H.write_pred(f"{d}/predictions/blind.parquet", cube, rs, pr(rs), "blind", man["model"], man["seed"]); print("blind", rid, len(rs), flush=True)


if __name__ == "__main__": main()
