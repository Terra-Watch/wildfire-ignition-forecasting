"""Fire-history parity: the live code must reproduce the dataset cube and the stored footprint shares."""
import json, os, sys, unittest
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "website", "forecast", "live"))
import fire  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "terrawatch-datasets", "dataset")
CUBE = f"{DATA}/forest_all_gfs"
HAVE = os.path.exists(f"{CUBE}/cube.npy")


@unittest.skipUnless(HAVE, "dataset not on this machine")
class FireParity(unittest.TestCase):
    def test_footprint_matches_stored_bnd333(self):
        det = pd.read_parquet(f"{DATA}/detections_forest.parquet").sample(20000, random_state=1)
        mask = np.load(f"{DATA}/forest_union_0p001.npy", mmap_mode="r")
        mine = fire.footprint_bnd333(det.latitude.values, det.longitude.values, mask)
        self.assertLess(np.abs(mine - det.bnd333.values).max(), 1e-6)

    def test_cell_id_matches_stored(self):
        det = pd.read_parquet(f"{DATA}/detections_forest.parquet").sample(20000, random_state=2)
        self.assertTrue((fire.cell_id(det.latitude.values, det.longitude.values) == det.cell_id.values).mean() > 0.99)

    def test_channels_match_cube(self):
        cells = pd.read_parquet(f"{CUBE}/cells.parquet"); cube = np.load(f"{CUBE}/cube.npy", mmap_mode="r")
        det = pd.concat([pd.read_parquet(f"{DATA}/detections_forest.parquet"), pd.read_parquet(f"{DATA}/detections_new.parquet")], ignore_index=True)
        A = f"{DATA}/../audit/false_detections"; hide = json.load(open(f"{A}/classifier_report_v2.json"))["thresholds_from_2019_2023"]["hide_keeps_98pct_of_confirmed"]
        det["p"] = pd.read_parquet(f"{A}/detection_scores_all.parquet").sort_values("rid").p_scar.values   # the final data drops detections the quality classifier flags
        det = det[(det.p >= hide) & det.cell_id.isin(set(cells.cell_id))]
        store = pd.DataFrame({"cell_id": det.cell_id.values, "date": det.date_utc.dt.normalize().values, "forest": det.bnd333.values >= 0.5}).groupby(["cell_id", "date"], as_index=False).forest.max(); t = 2250
        out = fire.fire_channels(store, cells.cell_id.values, pd.Timestamp("2019-01-01") + pd.Timedelta(days=t))
        names = [str(n) for n in np.load(f"{DATA}/scaler_forest_AG_train.npz", allow_pickle=True)["names"]]
        for k, v in out.items():
            ref = np.asarray(cube[t, :, names.index(k)], dtype=float)
            self.assertLess((v != ref).mean(), 0.005, k)   # the polygon rule is not applied live (see fire.py)


if __name__ == "__main__": unittest.main()
