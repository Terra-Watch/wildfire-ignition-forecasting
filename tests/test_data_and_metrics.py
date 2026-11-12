import os, sys, unittest
import numpy as np
import pytest
from wildfire_forecast import data, metrics

# The dataset is not in git: the tests that read it skip on a machine that does not have it (a CI runner, a fresh clone).
HAVE_DATA = os.path.exists(os.path.join(data.DATA, "forest_all_gfs", "rows.parquet"))
needs_data = unittest.skipUnless(HAVE_DATA, "dataset not on this machine")

class T(unittest.TestCase):
    def test_unseen_guard(self):
        os.environ.pop("WILDFIRE_FINAL", None); self.assertRaises(PermissionError, data.load_unseen)

    @needs_data
    def test_train_has_no_blind_states_and_counts_match_audit(self):
        import pandas as pd
        d = data.load_train(); self.assertFalse({"Uttar Pradesh", "West Bengal", "Kerala"} & set(d.rows.state))
        aud = pd.read_csv(os.path.join(data.DATA, "..", "audit", "G_merged_counts.csv")); aud = aud[aud.state.isin(set(d.rows.state))]
        r = d.rows[d.rows.valid_primary]; got = r.groupby(["state", "split_t"]).y_forest.agg(rows="size", positives="sum").reset_index()
        m = got.merge(aud, on=["state", "split_t"], suffixes=("", "_a")); self.assertEqual(len(m), len(got)); self.assertTrue((m.rows == m.rows_a).all() and (m.positives <= m.positives_a).all())   # removing false detections can only remove positives

    @needs_data
    def test_tabular_matches_cube(self):
        d = data.load_train(); r = d.select("train", ["D2"]).head(50); x = d.tabular(r); s = r.iloc[7]
        ref = np.nan_to_num((np.asarray(d.cube[s.t, s.cell_idx]) - d.mean) / d.sd); self.assertTrue(np.allclose(x[7], ref)); self.assertEqual(x.shape, (50, 41))

    def test_metrics(self):
        y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1.]); p = np.array([.9, .1, .2, .8, .3, .1, .2, .1, .2, .7])
        self.assertAlmostEqual(metrics.auprc(y, p), 1.0); self.assertEqual(metrics.topk(y, p, 0.3), (1.0, 1.0))
        w = np.repeat(np.arange(5), 2); d, lo, hi = metrics.week_bootstrap(y, p, p, w, draws=50); self.assertEqual((d, lo, hi), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(metrics.brier(np.array([1., 0]), np.array([1., 0])), 0.0)

if __name__ == "__main__": unittest.main()


@pytest.mark.skipif(not HAVE_DATA, reason="dataset not on this machine")
def test_gfs_dataset_has_41_channels_and_the_first_33_are_the_base_channels():
    a, b = data.load_train(), data.load_train(gfs=True)
    assert b.cube.shape[:2] == a.cube.shape[:2] and b.cube.shape[2] == 41 and len(b.mean) == 41 and len(b.sd) == 41
    assert np.array_equal(b.mean[:33], a.mean) and np.array_equal(b.sd[:33], a.sd)             # same scaler for the old channels
    r = a.select("train").sample(1000, random_state=0)
    assert np.allclose(b.tabular(r)[:, :33], a.tabular(r), equal_nan=True)                     # same rows, same first 33 numbers
    assert 295 < b.mean[33] < 310 and 5 < b.sd[33] < 9                                           # GFS max temperature scaler is in kelvin, not an accumulation error
