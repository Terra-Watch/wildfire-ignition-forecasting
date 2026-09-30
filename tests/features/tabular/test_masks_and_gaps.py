import unittest

import numpy as np
import pandas as pd

from wildfire.features.tabular import gap_days, masks

MS = np.array([18262 + 31 * i for i in range(130)], np.int64)


class GapDays(unittest.TestCase):
    def test_flag_gap_days(self):
        d = np.arange(10)
        s = np.array([100, 90, 0, 0, 40, 100, 0, 5, 100, 100])
        n = np.array([100, 100, 300, 400, 100, 100, 10, 10, 100, 100])
        self.assertEqual(list(gap_days.flag_gap_days(d, s, n, 0.5, 50)), [2, 3, 4])   # both thin (quiet days) are not flagged

    def test_touches_flagged(self):
        f = np.array([100, 101])
        self.assertEqual(list(gap_days.touches_flagged(np.array([99, 100, 103, 104, 105]), f, 3)), [False, True, True, True, False])
        self.assertEqual(list(gap_days.touches_flagged(np.array([5, 6]), np.array([], np.int64), 3)), [False, False])


class Masks(unittest.TestCase):
    def test_masked_weather_cell_makes_all_dependent_columns_nan(self):
        df = pd.DataFrame(dict(t2m=[20.0, np.nan, 25.0], ffmc=[80.0, 85.0, 90.0], rain7=[1.0, 0.0, 2.0],
                               dsr=[3.0, 0.0, 4.0], dmc=[5.0, np.nan, 6.0], other=[1.0, 2.0, 3.0]))
        n = masks.mask_dependent_columns(df, "t2m", ["t2m", "ffmc", "rain7", "dsr", "dmc"])
        self.assertEqual(n, 1)
        self.assertTrue(df.loc[1, ["t2m", "ffmc", "rain7", "dsr", "dmc"]].isna().all())
        self.assertEqual(df.loc[1, "other"], 2.0)
        self.assertEqual(list(df.ffmc.dropna()), [80.0, 90.0])

    def test_same_nan_rows_assertion(self):
        ok = pd.DataFrame(dict(a=[1.0, np.nan, 3.0], b=[4.0, np.nan, 6.0]))
        self.assertEqual(masks.assert_same_nan_rows(ok, ["a", "b"]), 1)
        bad = pd.DataFrame(dict(a=[1.0, np.nan, 3.0], b=[4.0, 5.0, 6.0]))
        with self.assertRaises(AssertionError):
            masks.assert_same_nan_rows(bad, ["a", "b"])

    def test_common_row_mask(self):
        df = pd.DataFrame(dict(x=[1.0, np.nan, 3.0, 4.0], y=[1.0, 2.0, np.nan, 4.0], z=[1.0, 2.0, 3.0, 4.0]))
        sets = {"A": ["x", "z"], "B": ["y", "z"], "C": ["z"]}
        self.assertEqual(list(masks.common_row_mask(df, sets)), [True, False, False, True])
        self.assertEqual(list(masks.common_row_mask(df, sets, ["A"])), [True, False, True, True])

    def test_percentile_rank_is_across_cells_with_average_ties(self):
        r = masks.percentile_rank(np.array([0.0, 0.0, 1.0, 2.0]))
        self.assertTrue(np.allclose(r, [0.375, 0.375, 0.75, 1.0]))

    def test_frozen_rank_has_no_year_signal(self):
        rank = masks.percentile_rank(np.array([0.0, 0.0, 1.0, 2.0, 3.0]))
        col = rank[np.array([0, 3, 1, 4, 2, 0])]          # the same cell in different years gets the same value
        self.assertTrue(np.allclose(col[[0, 5]], col[0]))
