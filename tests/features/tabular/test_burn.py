import datetime as dt
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wildfire.features.tabular import burn, climatology, gap_days

DEFAULTS = dict(min_frac=1.0, lookback=3652, cap=10.0, archive_start=11262, censor=True, lag=1)


def bf(*args, **kw):
    return burn.burn_features(*args, **{**DEFAULTS, **kw})


def climatology_fn(cell_on, day_on, flagged, row_cells, row_days, ncell):
    return climatology.cell_week_climatology(cell_on, day_on, flagged, row_cells, row_days, ncell, first_year=2019, years=7, week_cap=53)


def day(y, m, d):
    return (dt.date(y, m, d) - dt.date(1970, 1, 1)).days


def months(y0, m0, n):
    out = []
    y, m = y0, m0
    for _ in range(n):
        out.append(day(y, m, 1))
        y, m = y + (m == 12), m % 12 + 1
    return np.array(out, np.int64)


MS = months(2015, 1, 120)
ME = burn.month_end_days(MS)   # 2015-01 .. 2024-12


def idx(y, m):
    return int(np.where(MS == day(y, m, 1))[0][0])


class Burn(unittest.TestCase):
    def feats(self, D, o9, o1, covered_cell, covered_nb=None, **kw):
        kw.setdefault("censor", False)                                     # these tests exercise the strict (--burn-strict) semantics of bf_years_since; censoring is tested in YearsSinceCensoring
        return bf(np.array([D]), np.array(o9, np.int64), np.array(o1, np.int64), covered_nb if covered_nb is not None else covered_cell, covered_cell, MS, ME, **kw)
    def test_a_missing_month_gives_nan_not_zero(self):
        cov = np.ones(len(MS)); cov[idx(2020, 6)] = 0                          # June 2020 missing, no burns recorded anywhere
        D = day(2021, 3, 1); f = self.feats(D, [], [], cov)
        self.assertTrue(np.isnan(f["bf_log_n10_nbhd"][0]), "count of burns must be NaN when a window month is unobserved (0 would claim 'no burns')")
        self.assertTrue(np.isnan(f["bf_years_since"][0]))
        self.assertLess(f["burn_obs_frac"][0], 1.0)
        full = self.feats(D, [], [], np.ones(len(MS))); self.assertEqual(full["bf_log_n10_nbhd"][0], 0.0); self.assertEqual(full["burn_obs_frac"][0], 1.0)
    def test_a2_missing_month_outside_window_is_harmless(self):
        cov = np.ones(len(MS)); cov[idx(2015, 2)] = 0                          # Feb 2015 is before the look-back window (starts 2015-06-03) of a 2025-06-01 row
        f = self.feats(day(2025, 6, 1), [], [], cov); self.assertEqual(f["bf_log_n10_nbhd"][0], 0.0); self.assertEqual(f["burn_obs_frac"][0], 1.0)
        g = self.feats(day(2024, 12, 1), [], [], cov); self.assertTrue(np.isnan(g["bf_log_n10_nbhd"][0]), "same gap IS inside the window of a 2024-12-01 row")
    def test_a3_burn_after_gap_keeps_years_since(self):
        cov = np.ones(len(MS)); cov[idx(2019, 9)] = 0; b = day(2022, 5, 10)    # gap is BEFORE the last observed burn: years-since is still determined
        f = self.feats(day(2023, 5, 10), [b], [b], cov); self.assertAlmostEqual(f["bf_years_since"][0], 365 / 365.25, places=6)
        cov2 = cov.copy(); cov2[idx(2022, 11)] = 0                             # gap AFTER the last burn: a later burn cannot be excluded
        self.assertTrue(np.isnan(self.feats(day(2023, 5, 10), [b], [b], cov2)["bf_years_since"][0]))
    def test_a4_values_unchanged_when_fully_observed(self):
        cov = np.ones(len(MS)); b1, b2 = day(2018, 3, 3), day(2021, 4, 4); D = day(2022, 1, 1)
        f = self.feats(D, [b1, b2], [b2], cov); self.assertAlmostEqual(f["bf_log_n10_nbhd"][0], np.log1p(2)); self.assertAlmostEqual(f["bf_years_since"][0], (D - b2) / 365.25)
    def test_a5_min_frac_option(self):
        cov = np.ones(len(MS)); cov[idx(2021, 1)] = 0; D = day(2022, 1, 1)
        self.assertTrue(np.isnan(self.feats(D, [], [], cov, min_frac=1.0)["bf_log_n10_nbhd"][0]))
        f = self.feats(D, [], [], cov, min_frac=0.95); self.assertEqual(f["bf_log_n10_nbhd"][0], 0.0); self.assertGreater(f["burn_obs_frac"][0], 0.95)

class Window(unittest.TestCase):
    def test_b_window_coverage(self):
        cov = np.ones(len(MS)); cov[idx(2020, 2)] = 0                          # Feb 2020 = 29 days unobserved
        lo, hi = day(2020, 1, 1), day(2020, 4, 1)                              # Jan(31)+Feb(29)+Mar(31) = 91 days
        self.assertAlmostEqual(burn.window_coverage(cov, MS, ME, [lo], [hi])[0], 62 / 91)
        self.assertEqual(burn.window_coverage(np.ones(len(MS)), MS, ME, [lo], [hi])[0], 1.0)
        self.assertEqual(burn.window_coverage(cov, MS, ME, [day(2020, 3, 1)], [day(2020, 3, 1)])[0], 1.0, "empty window counts as observed")
        half = np.full(len(MS), 0.5); self.assertAlmostEqual(burn.window_coverage(half, MS, ME, [lo], [hi])[0], 0.5)   # fractional neighbourhood coverage
        self.assertAlmostEqual(burn.window_coverage(cov, MS, ME, [day(2020, 2, 15)], [day(2020, 3, 5)])[0], 4 / 19)     # 15 unobserved Feb days (15..29) + 4 observed Mar days
    def test_b2_month_end_days(self):
        self.assertEqual(ME[idx(2020, 2)] - MS[idx(2020, 2)], 29); self.assertEqual(ME[idx(2019, 2)] - MS[idx(2019, 2)], 28); self.assertEqual(ME[idx(2020, 12)], day(2021, 1, 1))

class Dates(unittest.TestCase):
    def test_c_dates_inside_month(self):
        b = np.zeros((len(MS), 3), np.int32); cov = np.ones((len(MS), 3), bool)
        for i in (idx(2018, 3), idx(2022, 12), idx(2020, 2)): b[i, 0] = MS[i]; b[i, 1] = ME[i] - 1                # first and last day of each month
        self.assertEqual(burn.check_burn_dates_in_month(b, cov, MS, ME), 0)
        bad = b.copy(); bad[idx(2018, 3), 2] = day(2018, 4, 1); self.assertEqual(burn.check_burn_dates_in_month(bad, cov, MS, ME), 1)          # day-of-year bug pattern: date spills into the next month
        old_format = b.copy(); old_format[idx(2022, 12), 2] = MS[idx(2022, 12)] + 334; self.assertEqual(burn.check_burn_dates_in_month(old_format, cov, MS, ME), 1)     # doy 335 added to a month start
        unc = cov.copy(); unc[idx(2018, 3), 0] = False; self.assertGreaterEqual(burn.check_burn_dates_in_month(b, unc, MS, ME), 1, "a burn in an uncovered cell-month is inconsistent")

class SaveLoad(unittest.TestCase):
    def test_d_flags_survive_save_load(self):
        M, NC = len(MS), 4; cov = np.ones((M, NC), bool); cov[idx(2023, 9)] = False; cov[idx(2024, 7)] = False; cov[idx(2025 - 1, 1), 2] = False if False else True; cov[idx(2020, 5), 3] = False
        status = np.array(["ok"] * M, dtype="U16"); status[idx(2023, 9)] = "source_missing"; status[idx(2024, 7)] = "source_missing"; b = np.zeros((M, NC), np.int32); b[idx(2019, 4), 1] = day(2019, 4, 9)
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "burn_cells.npz"; np.savez_compressed(p, burn_ord=b, fetched=cov.all(1), month_start=MS.astype(np.int32), cell_month_covered=cov, valid_month=cov.all(1), month_status=status)
            L = burn.load_burn(p); self.assertFalse(L["legacy"]); self.assertTrue((L["covered"] == cov).all()); self.assertTrue((L["month_end"] == ME).all()); self.assertEqual(list(L["month_status"]), list(status))
            self.assertEqual(int((~L["covered"].all(1)).sum()), 3); self.assertEqual(L["burn_ord"].dtype, np.int32)
    def test_d2_legacy_file_is_flagged(self):
        M, NC = len(MS), 2
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "old.npz"; f = np.ones(M, bool); f[5] = False; np.savez_compressed(p, burn_ord=np.zeros((M, NC), np.int32), fetched=f, month_start=MS.astype(np.int32)); L = burn.load_burn(p)
            self.assertTrue(L["legacy"]); self.assertFalse(L["covered"][5].any()); self.assertTrue(L["covered"][4].all())

class YearsSinceCensoring(unittest.TestCase):
    def test_censored_lower_bound_never_nan(self):
        cov = np.ones(len(MS)); cov[idx(2024, 7)] = 0; b = day(2022, 5, 10); D = day(2024, 11, 1)                # gap AFTER the last observed burn
        c = bf(np.array([D]), np.array([b], np.int64), np.array([b], np.int64), cov, cov, MS, ME, min_frac=0.95, censor=True)
        self.assertAlmostEqual(c["bf_years_since"][0], (D - b) / 365.25); self.assertTrue(c["bf_ys_censored"][0]); self.assertLess(c["bf_ys_obs_frac"][0], 1.0)   # lower bound + flag
        s = bf(np.array([D]), np.array([b], np.int64), np.array([b], np.int64), cov, cov, MS, ME, min_frac=1.0, censor=False); self.assertTrue(np.isnan(s["bf_years_since"][0]))            # --burn-strict
    def test_uncensored_when_fully_observed(self):
        cov = np.ones(len(MS)); b = day(2022, 5, 10); D = day(2023, 5, 10); c = bf(np.array([D]), np.array([b], np.int64), np.array([b], np.int64), cov, cov, MS, ME)
        self.assertFalse(c["bf_ys_censored"][0]); self.assertEqual(c["bf_ys_obs_frac"][0], 1.0); self.assertAlmostEqual(c["bf_years_since"][0], 365 / 365.25)
    def test_no_burn_after_gap_is_lower_bound_capped(self):
        cov = np.ones(len(MS)); cov[idx(2024, 7)] = 0; D = day(2024, 12, 1); c = bf(np.array([D]), np.array([], np.int64), np.array([], np.int64), cov, cov, MS, ME)
        self.assertEqual(c["bf_years_since"][0], 10.0); self.assertTrue(c["bf_ys_censored"][0]); self.assertFalse(np.isnan(c["bf_years_since"][0]))

class Threshold(unittest.TestCase):
    def frac_case(self, n_missing):
        cov = np.ones(len(MS)); D = day(2024, 12, 1)                            # window [2014-12-03, 2024-11-30): inside the 2015-01..2024-12 array
        for i in range(n_missing): cov[idx(2016, 1) + 3 * i] = 0                # n whole months missing, well inside the window
        f = bf(np.array([D]), np.array([], np.int64), np.array([], np.int64), cov, cov, MS, ME, min_frac=1.0); return cov, D, f["burn_obs_frac"][0]
    def test_threshold_behaviour(self):
        for n_missing, exp_ok in ((0, {1.0: True, 0.97: True, 0.95: True}), (3, {1.0: False, 0.97: True, 0.95: True}), (4, {1.0: False, 0.97: False, 0.95: True}), (7, {1.0: False, 0.97: False, 0.95: False})):
            cov, D, frac = self.frac_case(n_missing)
            for thr, ok in exp_ok.items():
                f = bf(np.array([D]), np.array([], np.int64), np.array([], np.int64), cov, cov, MS, ME, min_frac=thr)
                self.assertEqual(not np.isnan(f["bf_log_n10_nbhd"][0]), ok, f"{n_missing} missing months (obs {frac:.4f}) at min_obs_frac={thr}")
                self.assertAlmostEqual(f["burn_obs_frac"][0], frac)              # burn_obs_frac is written regardless of the threshold
    def test_zero_means_no_burn_only_when_observed(self):
        cov = np.ones(len(MS)); cov[idx(2020, 1):idx(2020, 1) + 5] = 0
        obs = bf(np.array([day(2024, 12, 1)]), np.array([], np.int64), np.array([], np.int64), cov, cov, MS, ME, min_frac=0.95)
        self.assertEqual(obs["bf_log_n10_nbhd"][0], 0.0)                          # 5 of 120 months missing = 0.958 >= 0.95 -> count kept but burn_obs_frac says < 1
        self.assertLess(obs["burn_obs_frac"][0], 1.0)

class Horizon(unittest.TestCase):
    def test_horizon_touching_flagged_day(self):
        F = np.array([1000]); D = np.array([999, 1000, 1001, 1003, 1004, 1007, 1008])
        self.assertEqual(list(gap_days.touches_flagged(D, F, 3)), [False, True, True, True, False, False, False])
        self.assertEqual(list(gap_days.touches_flagged(D, F, 7)), [False, True, True, True, True, True, False])      # D-7 = 1000 touches at D=1007; no forward window (D=999 stays kept)

class B0a(unittest.TestCase):
    def setUp(self):
        E0 = dt.date(1970, 1, 1); self.d = lambda y, m, dd: (dt.date(y, m, dd) - E0).days
    def run_b0a(self, onsets, flagged, rows):
        c = np.array([0] * len(onsets)); return climatology_fn(c, np.array([self.d(*o) for o in onsets]), np.array([self.d(*f) for f in flagged], np.int64), np.array([0] * len(rows)), np.array([self.d(*r) for r in rows]), 1)[0]
    def test_flagged_excluded_only_in_lookup_years(self):
        on = [(2019, 7, 10), (2019, 7, 11), (2020, 7, 7), (2020, 7, 8)]; fl = [(2019, 7, 11), (2020, 7, 7)]                # all four dates fall in ISO week 28 of their year
        r2020, r2021 = self.run_b0a(on, fl, [(2020, 7, 8), (2021, 7, 14)])
        self.assertAlmostEqual(r2020, 1 / 6, msg="2020 row: 2019 lookup only; 2019-07-11 excluded from numerator and denominator; the 2020 flagged day (own year) is never used")
        self.assertAlmostEqual(r2021, 2 / 12, msg="2021 row: lookup years 2019+2020, both flagged days excluded (numerator and denominator)")
        no_flag = self.run_b0a(on, [], [(2020, 7, 8), (2021, 7, 14)]); self.assertAlmostEqual(no_flag[0], 2 / 7); self.assertAlmostEqual(no_flag[1], 4 / 14)      # unchanged behaviour without flags
    def test_own_year_flag_does_not_change_row(self):
        on = [(2019, 7, 10)]; a = self.run_b0a(on, [], [(2020, 7, 8)]); b = self.run_b0a(on, [(2020, 7, 7), (2020, 7, 8), (2020, 7, 9)], [(2020, 7, 8)]); self.assertEqual(a[0], b[0])
    def test_first_year_is_nan_and_other_weeks_unaffected(self):
        on = [(2019, 3, 6)]; r = self.run_b0a(on, [(2019, 7, 11)], [(2019, 7, 10), (2020, 3, 4)]); self.assertTrue(np.isnan(r[0])); self.assertAlmostEqual(r[1], 1 / 7)



class LagAndCoverage(unittest.TestCase):
    def test_lag_30_excludes_recent_burns_and_moves_the_window(self):
        cov = np.ones(len(MS)); D = day(2023, 6, 30); b_recent = D - 20; b_ok = D - 31; b_edge = D - 30
        f = lambda bs, **k: bf(np.array([D]), np.array(bs, np.int64), np.array(bs, np.int64), cov, cov, MS, ME, lag=30, censor=False, **k)
        self.assertEqual(f([b_recent])["bf_log_n10_nbhd"][0], 0.0); self.assertEqual(f([b_edge])["bf_log_n10_nbhd"][0], 0.0)      # dated < D-30 only
        self.assertAlmostEqual(f([b_ok])["bf_log_n10_nbhd"][0], np.log1p(1)); self.assertAlmostEqual(f([b_ok])["bf_years_since"][0], 31 / 365.25)
        self.assertEqual(f([b_recent])["bf_years_since"][0], 10.0)                                                                # recent burn invisible -> cap
        g = bf(np.array([D]), np.array([b_recent], np.int64), np.array([b_recent], np.int64), cov, cov, MS, ME, lag=1, censor=False); self.assertAlmostEqual(g["bf_log_n10_nbhd"][0], np.log1p(1))   # lag 1 sees it
    def test_lag_days_are_not_counted_as_unobserved(self):
        cov = np.ones(len(MS)); cov[idx(2023, 6)] = 0; D = day(2023, 6, 30)                     # June 2023 unobserved but it lies inside the lag (D-30 = 2023-05-31)
        lo = bf(np.array([D]), np.array([], np.int64), np.array([], np.int64), cov, cov, MS, ME, lag=30)
        self.assertEqual(lo["burn_obs_frac"][0], 1.0); hi = bf(np.array([D]), np.array([], np.int64), np.array([], np.int64), cov, cov, MS, ME, lag=1); self.assertLess(hi["burn_obs_frac"][0], 1.0)
