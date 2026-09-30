import unittest

import numpy as np

from wildfire import config
from wildfire.evaluation import bootstrap, calibration, metrics, prior_shift, verdict
def H(p): p = np.clip(p, 1e-15, 1 - 1e-15); return -(p * np.log(p) + (1 - p) * np.log1p(-p))
class Perfect(unittest.TestCase):
    def test_perfect_scores(self):
        rng = np.random.default_rng(1); y = (rng.random(5000) < 0.05).astype(int); p = np.clip(y.astype(float), 1e-6, 1 - 1e-6)
        self.assertEqual(metrics.average_precision(y, y.astype(float)), 1.0); self.assertAlmostEqual(metrics.log_loss(y, p), -np.log(1 - 1e-6), places=9); self.assertAlmostEqual(metrics.brier(y, p), 1e-12, places=11)
    def test_worst_ranking(self):
        y = np.array([1, 1, 0, 0, 0, 0]); s = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.6]); self.assertAlmostEqual(metrics.average_precision(y, s), (1 / 5 + 2 / 6) / 2)   # positives ranked last: precision 1/5 and 2/6
class KnownAP(unittest.TestCase):
    def test_hand_example(self):
        self.assertAlmostEqual(metrics.average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]), (1 + 2 / 3) / 2)
    def test_all_tied_scores_equal_prevalence(self):
        y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0]); self.assertAlmostEqual(metrics.average_precision(y, np.zeros(10)), 0.2)
    def test_matches_sklearn_with_and_without_ties(self):
        from sklearn.metrics import average_precision_score
        rng = np.random.default_rng(2); y = (rng.random(20000) < 0.08).astype(int)
        for s in (rng.random(20000), np.round(rng.random(20000) * 0.3 + 0.6 * y * rng.random(20000), 1)): self.assertAlmostEqual(metrics.average_precision(y, s), average_precision_score(y, s), places=12)
    def test_no_positive_is_nan(self): self.assertTrue(np.isnan(metrics.average_precision([0, 0], [0.1, 0.2])))
class RandomBaseline(unittest.TestCase):
    def test_random_scores_ap_equals_prevalence_and_constant_prob_losses_are_exact(self):
        rng = np.random.default_rng(3); n = 200000; y = (rng.random(n) < 1 / 21).astype(int); prev = y.mean()
        self.assertAlmostEqual(metrics.average_precision(y, rng.random(n)), prev, delta=0.003)
        self.assertAlmostEqual(metrics.brier(y, np.full(n, prev)), prev * (1 - prev), places=12); self.assertAlmostEqual(metrics.log_loss(y, np.full(n, prev)), float(H(prev)), places=12)
class BinnedAP(unittest.TestCase):
    def test_histogram_ap_is_exact_with_one_bin_per_score_and_close_with_1000_quantile_bins(self):
        rng = np.random.default_rng(4); y = (rng.random(30000) < 0.05).astype(int); s = np.round(rng.random(30000) + 0.7 * y, 2); cl = rng.integers(0, 50, 30000)
        o = bootstrap.paired_week_blocked_bootstrap(y, cl, {"a": dict(score=s), "r": dict(score=rng.random(30000))}, "r", nboot=50, nbins=None); self.assertAlmostEqual(o["a"]["ap"]["point"], metrics.average_precision(y, s), places=12)
        _, ci = np.unique(cl, return_inverse=True); Hp, Hn = bootstrap.hist(s, y, ci, 50, None); self.assertAlmostEqual(float(bootstrap.ap_from_hist(Hp.sum(0, keepdims=True), Hn.sum(0, keepdims=True))[0]), metrics.average_precision(y, s), places=12)
        big = rng.random(200000); yb = (rng.random(200000) < 0.05 + 0.1 * big).astype(int); cb = rng.integers(0, 100, 200000); _, cib = np.unique(cb, return_inverse=True); Hp, Hn = bootstrap.hist(big, yb, cib, 100, 1000)
        self.assertAlmostEqual(float(bootstrap.ap_from_hist(Hp.sum(0, keepdims=True), Hn.sum(0, keepdims=True))[0]), metrics.average_precision(yb, big), delta=0.003)
class KnownShift(unittest.TestCase):
    def data(self, seed, n=60000, ncl=120):
        rng = np.random.default_rng(seed); p = rng.beta(0.6, 12, n); y = (rng.random(n) < p).astype(int); cl = rng.integers(0, ncl, n); return p, y, cl
    def expected(self, p):
        pbar = p.mean(); return dict(dll=float(H(p).mean() - H(pbar)), dbr=float((p * (1 - p)).mean() - (p * (1 - pbar) ** 2 + (1 - p) * pbar ** 2).mean()))
    def test_true_probability_beats_constant_and_the_bootstrap_brackets_the_known_delta(self):
        p, y, cl = self.data(5); o = bootstrap.paired_week_blocked_bootstrap(y, cl, {"arm": dict(score=p, prob=p), "ref": dict(score=np.random.default_rng(6).random(len(p)), prob=np.full(len(p), y.mean()))}, "ref", nboot=600, seed=1); e = self.expected(p); a = o["arm"]
        se_ll = (a["delta_logloss"]["hi"] - a["delta_logloss"]["lo"]) / 3.92; se_br = (a["delta_brier"]["hi"] - a["delta_brier"]["lo"]) / 3.92
        self.assertLess(abs(a["delta_logloss"]["point"] - e["dll"]), 4 * se_ll); self.assertLess(abs(a["delta_brier"]["point"] - e["dbr"]), 4 * se_br)
        self.assertTrue(a["delta_logloss"]["hi"] < 0 and a["delta_brier"]["hi"] < 0 and a["delta_ap"]["lo"] > 0); self.assertEqual(verdict.verdict_holdout(a), "PASS")
    def test_ap_shift_between_perfect_and_random_is_one_minus_prevalence(self):
        rng = np.random.default_rng(7); n = 40000; y = (rng.random(n) < 1 / 21).astype(int); cl = rng.integers(0, 80, n); o = bootstrap.paired_week_blocked_bootstrap(y, cl, {"arm": dict(score=y + 0.0), "ref": dict(score=rng.random(n))}, "ref", nboot=300)
        d = o["arm"]["delta_ap"]; self.assertAlmostEqual(d["point"], 1 - y.mean(), delta=0.01); self.assertTrue(d["lo"] <= d["point"] <= d["hi"]); self.assertGreater(d["lo"], 0.9)
    def test_null_ci_contains_zero_and_identical_arms_have_zero_delta_in_every_draw(self):
        rng = np.random.default_rng(8); n = 30000; y = (rng.random(n) < 0.05).astype(int); cl = rng.integers(0, 60, n); s = rng.random(n); o = bootstrap.paired_week_blocked_bootstrap(y, cl, {"a": dict(score=rng.random(n), prob=np.full(n, 0.05)), "b": dict(score=s, prob=np.full(n, 0.05)), "ref": dict(score=s, prob=np.full(n, 0.05)), "same": dict(score=s, prob=np.full(n, 0.05))}, "ref", nboot=400)
        self.assertTrue(o["a"]["delta_ap"]["lo"] <= 0 <= o["a"]["delta_ap"]["hi"]); self.assertEqual(o["same"]["delta_ap"]["lo"], 0.0); self.assertEqual(o["same"]["delta_ap"]["hi"], 0.0); self.assertEqual(o["same"]["delta_logloss"]["hi"], 0.0); self.assertEqual(verdict.verdict_holdout(o["a"]), "INCONCLUSIVE")
    def test_pairing_narrows_the_interval(self):
        rng = np.random.default_rng(9); n = 40000; p = rng.beta(0.6, 12, n); y = (rng.random(n) < p).astype(int); cl = rng.integers(0, 80, n); noisy = np.clip(p + rng.normal(0, 0.002, n), 1e-4, 0.99)
        o = bootstrap.paired_week_blocked_bootstrap(y, cl, {"arm": dict(score=noisy, prob=noisy), "ref": dict(score=p, prob=p)}, "ref", nboot=400); w_delta = o["arm"]["delta_logloss"]["hi"] - o["arm"]["delta_logloss"]["lo"]; w_arm = o["arm"]["logloss"]["hi"] - o["arm"]["logloss"]["lo"]; self.assertLess(w_delta, 0.25 * w_arm)
    def test_every_point_estimate_lies_inside_its_own_bootstrap_interval(self):
        p, y, cl = self.data(12, n=30000, ncl=90); o = bootstrap.paired_week_blocked_bootstrap(y, cl, {"arm": dict(score=p, prob=p), "ref": dict(score=np.random.default_rng(13).random(len(p)), prob=np.clip(np.random.default_rng(13).random(len(p)) * 0.3, 0.001, 0.9))}, "ref", nboot=300)
        for a in ("arm", "ref"):
            for m in ("ap", "logloss", "brier"): self.assertTrue(o[a][m]["lo"] <= o[a][m]["point"] <= o[a][m]["hi"], (a, m, o[a][m]))
        self.assertAlmostEqual(o["arm"]["brier"]["point"], metrics.brier(y, p), places=12); self.assertAlmostEqual(o["arm"]["logloss"]["point"], metrics.log_loss(y, p), places=12)
    def test_cluster_resampling_keeps_cluster_count_and_reports_it(self):
        rng = np.random.default_rng(10); y = (rng.random(2000) < 0.1).astype(int); cl = np.repeat(np.arange(40), 50); o = bootstrap.paired_week_blocked_bootstrap(y, cl, {"a": dict(score=rng.random(2000)), "r": dict(score=rng.random(2000))}, "r", nboot=20); self.assertEqual(o["_n_clusters"], 40)
    def test_coverage_of_the_known_delta(self):
        hit = 0; reps = 40
        for k in range(reps):
            p, y, cl = self.data(100 + k, n=8000, ncl=60); o = bootstrap.paired_week_blocked_bootstrap(y, cl, {"arm": dict(score=p, prob=p), "ref": dict(score=p, prob=np.full(len(p), y.mean()))}, "ref", nboot=250, seed=k); e = self.expected(p)["dll"]; hit += o["arm"]["delta_logloss"]["lo"] <= e <= o["arm"]["delta_logloss"]["hi"]
        self.assertGreaterEqual(hit / reps, 0.85, f"95% CI covered the known delta in {hit}/{reps} synthetic datasets")
class Calibration(unittest.TestCase):
    def test_isotonic_fit_on_training_years_recovers_a_monotone_truth_and_improves_loss(self):
        rng = np.random.default_rng(11); n = 120000; s = rng.random(n) * 0.5; f = lambda x: np.clip(2.5 * x ** 2, 0.001, 0.9); y = (rng.random(n) < f(s)).astype(int); yr = np.where(np.arange(n) < 100000, 2019 + np.arange(n) % 6, 2025)
        tr = yr <= 2024; p_te, m = calibration.calibrate_climatology(s[tr], y[tr], yr[tr], s[~tr]); self.assertTrue((np.diff(p_te[np.argsort(s[~tr])]) >= -1e-12).all()); self.assertLess(np.abs(p_te - f(s[~tr])).mean(), 0.03)
        self.assertLess(metrics.log_loss(y[~tr], p_te), metrics.log_loss(y[~tr], np.clip(s[~tr], 1e-6, 1 - 1e-6))); self.assertTrue(((p_te > 0) & (p_te < 1)).all())
    def test_holdout_year_is_refused_for_calibration(self):
        with self.assertRaises(ValueError): calibration.calibrate_climatology(np.array([0.1, 0.2]), np.array([0, 1]), np.array([2024, 2025]), np.array([0.3]))
    def test_nan_scores_are_ignored_in_fit_and_stay_nan(self):
        s = np.array([np.nan, 0.1, 0.2, 0.3, 0.4, 0.5]); y = np.array([0, 0, 0, 1, 1, 1]); yr = np.full(6, 2020); p, m = calibration.calibrate_climatology(s, y, yr, np.array([np.nan, 0.45])); self.assertTrue(np.isnan(p[0])); self.assertGreater(p[1], 0.5)
class Verdict(unittest.TestCase):
    def test_branches(self):
        mk = lambda ap, ll, br, pt=None: dict(delta_ap=dict(point=(ap[0] + ap[1]) / 2 if pt is None else pt, lo99=ap[0], hi99=ap[1]), delta_logloss=dict(lo99=ll[0], hi99=ll[1]), delta_brier=dict(lo99=br[0], hi99=br[1]))
        self.assertEqual(verdict.verdict_holdout(mk((0.01, 0.03), (-0.02, -0.005), (-0.01, -0.001))), "PASS"); self.assertEqual(verdict.verdict_holdout(mk((0.01, 0.03), (-0.02, 0.004), (-0.01, -0.001))), "INCONCLUSIVE")   # split: log loss CI reaches 0
        self.assertEqual(verdict.verdict_holdout(mk((0.001, 0.012), (-0.02, -0.005), (-0.01, -0.001))), "INCONCLUSIVE")                                     # point 0.0065 < min effect 0.010 although every CI clears 0
        self.assertEqual(verdict.verdict_holdout(mk((0.002, 0.03), (-0.02, -0.005), (-0.01, -0.001), pt=0.0099)), "INCONCLUSIVE"); self.assertEqual(verdict.verdict_holdout(mk((0.002, 0.03), (-0.02, -0.005), (-0.01, -0.001), pt=0.0100)), "PASS")
        self.assertEqual(verdict.verdict_holdout(mk((-0.03, -0.01), (0.0, 0.02), (0.0, 0.01))), "FAIL"); self.assertEqual(verdict.verdict_holdout(mk((-0.01, 0.02), (-0.01, 0.01), (-0.01, 0.01))), "INCONCLUSIVE")
    def test_min_effect_is_a_fifth_of_the_sampled_prevalence(self): self.assertAlmostEqual(config.load("evaluation")["verdict"]["min_ap_gain"] / (1 / 21), 0.21, places=2)
    def test_verdict_uses_the_99_percent_bounds(self):
        rng = np.random.default_rng(21); n = 30000; p = rng.beta(0.6, 12, n); y = (rng.random(n) < p).astype(int); cl = rng.integers(0, 60, n); o = bootstrap.paired_week_blocked_bootstrap(y, cl, {"arm": dict(score=p, prob=p), "ref": dict(score=rng.random(n), prob=np.full(n, y.mean()))}, "ref", nboot=500)
        d = o["arm"]["delta_logloss"]; self.assertLess(d["lo99"], d["lo"]); self.assertGreater(d["hi99"], d["hi"])              # 99% interval is wider than 95%
    def test_headline_m2_vs_m1_is_the_same_machinery_with_ref_m1(self):
        rng = np.random.default_rng(22); n = 30000; p = rng.beta(0.6, 12, n); y = (rng.random(n) < p).astype(int); cl = rng.integers(0, 60, n); p_ref = np.clip(0.5 * p + 0.5 * rng.beta(0.6, 12, n), 1e-4, 0.99)   # M1 = M2 blended with an independent draw: different ranking, worse calibration
        o = bootstrap.paired_week_blocked_bootstrap(y, cl, {"arm": dict(score=p, prob=p), "ref1": dict(score=p_ref, prob=p_ref), "ref0": dict(score=rng.random(n), prob=np.full(n, y.mean()))}, "ref1")
        self.assertIn("delta_ap", o["arm"]); self.assertNotIn("delta_ap", o["ref1"]); self.assertEqual(verdict.verdict_holdout(o["arm"]), "PASS")
class PriorShift(unittest.TestCase):
    def population(self, seed=30, N=1500000):
        rng = np.random.default_rng(seed); p = rng.beta(0.5, 30, N); y = (rng.random(N) < p).astype(int); return p, y
    def test_correction_recovers_the_true_probabilities_exactly(self):
        p, y = self.population(); rng = np.random.default_rng(31); pos = y == 1; nneg = int((~pos).sum()); r = 20 * pos.sum() / nneg; keep = pos | (rng.random(len(y)) < r); ps = y[keep].mean()
        odds_s = p[keep] / (1 - p[keep]) / r; p_sampled_world = odds_s / (1 + odds_s); pi_t = y.mean()                                        # what a perfectly calibrated model trained on the 1:20 sample outputs
        back = prior_shift.prior_shift_probs(p_sampled_world, pi_t, ps); self.assertLess(np.abs(back - p[keep]).max(), 0.02 * p.max() + 1e-3); self.assertAlmostEqual(float(np.mean(np.abs(back - p[keep]) / (p[keep] + 1e-9))), 0.0, delta=0.03)
        self.assertAlmostEqual(ps, 1 / 21, delta=0.001)
    def test_weighted_loss_on_the_sample_matches_the_population_loss_while_the_unweighted_one_does_not(self):
        p, y = self.population(32); rng = np.random.default_rng(33); pos = y == 1; r = 20 * pos.sum() / (~pos).sum(); keep = pos | (rng.random(len(y)) < r); ysamp = y[keep]; odds = p[keep] / (1 - p[keep]) / r; psw = odds / (1 + odds)
        frame = {"all": len(y)}; w = prior_shift.design_weights(ysamp, np.full(len(ysamp), "all"), frame); self.assertAlmostEqual(w.sum(), len(y), delta=1e-6 * len(y))            # weights sum to the frame size
        pop_ll = metrics.log_loss(y, p); pop_br = metrics.brier(y, p); corr = prior_shift.prior_shift_probs(psw, y.mean(), ysamp.mean())
        self.assertAlmostEqual(metrics.log_loss(ysamp, corr, w=w), pop_ll, delta=0.004 * pop_ll); self.assertAlmostEqual(metrics.brier(ysamp, corr, w=w), pop_br, delta=0.01 * pop_br)
        self.assertGreater(abs(metrics.log_loss(ysamp, psw) - pop_ll), 0.5 * pop_ll)                                                                       # before correction the sampled-scale loss is not the population loss
    def test_strata_weights_and_per_state_priors(self):
        y = np.array([1, 0, 0, 0, 1, 0]); st = np.array(["A", "A", "A", "A", "B", "B"]); w = prior_shift.design_weights(y, st, {"A": 103, "B": 11}); self.assertTrue(np.allclose(w, [1, 34, 34, 34, 1, 10]))
        pt = np.array([0.02, 0.02, 0.02, 0.02, 0.05, 0.05]); out = prior_shift.prior_shift_probs(np.full(6, 1 / 21), pt, 1 / 21); self.assertTrue(np.allclose(out, pt))                      # a sampled-scale probability equal to pi_s maps to pi_t
        self.assertTrue(np.allclose(prior_shift.prior_shift_probs(np.array([0.3]), 1 / 21, 1 / 21), 0.3))                                                      # pi_t == pi_s: identity
class Clusters(unittest.TestCase):
    def test_iso_week_main_vs_state_week_sensitivity(self):
        y = [2025, 2025, 2025, 2025]; wk = [3, 3, 4, 4]; st = ["OD", "MP", "OD", "MP"]
        self.assertEqual(len(set(prior_shift.cluster_labels(y, wk, st, "week"))), 2); self.assertEqual(len(set(prior_shift.cluster_labels(y, wk, st, "state_week"))), 4); self.assertEqual(prior_shift.cluster_labels(y, wk, st, "week")[0], "2025-W03")
    def test_week_clusters_are_fewer_than_state_week_clusters_in_the_bootstrap_report(self):
        rng = np.random.default_rng(40); n = 6000; wk = rng.integers(1, 32, n); stt = rng.choice(["OD", "MP", "TS", "CG", "MH", "JH"], n); y = (rng.random(n) < 0.05).astype(int); sc = rng.random(n)
        a = bootstrap.paired_week_blocked_bootstrap(y, prior_shift.cluster_labels(np.full(n, 2025), wk, stt, "week"), {"a": dict(score=sc), "r": dict(score=rng.random(n))}, "r", nboot=20); b = bootstrap.paired_week_blocked_bootstrap(y, prior_shift.cluster_labels(np.full(n, 2025), wk, stt, "state_week"), {"a": dict(score=sc), "r": dict(score=rng.random(n))}, "r", nboot=20)
        self.assertEqual(a["_n_clusters"], 31); self.assertEqual(b["_n_clusters"], 31 * 6)
    def test_weighted_bootstrap_point_matches_weighted_metric(self):
        rng = np.random.default_rng(41); n = 5000; y = (rng.random(n) < 0.05).astype(int); w = np.where(y == 1, 1.0, 3.0); p = np.clip(rng.random(n) * 0.2, 1e-4, 0.9); o = bootstrap.paired_week_blocked_bootstrap(y, rng.integers(0, 30, n), {"a": dict(score=p, prob=p), "r": dict(score=p, prob=np.full(n, 0.05))}, "r", nboot=50, weights=w)
        self.assertAlmostEqual(o["a"]["logloss"]["point"], metrics.log_loss(y, p, w=w), places=12); self.assertAlmostEqual(o["a"]["brier"]["point"], metrics.brier(y, p, w=w), places=12); self.assertNotAlmostEqual(o["a"]["brier"]["point"], metrics.brier(y, p), places=6)
