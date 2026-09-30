import numpy as np
import pandas as pd

from wildfire.evaluation import ladder, verdict


def test_top_k_capture_and_lift():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.1, 0.2, 0.3, 0.8, 0.0, 0.05, 0.4, 0.15, 0.25])
    out = ladder.top_k_capture(y, score, [0.2, 0.5])
    assert out["capture_20pct"] == 1.0 and out["lift_20pct"] == 5.0 and out["capture_50pct"] == 1.0


def test_reliability_bins_track_the_observed_rate():
    rng = np.random.default_rng(0)
    p = rng.random(50000) * 0.2
    y = (rng.random(50000) < p).astype(int)
    t = ladder.reliability(y, p, np.ones(len(p)), 10)
    assert len(t) == 10 and (abs(t.mean_predicted - t.observed_rate) < 0.02).all()


def test_verdict_rule_branches():
    mk = lambda point, se, lo, hi: dict(point=point, se=se, lo99=lo, hi99=hi)
    assert verdict.baseline_verdict(mk(0.03, 0.002, 0.02, 0.04), -0.01)[0] == "PASS"
    assert verdict.baseline_verdict(mk(0.03, 0.002, 0.02, 0.04), -0.05)[0] == "INCONCLUSIVE"     # a state is worse than the floor
    assert verdict.baseline_verdict(mk(0.001, 0.001, -0.002, 0.004), 0.0)[0] == "FAIL"           # upper bound below 0.010
    assert verdict.baseline_verdict(mk(0.03, 0.005, 0.01, 0.05), 0.0)[0] == "INCONCLUSIVE"        # power gate: MDE 0.017 > 0.010
    assert abs(verdict.minimum_detectable_effect(0.001) - 0.003417) < 1e-5


def test_contrast_recovers_a_real_gain():
    rng = np.random.default_rng(1)
    n = 60000
    rows = pd.DataFrame(dict(label=(rng.random(n) < 0.05).astype(int), state=rng.choice(["A", "B"], n),
                             year=rng.integers(2021, 2026, n), week=rng.integers(1, 53, n)))
    good = rows.label.values * 0.5 + rng.random(n)
    scores = dict(good=good, ref=rng.random(n))
    probs = dict(good=np.clip(good / 3, 0.001, 0.9), ref=np.full(n, 0.05))
    out = ladder.contrast(rows, scores, probs, np.ones(n), "good", "ref", ladder.cluster_labels(rows))
    assert out["delta_ap"] > 0.05 and out["ap_lo99"] > 0 and out["worst_state_delta_ap"] > 0
    assert out["verdict"] == ("PASS" if out["mde"] <= 0.01 else "INCONCLUSIVE")      # the power gate decides
