"""Decision rules for comparing an arm with a reference arm."""
from __future__ import annotations

from wildfire import config


def verdict_holdout(o: dict, min_ap_gain: float | None = None) -> str:
    """PASS iff the AP gain point >= min_ap_gain, its 99% lower bound > 0, and the 99% upper bounds of the log-loss and
    Brier differences are < 0. FAIL iff the AP gain's 99% upper bound < 0. Anything else is INCONCLUSIVE."""
    g = config.load("evaluation")["verdict"]["min_ap_gain"] if min_ap_gain is None else min_ap_gain
    if o["delta_ap"]["point"] >= g and o["delta_ap"]["lo99"] > 0 and o["delta_logloss"]["hi99"] < 0 and o["delta_brier"]["hi99"] < 0:
        return "PASS"
    if o["delta_ap"]["hi99"] < 0:
        return "FAIL"
    return "INCONCLUSIVE"


def verdict_paired_auc(o: dict, thr: float | None = None, floor: float | None = None) -> str:
    v = config.load("evaluation")["verdict"]
    thr = v["paired_auc_threshold"] if thr is None else thr
    floor = v["paired_auc_worst_state_floor"] if floor is None else floor
    worst = min(x["d"] for x in o["per"].values())
    if o["d"] >= thr and o["ci_lo"] > 0 and worst >= floor:
        return "PASS"
    if o["ci_hi"] < thr:
        return "FAIL"
    return "INCONCLUSIVE"
