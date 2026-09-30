"""The single evaluation harness. Every feature set and model goes through it."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from wildfire import config
from wildfire.evaluation import bootstrap, prior_shift, verdict, views
from wildfire.features.base import FeatureSet


def check_split(split: str) -> None:
    """Only configured development splits are available; the held-out geographic split is refused."""
    cfg = config.load("evaluation")["splits"]
    if split in cfg["refused"] or split not in cfg["available"]:
        raise ValueError(f"split '{split}' is not available for development")


def evaluate(feature_sets: Sequence[FeatureSet], model, split: str = "validation") -> dict:
    """Score `model` on the combined columns of `feature_sets`. Refuses unavailable splits."""
    check_split(split)
    raise NotImplementedError


def score_arms(rows: pd.DataFrame, scores: dict, probs: dict, ref: str, cluster, weights=None, nboot=None) -> dict:
    """Paired bootstrap of AP, log loss and Brier for every arm against `ref`, plus the ranking views of each arm."""
    arms = {a: dict(score=scores[a], prob=probs[a]) for a in scores}
    out = bootstrap.paired_week_blocked_bootstrap(rows.label.values, cluster, arms, ref, nboot=nboot, weights=weights)
    out["_views"] = pd.DataFrame([dict(arm=a, **views.views(rows, scores[a])) for a in scores])
    return out


def result_table(o: dict, arms: Sequence[str], ref: str, with_verdict: bool = True) -> pd.DataFrame:
    rows = []
    for a in arms:
        d = o[a]
        r = dict(arm=a, AP=d["ap"]["point"], AP_lo99=d["ap"]["lo99"], AP_hi99=d["ap"]["hi99"],
                 logloss=d["logloss"]["point"], brier=d["brier"]["point"])
        if a != ref:
            for k, nm in (("ap", "dAP"), ("logloss", "dLogLoss"), ("brier", "dBrier")):
                r.update({nm: d["delta_" + k]["point"], nm + "_lo99": d["delta_" + k]["lo99"], nm + "_hi99": d["delta_" + k]["hi99"]})
            if with_verdict:
                r["verdict"] = verdict.verdict_holdout(d)
        rows.append(r)
    return pd.DataFrame(rows)
