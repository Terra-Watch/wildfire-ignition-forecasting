import numpy as np
import pandas as pd

from wildfire.models import arms


def test_groups_and_the_42_inputs():
    g = arms.feature_groups()
    assert [len(g[k]) for k in ("weather", "fire_weather_codes", "static", "human", "season", "burn")] == [11, 6, 16, 5, 2, 2]
    cols = arms.arm_columns()
    assert len(cols["logistic_with_burn"]) == 42 and len(cols["boosting"]) == 42
    assert cols["boosting"] == cols["logistic_with_burn"]
    assert cols["logistic_with_burn"][:-2] == cols["logistic_weather_static"]


def test_arm_names_and_order():
    assert list(arms.feature_groups() and arms.arm_columns()) == [
        "climatology_cell", "landcover_season", "fwi_season", "logistic_weather_static", "logistic_with_burn", "boosting"]
    assert arms.reference_arm() == "climatology_cell"
    assert arms.fitted_arms() == ["landcover_season", "fwi_season", "logistic_weather_static", "logistic_with_burn", "boosting"]


def test_no_coverage_flags_or_same_day_inputs_are_model_inputs():
    all_cols = {c for v in arms.arm_columns().values() for c in v}
    assert not all_cols & {"burn_obs_frac", "bf_ys_obs_frac", "bf_ys_censored"}
    assert not any(c.endswith("_l0") or c.startswith("hbv_") or c.startswith("ex_") for c in all_cols)
    assert "bf_years_since" in all_cols and "hb_nl_rank" in all_cols


def test_human_layers_below_the_non_missing_share_are_dropped():
    g = arms.feature_groups()
    table = pd.DataFrame({c: np.ones(100) for c in g["human"]})
    table.loc[:10, "hb_road_km"] = np.nan                    # 89% present
    assert "hb_road_km" not in arms.feature_groups(table)["human"]
    assert "hb_nl_rank" in arms.feature_groups(table)["human"]
