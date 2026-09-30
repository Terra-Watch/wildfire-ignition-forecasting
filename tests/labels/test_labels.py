import numpy as np
import pandas as pd

from wildfire.labels import detections, domain, negatives


def test_onset_needs_quiet_days_before():
    d = pd.DataFrame(dict(cell=0, day=[10, 11, 12, 15, 16, 20, 30, 31], frp=1.0))
    cells = pd.DataFrame(dict(state=["OD"]))
    cd = detections.cell_day_table(d, cells)
    assert cd.day[cd.onset].tolist() == [10, 20, 30]      # 15 follows 12 by 3 days: not an onset; 20 follows 16 by 4: onset


def test_excluded_keys_cover_the_days_after_a_detection():
    cd = pd.DataFrame(dict(cell=[5], day=[100]))
    keys = negatives.excluded_keys(cd)
    assert keys.tolist() == [5 * negatives.KEY_STRIDE + d for d in (100, 101, 102, 103)]


def test_negatives_avoid_detections_are_distinct_and_reproducible():
    cd = pd.DataFrame(dict(cell=np.repeat(np.arange(3), 50), day=np.tile(np.arange(18300, 18350), 3)))
    ex = negatives.excluded_keys(cd)

    def draw():
        return negatives.draw_negatives(np.random.default_rng(1), ex, [1, 2, 3], 2020, np.arange(3), 40, "train", "OD", "A")

    a, b = draw(), draw()
    keys = a.cell.values * negatives.KEY_STRIDE + a.day.values
    assert len(a) == 40 and len(set(keys.tolist())) == 40 and not np.isin(keys, ex).any()
    assert a.equals(b)
    assert set(pd.to_datetime(a.day, unit="D").dt.month) <= {1, 2, 3}


def test_season_is_the_smallest_set_of_months_holding_the_share_of_onsets():
    month = [3] * 50 + [4] * 30 + [5] * 15 + [6] * 5
    cd2 = pd.DataFrame(dict(state="OD", month=month, onset=True, domain20=True, year=2020))
    assert domain.season_months(cd2)["OD"] == [3, 4, 5]     # 50+30 = 80% < 90%, adding May reaches 95%
