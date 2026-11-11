import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import refs


def test_fwi_matches_the_published_first_day_of_the_system():
    T, H, W, r = np.array([17.0]), np.array([42.0]), np.array([25.0]), np.array([0.0])          # reference day: 17 C, 42 %, 25 km/h, no rain, April, start codes 85 / 6 / 15
    F, P, D = refs.ffmc(T, H, W, r, np.array([85.0])), refs.dmc(T, H, r, np.array([6.0]), 4), refs.dc(T, r, np.array([15.0]), 4)
    isi, bui, fwi = refs.indices(F, P, D, W)
    for got, want in ((F, 87.692), (P, 8.545), (D, 19.013), (isi, 10.853), (bui, 8.5), (fwi, 10.096)): assert abs(got[0] - want) < 0.02, (got, want)


def test_fwi_rises_with_dryness_and_wind_and_falls_with_rain():
    n = 30; months = np.full(n, 4); base = dict(T=np.full((n, 1), 30.0), H=np.full((n, 1), 30.0), W=np.full((n, 1), 10.0), R=np.zeros((n, 1)))
    dry = refs.fwi_cube(months=months, **base)[-1, 0]
    windy = refs.fwi_cube(months=months, **{**base, "W": np.full((n, 1), 30.0)})[-1, 0]
    rain = np.zeros((n, 1)); rain[-3:] = 25.0; wet = refs.fwi_cube(months=months, **{**base, "R": rain})[-1, 0]
    assert windy > dry > wet


def test_fwi_keeps_the_codes_over_missing_days():
    n = 5; months = np.full(n, 3); T = np.full((n, 1), 28.0); H = np.full((n, 1), 35.0); W = np.full((n, 1), 12.0); R = np.zeros((n, 1)); H[2] = np.nan
    out = refs.fwi_cube(T, H, W, R, months); assert np.isfinite(out).all()


def test_persistence_and_climatology():
    tr = pd.DataFrame({"cell_idx": [0, 0, 0, 1], "label_date": pd.to_datetime(["2022-03-05", "2023-03-06", "2023-04-01", "2022-03-05"]), "y_forest": [1, 0, 0, 1]})
    rows = pd.DataFrame({"cell_idx": [0, 0, 1, 2], "label_date": pd.to_datetime(["2024-03-10", "2024-04-10", "2024-03-10", "2024-03-10"]), "had_det_t_forest": [1, 0, 0, 1]})
    p = refs.climatology(tr, rows)
    assert np.allclose(p[:3], [0.5, 0.0, 1.0]) and abs(p[3] - (2 / 3)) < 1e-6                 # cell 2 has no training rows: the overall rate of March (2 of 3)
    assert list(refs.persistence(rows)) == [1, 0, 0, 1]
