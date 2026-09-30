import numpy as np
import pandas as pd

from wildfire.features import weather_windows as ww
from wildfire.features.tabular import weather

FIRST = 20000


def make_series(days=200, points=3, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(days, len(weather.NAMES), points)).astype(np.float32)
    rain = (rng.random((days, points)) * 6).astype(np.float32)
    return dict(values=values, rain=rain, cell_point=np.array([0, 1, 2, 2]), first_day=FIRST)


def test_trailing_statistics():
    x = np.arange(10, dtype=np.float32)[:, None]
    m = ww.trailing(x, 3, "mean")
    assert np.isnan(m[:2]).all() and m[2, 0] == 1.0 and m[9, 0] == 8.0
    assert ww.trailing(x, 3, "max")[5, 0] == 5.0 and ww.trailing(x, 3, "sum")[5, 0] == 12.0


def test_days_since_rain_counts_and_caps():
    rain = np.array([[0.0], [5.0], [0.0], [0.0], [0.0]], np.float32)
    out = ww.days_since_rain(rain, 2.0, 60)[:, 0]
    assert out.tolist() == [60.0, 0.0, 1.0, 2.0, 3.0]
    assert ww.days_since_rain(np.zeros((100, 1), np.float32), 2.0, 60)[-1, 0] == 60.0


def test_window_matrix_shape_and_names():
    s = make_series()
    m = ww.window_matrix(s, np.array([0, 1, 3]), np.array([FIRST + 100, FIRST + 101, FIRST + 150]))
    assert m.shape == (3, len(ww.window_names())) and len(ww.window_names()) == 180


def test_a_value_on_the_target_day_does_not_change_its_features():
    """Shifted-feature leak test: features of day D may only use days up to D-1."""
    s = make_series()
    cell, day = np.array([1]), np.array([FIRST + 100])
    before = ww.window_matrix(s, cell, day)
    s["values"][day[0] - FIRST, :, :] += 100.0         # the target day itself
    s["rain"][day[0] - FIRST, :] += 100.0
    assert np.array_equal(before, ww.window_matrix(s, cell, day), equal_nan=True)
    after_next = ww.window_matrix(s, cell, day + 1)    # the next target day does see it
    assert not np.array_equal(ww.window_matrix(make_series(), cell, day + 1), after_next, equal_nan=True)


def test_a_feature_built_from_the_future_is_detected():
    s = make_series()
    cell, day = np.array([0]), np.array([FIRST + 100])
    honest = ww.window_matrix(s, cell, day)
    shifted = ww.window_matrix(s, cell, day + 1)          # deliberately one day late
    assert not np.array_equal(honest, shifted, equal_nan=True)


def test_anomaly_climatology_uses_training_years_only():
    days = 800
    dates = pd.to_datetime(FIRST + np.arange(days), unit="D")
    values = np.zeros((days, len(weather.NAMES), 1), np.float32)
    values[:, 0, 0] = dates.year - 2000                       # variable 0 equals the year, so the climate reveals the years used
    s = dict(values=values, rain=np.zeros((days, 1), np.float32), cell_point=np.array([0]), first_day=FIRST)
    years = sorted(set(dates.year))
    clim = ww.anomaly_climatology(s, [years[0]])
    doy = dates.dayofyear.values
    assert np.nanmax(np.abs(clim[doy[10], 0, 0] - (years[0] - 2000))) < 1e-6   # only the first year contributes
    a = ww.anomalies(s, clim, np.array([0]), np.array([FIRST + 500]))
    assert a.shape == (1, len(weather.NAMES))
