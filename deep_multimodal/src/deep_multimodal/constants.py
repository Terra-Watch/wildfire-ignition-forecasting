"""Names and sizes of the model inputs. The order of every list is the order of the columns in the saved input arrays."""

HISTORY_DAYS = 14                  # days of fire history before the target day
NEIGHBOURHOOD_RADIUS = 3           # cells on each side of the target cell: a 7 by 7 window
NEIGHBOURHOOD_SIDE = 2 * NEIGHBOURHOOD_RADIUS + 1
FIRE_CHANNELS = ("detected", "onset", "log_count", "log_power")   # four numbers per occupied (day, neighbour) entry
MAX_FIRE_TOKENS = 32               # occupied entries kept per row (nearest in space first, then most recent)

# 21 time-invariant columns: terrain, land cover shares, land structure, human layers
STATIC_COLUMNS = ["dem_mean", "dem_sd", "dem_slope", "dem_sin_aspect", "dem_cos_aspect", "dem_tpi",
                  "wc_10", "wc_20", "wc_30", "wc_40", "wc_50", "wc_60", "wc_80", "wc_90", "wc_edge_density",
                  "log_dist_built", "hb_log_pop", "hb_ghm_aa", "hb_ghm_bu", "hb_nl_rank", "hb_road_km"]
# 9 per-row columns: burn history, fire weather index, recent fire history of the cell and its ring, and the calendar
ROW_COLUMNS = ["bf_log_n10_nbhd", "bf_years_since", "fwi", "fh_detected_ring1_last7d", "fh_detected_ring3_last7d",
               "fh_detected_days_own_last14d", "fh_days_since_detection_own", "sin_doy", "cos_doy"]
BURN_COLUMNS = [0, 1]              # indices into ROW_COLUMNS that join the static tokens
OWN_HISTORY_COLUMNS = [1, 5, 6]    # indices into ROW_COLUMNS that enter the query
SKIP_COLUMNS = [1, 2, 3, 4, 5, 6]  # indices into ROW_COLUMNS read by the gated skip path
CALENDAR_COLUMNS = [7, 8]          # sine and cosine of the day of the year
STATIC_TOKENS = len(STATIC_COLUMNS) + len(BURN_COLUMNS)   # 23 static-feature tokens

# 20 daily reanalysis channels (17 weather and fire-weather variables, daily rain, wind u and v) plus 17 anomalies
WEATHER_CHANNELS = ["t2m", "vpd", "ws", "rh", "sw1", "sw2", "rain7", "dsr", "ffmc", "dmc", "dc", "isi", "bui", "fwi",
                    "tmax09", "sw1_anom", "sw2_anom", "rain", "u", "v"]
ANOMALY_CHANNELS = 17              # the first 17 channels also enter as anomalies against the fit-year climate
WEATHER_INPUTS = len(WEATHER_CHANNELS) + ANOMALY_CHANNELS   # 37 inputs per day
WEATHER_DAYS = 60
WEATHER_SEGMENTS = 12              # one token per 5 days of the 60-day sequence

# day-ahead forecast: 5 variables at 12 hourly steps (06 to 17 UTC of the target day) and 4 daily statistics of each
FORECAST_VARIABLES = ("t2m", "rh", "u10", "v10", "precip")
FORECAST_STATISTICS = ("min", "max", "mean", "sum")
FORECAST_HOURS = tuple(range(6, 18))
FORECAST_TOKENS = 6
FORECAST_DAILY_VALUES = len(FORECAST_VARIABLES) * len(FORECAST_STATISTICS)   # 20
