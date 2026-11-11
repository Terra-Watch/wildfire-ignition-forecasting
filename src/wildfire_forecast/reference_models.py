"""Forecasts that need no training: persistence, climatology and the Fire Weather Index (daily-proxy implementation).
Each returns one score per row of a row table, so they go through the same evaluation as the learned models.

  persistence : 1 if the cell had a forest detection on day t, else 0.
  climatology : share of detection days of the cell in that calendar month over the training years (rows with split_t == train).
  fwi         : the Canadian FWI system (FFMC, DMC, DC, ISI, BUI, FWI), one continuous recursion from the first cube day, started with the standard values
                (FFMC 85, DMC 6, DC 15). The cube holds daily values, so the temperature, relative humidity, wind and 24-hour rain inputs are
                t2m_max, rh_min_proxy, wind_speed_meanvec and precip_day. The standard day-length factors are used for every cell. Called
                "FWI (daily-proxy implementation)" in every table.
"""
import numpy as np, pandas as pd

LE = np.array([6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.2, 8.0, 7.0, 6.0])        # DMC day-length factor by month
LF = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6])        # DC day-length factor by month


def ffmc(T, H, W, r, F0):
    mo = 147.2 * (101 - F0) / (59.5 + F0)
    rf = np.maximum(r - 0.5, 0.0); wet = r > 0.5
    add = 42.5 * rf * np.exp(-100 / (251 - mo)) * (1 - np.exp(-6.93 / np.where(rf > 0, rf, 1)))
    add = np.where(mo > 150, add + 0.0015 * (mo - 150) ** 2 * np.sqrt(rf), add)
    mo = np.where(wet, np.minimum(mo + add, 250.0), mo)
    Ed = 0.942 * H ** 0.679 + 11 * np.exp((H - 100) / 10) + 0.18 * (21.1 - T) * (1 - np.exp(-0.115 * H))
    Ew = 0.618 * H ** 0.753 + 10 * np.exp((H - 100) / 10) + 0.18 * (21.1 - T) * (1 - np.exp(-0.115 * H))
    ko = 0.424 * (1 - (H / 100) ** 1.7) + 0.0694 * np.sqrt(W) * (1 - (H / 100) ** 8); kd = ko * 0.581 * np.exp(0.0365 * T)
    k1 = 0.424 * (1 - ((100 - H) / 100) ** 1.7) + 0.0694 * np.sqrt(W) * (1 - ((100 - H) / 100) ** 8); kw = k1 * 0.581 * np.exp(0.0365 * T)
    m = np.where(mo > Ed, Ed + (mo - Ed) * 10.0 ** (-kd), np.where(mo < Ew, Ew - (Ew - mo) * 10.0 ** (-kw), mo))
    return 59.5 * (250 - m) / (147.2 + m)


def dmc(T, H, r, P0, month):
    Po = np.maximum(P0, 1e-6)
    re = 0.92 * r - 1.27; Mo = 20 + np.exp(5.6348 - Po / 43.43)
    b = np.where(Po <= 33, 100 / (0.5 + 0.3 * Po), np.where(Po <= 65, 14 - 1.3 * np.log(Po), 6.2 * np.log(Po) - 17.2))
    Mr = Mo + 1000 * np.maximum(re, 0) / (48.77 + b * np.maximum(re, 0)); Pr = np.maximum(244.72 - 43.43 * np.log(np.maximum(Mr - 20, 1e-6)), 0.0)
    Pr = np.where(r > 1.5, Pr, P0)
    K = np.where(T > -1.1, 1.894 * (T + 1.1) * (100 - H) * LE[month - 1] * 1e-6, 0.0)
    return Pr + 100 * K


def dc(T, r, D0, month):
    rd = 0.83 * r - 1.27; Qo = 800 * np.exp(-D0 / 400); Qr = Qo + 3.937 * np.maximum(rd, 0); Dr = np.maximum(400 * np.log(800 / Qr), 0.0)
    Dr = np.where(r > 2.8, Dr, D0)
    V = np.maximum(0.36 * (T + 2.8) + LF[month - 1], 0.0)
    return Dr + 0.5 * V


def indices(F, P, D, W):
    m = 147.2 * (101 - F) / (59.5 + F); fF = 91.9 * np.exp(-0.1386 * m) * (1 + m ** 5.31 / 4.93e7); isi = 0.208 * np.exp(0.05039 * W) * fF
    den = np.maximum(P + 0.4 * D, 1e-9)
    bui = np.where(P <= 0.4 * D, 0.8 * P * D / den, P - (1 - 0.8 * D / den) * (0.92 + (0.0114 * P) ** 1.7))
    bui = np.maximum(bui, 0.0); fD = np.where(bui <= 80, 0.626 * bui ** 0.809 + 2, 1000 / (25 + 108.64 * np.exp(-0.023 * bui)))
    B = 0.1 * isi * fD; fwi = np.where(B > 1, np.exp(2.72 * (0.434 * np.log(np.maximum(B, 1e-9))) ** 0.647), B)
    return isi, bui, fwi


def fwi_cube(T, H, W, R, months):
    """T (deg C), H (percent), W (km/h), R (mm) are [days, cells] arrays; months is [days] (1 to 12). Returns the FWI as [days, cells]; NaN inputs keep the previous codes."""
    n, c = T.shape; F = np.full(c, 85.0); P = np.full(c, 6.0); D = np.full(c, 15.0); out = np.empty((n, c), np.float32)
    for t in range(n):
        ok = np.isfinite(T[t]) & np.isfinite(H[t]) & np.isfinite(W[t]) & np.isfinite(R[t])
        h = np.clip(np.where(ok, H[t], 50.0), 0.0, 100.0); tt = np.where(ok, T[t], 20.0); ww = np.where(ok, W[t], 5.0); rr = np.where(ok, np.maximum(R[t], 0.0), 0.0)
        F2, P2, D2 = ffmc(tt, h, ww, rr, F), dmc(tt, h, rr, P, int(months[t])), dc(tt, rr, D, int(months[t]))
        F, P, D = np.where(ok, F2, F), np.where(ok, P2, P), np.where(ok, D2, D)
        out[t] = indices(F, P, D, ww)[2]
    return out


def fwi_for_dataset(ds):
    """FWI (daily-proxy implementation) for every day and cell of a dataset's cube; reads t2m_max, rh_min_proxy, wind_speed_meanvec and precip_day."""
    names = [str(n) for n in np.load(ds.scaler_path, allow_pickle=True)["names"]]; ix = {k: names.index(k) for k in ("t2m_max", "rh_min_proxy", "wind_speed_meanvec", "precip_day")}
    d = np.asarray(ds.cube[:, :, [ix["t2m_max"], ix["rh_min_proxy"], ix["wind_speed_meanvec"], ix["precip_day"]]], dtype=np.float64)
    months = (pd.Timestamp("2019-01-01") + pd.to_timedelta(np.arange(d.shape[0]), "D")).month.values
    return fwi_cube(d[..., 0] - 273.15, d[..., 1], d[..., 2] * 3.6, d[..., 3], months)


def persistence(rows):
    return rows.had_det_t_forest.values.astype(np.float32)


def climatology(train_rows, rows):
    """Detection share per cell and calendar month of the label day over the training rows; a cell and month without training rows gets the month's overall share."""
    m = lambda r: pd.to_datetime(r.label_date).dt.month.values
    tr = pd.DataFrame({"cell": train_rows.cell_idx.values, "month": m(train_rows), "y": train_rows.y_forest.values})
    cm = tr.groupby(["cell", "month"]).y.mean().rename("p"); mm = tr.groupby("month").y.mean()
    key = pd.MultiIndex.from_arrays([rows.cell_idx.values, m(rows)]); p = cm.reindex(key).values
    return np.where(np.isnan(p), mm.reindex(m(rows)).values, p).astype(np.float32)
