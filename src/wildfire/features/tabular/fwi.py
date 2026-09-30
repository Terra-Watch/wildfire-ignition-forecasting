"""Vectorised numpy port of the Canadian fire-weather codes (FFMC, DMC, DC, ISI, BUI, FWI).

Translated line by line from the scalar reference implementation (Van Wagner and Pickett 1985; Van Wagner 1987).
"""
import numpy as np
FFMC_COEF = 250.0 * 59.5 / 101.0
ELL02 = np.array([7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8])   # DMC day length, 10 < lat <= 30 N (lat_adjust)
ELL01 = np.array([6.5, 7.5, 9, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8, 7, 6])       # lat > 30 N
FL01 = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5, 2.4, 0.4, -1.6, -1.6])   # DC day length, lat > 20 N
def ffmc(f0, temp, rh, ws, prec):
    wmo = FFMC_COEF * (101 - f0) / (59.5 + f0)
    ra = np.where(prec > 0.5, prec - 0.5, prec)
    with np.errstate(all="ignore"):
        rain = 42.5 * ra * np.exp(-100 / (251 - wmo)) * (1 - np.exp(-6.93 / ra))
        wr = np.where(wmo > 150, wmo + 0.0015 * (wmo - 150) ** 2 * np.sqrt(ra) + rain, wmo + rain)
    wmo = np.where(prec > 0.5, wr, wmo); wmo = np.minimum(wmo, 250)
    ed = 0.942 * rh ** 0.679 + 11 * np.exp((rh - 100) / 10) + 0.18 * (21.1 - temp) * (1 - 1 / np.exp(rh * 0.115))
    ew = 0.618 * rh ** 0.753 + 10 * np.exp((rh - 100) / 10) + 0.18 * (21.1 - temp) * (1 - 1 / np.exp(rh * 0.115))
    dry = (wmo < ed) & (wmo < ew)
    z = np.where(dry, 0.424 * (1 - ((100 - rh) / 100) ** 1.7) + 0.0694 * np.sqrt(ws) * (1 - ((100 - rh) / 100) ** 8), 0.0)
    x = z * 0.581 * np.exp(0.0365 * temp)
    wm = np.where(dry, ew - (ew - wmo) / 10 ** x, wmo)
    wet = wmo > ed
    z = np.where(wet, 0.424 * (1 - (rh / 100) ** 1.7) + 0.0694 * np.sqrt(ws) * (1 - (rh / 100) ** 8), z)
    x = z * 0.581 * np.exp(0.0365 * temp)
    wm = np.where(wet, ed + (wmo - ed) / 10 ** x, wm)
    return np.clip(59.5 * (250 - wm) / (FFMC_COEF + wm), 0, 101)
def dmc(d0, temp, rh, prec, lat, mon):
    temp = np.maximum(temp, -1.1); ell = ELL02 if 10 < lat <= 30 else ELL01
    rk = 1.894 * (temp + 1.1) * (100 - rh) * ell[mon - 1] * 1e-4
    with np.errstate(all="ignore"):
        rw = 0.92 * prec - 1.27; wmi = 20 + 280 / np.exp(0.023 * d0)
        b = np.where(d0 <= 33, 100 / (0.5 + 0.3 * d0), np.where(d0 <= 65, 14 - 1.3 * np.log(d0), 6.2 * np.log(d0) - 17.2))
        wmr = wmi + 1000 * rw / (48.77 + b * rw); pr_rain = 43.43 * (5.6348 - np.log(wmr - 20))
    pr = np.where(prec <= 1.5, d0, pr_rain); pr = np.maximum(pr, 0)
    return np.maximum(pr + rk, 0)
def dc(c0, temp, rh, prec, lat, mon):
    temp = np.maximum(temp, -2.8); fl = FL01[mon - 1] if lat > 20 else 1.4
    pe = np.maximum((0.36 * (temp + 2.8) + fl) / 2, 0)
    with np.errstate(all="ignore"):
        rw = 0.83 * prec - 1.27; smi = 800 * np.exp(-c0 / 400); dr0 = np.maximum(c0 - 400 * np.log(1 + 3.937 * rw / smi), 0)
    dr = np.where(prec <= 2.8, c0, dr0)
    return np.maximum(dr + pe, 0)
def isi(f, ws):
    fm = FFMC_COEF * (101 - f) / (59.5 + f)
    return 0.208 * np.exp(0.05039 * ws) * 91.9 * np.exp(-0.1386 * fm) * (1 + fm ** 5.31 / 49300000)
def bui(dm, dr):
    with np.errstate(all="ignore"):
        b1 = np.where((dm == 0) & (dr == 0), 0, 0.8 * dr * dm / (dm + 0.4 * dr)); p = np.where(dm == 0, 0, (dm - b1) / dm)
        cc = 0.92 + (0.0114 * dm) ** 1.7; b0 = np.maximum(dm - cc * p, 0)
    return np.where(b1 < dm, b0, b1)
def fwi(i, b):
    with np.errstate(all="ignore"):
        bb = np.where(b > 80, 0.1 * i * (1000 / (25 + 108.64 / np.exp(0.023 * b))), 0.1 * i * (0.626 * b ** 0.809 + 2))
        return np.where(bb <= 1.0, bb, np.exp(2.72 * (0.434 * np.log(bb)) ** 0.647))
