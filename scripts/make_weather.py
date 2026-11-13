#!/usr/bin/env python3
"""
make_weather.py -- builds data/weather_hourly.csv, an 8760-hour typical year for Abu Dhabi.

Why synthetic?  Hourly station archives were not reachable from the build machine, so the
typical year is constructed transparently from three published ingredients:

  1. Monthly climate normals for Abu Dhabi (data/abudhabi_monthly_normals.csv):
     mean daily maximum, mean daily minimum, mean relative humidity.
  2. The ASHRAE design-day diurnal temperature profile (ASHRAE Handbook-Fundamentals,
     Ch. 14, "fraction of daily temperature range" table, apparent solar time).
  3. The Haurwitz (1945) clear-sky model for global horizontal irradiance:
         GHI = 1098 cos(z) exp(-0.057 / cos(z))   [W/m2]
     with solar zenith z from latitude, declination (Cooper) and hour angle.

Dew point is held at its monthly mean through the day (constant moisture content).
That assumption is tested in the sensitivity analysis of run_analysis.py.

To use REAL hourly data instead, simply replace data/weather_hourly.csv with a file that
has the same columns (month, day, hour, T_C, Td_C, ghi_Wm2) -- nothing else changes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import carnot_penalty as cp  # noqa: E402

LAT, LON, TZ_MERIDIAN = 24.433, 54.651, 60.0   # Abu Dhabi Intl Airport; UTC+4 -> 60 deg E
# ASHRAE fraction of daily range at solar hours 1..24 (T = Tmax - f * daily_range)
ASHRAE_F = np.array([0.88, 0.92, 0.95, 0.98, 1.00, 0.98, 0.91, 0.74, 0.55, 0.38, 0.23, 0.13,
                     0.05, 0.00, 0.00, 0.06, 0.14, 0.24, 0.39, 0.50, 0.59, 0.68, 0.75, 0.82])
DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def ashrae_fraction(solar_hour):
    """Periodic linear interpolation of the ASHRAE table (defined at solar hours 1..24)."""
    xp = np.arange(0, 26)                      # 0 (=24), 1..24, 25 (=1)
    fp = np.r_[ASHRAE_F[-1], ASHRAE_F, ASHRAE_F[0]]
    return np.interp(np.mod(solar_hour, 24.0), xp, fp)


def haurwitz_ghi(day_of_year, solar_hour, lat_deg=LAT):
    decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + day_of_year) / 365.0)   # Cooper (1969)
    hour_angle = np.radians(15.0 * (solar_hour - 12.0))
    lat = np.radians(lat_deg)
    cosz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    cosz = np.clip(cosz, 0.0, None)
    return np.where(cosz > 0.01, 1098.0 * cosz * np.exp(-0.057 / np.maximum(cosz, 0.01)), 0.0)


def build():
    nm = pd.read_csv(ROOT / "data" / "abudhabi_monthly_normals.csv", comment="#")
    tmean = 0.5 * (nm.tmax_C + nm.tmin_C)
    nm["td_C"] = cp.dew_point_from_rh(tmean, nm.rh_pct)
    # monthly values sit at mid-month; interpolate linearly (periodically) to each day of the year
    mid = np.cumsum(DAYS_IN_MONTH) - DAYS_IN_MONTH / 2.0
    doy = np.arange(1, 366)

    def daily(col):
        return np.interp(doy, np.r_[mid[-1] - 365, mid, mid[0] + 365], np.r_[col.iloc[-1], col, col.iloc[0]])

    tmax, tmin, td = daily(nm.tmax_C), daily(nm.tmin_C), daily(nm.td_C)
    month = np.repeat(np.arange(1, 13), DAYS_IN_MONTH)
    dom = np.concatenate([np.arange(1, n + 1) for n in DAYS_IN_MONTH])
    rows = []
    for i, d in enumerate(doy):
        clock = np.arange(24) + 0.5                                  # middle of each clock hour
        solar = clock + (LON - TZ_MERIDIAN) / 15.0                   # longitude correction (-0.36 h)
        T = tmax[i] - ashrae_fraction(solar) * (tmax[i] - tmin[i])
        Td = np.minimum(td[i], T)
        rows.append(pd.DataFrame(dict(month=month[i], day=dom[i], hour=np.arange(24),
                                      T_C=np.round(T, 3), Td_C=np.round(Td, 3),
                                      ghi_Wm2=np.round(haurwitz_ghi(d, solar), 1))))
    wx = pd.concat(rows, ignore_index=True)
    out = ROOT / "data" / "weather_hourly.csv"
    with open(out, "w") as f:
        f.write("# Synthetic typical year for Abu Dhabi built by scripts/make_weather.py (see data/README.md)\n")
        wx.to_csv(f, index=False)
    print(f"wrote {out}  ({len(wx)} rows)")
    return wx


if __name__ == "__main__":
    build()
