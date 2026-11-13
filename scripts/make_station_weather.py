#!/usr/bin/env python3
"""
make_station_weather.py -- builds data/station_hourly/<STATION>_<YEAR>.csv: 8760-hour years of MEASURED
dry-bulb and dew-point temperature from airport weather reports, in the format run_analysis.py reads.

Inputs
  data/station/<STATION>_<YEAR>.csv
        Hourly airport observations (METAR) exactly as downloaded from the Iowa Environmental Mesonet
        archive. Columns: station, valid (local time, UTC+4), tmpc, dwpc, relh.
  data/reanalysis/abudhabi_reanalysis_<YEAR>.csv
        Hourly global horizontal irradiance for the same year (Open-Meteo reanalysis, Abu Dhabi airport
        grid cell). Airports do not report irradiance; it is used only as the hourly weight that says
        when the sun shines, so the same file serves all four airports (they lie within about 150 km of each other).

Conventions of the output
  Row "hour = h" stands for the clock hour [h, h+1).
  T_C and Td_C are the mean of the observations at h:00 and (h+1):00.
  ghi_Wm2 is the reanalysis value stamped (h+1):00, which is the mean over the preceding hour.
  Hours with no observation are filled by linear interpolation in time; the number of such hours and
  the longest gap are written in the first line of each output file.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, REAN, OUT = ROOT / "data" / "station", ROOT / "data" / "reanalysis", ROOT / "data" / "station_hourly"

STATIONS = {
    "OMAA": "Abu Dhabi International Airport (on the mainland, east of the city)",
    "OMAD": "Al Bateen Executive Airport (on Abu Dhabi island)",
    "OMDB": "Dubai International Airport",
    "OMSJ": "Sharjah International Airport",
}
SETS = [("OMAA", 2022), ("OMAA", 2023), ("OMAD", 2022), ("OMAD", 2023), ("OMDB", 2023), ("OMSJ", 2023)]
N = 8760


def hourly_observations(path, year):
    """Return arrays T, Td of length N+1 (top of every hour of the year, plus the first hour of the next
    year), NaN where there is no observation within 15 minutes of the top of the hour."""
    raw = pd.read_csv(path)
    raw["tmpc"] = pd.to_numeric(raw["tmpc"], errors="coerce")
    raw["dwpc"] = pd.to_numeric(raw["dwpc"], errors="coerce")
    raw = raw.dropna(subset=["tmpc", "dwpc"])
    when = pd.to_datetime(raw["valid"])
    k = (when - pd.Timestamp(year=year, month=1, day=1)).dt.total_seconds().values / 3600.0
    i = np.rint(k).astype(int)
    keep = (np.abs(k - i) <= 0.26) & (i >= 0) & (i <= N)
    i, t, d = i[keep], raw["tmpc"].values[keep], raw["dwpc"].values[keep]
    d = np.minimum(d, t)
    cnt = np.bincount(i, minlength=N + 1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.bincount(i, weights=t, minlength=N + 1) / cnt
        Td = np.bincount(i, weights=d, minlength=N + 1) / cnt
    return T, Td, cnt > 0


def longest_gap(ok):
    best = run = 0
    for v in ok:
        run = 0 if v else run + 1
        best = max(best, run)
    return best


def build_one(station, year):
    T, Td, ok = hourly_observations(RAW / f"{station}_{year}.csv", year)
    idx = np.arange(N + 1)
    missing, gap = int((~ok[:N]).sum()), longest_gap(ok[:N])
    T = np.interp(idx, idx[ok], T[ok])
    Td = np.interp(idx, idx[ok], Td[ok])
    rean = pd.read_csv(REAN / f"abudhabi_reanalysis_{year}.csv", comment="#")
    assert len(rean) == N
    g = rean["ghi_Wm2"].values.astype(float)
    g_mid = np.r_[g[1:], g[-1]]                       # value stamped (h+1):00 = mean over [h, h+1)
    T_mid = 0.5 * (T[:-1] + T[1:])
    Td_mid = np.minimum(0.5 * (Td[:-1] + Td[1:]), T_mid)
    stamp = pd.Timestamp(year=year, month=1, day=1) + pd.to_timedelta(np.arange(N), unit="h")
    wx = pd.DataFrame(dict(month=stamp.month, day=stamp.day, hour=stamp.hour,
                           T_C=np.round(T_mid, 3), Td_C=np.round(Td_mid, 3), ghi_Wm2=np.round(g_mid, 1)))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{station}_{year}.csv"
    with open(out, "w") as f:
        f.write(f"# Measured hourly weather, {STATIONS[station]}, {year}, local time. Built by "
                f"scripts/make_station_weather.py from data/station/{station}_{year}.csv; irradiance from "
                f"data/reanalysis. {missing} of {N} hours had no observation and were interpolated "
                f"(longest gap {gap} h).\n")
        wx.to_csv(f, index=False)
    print(f"wrote {out.relative_to(ROOT)}  missing hours {missing}, longest gap {gap} h")
    return wx, missing, gap


def build():
    return {f"{s}_{y}": build_one(s, y) for s, y in SETS}


if __name__ == "__main__":
    build()
    sys.exit(0)
