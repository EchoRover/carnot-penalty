# Data provenance

| File | What | Source | Licence / note |
|---|---|---|---|
| `Chillers.idf` | Performance curves of real electric chillers (manufacturer data fitted in the CoolTools project) | EnergyPlus `datasets/Chillers.idf`, https://github.com/NREL/EnergyPlus (branch `develop`), downloaded 13 Nov 2026 | EnergyPlus BSD-3-style licence; file unmodified |
| `abudhabi_monthly_normals.csv` | Mean daily max / min temperature and mean relative humidity, 12 months | Entered from the AI assistant's recollection of the published climate table for Abu Dhabi (UAE National Center of Meteorology normals); not downloaded | 36 numbers; not yet verified against the official table |
| `weather_hourly.csv` | 8760-hour typical year (dry-bulb, dew point, clear-sky irradiance) | **Generated** by `scripts/make_weather.py` from the normals + ASHRAE diurnal profile + Haurwitz clear-sky model | Synthetic; replace with measured hourly data when available |

Hourly station archives (NOAA ISD, Iowa Mesonet, Open-Meteo/ERA5, NASA POWER, PVGIS) were not
reachable from the machine on which the analysis was built, which is why the typical year is
synthesised from monthly normals. The analysis code accepts any 8760-hour file with the same
columns, so measured data can be dropped in without touching the model.
