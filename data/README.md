# Data provenance

Everything the analysis reads is in this folder, so the run needs no network access. All downloads were made on
13 November 2026.

| File | What | Source | Licence / note |
|---|---|---|---|
| `Chillers.idf` | Performance curves of real water-cooled electric chillers (manufacturer data) | EnergyPlus `datasets/Chillers.idf`, https://github.com/NREL/EnergyPlus (branch `develop`) | EnergyPlus BSD-3-style licence; file unmodified |
| `AirCooledChiller.idf` | Performance curves of real air-cooled electric chillers (manufacturers' catalogue data) | EnergyPlus `datasets/AirCooledChiller.idf`, same repository and branch | EnergyPlus BSD-3-style licence; file unmodified |
| `station/<STATION>_<YEAR>.csv` | Hourly airport weather reports (METAR): air temperature, dew point, relative humidity, local time. OMAA = Abu Dhabi International, OMAD = Al Bateen (Abu Dhabi island), OMDB = Dubai International, OMSJ = Sharjah International | Iowa Environmental Mesonet ASOS/METAR archive, https://mesonet.agron.iastate.edu/request/download.phtml (request below) | Freely available; credit the Iowa Environmental Mesonet. Files unmodified |
| `station_hourly/<STATION>_<YEAR>.csv` | The same observations as 8760-row years in the format the model reads | **Generated** by `scripts/make_station_weather.py` | Missing hours (at most 50 a year) are interpolated; the count is in each file's first line |
| `reanalysis/abudhabi_reanalysis_<YEAR>.csv` | Hourly dry-bulb, dew point and global horizontal irradiance for the Abu Dhabi airport grid cell (24.43N, 54.65E), local time | Open-Meteo Historical Weather API, https://open-meteo.com (request below), default best-match reanalysis | CC BY 4.0. A gridded model product, not a station record. Used for the hourly irradiance weights and as a comparison data set |
| `abudhabi_monthly_normals.csv` | Mean daily maximum, minimum and relative humidity, 12 months | Entered from the AI assistant's recollection of the published climate table for Abu Dhabi; not downloaded | Used only for the synthetic comparison year. Checked in this work against the 2022-2023 airport record: its maxima are on average 0.5 K below the record, its minima 3.2 K below |
| `weather_hourly.csv` | Synthetic typical year (dry-bulb, dew point, clear-sky irradiance) | **Generated** by `scripts/make_weather.py` from the normals, the ASHRAE diurnal profile and the Haurwitz clear-sky model | Comparison only: it is the weather the first version of the report was built on |

## The download requests

Station observations, one request per station and year (`report_type=3` is the routine hourly report):

    https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=OMAA&data=tmpc&data=dwpc&data=relh&year1=2023&month1=1&day1=1&year2=2024&month2=1&day2=1&tz=Asia%2FDubai&format=onlycomma&latlon=no&elev=no&missing=M&trace=T&direct=no&report_type=3

Reanalysis, one request per year (the JSON answer was written out as CSV without changing any value):

    https://archive-api.open-meteo.com/v1/archive?latitude=24.433&longitude=54.651&start_date=2023-01-01&end_date=2023-12-31&hourly=temperature_2m,dew_point_2m,relative_humidity_2m,shortwave_radiation&timezone=Asia%2FDubai
