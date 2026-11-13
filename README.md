# The Carnot Penalty: when should the Gulf make its cold?

Carnot 2026 (Afkar 2026, IIT Delhi – Abu Dhabi), team mathematical-modelling submission.

**Team Bueno**: Evan Tobias, Krishna Gaur, James Joshua

**Revised version, 14 November 2026.** It replaces the version submitted on 13 November, which is kept under the
git tag `submitted-2026-11-13`. What changed, and why, is in Section 6 of the report and at the end of this file.

**Question.** The UAE grid is filling up with midday solar, while most of its electricity goes to cooling that is
also needed after sunset. Should chillers make cold at solar noon and store it in a tank? Noon is when electricity
is cleanest, but it is also when the outdoor air is hottest, so a chiller should then be at its least efficient.

**Model.** We call the extra electricity the *Carnot penalty* and derive it in closed form:

    π = (T_sink,solar − T_sink,night) / L_night        (L = effective temperature lift of the chiller)

A cold tank beats a battery on energy when π < η_tank/η_battery − 1, and solar-following cold storage cuts emissions
when (solar-hour emissions)/(night emissions) < η_tank/(1+π).

**Answer.** With 268 real chillers and four station-years of measured Abu Dhabi weather:

| Cooling season, May to October | Air-cooled chiller | Water-cooled (district cooling) plant |
|---|---|---|
| Heat sink | outdoor dry-bulb | wet-bulb (cooling tower) |
| Carnot penalty π | about 15.5% | about −2.8% (a small bonus) |
| Cold tank returns | about 84% | about 100% |
| Against an 85% battery | a tie | the tank wins on every day |

The water-cooled result comes from the humidity: in Abu Dhabi the dew point falls by about 4.6 K in the solar hours,
and a cooling tower's heat sink follows the dew point much more than the air temperature.

**Read `report/report.pdf` first.**

## Folder map

| Path | What it is |
|---|---|
| `report/report.pdf` | The report (problem, model, assumptions, results, interpretation) |
| `report/report.tex`, `report/figures/` | LaTeX source of the report and the figures written by the code |
| `report/numbers.tex`, `report/table_*.tex` | Every number and table quoted in the report, written by the code |
| `src/carnot_penalty.py` | The whole model (about 200 lines): psychrometrics, chiller, Carnot penalty, storage choice |
| `scripts/make_station_weather.py` | Builds the measured 8760-hour years from the raw airport observations |
| `scripts/make_weather.py` | Builds the synthetic year from monthly normals (kept for comparison only) |
| `scripts/run_analysis.py` | Calibration, results, sensitivity, Monte Carlo, all figures, all report numbers |
| `tests/test_model.py` | 29 unit tests (psychrometric chart points, closed form, both chiller libraries, the station data, …) |
| `data/` | Input data and `data/README.md` with provenance |
| `results/` | `results.json`, `monthly_table.csv`, `datasets_table.csv`, `chiller_library.csv` produced by the code |
| `results/reanalysis_check/` | Outputs of the first version's model on the two reanalysis years, from the check added on 13 November; kept for the record |
| `reproduce.sh` | One command that does everything below |

## Reproduce everything (under a minute on a laptop)

```bash
python -m pip install -r requirements.txt     # numpy, pandas, matplotlib, pytest
python scripts/make_weather.py                # data/weather_hourly.csv (synthetic year, comparison only)
python scripts/make_station_weather.py        # data/station_hourly/*.csv (measured weather)
python -m pytest -q                           # 29 tests should pass
python scripts/run_analysis.py                # results/, report/figures/, report/numbers.tex
cd report && pdflatex report.tex && pdflatex report.tex   # optional: rebuild the PDF
```

or simply `bash reproduce.sh`. The run is deterministic (fixed seed 2026, no network access); re-running reproduces
`results/results.json` exactly.

## Using your own weather data

Put an 8760-row file with the columns `month, day, hour, T_C, Td_C, ghi_Wm2` into `data/station_hourly/` and add its
name to `AD_SETS` (the base case) or `OTHER_SETS` (comparison only) at the top of `scripts/run_analysis.py`.

## What changed since the version of 13 November

The first version used a synthetic typical year built from monthly climate normals and design values for the
air-cooled chiller. It reported a penalty of 18.5% for air-cooled and +3.5% for water-cooled plants. This version
tests the same model against measurements:

- **Weather is now measured**: hourly observations from Abu Dhabi International Airport and Al Bateen airport,
  2022 and 2023 (1460 days), with Dubai and Sharjah as a regional check.
- **Both chiller types are now calibrated on real machines**: the 111 air-cooled chillers of the EnergyPlus
  air-cooled library joined the 157 water-cooled ones.
- **Two numbers moved.** The water-cooled penalty went from +3.5% to about −2.8%, because the measured dew point
  falls in the solar hours and the synthetic year had held it constant. The air-cooled penalty went from 18.5% to
  about 15.5%: real nights are warmer than the normals table said, which lowers it, and real air-cooled chillers are
  more temperature-sensitive than the design values, which raises it.
- Every number in the report is now written by the code (`report/numbers.tex`), not typed.

## Tools and AI disclosure

Python (NumPy, pandas, Matplotlib, pytest) and LaTeX. An AI assistant (Anthropic's Claude) was used in this work
(problem framing, model development, code and tests, finding and preparing the data, figures, testing the first
version against measurements, drafting); the report's disclosure section lists the uses and the checks applied.
