# The Carnot Penalty — when should the Gulf make its cold?

Carnot 2026 (Afkar 2026, IIT Delhi – Abu Dhabi) — team mathematical-modelling submission.

**Team Bueno** — Evan Tobias, Krishna Gaur, James Joshua

**Question.** The UAE grid is filling up with midday solar, while most of its electricity
goes to cooling that is also needed after sunset. Should chillers make cold at solar noon
and store it in a tank? Noon is when electricity is cleanest — but it is also when the
outdoor heat sink is hottest, so every chiller is at its *least* efficient.

**Model.** We call the extra electricity the *Carnot penalty* and derive it in closed form:

    π = (T_sink,solar − T_sink,night) / L_night        (L = condensing − evaporating temperature)

A cold tank beats a battery on energy when π < η_tank/η_battery − 1, and solar-following
cold storage cuts emissions when (solar-hour emissions)/(night emissions) < η_tank/(1+π).

**Read `report/report.pdf` first.**

## Folder map

| Path | What it is |
|---|---|
| `report/report.pdf` | The report (problem, model, assumptions, results, interpretation) |
| `report/report.tex`, `report/figures/` | LaTeX source of the report and the figures written by the code |
| `src/carnot_penalty.py` | The whole model (≈200 lines): psychrometrics, chiller, Carnot penalty, storage choice |
| `scripts/make_weather.py` | Builds the 8760-hour Abu Dhabi typical year from the monthly normals |
| `scripts/run_analysis.py` | Calibration, results, sensitivity, Monte Carlo, all figures |
| `tests/test_model.py` | 19 unit tests (psychrometric chart points, second law, closed form, parser, …) |
| `data/` | Input data + `data/README.md` with provenance |
| `results/` | `results.json`, `monthly_table.csv`, `chiller_library.csv` produced by the code; `reanalysis_check/` holds the reruns on hourly reanalysis weather |
| `reproduce.sh` | One command that does everything below |

## Reproduce everything (≈1 minute on a laptop)

```bash
python -m pip install -r requirements.txt     # numpy, pandas, matplotlib, pytest
python -m pytest -q                           # 19 tests should pass
python scripts/make_weather.py                # data/weather_hourly.csv
python scripts/run_analysis.py                # results/ and report/figures/
cd report && pdflatex report.tex && pdflatex report.tex   # optional: rebuild the PDF
```

or simply `bash reproduce.sh`. The run is deterministic (fixed seed 2026, no network access);
re-running reproduces `results/results.json` exactly.

## Check against hourly weather for 2022 and 2023 (added 13 November, before the deadline)

The report's weather is a synthetic typical year (assumption A6), and the first next step it lists is a rerun on
hourly airport data. This is a first version of that rerun, on reanalysis rather than station observations.
`data/reanalysis/` holds two actual years, 2022 and 2023, of hourly dry-bulb,
dew point and irradiance for the Abu Dhabi airport grid cell, from the Open-Meteo Historical Weather API. The model
and the code are unchanged; only the weather file was swapped.

| May to October unless stated | Synthetic year (report) | Reanalysis 2022 | Reanalysis 2023 |
|---|---|---|---|
| Air-cooled: Carnot penalty, May to Oct (%) | 18.5 | 18.0 | 19.2 |
| Air-cooled: solar minus night sink temperature (K) | 7.3 | 7.2 | 7.8 |
| Air-cooled: effective round-trip efficiency of the tank (%) | 81.9 | 82.3 | 81.5 |
| Air-cooled: Monte Carlo chance the tank beats the battery (%) | 21.0 | 22.1 | 19.3 |
| Water-cooled: Carnot penalty, May to Oct (%) | +3.5 | -3.0 | -3.5 |
| Water-cooled: solar minus night sink temperature (K) | +1.7 | -1.5 | -1.8 |
| Water-cooled: effective round-trip efficiency of the tank (%) | 93.7 | 100.1 | 100.6 |
| Water-cooled: cooling-season days on which the tank beats the battery (%) | 100.0 | 99.5 | 100.0 |
| Water-cooled: Monte Carlo chance the tank beats the battery (%) | 91.1 | 99.9 | 100.0 |
| Dew point, solar hours minus night (K) | +0.0 | -6.0 | -6.5 |
| Psychrometric damping, dTwb/dT | 0.24 | 0.27 | 0.26 |

What it shows:

- **Air-cooled: confirmed.** The penalty on the reanalysis years is within one percentage point of the report's 18.5%, and a
  battery is still the better store.
- **Water-cooled: the conclusion holds and gets stronger, but the number changes.** In the reanalysis the air in the
  solar hours is about 6 K drier (dew point) than the night air. The report held the dew point constant through the day
  and tested only a shift of 3 K either way. With that humidity cycle the wet-bulb is slightly lower in the solar
  hours than at night, so the penalty becomes a small bonus of about 3%, and the tank beats the battery on almost every
  cooling-season day.
- The report's 3.5% for water-cooled plants should therefore be read as a conservative figure that follows from the
  constant-dew-point assumption, not as a measured one.

Limits of this check: reanalysis is a gridded model product, not a station record. The size and even the sign of the
day-night humidity cycle depend on distance from the coast (sea breeze), so the check should be repeated on station
observations before it is applied to a specific plant.

To reproduce: `cp data/reanalysis/abudhabi_reanalysis_2023.csv data/weather_hourly.csv && python scripts/run_analysis.py`.
That overwrites `results/` and `report/figures/`; `python scripts/make_weather.py` followed by
`python scripts/run_analysis.py` restores the synthetic year used in the report. The stored outputs of both runs are in
`results/reanalysis_check/`.

## Using your own weather data

Replace `data/weather_hourly.csv` with any 8760-row file that has the columns
`month, day, hour, T_C, Td_C, ghi_Wm2` (for example an hourly airport record or an EPW
typical year) and re-run `scripts/run_analysis.py`. Nothing else needs to change.

## Tools and AI disclosure

Python (NumPy, pandas, Matplotlib, pytest) and LaTeX. An AI assistant (Anthropic's Claude) was used in
this work (problem framing, model development, code and tests, data preparation, figures, drafting);
the report's disclosure section lists the uses and the checks applied.
