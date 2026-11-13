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
| `results/` | `results.json`, `monthly_table.csv`, `chiller_library.csv` produced by the code |
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

## Using your own weather data

Replace `data/weather_hourly.csv` with any 8760-row file that has the columns
`month, day, hour, T_C, Td_C, ghi_Wm2` (for example an hourly airport record or an EPW
typical year) and re-run `scripts/run_analysis.py`. Nothing else needs to change.

## Tools and AI disclosure

Python (NumPy, pandas, Matplotlib, pytest) and LaTeX. An AI assistant (Anthropic's Claude) was used in
this work (problem framing, model development, code and tests, data preparation, figures, drafting);
the report's disclosure section lists the uses and the checks applied.
