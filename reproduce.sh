#!/usr/bin/env bash
# Regenerates every number, table and figure of the report from the raw inputs in data/.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-$(command -v python3 || command -v python)}"
"$PY" scripts/make_weather.py            # data/weather_hourly.csv   (synthetic year, comparison only)
"$PY" scripts/make_station_weather.py    # data/station_hourly/*.csv (measured weather, the base case)
"$PY" -m pytest -q
"$PY" scripts/run_analysis.py > results/run_log.json
if command -v pdflatex >/dev/null; then
  (cd report && pdflatex -interaction=nonstopmode report.tex >/dev/null && pdflatex -interaction=nonstopmode report.tex >/dev/null)
  echo "report/report.pdf rebuilt"
fi
echo "done - see results/results.json"
