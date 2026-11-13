#!/usr/bin/env bash
# Regenerates every number, table and figure of the report from the raw inputs in data/.
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest -q
python scripts/make_weather.py
python scripts/run_analysis.py > results/run_log.json
if command -v pdflatex >/dev/null; then
  (cd report && pdflatex -interaction=nonstopmode report.tex >/dev/null && pdflatex -interaction=nonstopmode report.tex >/dev/null)
  echo "report/report.pdf rebuilt"
fi
echo "done - see results/results.json"
