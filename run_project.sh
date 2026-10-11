#!/usr/bin/env bash
# LA Collision Insurance Risk - runner for macOS / Linux.
# Usage:  ./run_project.sh [pipeline|tests|api|report|all]   (no argument = menu)
set -euo pipefail
cd "$(dirname "$0")"
GRAPH="$PWD/data/cache/test_grid.graphml"
PY=$(command -v python3 || command -v python || true)
[ -z "$PY" ] && { echo "Install Python 3.10+ first: https://www.python.org/downloads/"; exit 1; }

if [ ! -x .venv/bin/python ]; then
  echo "Creating Python environment in .venv (first run only, a few minutes) ..."
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements-dev.txt
fi
VPY=.venv/bin/python

pipeline() {
  "$VPY" -W ignore scripts/run_pipeline_offline.py
  "$VPY" -W ignore scripts/compare_oversamplers.py
  "$VPY" -W ignore scripts/make_report_figures.py
  echo "Pipeline finished. Results are in data/processed, models and reports."
}
tests() { "$VPY" -m pytest -q -p no:warnings; }
api() {
  [ -f models/risk_model.joblib ] && [ -f "$GRAPH" ] || pipeline
  echo "Dashboard at http://localhost:8000/dashboard (API docs: /docs)  (road network loads in ~20-60 s; Ctrl+C to stop)"
  CRS_RISK_GRAPHML="$GRAPH" "$VPY" -m uvicorn api.app:app --host 127.0.0.1 --port 8000
}
report() {
  command -v node >/dev/null || { echo "Install Node.js: https://nodejs.org"; return 1; }
  [ -f reports/model_comparison.csv ] || pipeline
  [ -d node_modules/docx ] || npm install docx
  "$VPY" -W ignore scripts/report/build_data.py
  node scripts/report/build_report.js
}

case "${1:-menu}" in
  pipeline) pipeline ;; tests) tests ;; api) api ;; report) report ;;
  all) pipeline && tests && api ;;
  *)
    echo "1) Full pipeline  2) Tests  3) Dashboard + API  4) Rebuild Word report  5) Everything"
    read -rp "Choose: " c
    case "$c" in 1) pipeline ;; 2) tests ;; 3) api ;; 4) report ;; 5) pipeline && tests && api ;; *) echo "Bye" ;; esac ;;
esac
