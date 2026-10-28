#!/bin/sh
# Full experiment chain; every stage skips up-to-date outputs, so it is safe to rerun.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-/home/box/work/drm-venv/bin/python}
W=${WORKERS:-7}
$PY scripts/run_grid.py tune --workers $W
$PY scripts/tune.py
$PY scripts/run_grid.py main --workers $W
$PY scripts/run_grid.py ksweep --workers $W
$PY scripts/aggregate.py
$PY scripts/demo_trajectories.py
$PY scripts/make_figures.py
