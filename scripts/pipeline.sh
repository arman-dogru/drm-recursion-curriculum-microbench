#!/bin/sh
# Full experiment chain; every stage skips up-to-date outputs, so it is safe to rerun.
#   PY       python interpreter with requirements.lock installed (default: python)
#   WORKERS  parallel single-threaded runs (default 7)
#   COMMIT=1 commit the dev tuning outputs before any eval run is launched (provenance)
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
W=${WORKERS:-7}
ts() { date -Iseconds; }
echo "$(ts) stage tune"
$PY scripts/run_grid.py tune --workers $W
# exit 3 if any chosen lr is at a grid edge -> stop, extend on dev, rerun
# FAIL_ON_EDGE=0 only after the pre-declared last grid extension (edges are then disclosed)
if [ "${FAIL_ON_EDGE:-1}" = 1 ]; then $PY scripts/tune.py --fail-on-edge; else $PY scripts/tune.py; fi
if [ "${COMMIT:-0}" = 1 ]; then
  git add results/tuning.json results/raw/dev logs
  git commit -q -m "Dev tuning outputs (committed before any eval run is launched)" || true
  echo "$(ts) committed tuning at $(git rev-parse --short HEAD)"
fi
echo "$(ts) stage main"
$PY scripts/run_grid.py main --workers $W
echo "$(ts) stage ksweep"
$PY scripts/run_grid.py ksweep --workers $W
echo "$(ts) stage aggregate"
$PY scripts/aggregate.py
$PY scripts/demo_trajectories.py
$PY scripts/make_figures.py
$PY scripts/claims.py
$PY scripts/render_readme.py
echo "$(ts) pipeline done"
