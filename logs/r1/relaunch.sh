#!/bin/sh
# Watcher used once on 2026-10-28: wait for the first pipeline (pid given as $1) to exit,
# then rerun the chain so the dev lr extension is picked up. Paths are repo-relative.
cd "$(dirname "$0")/../.."
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "$(date -Iseconds) first pipeline exited; relaunching with lr extension" >> logs/r1/pipeline.log
COMMIT=1 PY=${PY:-python} WORKERS=7 scripts/pipeline.sh > logs/r1/pipeline_b.log 2>&1
