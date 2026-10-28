#!/bin/sh
cd /home/box/work/drm-recursion-curriculum-microbench
while kill -0 85354 2>/dev/null; do sleep 30; done
echo "$(date -Iseconds) first pipeline exited; relaunching with lr extension" >> logs/r1/pipeline.log
COMMIT=1 PY=/home/box/work/drm-venv/bin/python WORKERS=7 scripts/pipeline.sh > logs/r1/pipeline_b.log 2>&1
