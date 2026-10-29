#!/bin/sh
# Strip machine-specific absolute paths from finished logs before they are committed
# (the repo is public). Only run on logs that are no longer being written.
#   sh scripts/sanitize_logs.sh logs/tune.log logs/r1/pipeline.log ...
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
for f in "$@"; do
  sed -i -e "s#$ROOT/##g" -e 's#/home/[A-Za-z0-9_.-]*/[A-Za-z0-9_./-]*/bin/python#$PY#g' \
         -e 's#/home/[A-Za-z0-9_.-]*/#~/#g' -e 's#/workspace/[A-Za-z0-9_./-]*#<scratch>#g' "$f"
done
