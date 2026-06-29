#!/usr/bin/env bash
set -euo pipefail
DEVICE=${DEVICE:-cuda}
OUT=${OUT:-runs/full}
if [[ "${PREPARE_FINEWEB:-0}" == "1" ]]; then
  python -m mewon.data.fineweb --out data/fineweb --tokens ${FINEWEB_TOKENS:-1000000000}
fi
mewon suite --config configs/full.yaml --out "$OUT" --device "$DEVICE"
mewon report --runs "$OUT" --out "$OUT/report"
mewon dashboard --runs "$OUT" --out "$OUT/dashboard"
