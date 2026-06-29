#!/usr/bin/env bash
set -euo pipefail
for seed in ${SEEDS:-0 1 2 3 4}; do
  mewon suite --config configs/quick.yaml --out runs/seed_${seed} --device ${DEVICE:-cpu}
done
mewon report --runs runs --out runs/report
