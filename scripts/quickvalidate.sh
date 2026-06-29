#!/usr/bin/env bash
set -euo pipefail
mewon validate --out runs/validate --device ${DEVICE:-cpu}
mewon report --runs runs/validate --out runs/validate_report
mewon dashboard --runs runs/validate --out runs/validate_dash
