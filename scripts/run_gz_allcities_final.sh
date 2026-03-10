#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE_ARG="${DEVICE_ARG:-}"

CONFIGS=(
  "configs/gz_rain_allcities_point_bimamba_final.yaml"
  "configs/gz_rain_allcities_quantile_bimamba_final.yaml"
)

echo "Using python: ${PYTHON_BIN}"
echo "Project root: ${ROOT_DIR}"

for cfg in "${CONFIGS[@]}"; do
  echo "============================================================"
  echo "Running config: ${cfg}"
  if [[ -n "${DEVICE_ARG}" ]]; then
    "${PYTHON_BIN}" scripts/train.py --config "${cfg}" --device "${DEVICE_ARG}"
  else
    "${PYTHON_BIN}" scripts/train.py --config "${cfg}"
  fi
done

echo "Done. All-city point and quantile runs finished."
