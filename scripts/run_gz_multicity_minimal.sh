#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIGS=(
  "configs/gz_rain_guiyang_point_bimamba_opt.yaml"
  "configs/gz_rain_guiyang_quantile_bimamba_asinh_v5.yaml"
  "configs/gz_rain_zunyi_point_bimamba_opt.yaml"
  "configs/gz_rain_zunyi_quantile_bimamba_asinh_v5.yaml"
  "configs/gz_rain_liupanshui_point_bimamba_opt.yaml"
  "configs/gz_rain_liupanshui_quantile_bimamba_asinh_v5.yaml"
  "configs/gz_rain_tongren_point_bimamba_opt.yaml"
  "configs/gz_rain_tongren_quantile_bimamba_asinh_v5.yaml"
  "configs/gz_rain_qiandongnan_point_bimamba_opt.yaml"
  "configs/gz_rain_qiandongnan_quantile_bimamba_asinh_v5.yaml"
)

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE_ARG="${DEVICE_ARG:-}"

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

echo "============================================================"
echo "Training complete. Summarizing metrics..."
"${PYTHON_BIN}" scripts/summarize_multicity_minimal.py
echo "Done."
