#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash scripts/colab_2cities_mamba_setup.sh <repo_url> <drive_xlsx_path>
# Example:
# bash scripts/colab_2cities_mamba_setup.sh \
#   https://github.com/yourname/260219_ML-rainfall-time-prediction.git \
#   /content/drive/MyDrive/rainfall/20152024.xlsx

REPO_URL="${1:-}"
DRIVE_XLSX_PATH="${2:-}"

if [[ -z "${REPO_URL}" || -z "${DRIVE_XLSX_PATH}" ]]; then
  echo "Missing args."
  echo "Usage: bash scripts/colab_2cities_mamba_setup.sh <repo_url> <drive_xlsx_path>"
  exit 1
fi

cd /content
if [[ -d rainfall ]]; then
  rm -rf rainfall
fi
git clone "${REPO_URL}" rainfall
cd rainfall

cp "${DRIVE_XLSX_PATH}" /content/rainfall/20152024.xlsx
ls -lh /content/rainfall/20152024.xlsx

python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install mamba-ssm

python - <<'PY'
import importlib.util
import torch
print("cuda_available =", torch.cuda.is_available())
print("mamba_installed =", importlib.util.find_spec("mamba_ssm") is not None)
if not torch.cuda.is_available():
    raise SystemExit("GPU is not available in this runtime.")
if importlib.util.find_spec("mamba_ssm") is None:
    raise SystemExit("mamba_ssm is not installed.")
PY

export PYTHONPATH=.
python -u -m scripts.train --config configs/gz_rain_quick_2cities_mamba.yaml --device cuda 2>&1 | tee runs/train_2cities_mamba.log

python scripts/plot_results.py --exp_dir runs/gz_rain_da_deformmamba_2cities_mamba --horizon 1 --tail_points 300

mkdir -p "/content/drive/MyDrive/rainfall_outputs"
cp -r runs/gz_rain_da_deformmamba_2cities_mamba "/content/drive/MyDrive/rainfall_outputs/"
tar -czvf /content/gz_rain_da_deformmamba_2cities_mamba.tgz -C runs gz_rain_da_deformmamba_2cities_mamba

echo "Done. Artifacts:"
echo "  /content/drive/MyDrive/rainfall_outputs/gz_rain_da_deformmamba_2cities_mamba"
echo "  /content/gz_rain_da_deformmamba_2cities_mamba.tgz"
