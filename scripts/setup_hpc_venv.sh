#!/bin/bash
set -euo pipefail

# Plain Python environment setup for the world-model HPC jobs.
#
# Usage from repo root:
#   bash scripts/setup_hpc_venv.sh
#
# Optional overrides:
#   PYTHON_BIN=python3.12
#   VENV_DIR=.venv
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
#   TORCH_PACKAGE=torch==2.5.1
#   TORCHVISION_PACKAGE=torchvision==0.20.1
#
# If your cluster provides a CUDA module, load it before running this script:
#   module load cuda/12.1

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
	if command -v python3.12 >/dev/null 2>&1; then
		PYTHON_BIN=python3.12
	elif command -v python3.11 >/dev/null 2>&1; then
		PYTHON_BIN=python3.11
	else
		PYTHON_BIN=python3
	fi
fi
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_PACKAGE="${TORCH_PACKAGE:-torch==2.5.1}"
TORCHVISION_PACKAGE="${TORCHVISION_PACKAGE:-torchvision==0.20.1}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ -n "$TORCH_INDEX_URL" ]]; then
	python -m pip install "$TORCH_PACKAGE" "$TORCHVISION_PACKAGE" --index-url "$TORCH_INDEX_URL"
else
	python -m pip install "$TORCH_PACKAGE" "$TORCHVISION_PACKAGE"
fi

python -m pip install -r requirements.txt

python - <<'PY'
import torch
import torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
	print("cuda_device", torch.cuda.get_device_name(0))
PY
