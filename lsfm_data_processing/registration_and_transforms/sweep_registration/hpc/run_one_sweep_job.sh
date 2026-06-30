#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$1"
REGISTRATION_CONFIG="$2"
IMAGE_KEY="$3"
TEMPLATE_KEY="$4"
PRESET_NAME="$5"
CONDA_ENV="$6"
PYTHON_EXECUTABLE="$7"

set +u
source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
conda activate "$CONDA_ENV"
set -u

cd "$PROJECT_DIR"

"$PYTHON_EXECUTABLE" -m lsfm_data_processing.registration_and_transforms.sweep_registration.hpc.run_one_sweep_job \
  --registration-config "$REGISTRATION_CONFIG" \
  --image-key "$IMAGE_KEY" \
  --template-key "$TEMPLATE_KEY" \
  --preset-name "$PRESET_NAME"
