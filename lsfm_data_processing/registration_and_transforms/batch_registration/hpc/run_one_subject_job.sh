#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$1"
REGISTRATION_CONFIG="$2"
SUBJECT_DIR="$3"
CONDA_ENV="$4"
PYTHON_EXECUTABLE="$5"

# Some cluster bash startup files are not nounset-safe, so relax `-u`
# only while sourcing the shell environment and activating conda.
set +u
source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
conda activate "$CONDA_ENV"
set -u

cd "$PROJECT_DIR"

"$PYTHON_EXECUTABLE" -m lsfm_data_processing.registration_and_transforms.batch_registration.hpc.run_one_subject \
  --registration-config "$REGISTRATION_CONFIG" \
  --subject-dir "$SUBJECT_DIR"
