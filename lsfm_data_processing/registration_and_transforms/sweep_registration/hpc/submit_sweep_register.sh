#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"

"$PYTHON_EXECUTABLE" -m lsfm_data_processing.registration_and_transforms.sweep_registration.hpc.submit_sweep_register
