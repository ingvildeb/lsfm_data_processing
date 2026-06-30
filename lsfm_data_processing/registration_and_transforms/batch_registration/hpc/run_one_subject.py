from __future__ import annotations

import argparse
from pathlib import Path
from lsfm_data_processing.registration_and_transforms._batch_register_core import (
    load_batch_register_settings_from_path,
    run_batch_registration_for_subject,
)


parser = argparse.ArgumentParser(
    description="Run batch-registration logic for one subject on HPC.",
)
parser.add_argument(
    "--registration-config",
    required=True,
    help="Path to the copied batch registration TOML on the HPC filesystem.",
)
parser.add_argument(
    "--subject-dir",
    required=True,
    help="Path to the subject folder to process.",
)
args = parser.parse_args()

registration_config = Path(args.registration_config).resolve()
subject_dir = Path(args.subject_dir).resolve()

settings = load_batch_register_settings_from_path(registration_config)
result = run_batch_registration_for_subject(
    settings=settings,
    subject_folder=subject_dir,
)

if not result.success:
    raise SystemExit(1)
