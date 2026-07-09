from __future__ import annotations

import argparse
from pathlib import Path
from lsfm_data_processing.registration_and_transforms._batch_register_core import (
    get_batch_job_spec,
    load_batch_register_settings_from_path,
    run_batch_registration_for_job,
)


parser = argparse.ArgumentParser(
    description="Run batch-registration logic for one image/template job on HPC.",
)
parser.add_argument(
    "--registration-config",
    required=True,
    help="Path to the copied batch registration TOML on the HPC filesystem.",
)
parser.add_argument(
    "--job-index",
    required=True,
    type=int,
    help="Zero-based batch job index from the canonical registration plan.",
)
args = parser.parse_args()

registration_config = Path(args.registration_config).resolve()

settings = load_batch_register_settings_from_path(registration_config)
job_spec = get_batch_job_spec(settings, args.job_index)
result = run_batch_registration_for_job(
    settings=settings,
    job_spec=job_spec,
)

if not result.success:
    raise SystemExit(1)
