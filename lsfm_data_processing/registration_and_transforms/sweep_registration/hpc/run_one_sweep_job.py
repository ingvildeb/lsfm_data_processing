from __future__ import annotations

import argparse
from pathlib import Path
from lsfm_data_processing.registration_and_transforms._sweep_register_core import (
    SweepJobSpec,
    load_sweep_register_settings_from_path,
    run_sweep_registration_for_job,
)


parser = argparse.ArgumentParser(
    description="Run sweep-registration logic for one image/template/preset job on HPC.",
)
parser.add_argument(
    "--registration-config",
    required=True,
    help="Path to the copied sweep registration TOML on the HPC filesystem.",
)
parser.add_argument(
    "--image-key",
    required=True,
    help="Image key from the [images] section to process.",
)
parser.add_argument(
    "--template-key",
    required=True,
    help="Template key from the [templates] section to use.",
)
parser.add_argument(
    "--preset-name",
    required=True,
    help="Registration preset name to apply for this sweep job.",
)
args = parser.parse_args()

registration_config = Path(args.registration_config).resolve()

settings = load_sweep_register_settings_from_path(registration_config)
job_spec = SweepJobSpec(
    image_key=args.image_key,
    template_key=args.template_key,
    preset_name=args.preset_name,
)
result = run_sweep_registration_for_job(
    settings=settings,
    job_spec=job_spec,
)

if not result.success:
    raise SystemExit(1)
