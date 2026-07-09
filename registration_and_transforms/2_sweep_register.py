from __future__ import annotations

from pathlib import Path
import sys


def _resolve_script_path() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve()

    candidate = Path.cwd() / "registration_and_transforms" / "2_sweep_register.py"
    if candidate.exists():
        return candidate.resolve()

    raise RuntimeError(
        "Could not resolve 2_sweep_register.py. "
        "Run this file as a script or launch the interactive session from the "
        "lsfm_data_processing repo root."
    )


SCRIPT_PATH = _resolve_script_path()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lsfm_data_processing.registration_and_transforms._sweep_register_core import (
    get_sweep_job_specs,
    load_sweep_register_settings,
    run_sweep_registration_for_jobs,
)


test_mode = False
settings = load_sweep_register_settings(
    SCRIPT_PATH,
    "2_sweep_register",
    test_mode=test_mode,
)
job_specs = get_sweep_job_specs(settings)
results = run_sweep_registration_for_jobs(
    settings=settings,
    job_specs=job_specs,
)

successful_jobs = [result for result in results if result.success]
failed_jobs = [result for result in results if not result.success]

print()
print(f"Finished sweep registration. Successes: {len(successful_jobs)}")
if successful_jobs:
    print("Successful jobs:")
    for result in successful_jobs:
        print(f"  - {result.label}")

print(f"Failures: {len(failed_jobs)}")
if failed_jobs:
    print("Failed jobs:")
    for result in failed_jobs:
        print(f"  - {result.label}")
