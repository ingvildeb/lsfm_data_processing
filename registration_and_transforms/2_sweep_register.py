from __future__ import annotations

from pathlib import Path

from lsfm_data_processing.registration_and_transforms._sweep_register_core import (
    get_sweep_job_specs,
    load_sweep_register_settings,
    run_sweep_registration_for_jobs,
)


test_mode = False
settings = load_sweep_register_settings(
    Path(__file__),
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
        print(
            f"  - {result.image_key} -> {result.template_key} "
            f"({result.preset_name})"
        )

print(f"Failures: {len(failed_jobs)}")
if failed_jobs:
    print("Failed jobs:")
    for result in failed_jobs:
        print(
            f"  - {result.image_key} -> {result.template_key} "
            f"({result.preset_name})"
        )
