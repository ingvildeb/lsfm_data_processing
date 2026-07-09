from __future__ import annotations

from pathlib import Path

from lsfm_data_processing.registration_and_transforms._sweep_register_core import (
    get_sweep_job_specs,
    load_sweep_register_settings_from_path,
    run_sweep_registration_for_jobs,
)
from lsfm_data_processing.utils.io_helpers import require_file


def main() -> None:
    project_dir = Path.cwd()
    config_path = require_file(
        project_dir / "configs" / "sweep_register.toml",
        "Sweep registration config",
    )

    settings = load_sweep_register_settings_from_path(config_path)
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


if __name__ == "__main__":
    main()
