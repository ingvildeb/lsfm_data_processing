from __future__ import annotations

from pathlib import Path

from lsfm_data_processing.registration_and_transforms._batch_register_core import (
    get_batch_job_specs,
    load_batch_register_settings_from_path,
    run_batch_registration_for_jobs,
)
from lsfm_data_processing.utils.io_helpers import require_file


def main() -> None:
    project_dir = Path.cwd()
    config_path = require_file(
        project_dir / "configs" / "batch_register.toml",
        "Batch registration config",
    )

    settings = load_batch_register_settings_from_path(config_path)
    job_specs = get_batch_job_specs(settings)
    results = run_batch_registration_for_jobs(
        settings=settings,
        job_specs=job_specs,
    )

    successful_jobs = [result for result in results if result.success]
    failed_jobs = [result for result in results if not result.success]

    print()
    print(f"Finished batch registration. Successes: {len(successful_jobs)}")
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
