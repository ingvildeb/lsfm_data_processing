from __future__ import annotations

from pathlib import Path

from lsfm_data_processing.registration_and_transforms._batch_register_core import (
    get_configured_subject_folders,
    load_batch_register_settings_from_path,
    run_batch_registration_for_subjects,
)
from lsfm_data_processing.utils.io_helpers import require_file


def main() -> None:
    project_dir = Path.cwd()
    config_path = require_file(
        project_dir / "configs" / "batch_register.toml",
        "Batch registration config",
    )

    settings = load_batch_register_settings_from_path(config_path)
    subject_folders = get_configured_subject_folders(settings)
    results = run_batch_registration_for_subjects(
        settings=settings,
        subject_folders=subject_folders,
    )

    successful_subjects = [result.subject_id for result in results if result.success]
    failed_subjects = [result.subject_id for result in results if not result.success]

    print()
    print(f"Finished batch registration. Successes: {len(successful_subjects)}")
    if successful_subjects:
        print("Successful subjects:")
        for subject_id in successful_subjects:
            print(f"  - {subject_id}")

    print(f"Failures: {len(failed_subjects)}")
    if failed_subjects:
        print("Failed subjects:")
        for subject_id in failed_subjects:
            print(f"  - {subject_id}")


if __name__ == "__main__":
    main()
