from __future__ import annotations

from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from registration_and_transforms._batch_register_core import (
    get_configured_subject_folders,
    load_batch_register_settings,
    run_batch_registration_for_subjects,
)


test_mode = False
settings = load_batch_register_settings(
    Path(__file__),
    "1_batch_register",
    test_mode=test_mode,
)
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
