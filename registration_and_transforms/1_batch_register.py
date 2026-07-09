from __future__ import annotations

from pathlib import Path
import sys


def _resolve_script_path() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve()

    candidate = Path.cwd() / "registration_and_transforms" / "1_batch_register.py"
    if candidate.exists():
        return candidate.resolve()

    raise RuntimeError(
        "Could not resolve 1_batch_register.py. "
        "Run this file as a script or launch the interactive session from the "
        "lsfm_data_processing repo root."
    )


SCRIPT_PATH = _resolve_script_path()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lsfm_data_processing.registration_and_transforms._batch_register_core import (
    get_configured_subject_folders,
    load_batch_register_settings,
    run_batch_registration_for_subjects,
)


test_mode = False
settings = load_batch_register_settings(
    SCRIPT_PATH,
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
