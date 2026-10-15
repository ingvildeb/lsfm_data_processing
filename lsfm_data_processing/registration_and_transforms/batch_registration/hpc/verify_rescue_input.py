"""Verify a transferred rescue input against its local generation manifest."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys


RESCUE_GENERATION_MANIFEST_FILENAME = "rescue_generation_manifest.json"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_rescue_input(
    manifest_path: Path | str,
    config_filename: str,
) -> str:
    """Validate one HPC subject image and return a compact success message."""

    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid rescue generation manifest: {path}") from exc
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported rescue generation manifest schema: {manifest.get('schema_version')!r}"
        )
    jobs = manifest.get("jobs")
    if not isinstance(jobs, dict) or config_filename not in jobs:
        raise ValueError(
            f"Rescue manifest has no job for configuration: {config_filename}"
        )
    job = jobs[config_filename]
    input_record = job.get("subject_input") if isinstance(job, dict) else None
    if not isinstance(input_record, dict):
        raise ValueError(f"Rescue job has no subject-input record: {config_filename}")
    subject_id = str(job.get("subject_id", "")).strip()
    input_path_text = str(input_record.get("hpc_path", "")).strip()
    input_path = Path(input_path_text)
    expected_size = input_record.get("size_bytes")
    expected_sha256 = input_record.get("sha256")
    if not subject_id or not input_path_text or not isinstance(expected_size, int):
        raise ValueError(f"Incomplete rescue input record: {config_filename}")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"Invalid rescue input SHA256: {config_filename}")
    if not input_path.is_file():
        raise FileNotFoundError(
            f"{subject_id} input is missing on HPC: {input_path}. "
            "Transfer the generated subject input before submission."
        )
    observed_size = input_path.stat().st_size
    observed_sha256 = file_sha256(input_path)
    if observed_size != expected_size or observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"{subject_id} input differs from local provenance: {input_path}. "
            "Transfer the updated file before submission."
        )
    return f"{subject_id} input: {input_path.name}. verified sha256: yes"


def verify_rescue_input_for_config(
    registration_config: Path | str,
) -> str | None:
    """Verify a rescue input at job startup when its batch manifest exists."""

    config_path = Path(registration_config)
    manifest_path = config_path.parent / RESCUE_GENERATION_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid rescue generation manifest: {manifest_path}") from exc
    jobs = manifest.get("jobs") if isinstance(manifest, dict) else None
    if not isinstance(jobs, dict):
        raise ValueError(f"Invalid rescue generation manifest: {manifest_path}")
    if config_path.name not in jobs:
        return None
    return verify_rescue_input(manifest_path, config_path.name)


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2:
        raise SystemExit(
            "Usage: python -m "
            "lsfm_data_processing.registration_and_transforms.batch_registration.hpc."
            "verify_rescue_input <manifest.json> <config.toml>"
        )
    print(verify_rescue_input(Path(arguments[0]), arguments[1]))


if __name__ == "__main__":
    main()


__all__ = [
    "file_sha256",
    "verify_rescue_input",
    "verify_rescue_input_for_config",
]
