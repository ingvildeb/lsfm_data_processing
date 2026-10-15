"""Compact, human-readable summaries for registration submissions."""

from __future__ import annotations

from pathlib import Path

from atlasspace.runtime.registration import RegistrationJob


def missing_registration_inputs(job: RegistrationJob) -> tuple[tuple[str, Path], ...]:
    """Return every missing image or segmentation required by a job."""

    required = [
        ("subject image", Path(job.moving_image_config.image)),
        ("template image", Path(job.fixed_image_config.image)),
    ]
    required.extend(
        (f"subject segmentation {name!r}", Path(path))
        for name, path in job.moving_image_config.segmentations.items()
    )
    required.extend(
        (f"template segmentation {name!r}", Path(path))
        for name, path in job.fixed_image_config.segmentations.items()
    )
    return tuple((label, path) for label, path in required if not path.is_file())


def _number(value: float | int) -> str:
    return format(float(value), ".12g")


def format_ready_job(
    job: RegistrationJob,
    *,
    input_sha256_verified: bool,
    mem_gb: int,
) -> str:
    """Describe one preflighted registration job in one concise line."""

    subject_id = job.moving_image_config.image_id
    subject_input = Path(job.moving_image_config.image).name
    registration = job.parameters.registration
    padding = job.parameters.preprocessing.fixed_padding_um
    verification = " (SHA256 verified)" if input_sha256_verified else ""
    padding_text = "none" if padding is None else f"{_number(padding)} um"
    return (
        f"[ready] {subject_id} input found: {subject_input}{verification}; "
        f"template={job.fixed_image_config.image_id}; "
        f"resolution={_number(registration.working_resolution_um)} um; "
        f"gradient={_number(registration.syn_gradient_step)}; "
        f"padding={padding_text}; memory={mem_gb} GB"
    )


def format_missing_job(
    job: RegistrationJob,
    missing: tuple[tuple[str, Path], ...],
) -> str:
    """Describe missing inputs for one blocked registration job."""

    details = ", ".join(f"{label}: {path}" for label, path in missing)
    return f"[error] {job.moving_image_config.image_id} missing {details}"


__all__ = [
    "format_missing_job",
    "format_ready_job",
    "missing_registration_inputs",
]
