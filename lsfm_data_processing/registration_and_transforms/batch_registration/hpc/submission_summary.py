"""Compact, human-readable summaries for registration submissions."""

from __future__ import annotations

from pathlib import Path

from atlasspace.runtime.registration import RegistrationJob


def _image_configs_by_role(
    job: RegistrationJob,
    *,
    subject_image_id: str,
    template_image_id: str,
):
    configs = {
        job.fixed_image_config.image_id: job.fixed_image_config,
        job.moving_image_config.image_id: job.moving_image_config,
    }
    try:
        return configs[subject_image_id], configs[template_image_id]
    except KeyError as exc:
        raise ValueError(
            "Subject/template role metadata does not match the registration job: "
            f"subject={subject_image_id!r}, template={template_image_id!r}."
        ) from exc


def missing_registration_inputs(
    job: RegistrationJob,
    *,
    subject_image_id: str,
    template_image_id: str,
) -> tuple[tuple[str, Path], ...]:
    """Return every missing image or segmentation required by a job."""

    subject, template = _image_configs_by_role(
        job,
        subject_image_id=subject_image_id,
        template_image_id=template_image_id,
    )
    required = [
        ("subject image", Path(subject.image)),
        ("template image", Path(template.image)),
    ]
    required.extend(
        (f"subject segmentation {name!r}", Path(path))
        for name, path in subject.segmentations.items()
    )
    required.extend(
        (f"template segmentation {name!r}", Path(path))
        for name, path in template.segmentations.items()
    )
    return tuple((label, path) for label, path in required if not path.is_file())


def _number(value: float | int) -> str:
    return format(float(value), ".12g")


def format_ready_job(
    job: RegistrationJob,
    *,
    subject_image_id: str,
    template_image_id: str,
    input_sha256_verified: bool,
    mem_gb: int,
) -> str:
    """Describe one preflighted registration job in one concise line."""

    subject, _ = _image_configs_by_role(
        job,
        subject_image_id=subject_image_id,
        template_image_id=template_image_id,
    )
    registration = job.parameters.registration
    padding = job.parameters.preprocessing.fixed_padding_um
    input_text = "input found"
    if input_sha256_verified:
        input_text += f": {Path(subject.image).name}; verified sha256=yes"
    padding_text = "none" if padding is None else f"{_number(padding)} um"
    return (
        f"[ready] {subject_image_id}; {input_text}; "
        f"template={template_image_id}; "
        f"resolution={_number(registration.working_resolution_um)} um; "
        f"gradient={_number(registration.syn_gradient_step)}; "
        f"padding={padding_text}; memory={mem_gb} GB"
    )


def format_missing_job(
    job: RegistrationJob,
    missing: tuple[tuple[str, Path], ...],
    *,
    subject_image_id: str,
) -> str:
    """Describe missing inputs for one blocked registration job."""

    details = ", ".join(f"{label}: {path}" for label, path in missing)
    return f"[error] {subject_image_id} missing {details}"


__all__ = [
    "format_missing_job",
    "format_ready_job",
    "missing_registration_inputs",
]
