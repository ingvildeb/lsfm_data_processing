from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atlasspace import ImageConfig, SpaceDefinition, registration, transforms

from lsfm_data_processing.utils.io_helpers import (
    CanonicalConfigTemplate,
    load_toml_config,
    load_script_config,
    normalize_user_path,
    require_dir,
    require_file,
)
from lsfm_data_processing.utils.naming import get_underscore_token


BATCH_LOCAL_CANONICAL_TEMPLATE = CanonicalConfigTemplate(
    resource_package="lsfm_data_processing.registration_and_transforms.batch_registration",
    resource_parts=("config_templates", "batch_register.toml"),
    config_label="Batch registration config",
)


@dataclass
class BatchRegisterSettings:
    registration_preset: str
    template_role: str
    output_subdir: str
    orientation_alignment: str
    write_input_images: bool
    subjects_dir: Path
    subject_image_name: str
    subject_orientation: str
    subject_resolution_um: float
    underscores_to_id: int
    transform_segmentations: bool
    segmentation_output_subdir: str
    segmentation_interpolation: str
    write_transform_intermediates: bool
    templates_cfg: dict[str, Any]
    subject_to_template_cfg: dict[str, str]


@dataclass
class SubjectRunResult:
    subject_id: str
    subject_folder: Path
    success: bool
    output_dir: Path | None
    error_message: str | None


def _settings_from_cfg(cfg: dict[str, Any]) -> BatchRegisterSettings:
    run_cfg = cfg["run"]
    subject_defaults_cfg = cfg["subject_defaults"]
    segmentation_transform_cfg = cfg.get("segmentation_transform", {})

    return BatchRegisterSettings(
        registration_preset=run_cfg["registration_preset"],
        template_role=run_cfg.get("template_role", "moving"),
        output_subdir=run_cfg.get("output_subdir", "_01_registration"),
        orientation_alignment=run_cfg.get("orientation_alignment", "none"),
        write_input_images=run_cfg.get("write_input_images", False),
        subjects_dir=normalize_user_path(subject_defaults_cfg["subjects_dir"]),
        subject_image_name=subject_defaults_cfg["image_name"],
        subject_orientation=subject_defaults_cfg["orientation"],
        subject_resolution_um=float(subject_defaults_cfg["resolution_um"]),
        underscores_to_id=subject_defaults_cfg["underscores_to_id"],
        transform_segmentations=segmentation_transform_cfg.get("enabled", False),
        segmentation_output_subdir=segmentation_transform_cfg.get(
            "output_subdir",
            "_02_template_segmentations",
        ),
        segmentation_interpolation=segmentation_transform_cfg.get(
            "interpolation",
            "genericLabel",
        ),
        write_transform_intermediates=segmentation_transform_cfg.get(
            "write_intermediates",
            False,
        ),
        templates_cfg=cfg["templates"],
        subject_to_template_cfg=cfg["subject_to_template"],
    )


def load_batch_register_settings(
    script_path: Path,
    config_basename: str,
    *,
    test_mode: bool = False,
) -> BatchRegisterSettings:
    cfg = load_script_config(
        script_path,
        config_basename,
        test_mode=test_mode,
        canonical_template=BATCH_LOCAL_CANONICAL_TEMPLATE,
        warn_on_stale=True,
    )
    return _settings_from_cfg(cfg)


def load_batch_register_settings_from_path(config_path: Path) -> BatchRegisterSettings:
    resolved_path = require_file(config_path, "Batch registration config")
    cfg = load_toml_config(resolved_path)
    return _settings_from_cfg(cfg)


def get_configured_subject_folders(settings: BatchRegisterSettings) -> list[Path]:
    return [
        require_dir(settings.subjects_dir / subject_id, "Subject folder")
        for subject_id in settings.subject_to_template_cfg
    ]


def _build_parameters(settings: BatchRegisterSettings):
    parameters = registration.load_preset(settings.registration_preset)
    parameters.execution.write_input_images = settings.write_input_images
    return parameters


def _make_space_definition(
    *,
    space_name: str,
    orientation: str,
    resolution_um: float,
) -> SpaceDefinition:
    return SpaceDefinition(
        space_name=space_name,
        orientation=orientation,
        resolution_um=(resolution_um, resolution_um, resolution_um),
    )


def _parse_subject_id(subject_folder: Path, token_index: int) -> str:
    return get_underscore_token(subject_folder.name, token_index, "subject_id")


def _build_subject_image_config(
    settings: BatchRegisterSettings,
    subject_folder: Path,
) -> ImageConfig:
    subject_id = _parse_subject_id(subject_folder, settings.underscores_to_id)
    subject_image_path = require_file(
        subject_folder / settings.subject_image_name,
        f"Subject image for {subject_id}",
    )

    return ImageConfig(
        image_id=subject_id,
        image=subject_image_path,
        space=_make_space_definition(
            space_name=subject_id,
            orientation=settings.subject_orientation,
            resolution_um=settings.subject_resolution_um,
        ),
    )


def _build_template_image_config(
    settings: BatchRegisterSettings,
    template_key: str,
) -> ImageConfig:
    template_cfg = settings.templates_cfg[template_key]
    template_space_name = template_cfg.get("space_name", template_key)
    template_resolution_um = float(template_cfg["resolution_um"])
    template_image_path = require_file(
        normalize_user_path(template_cfg["image"]),
        f"Template image for {template_key}",
    )

    return ImageConfig(
        image_id=template_key,
        image=template_image_path,
        space=_make_space_definition(
            space_name=template_space_name,
            orientation=template_cfg["orientation"],
            resolution_um=template_resolution_um,
        ),
    )


def _build_segmentation_image_config(
    settings: BatchRegisterSettings,
    template_key: str,
    segmentation_name: str,
    segmentation_path_value: str,
) -> ImageConfig:
    template_cfg = settings.templates_cfg[template_key]
    template_space_name = template_cfg.get("space_name", template_key)
    template_resolution_um = float(template_cfg["resolution_um"])
    segmentation_path = require_file(
        normalize_user_path(segmentation_path_value),
        f"Template segmentation '{segmentation_name}' for {template_key}",
    )

    return ImageConfig(
        image_id=segmentation_name,
        image=segmentation_path,
        space=_make_space_definition(
            space_name=template_space_name,
            orientation=template_cfg["orientation"],
            resolution_um=template_resolution_um,
        ),
    )


def _build_registration_job(
    settings: BatchRegisterSettings,
    *,
    subject_folder: Path,
    subject_config: ImageConfig,
    template_config: ImageConfig,
    parameters,
) -> registration.RegistrationJob:
    output_dir = subject_folder / settings.output_subdir

    if settings.template_role == "moving":
        fixed_image_config = subject_config
        moving_image_config = template_config
    elif settings.template_role == "fixed":
        fixed_image_config = template_config
        moving_image_config = subject_config
    else:
        raise RuntimeError(
            f"template_role must be 'moving' or 'fixed'. Got: {settings.template_role}"
        )

    return registration.RegistrationJob(
        fixed_image_config=fixed_image_config,
        moving_image_config=moving_image_config,
        output_dir=output_dir,
        parameters=parameters,
        orientation_alignment=settings.orientation_alignment,
    )


def _segmentation_transform_direction(settings: BatchRegisterSettings) -> str:
    if settings.template_role == "moving":
        return "forward"
    if settings.template_role == "fixed":
        return "inverse"
    raise RuntimeError(f"Unsupported template_role: {settings.template_role}")


def _segmentation_output_name(
    settings: BatchRegisterSettings,
    segmentation_name: str,
) -> str:
    direction = _segmentation_transform_direction(settings)
    suffix = (
        "WarpedSegmentation"
        if direction == "forward"
        else "InverseWarpedSegmentation"
    )
    return f"{segmentation_name}_{suffix}.nii.gz"


def transform_template_segmentations_for_subject(
    *,
    settings: BatchRegisterSettings,
    subject_folder: Path,
    subject_config: ImageConfig,
    template_key: str,
    registration_result,
) -> None:
    template_cfg = settings.templates_cfg[template_key]
    segmentations_cfg = template_cfg.get("segmentations", {})

    if not segmentations_cfg:
        print(
            f"  No template segmentations configured for {template_key}. "
            "Skipping segmentation transforms."
        )
        return

    output_dir = subject_folder / settings.segmentation_output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    transform_sequence = transforms.TransformSequence.from_registration_result(
        registration_result
    )
    direction = _segmentation_transform_direction(settings)

    for segmentation_name, segmentation_path_value in segmentations_cfg.items():
        segmentation_config = _build_segmentation_image_config(
            settings,
            template_key,
            segmentation_name,
            segmentation_path_value,
        )
        output_path = output_dir / _segmentation_output_name(
            settings,
            segmentation_name,
        )
        transforms.transform_segmentation(
            segmentation_config,
            transform_sequence,
            subject_config,
            direction=direction,
            interpolation=settings.segmentation_interpolation,
            output_path=output_path,
            write_intermediates=settings.write_transform_intermediates,
        )
        print(f"  Wrote transformed segmentation: {output_path.name}")


def _resolve_template_key(
    settings: BatchRegisterSettings,
    subject_id: str,
) -> str:
    template_key = settings.subject_to_template_cfg.get(subject_id)
    if template_key is None:
        raise RuntimeError(
            "Subject id was not found in subject_to_template mapping.\n"
            f"Subject id: {subject_id}"
        )
    if template_key not in settings.templates_cfg:
        raise RuntimeError(
            f"Template key '{template_key}' is not defined under [templates]."
        )
    return template_key


def _run_batch_registration_for_subject_impl(
    *,
    settings: BatchRegisterSettings,
    subject_folder: Path,
    parameters,
) -> SubjectRunResult:
    subject_folder = require_dir(subject_folder, "Subject folder")
    subject_id = _parse_subject_id(subject_folder, settings.underscores_to_id)
    template_key = _resolve_template_key(settings, subject_id)

    print(f"Running registration for {subject_id} using template '{template_key}' ...")

    try:
        subject_config = _build_subject_image_config(settings, subject_folder)
        template_config = _build_template_image_config(settings, template_key)
        job = _build_registration_job(
            settings,
            subject_folder=subject_folder,
            subject_config=subject_config,
            template_config=template_config,
            parameters=parameters,
        )

        result = registration.run_antspy_registration(job)

        if not result.success:
            print(f"  Registration failed for {subject_id}: {result.error_message}")
            return SubjectRunResult(
                subject_id=subject_id,
                subject_folder=subject_folder,
                success=False,
                output_dir=job.output_dir,
                error_message=result.error_message,
            )

        print(f"  Registration finished: {job.output_dir}")

        if settings.transform_segmentations:
            transform_template_segmentations_for_subject(
                settings=settings,
                subject_folder=subject_folder,
                subject_config=subject_config,
                template_key=template_key,
                registration_result=result,
            )

        return SubjectRunResult(
            subject_id=subject_id,
            subject_folder=subject_folder,
            success=True,
            output_dir=job.output_dir,
            error_message=None,
        )
    except Exception as exc:
        print(f"  Failed while processing {subject_id}: {exc}")
        return SubjectRunResult(
            subject_id=subject_id,
            subject_folder=subject_folder,
            success=False,
            output_dir=None,
            error_message=str(exc),
        )


def run_batch_registration_for_subject(
    *,
    settings: BatchRegisterSettings,
    subject_folder: Path,
) -> SubjectRunResult:
    parameters = _build_parameters(settings)
    return _run_batch_registration_for_subject_impl(
        settings=settings,
        subject_folder=subject_folder,
        parameters=parameters,
    )


def run_batch_registration_for_subjects(
    *,
    settings: BatchRegisterSettings,
    subject_folders: list[Path],
) -> list[SubjectRunResult]:
    parameters = _build_parameters(settings)
    print(f"Using registration preset: {settings.registration_preset}")
    return [
        _run_batch_registration_for_subject_impl(
            settings=settings,
            subject_folder=subject_folder,
            parameters=parameters,
        )
        for subject_folder in subject_folders
    ]
