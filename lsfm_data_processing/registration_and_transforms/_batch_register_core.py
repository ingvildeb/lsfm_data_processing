from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


GITHUB_ROOT = Path(__file__).resolve().parents[3]
ATLASSPACE_SRC = GITHUB_ROOT / "atlasspace" / "src"
if ATLASSPACE_SRC.exists() and str(ATLASSPACE_SRC) not in sys.path:
    sys.path.insert(0, str(ATLASSPACE_SRC))

from atlasspace import registration

from lsfm_data_processing.utils.io_helpers import (
    CanonicalConfigTemplate,
    load_toml_config,
    prepare_script_config_path,
    require_dir,
    require_file,
)


BATCH_LOCAL_CANONICAL_TEMPLATE = CanonicalConfigTemplate(
    resource_package="atlasspace.config_templates",
    resource_parts=("registration_batch_template.toml",),
    config_label="Batch registration config",
    compatible_template_ids=("registration_and_transforms/1_batch_register",),
)


@dataclass
class BatchRegisterSettings:
    config_path: Path
    jobs: list[registration.RegistrationJob]


@dataclass(frozen=True)
class BatchJobSpec:
    job_index: int
    fixed_image_id: str
    moving_image_id: str
    preset_name: str
    output_dir: Path

    @property
    def label(self) -> str:
        return f"{self.moving_image_id} -> {self.fixed_image_id} ({self.preset_name})"


@dataclass(frozen=True)
class BatchRunResult:
    job_index: int
    fixed_image_id: str
    moving_image_id: str
    preset_name: str
    success: bool
    output_dir: Path
    error_message: str | None

    @property
    def label(self) -> str:
        return f"{self.moving_image_id} -> {self.fixed_image_id} ({self.preset_name})"


SubjectRunResult = BatchRunResult


def load_batch_register_settings(
    script_path: Path,
    config_basename: str,
    *,
    test_mode: bool = False,
) -> BatchRegisterSettings:
    config_path = prepare_script_config_path(
        script_path,
        config_basename,
        test_mode=test_mode,
        canonical_template=BATCH_LOCAL_CANONICAL_TEMPLATE,
        warn_on_stale=True,
    )
    print(f"Using config: {config_path.name}")
    return load_batch_register_settings_from_path(config_path)


def load_batch_register_settings_from_path(config_path: Path) -> BatchRegisterSettings:
    resolved_path = require_file(config_path, "Batch registration config")
    cfg = load_toml_config(resolved_path)
    if _looks_like_legacy_batch_config(cfg):
        _raise_legacy_batch_config_error(resolved_path)

    plan = registration.load_registration_plan_from_dict(
        cfg,
        config_path=resolved_path,
    )
    jobs = registration.build_jobs_from_plan(plan)
    if not jobs:
        raise RuntimeError("No batch jobs were produced by the registration plan.")
    return BatchRegisterSettings(
        config_path=resolved_path,
        jobs=jobs,
    )


def get_batch_job_spec(
    settings: BatchRegisterSettings,
    job_index: int,
) -> BatchJobSpec:
    try:
        job = settings.jobs[job_index]
    except IndexError as exc:
        raise RuntimeError(
            f"Batch job index {job_index} is out of range for {len(settings.jobs)} jobs."
        ) from exc

    return BatchJobSpec(
        job_index=job_index,
        fixed_image_id=job.fixed_image_config.image_id,
        moving_image_id=job.moving_image_config.image_id,
        preset_name=job.parameters.name,
        output_dir=job.output_dir,
    )


def get_batch_job_specs(settings: BatchRegisterSettings) -> list[BatchJobSpec]:
    return [
        get_batch_job_spec(settings, job_index)
        for job_index in range(len(settings.jobs))
    ]


def get_configured_subject_folders(settings: BatchRegisterSettings) -> list[Path]:
    resolved_paths: list[Path] = []
    seen_paths: set[Path] = set()

    for job_spec in get_batch_job_specs(settings):
        subject_folder = require_dir(job_spec.output_dir.parent, "Batch run folder")
        normalized_subject_folder = subject_folder.resolve(strict=False)
        if normalized_subject_folder in seen_paths:
            continue
        seen_paths.add(normalized_subject_folder)
        resolved_paths.append(subject_folder)

    return resolved_paths


def run_batch_registration_for_job(
    *,
    settings: BatchRegisterSettings,
    job_spec: BatchJobSpec,
) -> BatchRunResult:
    job = _resolve_registration_job(settings, job_spec)
    return _run_batch_job_impl(job_spec=job_spec, job=job)


def run_batch_registration_for_jobs(
    *,
    settings: BatchRegisterSettings,
    job_specs: list[BatchJobSpec],
) -> list[BatchRunResult]:
    preset_names = list(dict.fromkeys(job.parameters.name for job in settings.jobs))
    print(f"Using registration presets: {', '.join(preset_names)}")
    return [
        run_batch_registration_for_job(
            settings=settings,
            job_spec=job_spec,
        )
        for job_spec in job_specs
    ]


def run_batch_registration_for_subject(
    *,
    settings: BatchRegisterSettings,
    subject_folder: Path,
) -> BatchRunResult:
    job_spec = _find_job_spec_for_subject_folder(settings, subject_folder)
    return run_batch_registration_for_job(
        settings=settings,
        job_spec=job_spec,
    )


def run_batch_registration_for_subjects(
    *,
    settings: BatchRegisterSettings,
    subject_folders: list[Path],
) -> list[BatchRunResult]:
    job_specs = [
        _find_job_spec_for_subject_folder(settings, subject_folder)
        for subject_folder in subject_folders
    ]
    return run_batch_registration_for_jobs(
        settings=settings,
        job_specs=job_specs,
    )


def _resolve_registration_job(
    settings: BatchRegisterSettings,
    job_spec: BatchJobSpec,
) -> registration.RegistrationJob:
    job = settings.jobs[job_spec.job_index]
    expected_spec = get_batch_job_spec(settings, job_spec.job_index)

    if (
        job_spec.fixed_image_id != expected_spec.fixed_image_id
        or job_spec.moving_image_id != expected_spec.moving_image_id
        or job_spec.preset_name != expected_spec.preset_name
        or job_spec.output_dir != expected_spec.output_dir
    ):
        raise RuntimeError(
            "Batch job spec does not match the current registration config. "
            f"Expected {expected_spec.label}, got {job_spec.label}."
        )

    return job


def _run_batch_job_impl(
    *,
    job_spec: BatchJobSpec,
    job: registration.RegistrationJob,
) -> BatchRunResult:
    print(f"Running batch registration for {job_spec.label} ...")

    try:
        result = registration.run_antspy_registration(job)

        if not result.success:
            print(f"  Registration failed for {job_spec.label}: {result.error_message}")
            return BatchRunResult(
                job_index=job_spec.job_index,
                fixed_image_id=job_spec.fixed_image_id,
                moving_image_id=job_spec.moving_image_id,
                preset_name=job_spec.preset_name,
                success=False,
                output_dir=job_spec.output_dir,
                error_message=result.error_message,
            )

        print(f"  Registration finished: {job_spec.output_dir}")
        return BatchRunResult(
            job_index=job_spec.job_index,
            fixed_image_id=job_spec.fixed_image_id,
            moving_image_id=job_spec.moving_image_id,
            preset_name=job_spec.preset_name,
            success=True,
            output_dir=job_spec.output_dir,
            error_message=None,
        )
    except Exception as exc:
        print(f"  Failed while processing {job_spec.label}: {exc}")
        return BatchRunResult(
            job_index=job_spec.job_index,
            fixed_image_id=job_spec.fixed_image_id,
            moving_image_id=job_spec.moving_image_id,
            preset_name=job_spec.preset_name,
            success=False,
            output_dir=job_spec.output_dir,
            error_message=str(exc),
        )


def _find_job_spec_for_subject_folder(
    settings: BatchRegisterSettings,
    subject_folder: Path,
) -> BatchJobSpec:
    resolved_subject_folder = require_dir(subject_folder, "Batch run folder").resolve(
        strict=False
    )
    matching_job_specs = [
        job_spec
        for job_spec in get_batch_job_specs(settings)
        if job_spec.output_dir.parent.resolve(strict=False) == resolved_subject_folder
    ]

    if not matching_job_specs:
        raise RuntimeError(
            "No batch registration job matched the requested subject folder.\n"
            f"Subject folder: {resolved_subject_folder}"
        )

    if len(matching_job_specs) > 1:
        matching_labels = ", ".join(job_spec.label for job_spec in matching_job_specs)
        raise RuntimeError(
            "Multiple batch registration jobs matched the requested subject folder. "
            "The canonical batch config currently expects one run image per output "
            f"folder.\nSubject folder: {resolved_subject_folder}\nJobs: {matching_labels}"
        )

    return matching_job_specs[0]


def _looks_like_legacy_batch_config(cfg: dict) -> bool:
    run_cfg = cfg.get("run", {})
    return (
        "subject_defaults" in cfg
        or "templates" in cfg
        or "subject_to_template" in cfg
        or "segmentation_transform" in cfg
        or (isinstance(run_cfg, dict) and "registration_preset" in run_cfg)
    )


def _raise_legacy_batch_config_error(config_path: Path) -> None:
    raise RuntimeError(
        "This batch registration workflow now uses the canonical atlasspace "
        f"batch TOML schema.\nConfig: {config_path}\n"
        "The file still looks like the legacy lsfm_data_processing batch schema "
        "([subject_defaults], [templates], [subject_to_template], "
        "[segmentation_transform], or [run].registration_preset).\n"
        "Please recreate or migrate this config using the canonical template:\n"
        "atlasspace/src/atlasspace/config_templates/registration_batch_template.toml"
    )
