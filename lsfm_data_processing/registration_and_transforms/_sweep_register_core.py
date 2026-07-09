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
    load_toml_config,
    require_file,
    resolve_script_config_path,
)


@dataclass
class SweepRegisterSettings:
    config_path: Path
    jobs: list[registration.RegistrationJob]


@dataclass(frozen=True)
class SweepJobSpec:
    job_index: int
    fixed_image_id: str
    moving_image_id: str
    preset_name: str
    output_dir: Path

    @property
    def label(self) -> str:
        return f"{self.moving_image_id} -> {self.fixed_image_id} ({self.preset_name})"


@dataclass(frozen=True)
class SweepRunResult:
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


def load_sweep_register_settings(
    script_path: Path,
    config_basename: str,
    *,
    test_mode: bool = False,
) -> SweepRegisterSettings:
    config_path = resolve_script_config_path(
        script_path,
        config_basename,
        test_mode=test_mode,
    )
    print(f"Using config: {config_path.name}")
    return load_sweep_register_settings_from_path(config_path)


def load_sweep_register_settings_from_path(config_path: Path) -> SweepRegisterSettings:
    resolved_path = require_file(config_path, "Sweep registration config")
    cfg = load_toml_config(resolved_path)
    plan = registration.load_registration_plan_from_dict(
        cfg,
        config_path=resolved_path,
    )
    jobs = registration.build_jobs_from_plan(plan)
    if not jobs:
        raise RuntimeError("No sweep jobs were produced by the registration plan.")
    return SweepRegisterSettings(
        config_path=resolved_path,
        jobs=jobs,
    )


def get_sweep_job_spec(
    settings: SweepRegisterSettings,
    job_index: int,
) -> SweepJobSpec:
    try:
        job = settings.jobs[job_index]
    except IndexError as exc:
        raise RuntimeError(
            f"Sweep job index {job_index} is out of range for {len(settings.jobs)} jobs."
        ) from exc

    return SweepJobSpec(
        job_index=job_index,
        fixed_image_id=job.fixed_image_config.image_id,
        moving_image_id=job.moving_image_config.image_id,
        preset_name=job.parameters.name,
        output_dir=job.output_dir,
    )


def get_sweep_job_specs(settings: SweepRegisterSettings) -> list[SweepJobSpec]:
    return [
        get_sweep_job_spec(settings, job_index)
        for job_index in range(len(settings.jobs))
    ]


def run_sweep_registration_for_job(
    *,
    settings: SweepRegisterSettings,
    job_spec: SweepJobSpec,
) -> SweepRunResult:
    job = _resolve_registration_job(settings, job_spec)
    return _run_sweep_job_impl(job_spec=job_spec, job=job)


def run_sweep_registration_for_jobs(
    *,
    settings: SweepRegisterSettings,
    job_specs: list[SweepJobSpec],
) -> list[SweepRunResult]:
    preset_names = list(dict.fromkeys(job.parameters.name for job in settings.jobs))
    print(f"Using registration presets: {', '.join(preset_names)}")
    return [
        run_sweep_registration_for_job(
            settings=settings,
            job_spec=job_spec,
        )
        for job_spec in job_specs
    ]


def _resolve_registration_job(
    settings: SweepRegisterSettings,
    job_spec: SweepJobSpec,
) -> registration.RegistrationJob:
    job = settings.jobs[job_spec.job_index]
    expected_spec = get_sweep_job_spec(settings, job_spec.job_index)

    if (
        job_spec.fixed_image_id != expected_spec.fixed_image_id
        or job_spec.moving_image_id != expected_spec.moving_image_id
        or job_spec.preset_name != expected_spec.preset_name
        or job_spec.output_dir != expected_spec.output_dir
    ):
        raise RuntimeError(
            "Sweep job spec does not match the current registration config. "
            f"Expected {expected_spec.label}, got {job_spec.label}."
        )

    return job


def _run_sweep_job_impl(
    *,
    job_spec: SweepJobSpec,
    job: registration.RegistrationJob,
) -> SweepRunResult:
    print(f"Running sweep registration for {job_spec.label} ...")

    try:
        result = registration.run_antspy_registration(job)

        if not result.success:
            print(f"  Registration failed for {job_spec.label}: {result.error_message}")
            return SweepRunResult(
                job_index=job_spec.job_index,
                fixed_image_id=job_spec.fixed_image_id,
                moving_image_id=job_spec.moving_image_id,
                preset_name=job_spec.preset_name,
                success=False,
                output_dir=job_spec.output_dir,
                error_message=result.error_message,
            )

        print(f"  Registration finished: {job_spec.output_dir}")
        return SweepRunResult(
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
        return SweepRunResult(
            job_index=job_spec.job_index,
            fixed_image_id=job_spec.fixed_image_id,
            moving_image_id=job_spec.moving_image_id,
            preset_name=job_spec.preset_name,
            success=False,
            output_dir=job_spec.output_dir,
            error_message=str(exc),
        )
