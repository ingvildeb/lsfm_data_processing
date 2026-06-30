from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re
import tomllib

from atlasspace import ImageConfig, SpaceDefinition, registration

from lsfm_data_processing.utils.io_helpers import (
    load_script_config,
    normalize_user_path,
    require_dir,
    require_file,
)


@dataclass
class SweepRegisterSettings:
    output_root: Path
    orientation_alignment: str
    write_input_images: bool
    preset_names: list[str]
    templates_cfg: dict[str, Any]
    images_cfg: dict[str, Any]
    image_to_template_cfg: dict[str, list[str]]


@dataclass
class SweepJobSpec:
    image_key: str
    template_key: str
    preset_name: str


@dataclass
class SweepRunResult:
    image_key: str
    template_key: str
    preset_name: str
    success: bool
    output_dir: Path | None
    error_message: str | None


def _normalize_backslashes_in_toml_strings(config_text: str) -> str:
    string_pattern = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')

    def replace_match(match: re.Match[str]) -> str:
        string_content = match.group(1).replace("\\", "/")
        return f'"{string_content}"'

    return string_pattern.sub(replace_match, config_text)


def _load_toml_dict(config_path: Path) -> dict[str, Any]:
    config_text = config_path.read_text(encoding="utf-8")

    try:
        return tomllib.loads(config_text)
    except tomllib.TOMLDecodeError as exc:
        if "Unescaped '\\' in a string" not in str(exc):
            raise
        normalized_text = _normalize_backslashes_in_toml_strings(config_text)
        return tomllib.loads(normalized_text)


def _normalize_image_to_template_cfg(
    raw_mapping: dict[str, Any],
) -> dict[str, list[str]]:
    normalized_mapping: dict[str, list[str]] = {}

    for image_key, template_value in raw_mapping.items():
        if isinstance(template_value, str):
            normalized_mapping[image_key] = [template_value]
        elif isinstance(template_value, list) and all(
            isinstance(item, str) for item in template_value
        ):
            if not template_value:
                raise RuntimeError(
                    f"[image_to_template].{image_key} must not be an empty list."
                )
            normalized_mapping[image_key] = template_value
        else:
            raise RuntimeError(
                "Each [image_to_template] value must be either a template key string "
                "or a list of template key strings.\n"
                f"Got {template_value!r} for image key '{image_key}'."
            )

    return normalized_mapping


def _settings_from_cfg(cfg: dict[str, Any]) -> SweepRegisterSettings:
    run_cfg = cfg["run"]
    image_to_template_cfg = _normalize_image_to_template_cfg(cfg["image_to_template"])

    return SweepRegisterSettings(
        output_root=normalize_user_path(run_cfg["output_root"]),
        orientation_alignment=run_cfg.get("orientation_alignment", "none"),
        write_input_images=run_cfg.get("write_input_images", False),
        preset_names=list(cfg["presets"]["names"]),
        templates_cfg=cfg["templates"],
        images_cfg=cfg["images"],
        image_to_template_cfg=image_to_template_cfg,
    )


def load_sweep_register_settings(
    script_path: Path,
    config_basename: str,
    *,
    test_mode: bool = False,
) -> SweepRegisterSettings:
    cfg = load_script_config(
        script_path,
        config_basename,
        test_mode=test_mode,
    )
    return _settings_from_cfg(cfg)


def load_sweep_register_settings_from_path(config_path: Path) -> SweepRegisterSettings:
    resolved_path = require_file(config_path, "Sweep registration config")
    cfg = _load_toml_dict(resolved_path)
    return _settings_from_cfg(cfg)


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


def _build_image_config(
    image_cfg: dict[str, Any],
    image_key: str,
) -> ImageConfig:
    image_space_name = image_cfg.get("space_name", image_key)
    image_resolution_um = float(image_cfg["resolution_um"])
    image_path = require_file(
        normalize_user_path(image_cfg["image"]),
        f"Image for {image_key}",
    )

    return ImageConfig(
        image_id=image_key,
        image=image_path,
        space=_make_space_definition(
            space_name=image_space_name,
            orientation=image_cfg["orientation"],
            resolution_um=image_resolution_um,
        ),
    )


def _build_template_image_config(
    settings: SweepRegisterSettings,
    template_key: str,
) -> ImageConfig:
    template_cfg = settings.templates_cfg[template_key]
    return _build_image_config(template_cfg, template_key)


def _build_run_image_config(
    settings: SweepRegisterSettings,
    image_key: str,
) -> ImageConfig:
    image_cfg = settings.images_cfg[image_key]
    return _build_image_config(image_cfg, image_key)


def get_sweep_job_specs(settings: SweepRegisterSettings) -> list[SweepJobSpec]:
    job_specs: list[SweepJobSpec] = []

    for image_key, template_keys in settings.image_to_template_cfg.items():
        if image_key not in settings.images_cfg:
            raise RuntimeError(
                f"Image key '{image_key}' is not defined under [images]."
            )

        for template_key in template_keys:
            if template_key not in settings.templates_cfg:
                raise RuntimeError(
                    f"Template key '{template_key}' is not defined under [templates]."
                )

            for preset_name in settings.preset_names:
                job_specs.append(
                    SweepJobSpec(
                        image_key=image_key,
                        template_key=template_key,
                        preset_name=preset_name,
                    )
                )

    return job_specs


def _build_parameters(settings: SweepRegisterSettings, preset_name: str):
    parameters = registration.load_preset(preset_name)
    parameters.execution.write_input_images = settings.write_input_images
    return parameters


def _pair_output_dir(
    settings: SweepRegisterSettings,
    *,
    image_key: str,
    template_key: str,
    preset_name: str,
) -> Path:
    return settings.output_root / image_key / template_key / preset_name


def _build_registration_job(
    settings: SweepRegisterSettings,
    *,
    image_key: str,
    template_key: str,
    run_image_config: ImageConfig,
    template_config: ImageConfig,
    parameters,
) -> registration.RegistrationJob:
    output_dir = _pair_output_dir(
        settings,
        image_key=image_key,
        template_key=template_key,
        preset_name=parameters.name,
    )

    return registration.RegistrationJob(
        fixed_image_config=run_image_config,
        moving_image_config=template_config,
        output_dir=output_dir,
        parameters=parameters,
        orientation_alignment=settings.orientation_alignment,
    )


def _run_sweep_job_impl(
    *,
    settings: SweepRegisterSettings,
    job_spec: SweepJobSpec,
    parameters,
) -> SweepRunResult:
    print(
        "Running sweep registration for "
        f"{job_spec.image_key} -> {job_spec.template_key} "
        f"with preset '{job_spec.preset_name}' ..."
    )

    try:
        run_image_config = _build_run_image_config(settings, job_spec.image_key)
        template_config = _build_template_image_config(settings, job_spec.template_key)
        job = _build_registration_job(
            settings,
            image_key=job_spec.image_key,
            template_key=job_spec.template_key,
            run_image_config=run_image_config,
            template_config=template_config,
            parameters=parameters,
        )

        result = registration.run_antspy_registration(job)

        if not result.success:
            print(
                f"  Registration failed for {job_spec.image_key} "
                f"with preset '{job_spec.preset_name}': {result.error_message}"
            )
            return SweepRunResult(
                image_key=job_spec.image_key,
                template_key=job_spec.template_key,
                preset_name=job_spec.preset_name,
                success=False,
                output_dir=job.output_dir,
                error_message=result.error_message,
            )

        print(f"  Registration finished: {job.output_dir}")

        return SweepRunResult(
            image_key=job_spec.image_key,
            template_key=job_spec.template_key,
            preset_name=job_spec.preset_name,
            success=True,
            output_dir=job.output_dir,
            error_message=None,
        )
    except Exception as exc:
        print(
            f"  Failed while processing {job_spec.image_key} "
            f"with preset '{job_spec.preset_name}': {exc}"
        )
        return SweepRunResult(
            image_key=job_spec.image_key,
            template_key=job_spec.template_key,
            preset_name=job_spec.preset_name,
            success=False,
            output_dir=None,
            error_message=str(exc),
        )


def run_sweep_registration_for_job(
    *,
    settings: SweepRegisterSettings,
    job_spec: SweepJobSpec,
) -> SweepRunResult:
    parameters = _build_parameters(settings, job_spec.preset_name)
    return _run_sweep_job_impl(
        settings=settings,
        job_spec=job_spec,
        parameters=parameters,
    )


def run_sweep_registration_for_jobs(
    *,
    settings: SweepRegisterSettings,
    job_specs: list[SweepJobSpec],
) -> list[SweepRunResult]:
    parameters_by_preset = {
        preset_name: _build_parameters(settings, preset_name)
        for preset_name in settings.preset_names
    }
    print(f"Using registration presets: {', '.join(settings.preset_names)}")
    return [
        _run_sweep_job_impl(
            settings=settings,
            job_spec=job_spec,
            parameters=parameters_by_preset[job_spec.preset_name],
        )
        for job_spec in job_specs
    ]
