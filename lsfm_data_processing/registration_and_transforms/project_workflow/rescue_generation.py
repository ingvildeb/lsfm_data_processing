"""Plan and apply immutable rescue-registration artifacts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

import yaml

from .generation import (
    HpcSubmissionEntry,
    RegistrationBatchSpec,
    render_hpc_submission_helper,
    render_registration_batch_config,
)
from .rescue_catalog import ExpandedRescueJob


PARAMETER_SECTIONS = {
    "syn_gradient_step": "registration",
    "working_resolution_um": "registration",
    "fixed_padding_um": "preprocessing",
}


@dataclass(frozen=True)
class RescueGenerationRequest:
    """A concrete rescue job plus project-owned image/template details."""

    job: ExpandedRescueJob
    registration_spec: RegistrationBatchSpec
    base_preset: Mapping[str, Any]
    content_addressed_preset: bool = False


@dataclass(frozen=True)
class PlannedRescueArtifact:
    destination: Path
    content: bytes
    state: str
    purpose: str


@dataclass(frozen=True)
class PlannedRescueRun:
    request_name: str
    variant: str
    subject_ids: tuple[str, ...]
    config_filename: str
    output_subdir: str
    overrides: dict[str, float | int]
    source_run: str = ""
    hpc_mem_gb: int | None = None


@dataclass(frozen=True)
class RescueGenerationPlan:
    batch_id: str
    evaluation_workbook: Path
    runs: tuple[PlannedRescueRun, ...]
    files: tuple[PlannedRescueArtifact, ...]

    @property
    def subject_job_count(self) -> int:
        return len(self.runs)


def load_registration_preset(reference: str) -> dict[str, Any]:
    """Load a built-in or file-based AtlasSpace preset as plain data."""

    try:
        from atlasspace.registration import load_preset
    except ImportError as exc:
        raise RuntimeError(
            "atlasspace is required to materialize rescue presets. "
            "Install lsfm-data-processing with the registration extra."
        ) from exc
    return load_preset(reference).model_dump(mode="json")


def load_registration_parameter_snapshot(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Registration parameter snapshot is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid registration parameter snapshot: {path}")
    return data


def _slug(value: object, *, field: str) -> str:
    result = re.sub(r"[^a-z0-9_-]+", "_", str(value).strip().casefold()).strip("_")
    if not result:
        raise ValueError(f"{field} must contain at least one letter or number")
    return result


def _number_token(value: float | int) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def _override_token(overrides: Mapping[str, float | int]) -> str:
    tokens: list[str] = []
    if "working_resolution_um" in overrides:
        tokens.append(f"wr{_number_token(overrides['working_resolution_um'])}um")
    if "syn_gradient_step" in overrides:
        tokens.append(f"gs{_number_token(overrides['syn_gradient_step'])}")
    if "fixed_padding_um" in overrides:
        tokens.append(f"pad{_number_token(overrides['fixed_padding_um'])}um")
    return "_".join(tokens) or "unchanged"


def _scientific_preset(
    base_preset: Mapping[str, Any], overrides: Mapping[str, float | int]
) -> dict[str, Any]:
    content = deepcopy(dict(base_preset))
    for parameter, value in overrides.items():
        section = PARAMETER_SECTIONS.get(parameter)
        if section is None:
            raise ValueError(f"Unsupported rescue parameter: {parameter}")
        table = content.get(section)
        if not isinstance(table, dict):
            raise ValueError(f"Base preset has no {section!r} table")
        table[parameter] = value
    return content


def _scientific_payload(preset: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(preset))
    payload.pop("name", None)
    payload.pop("description", None)
    return payload


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _preset_name(
    preset: Mapping[str, Any],
    overrides: Mapping[str, float | int],
    *,
    content_addressed: bool,
) -> str:
    scientific_hash = _stable_hash(_scientific_payload(preset))
    if content_addressed:
        return f"registration_{scientific_hash[:12]}"
    base_name = _slug(preset.get("name", "registration"), field="preset name")
    return f"{base_name}_{_override_token(overrides)}"


def _render_preset(
    preset: Mapping[str, Any], *, name: str
) -> bytes:
    content = deepcopy(dict(preset))
    content["name"] = name
    content["description"] = (
        "Generated registration rescue preset; regenerate from the canonical "
        "evaluation workbook instead of editing this file."
    )
    header = "# Generated rescue preset. Do not edit manually.\n\n"
    return (
        header + yaml.safe_dump(content, sort_keys=False, allow_unicode=True)
    ).encode("utf-8")


def _artifact_state(
    destination: Path, content: bytes, *, replace_generated: bool = False
) -> str:
    if not destination.exists():
        return "write"
    if not destination.is_file():
        raise FileExistsError(f"Rescue destination is not a file: {destination}")
    if destination.read_bytes() == content:
        return "already_present"
    if replace_generated:
        return "replace"
    raise FileExistsError(
        f"Rescue destination exists and differs; it will not be overwritten: "
        f"{destination}"
    )


def _add_artifact(
    planned: dict[Path, PlannedRescueArtifact],
    *,
    destination: Path,
    content: bytes,
    purpose: str,
    replace_generated: bool = False,
) -> None:
    existing = planned.get(destination)
    if existing is not None:
        if existing.content != content:
            raise ValueError(f"Differing generated artifacts map to {destination}")
        return
    planned[destination] = PlannedRescueArtifact(
        destination=destination,
        content=content,
        state=_artifact_state(
            destination, content, replace_generated=replace_generated
        ),
        purpose=purpose,
    )


def plan_rescue_generation(
    *,
    batch_id: str,
    evaluation_workbook: Path,
    configs_dir: Path,
    helper_path: Path,
    requests: tuple[RescueGenerationRequest, ...],
    job_name_prefix: str,
    log_dir: str = "logs/rescue",
) -> RescueGenerationPlan:
    """Plan every preset, TOML, provenance file, and submission helper."""

    if not requests:
        return RescueGenerationPlan(batch_id, Path(evaluation_workbook), (), ())
    planned: dict[Path, PlannedRescueArtifact] = {}
    runs: list[PlannedRescueRun] = []
    helper_entries: list[HpcSubmissionEntry] = []
    for request in requests:
        job = request.job
        overrides = job.override_mapping
        scientific = _scientific_preset(request.base_preset, overrides)
        scientific_hash = _stable_hash(_scientific_payload(scientific))
        preset_name = _preset_name(
            scientific,
            overrides,
            content_addressed=request.content_addressed_preset,
        )
        preset_relative = PurePosixPath("registration_presets") / f"{preset_name}.yaml"
        preset_path = Path(configs_dir) / Path(preset_relative.as_posix())
        preset_content = _render_preset(
            scientific, name=preset_name
        )
        _add_artifact(
            planned,
            destination=preset_path,
            content=preset_content,
            purpose="generated registration preset",
        )

        base_hash = _stable_hash(_scientific_payload(request.base_preset))
        preset_provenance = {
            "schema_version": 1,
            "base_preset_name": (
                None
                if request.content_addressed_preset
                else str(request.base_preset.get("name", ""))
            ),
            "base_scientific_sha256": base_hash,
            "generated_preset": preset_name,
            "scientific_sha256": scientific_hash,
            "overrides": overrides,
        }
        preset_provenance_content = (
            json.dumps(preset_provenance, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        _add_artifact(
            planned,
            destination=preset_path.with_suffix(".provenance.json"),
            content=preset_provenance_content,
            purpose="preset provenance",
        )

        request_slug = _slug(job.strategy_id, field="request name")
        subject_slug = _slug(job.subject_id, field="subject ID")
        variant_slug = _slug(job.variant, field="variant")
        output_subdir = (
            PurePosixPath("registration_runs") / job.output_group / variant_slug
        ).as_posix()
        config_filename = f"{request_slug}_{subject_slug}_{variant_slug}.toml"
        config_path = Path(configs_dir) / config_filename
        registration_spec = replace(
            request.registration_spec,
            registration_presets=(preset_relative.as_posix(),),
            output_subdir=output_subdir,
        )
        config_content = render_registration_batch_config(registration_spec).encode(
            "utf-8"
        )
        _add_artifact(
            planned,
            destination=config_path,
            content=config_content,
            purpose="registration config",
        )

        subject_images = [
            image
            for image in registration_spec.images
            if image.image_id == job.subject_id
        ]
        if len(subject_images) != 1:
            raise ValueError(
                f"Registration spec must contain exactly one image for {job.subject_id}"
            )
        job_provenance = {
            "schema_version": 1,
            "subject_id": job.subject_id,
            "strategy_id": job.strategy_id,
            "variant": job.variant,
            "source_run": job.source_run or None,
            "output_subdir": output_subdir,
            "registration_preset": preset_relative.as_posix(),
            "subject_image": str(subject_images[0].path),
            "scientific_sha256": scientific_hash,
            "parameter_overrides": overrides,
            "hpc_mem_gb": job.hpc_mem_gb,
            "rationale": job.rationale,
            "config_sha256": sha256(config_content).hexdigest(),
        }
        job_provenance_content = (
            json.dumps(job_provenance, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        _add_artifact(
            planned,
            destination=config_path.with_suffix(".provenance.json"),
            content=job_provenance_content,
            purpose="registration job provenance",
        )
        runs.append(
            PlannedRescueRun(
                request_name=job.strategy_id,
                variant=job.variant,
                subject_ids=(job.subject_id,),
                config_filename=config_filename,
                output_subdir=output_subdir,
                overrides=overrides,
                source_run=job.source_run,
                hpc_mem_gb=job.hpc_mem_gb,
            )
        )
        helper_entries.append(HpcSubmissionEntry(config_filename, job.hpc_mem_gb))

    helper_content = render_hpc_submission_helper(
        batch_id=batch_id,
        entries=tuple(helper_entries),
        helper_filename=Path(helper_path).name,
        job_name_prefix=job_name_prefix,
        log_dir=log_dir,
    ).encode("utf-8")
    _add_artifact(
        planned,
        destination=Path(helper_path),
        content=helper_content,
        purpose="HPC submission helper",
        replace_generated=True,
    )
    return RescueGenerationPlan(
        batch_id=batch_id,
        evaluation_workbook=Path(evaluation_workbook),
        runs=tuple(runs),
        files=tuple(planned.values()),
    )


def summarize_rescue_generation_plan(plan: RescueGenerationPlan) -> str:
    write_count = sum(item.state == "write" for item in plan.files)
    replace_count = sum(item.state == "replace" for item in plan.files)
    existing_count = sum(item.state == "already_present" for item in plan.files)
    lines = [
        f"Registration rescue preparation plan: {plan.batch_id}",
        f"Rescue configurations: {len(plan.runs)}",
        f"Expanded subject jobs: {plan.subject_job_count}",
        f"Files to write: {write_count}",
        f"Generated helpers to refresh: {replace_count}",
        f"Identical files already present: {existing_count}",
        f"Evaluation source: {plan.evaluation_workbook}",
    ]
    for run in plan.runs:
        changes = ", ".join(f"{key}={value}" for key, value in run.overrides.items())
        lines.append(
            f"  {run.request_name}/{run.variant}: {run.subject_ids[0]}; "
            f"changes={changes}; output={run.output_subdir}"
        )
    return "\n".join(lines)


def apply_rescue_generation_plan(plan: RescueGenerationPlan) -> None:
    """Apply a preflighted plan with immutable artifacts and resumable writes."""

    for item in plan.files:
        if item.destination.exists():
            if item.destination.is_file() and item.destination.read_bytes() == item.content:
                continue
            if item.state == "replace" and item.purpose == "HPC submission helper":
                temporary = item.destination.with_name(f".{item.destination.name}.tmp")
                temporary.write_bytes(item.content)
                temporary.replace(item.destination)
                continue
            raise FileExistsError(
                "Destination changed after preflight and will not be overwritten: "
                f"{item.destination}"
            )
        item.destination.parent.mkdir(parents=True, exist_ok=True)
        with item.destination.open("xb") as handle:
            handle.write(item.content)
        if item.destination.read_bytes() != item.content:
            raise IOError(f"Written rescue file failed verification: {item.destination}")
