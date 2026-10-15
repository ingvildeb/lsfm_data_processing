"""Plan and apply immutable rescue-registration artifacts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
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
from .models import normalize_run_path
from .rescue_catalog import ExpandedRescueJob


PARAMETER_SECTIONS = {
    "syn_gradient_step": "registration",
    "working_resolution_um": "registration",
    "fixed_padding_um": "preprocessing",
}
RESCUE_GENERATION_MANIFEST_FILENAME = "rescue_generation_manifest.json"


@dataclass(frozen=True)
class RescueGenerationRequest:
    """A concrete rescue job plus project-owned image/template details."""

    job: ExpandedRescueJob
    registration_spec: RegistrationBatchSpec
    base_preset: Mapping[str, Any]
    subject_image_source: Path | None = None
    content_addressed_preset: bool = False


@dataclass(frozen=True)
class RescueSourceContext:
    """Parameters and subject-image input inherited from one registration run."""

    source_run: str
    parameter_snapshot: Mapping[str, Any]
    subject_image: str


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


def load_rescue_source_context(
    *,
    subject_id: str,
    subject_dir: Path,
    source_run: str,
) -> RescueSourceContext:
    """Load the complete reusable execution context of a prior run."""

    normalized_run = normalize_run_path(source_run)
    parts = Path(normalized_run).parts
    if (
        not parts
        or parts[0] != "registration_runs"
        or ".." in parts
    ):
        raise ValueError(
            "Source run must be a subject-relative path under registration_runs/: "
            f"{source_run!r}"
        )
    run_dir = Path(subject_dir) / Path(normalized_run)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Source registration run is missing: {run_dir}")

    try:
        from atlasspace.registration import load_registration_result_manifest
    except ImportError as exc:
        raise RuntimeError(
            "atlasspace is required to resolve registration source runs. "
            "Install lsfm-data-processing with the registration extra."
        ) from exc
    manifest = load_registration_result_manifest(run_dir)
    images = (manifest.fixed_image, manifest.moving_image)
    matches = [image for image in images if image.image_id == subject_id]
    if len(matches) != 1:
        raise ValueError(
            f"Source run must identify subject {subject_id!r} exactly once; "
            f"found {len(matches)} matching manifest images in {run_dir}."
        )
    snapshot_value = manifest.parameters_snapshot
    if not snapshot_value:
        raise ValueError(
            f"Source registration manifest has no parameter snapshot: {run_dir}"
        )
    snapshot_path = Path(snapshot_value)
    if not snapshot_path.is_absolute():
        snapshot_path = run_dir / snapshot_path
    parameters = load_registration_parameter_snapshot(snapshot_path)
    return RescueSourceContext(
        source_run=normalized_run,
        parameter_snapshot=parameters,
        subject_image=matches[0].image,
    )


def source_derived_generation_request(
    *,
    job: ExpandedRescueJob,
    registration_spec: RegistrationBatchSpec,
    baseline_preset: Mapping[str, Any],
    subject_dir: Path,
    subject_image_source: Path | None = None,
    subject_image_override: str | PurePosixPath | None = None,
    content_addressed_preset: bool = False,
) -> RescueGenerationRequest:
    """Build a request, inheriting full context when ``job.source_run`` is set.

    ``subject_image_override`` explicitly supports reusing an accepted masked
    image while parameters come from a different source run. It is never
    inferred from other runs or subject history.
    """

    if not job.source_run:
        if subject_image_override is not None:
            raise ValueError(
                "subject_image_override requires an explicit source run"
            )
        return RescueGenerationRequest(
            job=job,
            registration_spec=registration_spec,
            base_preset=baseline_preset,
            subject_image_source=subject_image_source,
            content_addressed_preset=content_addressed_preset,
        )
    context = load_rescue_source_context(
        subject_id=job.subject_id,
        subject_dir=subject_dir,
        source_run=job.source_run,
    )
    matching = [
        image for image in registration_spec.images if image.image_id == job.subject_id
    ]
    if len(matching) != 1:
        raise ValueError(
            "Registration spec must contain exactly one image for "
            f"{job.subject_id}; found {len(matching)}."
        )
    inherited_image = (
        context.subject_image
        if subject_image_override is None
        else subject_image_override
    )
    inherited_images = tuple(
        replace(image, path=inherited_image)
        if image.image_id == job.subject_id
        else image
        for image in registration_spec.images
    )
    return RescueGenerationRequest(
        job=job,
        registration_spec=replace(registration_spec, images=inherited_images),
        base_preset=context.parameter_snapshot,
        subject_image_source=subject_image_source,
        content_addressed_preset=True,
    )


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


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version() -> str:
    try:
        return version("lsfm-data-processing")
    except PackageNotFoundError:
        return "unknown"


def _load_generation_manifest(path: Path, *, batch_id: str) -> dict[str, Any]:
    if not path.exists():
        timestamp = datetime.now(timezone.utc).isoformat()
        return {
            "schema_version": 1,
            "batch_id": batch_id,
            "created_at": timestamp,
            "generator": {
                "package": "lsfm-data-processing",
                "version": _package_version(),
            },
            "presets": {},
            "jobs": {},
        }
    if not path.is_file():
        raise FileExistsError(f"Rescue generation manifest is not a file: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid rescue generation manifest: {path}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("batch_id") != batch_id
        or not isinstance(manifest.get("presets"), dict)
        or not isinstance(manifest.get("jobs"), dict)
    ):
        raise ValueError(f"Unsupported rescue generation manifest: {path}")
    return manifest


def _merge_manifest_record(
    records: dict[str, Any],
    *,
    key: str,
    record: dict[str, Any],
    label: str,
) -> None:
    existing = records.get(key)
    if existing is not None and existing != record:
        raise FileExistsError(
            f"Existing rescue {label} provenance differs for {key}"
        )
    records[key] = record


def _merge_job_manifest_record(
    records: dict[str, Any],
    *,
    key: str,
    record: dict[str, Any],
) -> None:
    existing = records.get(key)
    if existing is None:
        records[key] = record
        return
    if not isinstance(existing, dict):
        raise FileExistsError(f"Existing rescue job provenance is invalid for {key}")
    mutable = {"generated_at", "input_updated_at", "rationale", "subject_input"}
    old_immutable = {name: value for name, value in existing.items() if name not in mutable}
    new_immutable = {name: value for name, value in record.items() if name not in mutable}
    if old_immutable != new_immutable:
        raise FileExistsError(f"Existing rescue job provenance differs for {key}")
    record["generated_at"] = existing.get("generated_at", record["generated_at"])
    if existing.get("subject_input") != record.get("subject_input"):
        record["input_updated_at"] = datetime.now(timezone.utc).isoformat()
    elif "input_updated_at" in existing:
        record["input_updated_at"] = existing["input_updated_at"]
    records[key] = record


def _terminal_registration_result(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_file():
        raise FileExistsError(f"Registration result is not a file: {path}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid registration result manifest: {path}") from exc
    if not isinstance(result.get("success"), bool):
        raise ValueError(f"Registration result has no Boolean success field: {path}")
    return True


def _planned_run_from_manifest(record: Mapping[str, Any]) -> PlannedRescueRun:
    return PlannedRescueRun(
        request_name=str(record["strategy_id"]),
        variant=str(record["variant"]),
        subject_ids=(str(record["subject_id"]),),
        config_filename=str(record["config_filename"]),
        output_subdir=str(record["output_subdir"]),
        overrides=dict(record.get("parameter_overrides", {})),
        source_run=str(record.get("source_run") or ""),
        hpc_mem_gb=record.get("hpc_mem_gb"),
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
    """Plan immutable rescue artifacts and reconstruct unfinished submissions."""

    configs_dir = Path(configs_dir)
    manifest_path = configs_dir / RESCUE_GENERATION_MANIFEST_FILENAME
    if not requests and not manifest_path.is_file():
        return RescueGenerationPlan(batch_id, Path(evaluation_workbook), (), ())
    planned: dict[Path, PlannedRescueArtifact] = {}
    manifest = _load_generation_manifest(manifest_path, batch_id=batch_id)
    presets = manifest["presets"]
    jobs = manifest["jobs"]
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
        preset_key = preset_relative.as_posix()
        preset_record = {
            "generated_preset": preset_name,
            "scientific_sha256": scientific_hash,
            "content_sha256": sha256(preset_content).hexdigest(),
        }
        if not request.content_addressed_preset:
            preset_record.update(
                {
                    "base_preset_name": str(request.base_preset.get("name", "")),
                    "base_scientific_sha256": base_hash,
                    "overrides": overrides,
                }
            )
        _merge_manifest_record(
            presets,
            key=preset_key,
            record=preset_record,
            label="preset",
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
        if request.subject_image_source is None:
            raise ValueError(
                f"Rescue request for {job.subject_id} must provide subject_image_source"
            )
        subject_image_source = Path(request.subject_image_source)
        if not subject_image_source.is_file():
            raise FileNotFoundError(
                f"Local rescue subject image is missing: {subject_image_source}"
            )
        existing_job = jobs.get(config_filename)
        generated_at = (
            existing_job.get("generated_at")
            if isinstance(existing_job, dict) and existing_job.get("generated_at")
            else datetime.now(timezone.utc).isoformat()
        )
        job_record = {
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
            "config_filename": config_filename,
            "generated_at": generated_at,
            "subject_input": {
                "local_source": str(subject_image_source),
                "hpc_path": str(subject_images[0].path),
                "size_bytes": subject_image_source.stat().st_size,
                "sha256": _file_sha256(subject_image_source),
            },
        }
        _merge_job_manifest_record(jobs, key=config_filename, record=job_record)

    manifest_content = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _add_artifact(
        planned,
        destination=manifest_path,
        content=manifest_content,
        purpose="rescue generation manifest",
        replace_generated=True,
    )

    runs: list[PlannedRescueRun] = []
    helper_entries: list[HpcSubmissionEntry] = []
    batch_root = configs_dir.parent
    for config_filename, record in sorted(jobs.items()):
        if not isinstance(record, dict):
            raise ValueError(
                f"Invalid rescue job provenance for {config_filename}"
            )
        config_path = configs_dir / config_filename
        planned_config = planned.get(config_path)
        if not config_path.is_file() and planned_config is None:
            raise FileNotFoundError(
                f"Generated rescue config is missing: {config_path}"
            )
        output_manifest = (
            batch_root
            / "subjects"
            / str(record["subject_id"])
            / Path(str(record["output_subdir"]))
            / "registration_result.json"
        )
        if _terminal_registration_result(output_manifest):
            continue
        run = _planned_run_from_manifest(record)
        runs.append(run)
        helper_entries.append(
            HpcSubmissionEntry(run.config_filename, run.hpc_mem_gb)
        )

    helper_content = render_hpc_submission_helper(
        batch_id=batch_id,
        entries=tuple(helper_entries),
        helper_filename=Path(helper_path).name,
        job_name_prefix=job_name_prefix,
        log_dir=log_dir,
        input_manifest=(
            PurePosixPath("configs") / RESCUE_GENERATION_MANIFEST_FILENAME
        ).as_posix(),
        allow_empty=True,
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
            if item.state == "replace" and item.purpose in {
                "HPC submission helper",
                "rescue generation manifest",
            }:
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
