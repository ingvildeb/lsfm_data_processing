"""Plan and apply canonical baseline registration artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePath
import tomllib

from .generation import (
    HpcConfigSpec,
    HpcSubmissionEntry,
    RegistrationBatchSpec,
    render_hpc_config,
    render_hpc_submission_helper,
    render_registration_batch_config,
)


@dataclass(frozen=True)
class PlannedBaselineArtifact:
    destination: Path
    content: bytes
    state: str
    purpose: str
    replaceable: bool = False


@dataclass(frozen=True)
class BaselineGenerationPlan:
    batch_id: str
    files: tuple[PlannedBaselineArtifact, ...]


def _state(destination: Path, content: bytes, *, replaceable: bool) -> str:
    if not destination.exists():
        return "write"
    if not destination.is_file():
        raise FileExistsError(f"Generated destination is not a file: {destination}")
    if destination.read_bytes() == content:
        return "already_present"
    if replaceable:
        return "replace"
    raise FileExistsError(
        "Immutable baseline artifact exists and differs; it will not be "
        f"overwritten: {destination}"
    )


def _artifact(
    destination: Path,
    content: bytes,
    purpose: str,
    *,
    replaceable: bool = False,
) -> PlannedBaselineArtifact:
    return PlannedBaselineArtifact(
        destination=Path(destination),
        content=content,
        state=_state(Path(destination), content, replaceable=replaceable),
        purpose=purpose,
        replaceable=replaceable,
    )


def _registration_spec_payload(spec: RegistrationBatchSpec) -> dict[str, object]:
    """Return JSON-safe scientific inputs without project tracker metadata."""

    def normalize(value: object) -> object:
        if isinstance(value, PurePath):
            return str(value)
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    return normalize(asdict(spec))  # type: ignore[return-value]


def plan_baseline_generation(
    *,
    batch_id: str,
    configs_dir: Path,
    helper_path: Path,
    registration_spec: RegistrationBatchSpec,
    hpc_spec: HpcConfigSpec,
) -> BaselineGenerationPlan:
    """Preflight the immutable baseline and replaceable operational bundle."""

    configs_dir = Path(configs_dir)
    helper_path = Path(helper_path)
    baseline_path = configs_dir / "baseline.toml"
    provenance_path = configs_dir / "baseline.provenance.json"
    hpc_path = configs_dir / "hpc.toml"

    rendered_baseline_content = render_registration_batch_config(
        registration_spec
    ).encode("utf-8")
    baseline_content = rendered_baseline_content
    adopted_existing_config = False
    if baseline_path.exists():
        if not baseline_path.is_file():
            raise FileExistsError(
                f"Baseline registration destination is not a file: {baseline_path}"
            )
        existing_content = baseline_path.read_bytes()
        if existing_content != rendered_baseline_content:
            try:
                existing_toml = tomllib.loads(existing_content.decode("utf-8"))
                rendered_toml = tomllib.loads(
                    rendered_baseline_content.decode("utf-8")
                )
            except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
                raise FileExistsError(
                    "Immutable baseline artifact exists but is not valid compatible "
                    f"TOML: {baseline_path}"
                ) from exc
            if existing_toml != rendered_toml:
                raise FileExistsError(
                    "Immutable baseline artifact exists with different scientific "
                    f"or job settings: {baseline_path}"
                )
            baseline_content = existing_content
            adopted_existing_config = True
    spec_payload = _registration_spec_payload(registration_spec)
    provenance = {
        "schema_version": 1,
        "batch_id": batch_id,
        "registration_config": baseline_path.name,
        "registration_config_sha256": sha256(baseline_content).hexdigest(),
        "registration_spec": spec_payload,
        "adopted_existing_config": adopted_existing_config,
    }
    provenance_content = (
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    hpc_content = render_hpc_config(hpc_spec).encode("utf-8")
    helper_content = render_hpc_submission_helper(
        batch_id=batch_id,
        entries=(HpcSubmissionEntry(baseline_path.name),),
        helper_filename=helper_path.name,
        job_name_prefix=hpc_spec.job_name_prefix,
        log_dir=hpc_spec.log_dir,
    ).encode("utf-8")
    files = (
        _artifact(baseline_path, baseline_content, "baseline registration config"),
        _artifact(provenance_path, provenance_content, "baseline provenance"),
        _artifact(
            hpc_path,
            hpc_content,
            "operational HPC config",
            replaceable=True,
        ),
        _artifact(
            helper_path,
            helper_content,
            "baseline HPC submission helper",
            replaceable=True,
        ),
    )
    return BaselineGenerationPlan(batch_id=batch_id, files=files)


def summarize_baseline_generation_plan(plan: BaselineGenerationPlan) -> str:
    counts = {
        state: sum(item.state == state for item in plan.files)
        for state in ("write", "replace", "already_present")
    }
    return "\n".join(
        [
            f"Baseline artifact plan: {plan.batch_id}",
            f"Files to write: {counts['write']}",
            f"Operational files to refresh: {counts['replace']}",
            f"Identical files already present: {counts['already_present']}",
        ]
    )


def _atomic_replace(destination: Path, content: bytes) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def apply_baseline_generation_plan(plan: BaselineGenerationPlan) -> None:
    """Apply a preflighted plan with resumable, contract-aware writes."""

    for item in plan.files:
        item.destination.parent.mkdir(parents=True, exist_ok=True)
        if item.destination.exists():
            if item.destination.is_file() and item.destination.read_bytes() == item.content:
                continue
            if item.replaceable:
                _atomic_replace(item.destination, item.content)
                continue
            raise FileExistsError(
                "Immutable baseline artifact changed after preflight: "
                f"{item.destination}"
            )
        with item.destination.open("xb") as handle:
            handle.write(item.content)
        if item.destination.read_bytes() != item.content:
            raise IOError(f"Written baseline file failed verification: {item.destination}")
