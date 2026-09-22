"""Resumable distribution of finalized registration runs into project data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import filecmp
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping

from .models import FinalDecision, FinalDecisionKind, normalize_subject_id


TRANSFERRED_SUFFIX = "_transferred"
SELECTION_METADATA_FILENAME = "registration_selection.json"
EXCLUSION_METADATA_FILENAME = "registration_exclusion.json"
IN_PROGRESS_METADATA_FILENAME = ".registration_distribution_in_progress.json"


@dataclass(frozen=True)
class RegistrationCopyOperation:
    source: Path
    destination: Path
    state: str


@dataclass(frozen=True)
class SubjectRegistrationDistribution:
    subject_id: str
    source_subject_dir: Path
    destination_root: Path
    decision: str
    decision_run_path: str
    selected_run_path: str | None
    selected_destination: Path | None
    archived_run_paths: tuple[str, ...]
    legacy_canonical_archive: tuple[Path, Path] | None
    decision_metadata: dict[str, Any]
    decision_metadata_path: Path
    operations: tuple[RegistrationCopyOperation, ...]

    @property
    def copies(self) -> tuple[RegistrationCopyOperation, ...]:
        return tuple(item for item in self.operations if item.state == "copy")

    @property
    def existing(self) -> tuple[RegistrationCopyOperation, ...]:
        return tuple(
            item for item in self.operations if item.state == "already_present"
        )


@dataclass(frozen=True)
class RegistrationDistributionPlan:
    batch_id: str
    subjects: tuple[SubjectRegistrationDistribution, ...]
    skipped_no_selection: tuple[str, ...]
    skipped_transferred: tuple[str, ...]

    @property
    def operations(self) -> tuple[RegistrationCopyOperation, ...]:
        return tuple(item for subject in self.subjects for item in subject.operations)

    @property
    def required_confirmation(self) -> str:
        return f"DISTRIBUTE REGISTRATIONS {self.batch_id}"


SubjectFinalizer = Callable[[SubjectRegistrationDistribution], None]


def build_registration_distribution_plan(
    *,
    batch_id: str,
    subjects_root: Path | str,
    destination_roots: Mapping[str, Path | str],
    decisions: Mapping[str, FinalDecision],
) -> RegistrationDistributionPlan:
    """Plan canonical installation and archival without modifying files."""

    source_root = Path(subjects_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Batch subjects directory is missing: {source_root}")
    normalized_destinations = {
        normalize_subject_id(subject_id): Path(path)
        for subject_id, path in destination_roots.items()
    }
    active, transferred = _discover_subject_directories(source_root)
    planned: list[SubjectRegistrationDistribution] = []
    skipped: list[str] = []
    errors: list[str] = []
    for subject_id, source_dir in active.items():
        decision = decisions.get(subject_id)
        if decision is None or decision.kind is FinalDecisionKind.UNRESOLVED:
            skipped.append(subject_id)
            continue
        destination_root = normalized_destinations.get(subject_id)
        if destination_root is None:
            errors.append(f"{subject_id}: destination root is unavailable")
            continue
        if not destination_root.parent.is_dir():
            errors.append(
                f"{subject_id}: destination parent directory is missing: "
                f"{destination_root.parent}"
            )
            continue
        try:
            planned.append(
                _plan_subject(
                    batch_id=batch_id,
                    subject_id=subject_id,
                    source_dir=source_dir,
                    destination_root=destination_root,
                    decision=decision,
                )
            )
        except (FileNotFoundError, FileExistsError, ValueError) as exc:
            errors.append(f"{subject_id}: {exc}")
    if errors:
        raise RuntimeError(
            "Registration distribution preflight failed. No files were copied.\n"
            + "\n".join(errors)
        )
    return RegistrationDistributionPlan(
        batch_id=batch_id,
        subjects=tuple(planned),
        skipped_no_selection=tuple(sorted(skipped)),
        skipped_transferred=tuple(sorted(transferred)),
    )


def summarize_registration_distribution_plan(
    plan: RegistrationDistributionPlan,
) -> str:
    copies = sum(len(subject.copies) for subject in plan.subjects)
    existing = sum(len(subject.existing) for subject in plan.subjects)
    lines = [
        f"Registration distribution plan: {plan.batch_id}",
        f"Subjects ready to transfer: {len(plan.subjects):,}",
        f"Files to copy: {copies:,}",
        f"Identical files already present: {existing:,}",
        f"Skipped without a selected run: {len(plan.skipped_no_selection):,}",
        "Archival-only excluded subjects: "
        f"{sum(item.decision == 'excluded' for item in plan.subjects):,}",
        f"Skipped already transferred: {len(plan.skipped_transferred):,}",
    ]
    if plan.skipped_no_selection:
        lines.append("  no selection: " + ", ".join(plan.skipped_no_selection))
    if plan.skipped_transferred:
        lines.append("  transferred: " + ", ".join(plan.skipped_transferred))
    for subject in plan.subjects:
        archive_label = (
            "archived runs"
            if subject.decision == "excluded"
            else "archived alternatives"
        )
        lines.extend(
            (
                f"  {subject.subject_id}: decision={subject.decision}, "
                f"run={subject.decision_run_path}",
                f"    {archive_label}={len(subject.archived_run_paths):,}",
                "    archive existing canonical="
                + ("yes" if subject.legacy_canonical_archive else "no"),
                f"    copy={len(subject.copies):,}, "
                f"already present={len(subject.existing):,}",
                f"    -> {subject.decision_metadata_path.parent}",
            )
        )
    return "\n".join(lines)


def apply_registration_distribution_plan(
    plan: RegistrationDistributionPlan,
    *,
    confirmation: str,
    finalize_subject: SubjectFinalizer | None = None,
) -> None:
    """Apply a plan, finalizing each subject only after validation and hooks."""

    if confirmation != plan.required_confirmation:
        raise ValueError(f"Required confirmation: {plan.required_confirmation!r}")
    for index, subject in enumerate(plan.subjects, start=1):
        print(f"[{index}/{len(plan.subjects)}] {subject.subject_id}: {subject.decision}")
        _apply_legacy_archive(subject)
        metadata_text = _metadata_text(subject.decision_metadata)
        progress_path = _ensure_progress_marker(subject, metadata_text)
        for operation in subject.operations:
            copy_and_verify(operation)
        if finalize_subject is not None:
            finalize_subject(subject)
        _write_new_or_validate_identical(subject.decision_metadata_path, metadata_text)
        if progress_path is not None:
            progress_path.unlink()
        transferred = subject.source_subject_dir.with_name(
            f"{subject.source_subject_dir.name}{TRANSFERRED_SUFFIX}"
        )
        if transferred.exists():
            raise FileExistsError(f"Transferred destination exists: {transferred}")
        subject.source_subject_dir.rename(transferred)
        print(
            f"    marked transferred: {subject.source_subject_dir.name} -> "
            f"{transferred.name}"
        )


def files_match(source: Path | str, destination: Path | str) -> bool:
    source_path = Path(source)
    destination_path = Path(destination)
    return (
        source_path.is_file()
        and destination_path.is_file()
        and source_path.stat().st_size == destination_path.stat().st_size
        and filecmp.cmp(source_path, destination_path, shallow=False)
    )


def plan_verified_copy(
    source: Path | str,
    destination: Path | str,
    *,
    assume_destination_absent: bool = False,
) -> RegistrationCopyOperation:
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(f"Copy source is missing: {source_path}")
    if destination_path.exists() and not assume_destination_absent:
        if not files_match(source_path, destination_path):
            raise FileExistsError(
                "Destination exists but differs from its source; it will not be "
                f"overwritten: {destination_path}"
            )
        state = "already_present"
    else:
        state = "copy"
    return RegistrationCopyOperation(source_path, destination_path, state)


def copy_and_verify(operation: RegistrationCopyOperation) -> None:
    if operation.state == "already_present":
        if not files_match(operation.source, operation.destination):
            raise FileExistsError(
                f"Existing destination changed after preflight: {operation.destination}"
            )
        return
    if operation.state != "copy":
        raise ValueError(f"Unsupported copy operation state: {operation.state!r}")
    operation.destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with operation.source.open("rb") as source_handle:
            with operation.destination.open("xb") as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle)
        shutil.copystat(operation.source, operation.destination)
    except FileExistsError:
        if not files_match(operation.source, operation.destination):
            raise
    if not files_match(operation.source, operation.destination):
        raise IOError(f"Copied file failed verification: {operation.destination}")


def _discover_subject_directories(
    subjects_root: Path,
) -> tuple[dict[str, Path], tuple[str, ...]]:
    active: dict[str, Path] = {}
    transferred: list[str] = []
    for directory in sorted(subjects_root.iterdir()):
        if not directory.is_dir():
            continue
        if directory.name.endswith(TRANSFERRED_SUFFIX):
            subject_id = normalize_subject_id(
                directory.name.removesuffix(TRANSFERRED_SUFFIX)
            )
            if subject_id in transferred:
                raise ValueError(f"Duplicate transferred subject directory: {subject_id}")
            transferred.append(subject_id)
        else:
            subject_id = normalize_subject_id(directory.name)
            if subject_id in active:
                raise ValueError(f"Duplicate active subject directory: {subject_id}")
            active[subject_id] = directory
    duplicate_states = tuple(sorted(set(active).intersection(transferred)))
    if duplicate_states:
        raise ValueError(
            "Subjects have both active and transferred directories; explicitly "
            "resolve the conflicting state before continuing: "
            + ", ".join(duplicate_states)
        )
    return active, tuple(transferred)


def _plan_subject(
    *,
    batch_id: str,
    subject_id: str,
    source_dir: Path,
    destination_root: Path,
    decision: FinalDecision,
) -> SubjectRegistrationDistribution:
    if decision.evaluation is None or decision.run_path is None:
        raise ValueError("final decision lacks its evaluated run")
    evaluation = decision.evaluation
    runs = _manifested_runs(source_dir)
    selected_run = (
        decision.run_path if decision.kind is FinalDecisionKind.SELECTED else None
    )
    if selected_run is not None:
        selected_dir = runs.get(selected_run)
        if selected_dir is None:
            raise FileNotFoundError(f"Selected run is missing: {selected_run}")
        _validate_successful_manifest(selected_dir)

    canonical = destination_root / "registration"
    archive_root = destination_root / "registration_runs"
    metadata = _decision_metadata(batch_id, decision)
    legacy_archive = _plan_legacy_archive(canonical, archive_root, metadata)
    operations: list[RegistrationCopyOperation] = []
    archived: list[str] = []
    seen: set[Path] = set()
    for run_path, run_dir in runs.items():
        if selected_run is not None and run_path == selected_run:
            run_destination = canonical
        else:
            run_destination = archive_root / Path(run_path).relative_to(
                "registration_runs"
            )
            archived.append(run_path)
        for source in _files(run_dir):
            destination = run_destination / source.relative_to(run_dir)
            if destination in seen:
                raise ValueError(f"Multiple files map to destination: {destination}")
            seen.add(destination)
            operations.append(
                plan_verified_copy(
                    source,
                    destination,
                    assume_destination_absent=(
                        legacy_archive is not None and run_path == selected_run
                    ),
                )
            )
    metadata_path = (
        canonical / SELECTION_METADATA_FILENAME
        if selected_run is not None
        else archive_root / EXCLUSION_METADATA_FILENAME
    )
    _validate_metadata_destination(metadata_path, metadata)
    return SubjectRegistrationDistribution(
        subject_id=subject_id,
        source_subject_dir=source_dir,
        destination_root=destination_root,
        decision=decision.kind.value,
        decision_run_path=decision.run_path,
        selected_run_path=selected_run,
        selected_destination=canonical if selected_run is not None else None,
        archived_run_paths=tuple(sorted(archived)),
        legacy_canonical_archive=legacy_archive,
        decision_metadata=metadata,
        decision_metadata_path=metadata_path,
        operations=tuple(operations),
    )


def _manifested_runs(subject_dir: Path) -> dict[str, Path]:
    runs_dir = subject_dir / "registration_runs"
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Registration runs directory is missing: {runs_dir}")
    runs: dict[str, Path] = {}
    for manifest_path in sorted(runs_dir.rglob("registration_result.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Invalid registration manifest: {manifest_path}") from exc
        if not isinstance(manifest.get("success"), bool):
            raise ValueError(
                f"Registration manifest lacks a Boolean success field: {manifest_path}"
            )
        run_dir = manifest_path.parent
        runs[run_dir.relative_to(subject_dir).as_posix()] = run_dir
    if not runs:
        raise FileNotFoundError(f"No registration manifests found: {runs_dir}")
    orphaned = [
        path
        for path in _files(runs_dir)
        if not any(run_dir == path.parent or run_dir in path.parents for run_dir in runs.values())
    ]
    if orphaned:
        preview = ", ".join(str(path) for path in orphaned[:3])
        raise ValueError(
            "Registration files exist outside a manifested run and cannot be "
            f"distributed safely: {preview}"
        )
    return runs


def _validate_successful_manifest(run_dir: Path) -> None:
    manifest = json.loads(
        (run_dir / "registration_result.json").read_text(encoding="utf-8")
    )
    if not bool(manifest.get("success")):
        raise ValueError(f"Selected run did not complete successfully: {run_dir}")


def _files(directory: Path) -> list[Path]:
    result: list[Path] = []
    for root, _, filenames in os.walk(directory):
        result.extend(Path(root) / filename for filename in filenames)
    return sorted(result)


def _decision_metadata(batch_id: str, decision: FinalDecision) -> dict[str, Any]:
    assert decision.evaluation is not None
    evaluation = decision.evaluation
    return {
        "schema_version": 1,
        "subject_id": decision.subject_id,
        "batch_id": batch_id,
        "decision": decision.kind.value,
        "decision_run_path": decision.run_path,
        "selected_run_path": (
            decision.run_path
            if decision.kind is FinalDecisionKind.SELECTED
            else None
        ),
        "preset_name": _json_value(evaluation.preset_name),
        "result": _json_value(evaluation.result),
        "comments": _json_value(evaluation.comments),
        "evaluator": _json_value(evaluation.evaluator),
        "evaluation_date": _json_value(evaluation.evaluation_date),
    }


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    try:
        if value != value:
            return None
    except (TypeError, ValueError):
        pass
    return value


def _plan_legacy_archive(
    canonical: Path,
    archive_root: Path,
    metadata: Mapping[str, Any],
) -> tuple[Path, Path] | None:
    progress = canonical / IN_PROGRESS_METADATA_FILENAME
    selection = canonical / SELECTION_METADATA_FILENAME
    if progress.is_file() and progress.read_text(encoding="utf-8") != _metadata_text(
        metadata
    ):
        raise FileExistsError(f"Distribution marker differs: {progress}")
    if canonical.is_dir() and not selection.is_file() and not progress.is_file():
        legacy_destination = archive_root / "legacy_pre_batch"
        if not legacy_destination.exists():
            return canonical, legacy_destination
    elif canonical.exists() and not canonical.is_dir():
        raise FileExistsError(
            f"Canonical registration destination is not a directory: {canonical}"
        )
    return None


def _metadata_text(metadata: Mapping[str, Any]) -> str:
    return json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n"


def _validate_metadata_destination(
    path: Path, metadata: Mapping[str, Any]
) -> None:
    if not path.exists():
        return
    if not path.is_file() or path.read_text(encoding="utf-8") != _metadata_text(
        metadata
    ):
        raise FileExistsError(
            f"Existing decision metadata differs and will not be overwritten: {path}"
        )


def _apply_legacy_archive(subject: SubjectRegistrationDistribution) -> None:
    if subject.legacy_canonical_archive is None:
        return
    source, destination = subject.legacy_canonical_archive
    if destination.exists():
        raise FileExistsError(f"Legacy archive appeared after preflight: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    print(f"    archived legacy registration: {destination}")


def _ensure_progress_marker(
    subject: SubjectRegistrationDistribution, metadata_text: str
) -> Path | None:
    if subject.selected_destination is None:
        return None
    subject.selected_destination.mkdir(parents=True, exist_ok=True)
    progress = subject.selected_destination / IN_PROGRESS_METADATA_FILENAME
    _write_new_or_validate_identical(progress, metadata_text)
    return progress


def _write_new_or_validate_identical(path: Path, content: str) -> None:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Existing metadata differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


__all__ = [
    "EXCLUSION_METADATA_FILENAME",
    "IN_PROGRESS_METADATA_FILENAME",
    "RegistrationCopyOperation",
    "RegistrationDistributionPlan",
    "SELECTION_METADATA_FILENAME",
    "SubjectFinalizer",
    "SubjectRegistrationDistribution",
    "TRANSFERRED_SUFFIX",
    "apply_registration_distribution_plan",
    "build_registration_distribution_plan",
    "copy_and_verify",
    "files_match",
    "plan_verified_copy",
    "summarize_registration_distribution_plan",
]
