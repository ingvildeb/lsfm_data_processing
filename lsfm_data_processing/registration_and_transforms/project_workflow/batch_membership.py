"""Canonical registration-batch workbook and membership manifest handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PureWindowsPath
import re
import shutil
from typing import Mapping
from io import BytesIO

from openpyxl import load_workbook


BATCH_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BatchMember:
    subject_id: str
    recorded_age: str
    template_age: str
    session_path: str


@dataclass(frozen=True)
class BatchMembershipPlan:
    batch_id: str
    canonical_workbook: Path
    manifest_path: Path
    members: tuple[BatchMember, ...]
    canonical_source: Path | None
    canonical_state: str
    manifest_content: bytes
    manifest_state: str
    observed_workbook_sha256: str
    source_workbook_sha256: str | None
    observed_manifest_sha256: str | None
    legacy_files_to_retire: tuple[Path, ...]
    canonical_content: bytes

    @property
    def required_confirmation(self) -> str:
        return f"PREPARE REGISTRATION {self.batch_id}"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_subject_id(value: object) -> str:
    subject_id = str(value).strip().upper()
    if not subject_id:
        raise ValueError("subject ID is empty")
    return subject_id


def _normalize_age(value: object) -> str:
    age = str(value).strip().upper()
    if not re.fullmatch(r"P\d+", age):
        raise ValueError(f"age must use P<number>, got {value!r}")
    return age


def _session_contains_subject(path: str, subject_id: str) -> bool:
    name = PureWindowsPath(path).name
    pattern = rf"(?<![A-Za-z0-9]){re.escape(subject_id)}(?![A-Za-z0-9])"
    return re.search(pattern, name, flags=re.IGNORECASE) is not None


def _worksheet_rows(path: Path) -> tuple[dict[str, int], list[tuple[object, ...]]]:
    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        raise ValueError(f"Could not read batch workbook: {path}") from exc
    try:
        sheet = workbook[workbook.sheetnames[0]]
        rows = list(sheet.iter_rows(values_only=True))
    finally:
        workbook.close()
    if not rows:
        raise ValueError(f"Batch workbook is empty: {path}")
    headers: dict[str, int] = {}
    for index, value in enumerate(rows[0]):
        name = str(value).strip().casefold() if value is not None else ""
        if name:
            if name in headers:
                raise ValueError(f"Duplicate batch workbook column {name!r}: {path}")
            headers[name] = index
    return headers, rows[1:]


def _read_members(
    path: Path,
    *,
    age_to_template: Mapping[str, str],
    available_templates: set[str],
    legacy_record: bool,
) -> tuple[BatchMember, ...]:
    headers, rows = _worksheet_rows(path)
    if legacy_record:
        required = ("subject id", "recorded age", "template age", "session path")
    else:
        required = ("id", "age", "path")
    missing = [column for column in required if column not in headers]
    if missing:
        raise ValueError(f"Batch workbook is missing columns {missing}: {path}")

    members: list[BatchMember] = []
    errors: list[str] = []
    for excel_row, row in enumerate(rows, start=2):
        if all(value is None or not str(value).strip() for value in row):
            continue
        try:
            subject_id = _normalize_subject_id(
                row[headers["subject id" if legacy_record else "id"]]
            )
            recorded_age = _normalize_age(
                row[headers["recorded age" if legacy_record else "age"]]
            )
            if legacy_record:
                template_age = _normalize_age(row[headers["template age"]])
            else:
                template_age = age_to_template.get(recorded_age, "")
                if not template_age:
                    raise ValueError(
                        f"age {recorded_age} has no configured template mapping"
                    )
            session_path = str(
                row[headers["session path" if legacy_record else "path"]]
            ).strip()
            if not session_path:
                raise ValueError("session path is empty")
            if not _session_contains_subject(session_path, subject_id):
                raise ValueError(
                    f"subject ID {subject_id} is absent from session folder "
                    f"{PureWindowsPath(session_path).name!r}"
                )
            if template_age not in available_templates:
                raise ValueError(f"template age {template_age} is not configured")
            members.append(
                BatchMember(
                    subject_id,
                    recorded_age,
                    template_age,
                    session_path,
                )
            )
        except (IndexError, TypeError, ValueError) as exc:
            errors.append(f"row {excel_row}: {exc}")
    duplicates = sorted(
        subject_id
        for subject_id in {member.subject_id for member in members}
        if sum(item.subject_id == subject_id for item in members) > 1
    )
    if duplicates:
        errors.append(f"duplicate subject IDs: {duplicates}")
    if not members:
        errors.append("workbook contains no subjects")
    if errors:
        raise ValueError(
            f"Batch workbook validation failed: {path}\n" + "\n".join(errors)
        )
    return tuple(members)


def _normalized_members(members: tuple[BatchMember, ...]) -> list[dict[str, str]]:
    return [
        {
            "subject_id": member.subject_id,
            "recorded_age": member.recorded_age,
            "template_age": member.template_age,
            "session_path": member.session_path,
        }
        for member in sorted(members, key=lambda item: item.subject_id)
    ]


def _membership_hash(members: tuple[BatchMember, ...]) -> str:
    content = json.dumps(
        _normalized_members(members), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(content).hexdigest()


def _membership_difference(
    expected: tuple[BatchMember, ...], observed: tuple[BatchMember, ...]
) -> str:
    expected_by_id = {item.subject_id: item for item in expected}
    observed_by_id = {item.subject_id: item for item in observed}
    added = sorted(observed_by_id.keys() - expected_by_id.keys())
    removed = sorted(expected_by_id.keys() - observed_by_id.keys())
    changed = sorted(
        subject_id
        for subject_id in expected_by_id.keys() & observed_by_id.keys()
        if expected_by_id[subject_id] != observed_by_id[subject_id]
    )
    parts = []
    if added:
        parts.append(f"added={added}")
    if removed:
        parts.append(f"removed={removed}")
    if changed:
        parts.append(f"changed={changed}")
    return "; ".join(parts) or "unknown membership difference"


def _members_from_manifest(
    path: Path, *, expected_batch_id: str
) -> tuple[BatchMember, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid batch manifest: {path}") from exc
    if payload.get("schema_version") != BATCH_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported batch manifest schema "
            f"{payload.get('schema_version')!r}: {path}"
        )
    if payload.get("batch_id") != expected_batch_id:
        raise ValueError(
            f"Batch manifest identifies {payload.get('batch_id')!r}, expected "
            f"{expected_batch_id!r}: {path}"
        )
    try:
        members = tuple(
            BatchMember(
                str(item["subject_id"]),
                str(item["recorded_age"]),
                str(item["template_age"]),
                str(item["session_path"]),
            )
            for item in payload["subjects"]
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid batch manifest subjects: {path}") from exc
    if payload.get("normalized_membership_sha256") != _membership_hash(members):
        raise ValueError(f"Batch manifest membership hash is invalid: {path}")
    return members


def read_batch_manifest_members(
    path: Path, *, expected_batch_id: str
) -> tuple[BatchMember, ...]:
    """Read and validate canonical frozen membership for cross-batch audits."""

    return _members_from_manifest(
        Path(path), expected_batch_id=expected_batch_id
    )


def _manifest_content(
    *,
    batch_id: str,
    members: tuple[BatchMember, ...],
    workbook_sha256: str,
    source_workbook: Path,
    migration_mode: str,
) -> bytes:
    template_counts: dict[str, int] = {}
    for member in members:
        template_counts[member.template_age] = template_counts.get(member.template_age, 0) + 1
    payload = {
        "schema_version": BATCH_MANIFEST_SCHEMA_VERSION,
        "batch_id": batch_id,
        "normalized_membership_sha256": _membership_hash(members),
        "workbook_sha256_at_definition": workbook_sha256,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_workbook": str(source_workbook),
        "migration_mode": migration_mode,
        "subject_count": len(members),
        "template_counts": dict(sorted(template_counts.items())),
        "subjects": _normalized_members(members),
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _canonical_workbook_content(members: tuple[BatchMember, ...]) -> bytes:
    """Render the minimal canonical membership workbook for legacy migration."""
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Batch subjects"
    sheet.append(["ID", "age", "path"])
    for member in members:
        sheet.append([member.subject_id, member.recorded_age, member.session_path])
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


def plan_batch_membership(
    *,
    batch_id: str,
    batch_dir: Path,
    age_to_template: Mapping[str, str],
    available_templates: set[str],
) -> BatchMembershipPlan:
    """Plan a new canonical definition or a verified legacy migration."""

    batch_dir = Path(batch_dir)
    canonical = batch_dir / f"{batch_id}.xlsx"
    manifest = batch_dir / "batch_manifest.json"
    legacy_input = batch_dir / f"{batch_id}_input.xlsx"
    legacy_record = batch_dir / f"{batch_id}_subjects.xlsx"
    legacy_existing = tuple(
        path for path in (legacy_input, legacy_record) if path.exists()
    )

    canonical_source: Path | None = None
    canonical_state = "already_present"
    canonical_content: bytes
    source_workbook_sha256: str | None = None
    if canonical.is_file():
        members = _read_members(
            canonical,
            age_to_template=age_to_template,
            available_templates=available_templates,
            legacy_record=False,
        )
        workbook_sha = _file_sha256(canonical)
        canonical_content = canonical.read_bytes()
        source_workbook_sha256 = workbook_sha
    elif legacy_input.is_file() and legacy_record.is_file():
        members = _read_members(
            legacy_input,
            age_to_template=age_to_template,
            available_templates=available_templates,
            legacy_record=False,
        )
        frozen = _read_members(
            legacy_record,
            age_to_template=age_to_template,
            available_templates=available_templates,
            legacy_record=True,
        )
        if _membership_hash(members) != _membership_hash(frozen):
            raise ValueError(
                "Legacy batch input and frozen record disagree: "
                + _membership_difference(frozen, members)
            )
        canonical_source = legacy_input
        canonical_state = "copy_legacy_input"
        workbook_sha = _file_sha256(legacy_input)
        canonical_content = legacy_input.read_bytes()
        source_workbook_sha256 = workbook_sha
    elif legacy_record.is_file():
        members = _read_members(
            legacy_record,
            age_to_template=age_to_template,
            available_templates=available_templates,
            legacy_record=True,
        )
        canonical_source = legacy_record
        canonical_state = "convert_legacy_record"
        canonical_content = _canonical_workbook_content(members)
        workbook_sha = sha256(canonical_content).hexdigest()
        source_workbook_sha256 = _file_sha256(legacy_record)
    elif legacy_existing:
        raise ValueError(
            "Incomplete legacy batch definition; expected both "
            f"{legacy_input.name} and {legacy_record.name}"
        )
    else:
        raise FileNotFoundError(
            f"Canonical batch workbook is missing: {canonical}. Create this file "
            "with ID, age, and path columns."
        )

    if canonical.is_file() and legacy_record.is_file():
        frozen = _read_members(
            legacy_record,
            age_to_template=age_to_template,
            available_templates=available_templates,
            legacy_record=True,
        )
        if _membership_hash(members) != _membership_hash(frozen):
            raise ValueError(
                "Canonical workbook and legacy frozen record disagree: "
                + _membership_difference(frozen, members)
            )

    manifest_content = _manifest_content(
        batch_id=batch_id,
        members=members,
        workbook_sha256=workbook_sha,
        source_workbook=canonical_source or canonical,
        migration_mode=canonical_state,
    )
    observed_manifest_sha: str | None = None
    if manifest.exists():
        if not manifest.is_file():
            raise FileExistsError(f"Batch manifest is not a file: {manifest}")
        frozen = _members_from_manifest(
            manifest, expected_batch_id=batch_id
        )
        if _membership_hash(members) != _membership_hash(frozen):
            raise ValueError(
                "Canonical workbook differs from immutable batch manifest: "
                + _membership_difference(frozen, members)
                + ". Create a new batch instead of changing membership."
            )
        manifest_state = "already_present"
        observed_manifest_sha = _file_sha256(manifest)
    else:
        subjects_dir = batch_dir / "subjects"
        established = (
            (batch_dir / "configs" / "baseline.toml").exists()
            or (batch_dir / "registration_evaluation.xlsx").exists()
            or (subjects_dir.is_dir() and any(subjects_dir.iterdir()))
        )
        if established and not legacy_record.is_file():
            raise FileNotFoundError(
                "Established batch has no manifest or legacy frozen record; "
                "membership cannot be adopted safely."
            )
        manifest_state = "write"

    return BatchMembershipPlan(
        batch_id=batch_id,
        canonical_workbook=canonical,
        manifest_path=manifest,
        members=members,
        canonical_source=canonical_source,
        canonical_state=canonical_state,
        manifest_content=manifest_content,
        manifest_state=manifest_state,
        observed_workbook_sha256=workbook_sha,
        source_workbook_sha256=source_workbook_sha256,
        observed_manifest_sha256=observed_manifest_sha,
        legacy_files_to_retire=legacy_existing,
        canonical_content=canonical_content,
    )


def summarize_batch_membership_plan(plan: BatchMembershipPlan) -> str:
    return "\n".join(
        [
            f"Batch membership plan: {plan.batch_id}",
            f"Subjects: {len(plan.members)}",
            f"Canonical workbook: {plan.canonical_state}: "
            f"{plan.canonical_workbook}",
            f"Batch manifest: {plan.manifest_state}: {plan.manifest_path}",
            f"Legacy files to retire: {len(plan.legacy_files_to_retire)}",
        ]
    )


def apply_batch_membership_plan(
    plan: BatchMembershipPlan, *, confirmation: str
) -> None:
    """Apply migration/manifest writes, retiring legacy files only at the end."""

    if confirmation != plan.required_confirmation:
        raise ValueError(f"Required confirmation: {plan.required_confirmation!r}")
    if plan.canonical_state in {"copy_legacy_input", "convert_legacy_record"}:
        if plan.canonical_source is None:
            raise ValueError("Legacy migration has no canonical source workbook.")
        if plan.source_workbook_sha256 is None or (
            _file_sha256(plan.canonical_source) != plan.source_workbook_sha256
        ):
            raise RuntimeError("Legacy input workbook changed after preflight.")
        plan.canonical_workbook.parent.mkdir(parents=True, exist_ok=True)
        temporary = plan.canonical_workbook.with_name(
            f".{plan.canonical_workbook.name}.tmp"
        )
        try:
            if plan.canonical_state == "copy_legacy_input":
                shutil.copy2(plan.canonical_source, temporary)
            else:
                temporary.write_bytes(plan.canonical_content)
            if _file_sha256(temporary) != plan.observed_workbook_sha256:
                raise IOError("Canonical workbook copy failed verification.")
            temporary.replace(plan.canonical_workbook)
        finally:
            temporary.unlink(missing_ok=True)
    elif _file_sha256(plan.canonical_workbook) != plan.observed_workbook_sha256:
        raise RuntimeError("Canonical batch workbook changed after preflight.")

    if plan.manifest_state == "write":
        if plan.manifest_path.exists():
            raise FileExistsError(
                f"Batch manifest appeared after preflight: {plan.manifest_path}"
            )
        temporary = plan.manifest_path.with_name(f".{plan.manifest_path.name}.tmp")
        try:
            temporary.write_bytes(plan.manifest_content)
            temporary.replace(plan.manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
    elif (
        plan.observed_manifest_sha256 is None
        or _file_sha256(plan.manifest_path) != plan.observed_manifest_sha256
    ):
        raise RuntimeError("Batch manifest changed after preflight.")

    verified = _members_from_manifest(
        plan.manifest_path, expected_batch_id=plan.batch_id
    )
    if _membership_hash(verified) != _membership_hash(plan.members):
        raise RuntimeError("Written batch manifest failed membership verification.")
    for legacy in plan.legacy_files_to_retire:
        legacy.unlink(missing_ok=True)
