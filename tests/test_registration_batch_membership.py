from __future__ import annotations

import json
from pathlib import Path

from openpyxl import Workbook
import pytest

from lsfm_data_processing.registration_and_transforms.project_workflow.batch_membership import (
    apply_batch_membership_plan,
    plan_batch_membership,
)


AGE_MAP = {"P7": "P8", "P12": "P12"}
TEMPLATES = {"P8", "P12"}


def _write_input(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["ID", "age", "path"])
    for row in rows:
        sheet.append(row)
    workbook.save(path)


def _write_legacy_record(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Batch subjects"
    sheet.append(["Subject ID", "Recorded age", "Template age", "Session path"])
    for row in rows:
        sheet.append(row)
    workbook.save(path)


def _plan(batch: Path):
    return plan_batch_membership(
        batch_id="batch001",
        batch_dir=batch,
        age_to_template=AGE_MAP,
        available_templates=TEMPLATES,
    )


def test_new_canonical_workbook_defines_immutable_manifest(tmp_path: Path) -> None:
    batch = tmp_path / "batch001"
    workbook = batch / "batch001.xlsx"
    rows = [
        ("A001", "P7", str(tmp_path / "session_A001")),
        ("A002", "P12", str(tmp_path / "session_A002")),
    ]
    _write_input(workbook, rows)
    plan = _plan(batch)
    assert plan.canonical_state == "already_present"
    assert plan.manifest_state == "write"

    apply_batch_membership_plan(plan, confirmation=plan.required_confirmation)
    payload = json.loads((batch / "batch_manifest.json").read_text())
    assert [item["subject_id"] for item in payload["subjects"]] == ["A001", "A002"]

    _write_input(workbook, list(reversed(rows)))
    resumed = _plan(batch)
    assert resumed.manifest_state == "already_present"


def test_membership_change_requires_a_new_batch(tmp_path: Path) -> None:
    batch = tmp_path / "batch001"
    workbook = batch / "batch001.xlsx"
    _write_input(
        workbook,
        [("A001", "P7", str(tmp_path / "session_A001"))],
    )
    first = _plan(batch)
    apply_batch_membership_plan(first, confirmation=first.required_confirmation)
    _write_input(
        workbook,
        [
            ("A001", "P7", str(tmp_path / "session_A001")),
            ("A002", "P12", str(tmp_path / "session_A002")),
        ],
    )

    with pytest.raises(ValueError, match=r"added=\['A002'\].*new batch"):
        _plan(batch)


def test_legacy_pair_migrates_then_retires_only_after_verification(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "batch001"
    session = str(tmp_path / "session_A001")
    legacy_input = batch / "batch001_input.xlsx"
    legacy_record = batch / "batch001_subjects.xlsx"
    _write_input(legacy_input, [("A001", "P7", session)])
    _write_legacy_record(legacy_record, [("A001", "P7", "P8", session)])

    plan = _plan(batch)
    assert plan.canonical_state == "copy_legacy_input"
    assert set(plan.legacy_files_to_retire) == {legacy_input, legacy_record}
    assert legacy_input.exists() and legacy_record.exists()

    apply_batch_membership_plan(plan, confirmation=plan.required_confirmation)
    assert (batch / "batch001.xlsx").is_file()
    assert (batch / "batch_manifest.json").is_file()
    assert not legacy_input.exists()
    assert not legacy_record.exists()


def test_legacy_record_alone_is_planned_for_canonical_conversion(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "batch001"
    legacy_record = batch / "batch001_subjects.xlsx"
    _write_legacy_record(
        legacy_record,
        [("A001", "P7", "P8", str(tmp_path / "session_A001"))],
    )

    plan = _plan(batch)

    assert plan.canonical_state == "convert_legacy_record"
    assert plan.canonical_source == legacy_record
    assert plan.canonical_content


def test_disagreeing_legacy_files_stop_migration(tmp_path: Path) -> None:
    batch = tmp_path / "batch001"
    _write_input(
        batch / "batch001_input.xlsx",
        [("A001", "P7", str(tmp_path / "session_A001"))],
    )
    _write_legacy_record(
        batch / "batch001_subjects.xlsx",
        [("A002", "P12", "P12", str(tmp_path / "session_A002"))],
    )

    with pytest.raises(ValueError, match="Legacy batch input and frozen record disagree"):
        _plan(batch)
