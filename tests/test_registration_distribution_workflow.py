from __future__ import annotations

import json
from pathlib import Path

import pytest

from lsfm_data_processing.registration_and_transforms.project_workflow.distribution import (
    IN_PROGRESS_METADATA_FILENAME,
    apply_registration_distribution_plan,
    build_registration_distribution_plan,
)
from lsfm_data_processing.registration_and_transforms.project_workflow.decisions import (
    resolve_final_decisions,
)
from lsfm_data_processing.registration_and_transforms.project_workflow.models import RunEvaluation


def _run(
    subject_dir: Path,
    relative: str,
    *,
    success: bool,
    payload: bytes,
) -> None:
    run_dir = subject_dir / relative
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "registration_result.json").write_text(
        json.dumps({"success": success, "preset_name": relative}),
        encoding="utf-8",
    )
    (run_dir / "output.nii.gz").write_bytes(payload)


def _decision(
    subject_id: str,
    run_path: str,
    *,
    excluded: bool = False,
):
    evaluation = RunEvaluation(
        subject_id,
        run_path,
        preset_name="preset",
        manifest_status="failed" if excluded else "success",
        result="exclude" if excluded else "excellent",
        selected=True,
        evaluator="Reviewer",
    )
    return resolve_final_decisions((evaluation,))[subject_id]


def test_distribution_installs_selection_and_archives_every_other_run(
    tmp_path: Path,
) -> None:
    subjects = tmp_path / "subjects"
    selected_source = subjects / "A"
    excluded_source = subjects / "B"
    _run(
        selected_source,
        "registration_runs/baseline",
        success=True,
        payload=b"selected",
    )
    _run(
        selected_source,
        "registration_runs/rescue/gs0p01",
        success=True,
        payload=b"alternative",
    )
    _run(
        excluded_source,
        "registration_runs/baseline",
        success=False,
        payload=b"failed-but-archived",
    )
    destination_parent = tmp_path / "sessions"
    destination_parent.mkdir()
    destinations = {
        "A": destination_parent / "A" / "_analysis",
        "B": destination_parent / "B" / "_analysis",
    }
    for root in destinations.values():
        root.parent.mkdir()
    plan = build_registration_distribution_plan(
        batch_id="batch001",
        subjects_root=subjects,
        destination_roots=destinations,
        decisions={
            "A": _decision("A", "registration_runs/baseline"),
            "B": _decision("B", "registration_runs/baseline", excluded=True),
        },
    )

    apply_registration_distribution_plan(
        plan, confirmation=plan.required_confirmation
    )

    assert (
        destinations["A"] / "registration" / "output.nii.gz"
    ).read_bytes() == b"selected"
    assert (
        destinations["A"]
        / "registration_runs"
        / "rescue"
        / "gs0p01"
        / "output.nii.gz"
    ).read_bytes() == b"alternative"
    assert not (destinations["B"] / "registration").exists()
    assert (
        destinations["B"] / "registration_runs" / "baseline" / "output.nii.gz"
    ).read_bytes() == b"failed-but-archived"
    assert (subjects / "A_transferred").is_dir()
    assert (subjects / "B_transferred").is_dir()


def test_failed_finalizer_leaves_subject_resumable(tmp_path: Path) -> None:
    subjects = tmp_path / "subjects"
    source = subjects / "A"
    _run(
        source,
        "registration_runs/baseline",
        success=True,
        payload=b"selected",
    )
    session = tmp_path / "session"
    session.mkdir()
    destination = session / "_analysis"
    decision = _decision("A", "registration_runs/baseline")
    first = build_registration_distribution_plan(
        batch_id="batch001",
        subjects_root=subjects,
        destination_roots={"A": destination},
        decisions={"A": decision},
    )

    def fail_after_copy(_subject) -> None:
        raise RuntimeError("postprocessing failed")

    with pytest.raises(RuntimeError, match="postprocessing failed"):
        apply_registration_distribution_plan(
            first,
            confirmation=first.required_confirmation,
            finalize_subject=fail_after_copy,
        )

    progress = destination / "registration" / IN_PROGRESS_METADATA_FILENAME
    assert progress.is_file()
    assert source.is_dir()
    resumed = build_registration_distribution_plan(
        batch_id="batch001",
        subjects_root=subjects,
        destination_roots={"A": destination},
        decisions={"A": decision},
    )
    assert all(item.state == "already_present" for item in resumed.operations)
    apply_registration_distribution_plan(
        resumed, confirmation=resumed.required_confirmation
    )
    assert not progress.exists()
    assert (subjects / "A_transferred").is_dir()


def test_differing_partial_destination_is_a_conflict(tmp_path: Path) -> None:
    subjects = tmp_path / "subjects"
    source = subjects / "A"
    _run(
        source,
        "registration_runs/baseline",
        success=True,
        payload=b"selected",
    )
    session = tmp_path / "session"
    session.mkdir()
    destination = session / "_analysis"
    decision = _decision("A", "registration_runs/baseline")
    initial = build_registration_distribution_plan(
        batch_id="batch001",
        subjects_root=subjects,
        destination_roots={"A": destination},
        decisions={"A": decision},
    )
    conflicting = destination / "registration" / "output.nii.gz"
    conflicting.parent.mkdir(parents=True)
    conflicting.write_bytes(b"different")
    progress = conflicting.parent / IN_PROGRESS_METADATA_FILENAME
    progress.write_text(
        json.dumps(initial.subjects[0].decision_metadata, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="exists but differs"):
        build_registration_distribution_plan(
            batch_id="batch001",
            subjects_root=subjects,
            destination_roots={"A": destination},
            decisions={"A": decision},
        )


def test_active_and_transferred_directories_are_a_conflict(
    tmp_path: Path,
) -> None:
    subjects = tmp_path / "subjects"
    _run(
        subjects / "A",
        "registration_runs/baseline",
        success=True,
        payload=b"recreated",
    )
    (subjects / "A_transferred").mkdir()
    session = tmp_path / "session"
    session.mkdir()

    with pytest.raises(ValueError, match="both active and transferred"):
        build_registration_distribution_plan(
            batch_id="batch001",
            subjects_root=subjects,
            destination_roots={"A": session / "_analysis"},
            decisions={"A": _decision("A", "registration_runs/baseline")},
        )


def test_unmanifested_run_files_block_distribution(tmp_path: Path) -> None:
    subjects = tmp_path / "subjects"
    source = subjects / "A"
    _run(
        source,
        "registration_runs/baseline",
        success=True,
        payload=b"selected",
    )
    incomplete = source / "registration_runs" / "rescue" / "incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "partial_output.nii.gz").write_bytes(b"partial")
    session = tmp_path / "session"
    session.mkdir()

    with pytest.raises(RuntimeError, match="outside a manifested run"):
        build_registration_distribution_plan(
            batch_id="batch001",
            subjects_root=subjects,
            destination_roots={"A": session / "_analysis"},
            decisions={"A": _decision("A", "registration_runs/baseline")},
        )
