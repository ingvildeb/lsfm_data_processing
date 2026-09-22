from __future__ import annotations

import json
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np

from lsfm_data_processing.registration_and_transforms.project_workflow.decisions import (
    resolve_final_decisions,
    resolve_rescue_requests,
)
from lsfm_data_processing.registration_and_transforms.project_workflow.masking import (
    MaskingSettings,
    apply_masking_plans,
    build_masking_plans,
)
from lsfm_data_processing.registration_and_transforms.project_workflow.models import (
    RescueRequest,
    RescueRequestStatus,
    RescueStrategy,
    RescueStrategyKind,
    SourceRunRequirement,
)


def _strategies() -> tuple[RescueStrategy, ...]:
    return (
        RescueStrategy("lower_gradient", "Lower gradient step"),
        RescueStrategy(
            "masking",
            "Masking",
            SourceRunRequirement.REQUIRED,
            RescueStrategyKind.MASKING,
        ),
    )


def _request_plan(*requests: RescueRequest):
    subjects = tuple(sorted({request.subject_id for request in requests}))
    return resolve_rescue_requests(
        requests,
        strategies=_strategies(),
        decisions=resolve_final_decisions((), subject_ids=subjects),
        known_runs={subject: {"registration_runs/baseline"} for subject in subjects},
    )


def _write_nifti(
    path: Path, data: np.ndarray, *, affine: np.ndarray | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.diag((0.02, 0.02, 0.02, 1.0))
    nib.save(nib.Nifti1Image(data, affine), path)


def _write_registration_inputs(subject_dir: Path) -> Path:
    run_dir = subject_dir / "registration_runs" / "baseline"
    effective_shape = (20, 25, 30)
    original_shape = (20, 30, 25)
    lsp_affine = np.array(
        (
            (0.02, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.02, 0.0),
            (0.0, -0.02, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    labels = np.zeros(effective_shape, dtype=np.uint16)
    labels[3:17, 4:21, 5:25] = 1
    _write_nifti(
        run_dir / "labels_WarpedSegmentation.nii.gz", labels, affine=lsp_affine
    )
    _write_nifti(
        run_dir / "fixed_normalized_for_registration.nii.gz",
        np.arange(np.prod(effective_shape), dtype=np.float32).reshape(effective_shape),
        affine=lsp_affine,
    )
    original = np.arange(np.prod(original_shape), dtype=np.uint16).reshape(original_shape)
    _write_nifti(subject_dir / "ch1_native_20um.nii.gz", original)
    (run_dir / "registration_parameters.yaml").write_text(
        "name: baseline\n", encoding="utf-8"
    )
    space_las = {
        "space_name": "subject",
        "orientation": "las",
        "axis_labels": ["x", "y", "z"],
        "units": "voxel",
        "resolution_um": [20.0, 20.0, 20.0],
        "shape": list(original_shape),
    }
    space_lsp = {
        "space_name": "subject",
        "orientation": "lsp",
        "axis_labels": ["x", "z", "y"],
        "units": "voxel",
        "resolution_um": [20.0, 20.0, 20.0],
        "shape": list(effective_shape),
    }
    manifest = {
        "schema_version": 1,
        "success": True,
        "preset_name": "baseline",
        "parameters_snapshot": "registration_parameters.yaml",
        "orientation_alignment": "fixed_to_moving",
        "fixed_image": {
            "image_id": "subject",
            "image": str(subject_dir / "ch1_native_20um.nii.gz"),
            "space": space_las,
            "normalized_image": "fixed_normalized_for_registration.nii.gz",
            "segmentations": {},
        },
        "moving_image": {
            "image_id": "template",
            "image": "template.nii.gz",
            "space": space_lsp,
            "normalized_image": None,
            "segmentations": {},
        },
        "effective_fixed_space": space_lsp,
        "effective_moving_space": space_lsp,
        "forward_transforms": [],
        "inverse_transforms": [],
        "transformed_segmentations": {
            "labels": "labels_WarpedSegmentation.nii.gz"
        },
    }
    (run_dir / "registration_result.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return run_dir


def test_masking_waits_only_for_same_subject_rescues(tmp_path: Path) -> None:
    _write_registration_inputs(tmp_path / "A")
    plan = _request_plan(
        RescueRequest("A", "Masking", source_run="registration_runs/baseline"),
        RescueRequest("B", "Masking", source_run="registration_runs/baseline"),
        RescueRequest("B", "Lower gradient step"),
    )

    masking = build_masking_plans(
        plan,
        subjects_root=tmp_path,
        original_fixed_filename="ch1_native_20um.nii.gz",
    )

    by_subject = {item.subject_id: item for item in masking}
    assert by_subject["A"].action == "create_review"
    assert by_subject["B"].action == "blocked"
    assert by_subject["B"].state is RescueRequestStatus.BLOCKED_PENDING_RESCUES


def test_completed_mask_creates_separate_native_input_and_preserves_original(
    tmp_path: Path,
) -> None:
    subject_dir = tmp_path / "A"
    _write_registration_inputs(subject_dir)
    plan = _request_plan(
        RescueRequest("A", "Masking", source_run="registration_runs/baseline")
    )
    settings = MaskingSettings(review_resolution_um=50.0)
    original_path = subject_dir / "ch1_native_20um.nii.gz"
    original_bytes = original_path.read_bytes()

    first = build_masking_plans(
        plan,
        subjects_root=tmp_path,
        original_fixed_filename=original_path.name,
        settings=settings,
    )
    apply_masking_plans(first, settings=settings)
    awaiting = build_masking_plans(
        plan,
        subjects_root=tmp_path,
        original_fixed_filename=original_path.name,
        settings=settings,
    )
    assert awaiting[0].action == "await"
    shutil.copyfile(awaiting[0].draft_mask, awaiting[0].completed_mask)

    completed = build_masking_plans(
        plan,
        subjects_root=tmp_path,
        original_fixed_filename=original_path.name,
        settings=settings,
    )
    assert completed[0].action == "apply_completed"
    apply_masking_plans(completed, settings=settings)
    ready = build_masking_plans(
        plan,
        subjects_root=tmp_path,
        original_fixed_filename=original_path.name,
        settings=settings,
    )

    assert ready[0].state is RescueRequestStatus.MASK_READY
    assert ready[0].action == "ready"
    assert ready[0].native_mask.is_file()
    assert ready[0].masked_fixed_image.is_file()
    assert nib.load(ready[0].masked_fixed_image).shape == nib.load(original_path).shape
    assert original_path.read_bytes() == original_bytes
