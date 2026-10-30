from __future__ import annotations

import json
from pathlib import Path

import pytest

from lsfm_data_processing.registration_and_transforms.project_workflow.run_inventory import (
    discover_registration_run_inventory,
)


def _write_config(
    batch: Path,
    *,
    subject_id: str = "A",
    run_path: str = "registration_runs/baseline",
    preset: str = "planned_preset",
) -> None:
    configs = batch / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / f"{subject_id}_{Path(run_path).name}.toml").write_text(
        "\n".join(
            (
                "[run]",
                f'registration_presets = ["{preset}"]',
                f'output_subdir = "{run_path}"',
                "[batch.image_to_template]",
                f'{subject_id} = "template"',
            )
        ),
        encoding="utf-8",
    )


def _space(name: str) -> dict[str, object]:
    return {
        "space_name": name,
        "orientation": "lsp",
        "axis_labels": ["x", "z", "y"],
        "units": "voxel",
        "resolution_um": [20.0, 20.0, 20.0],
        "shape": [2, 3, 4],
    }


def _write_manifest(
    subject_dir: Path,
    *,
    run_path: str = "registration_runs/baseline",
    preset: str = "executed_preset",
    success: bool = True,
) -> Path:
    run_dir = subject_dir / run_path
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "success": success,
        "preset_name": preset,
        "parameters_snapshot": "registration_parameters.yaml",
        "orientation_alignment": "fixed_to_moving",
        "fixed_image": {
            "image_id": "subject",
            "image": "subject.nii.gz",
            "space": _space("subject"),
            "segmentations": {},
        },
        "moving_image": {
            "image_id": "template",
            "image": "template.nii.gz",
            "space": _space("template"),
            "segmentations": {},
        },
        "effective_fixed_space": _space("subject"),
        "effective_moving_space": _space("template"),
        "forward_transforms": [],
        "inverse_transforms": [],
        "transformed_segmentations": {},
    }
    path = run_dir / "registration_result.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_inventory_classifies_planned_and_missing_manifest_runs(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path, subject_id="A")
    _write_config(
        tmp_path,
        subject_id="B",
        run_path="registration_runs/rescue/gs0p01",
    )
    (tmp_path / "subjects" / "B" / "registration_runs/rescue/gs0p01").mkdir(
        parents=True
    )

    inventory = discover_registration_run_inventory(batch_dir=tmp_path)
    by_subject = {item.subject_id: item for item in inventory}

    assert by_subject["A"].manifest_status == "planned"
    assert by_subject["B"].manifest_status == "missing_manifest"


@pytest.mark.parametrize("success,status", ((True, "success"), (False, "failed")))
def test_manifest_is_authoritative_for_status_and_preset(
    tmp_path: Path, success: bool, status: str
) -> None:
    _write_config(tmp_path, preset="planned_preset")
    _write_manifest(
        tmp_path / "subjects" / "A",
        preset="executed_preset",
        success=success,
    )

    inventory = discover_registration_run_inventory(batch_dir=tmp_path)

    assert len(inventory) == 1
    assert inventory[0].manifest_status == status
    assert inventory[0].preset_name == "executed_preset"


def test_transferred_subject_runs_remain_discoverable(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "subjects" / "A_transferred")

    inventory = discover_registration_run_inventory(batch_dir=tmp_path)

    assert [(item.subject_id, item.manifest_status) for item in inventory] == [
        ("A", "success")
    ]


def test_active_and_transferred_subject_directories_stop_refresh(
    tmp_path: Path,
) -> None:
    _write_manifest(tmp_path / "subjects" / "A")
    _write_manifest(tmp_path / "subjects" / "A_transferred")

    with pytest.raises(ValueError, match="both active and transferred"):
        discover_registration_run_inventory(batch_dir=tmp_path)


@pytest.mark.parametrize(
    "invalid_content",
    (
        "not json",
        json.dumps({"schema_version": 1, "success": "yes"}),
        json.dumps({"schema_version": 999, "success": True}),
    ),
)
def test_invalid_manifest_stops_refresh(
    tmp_path: Path, invalid_content: str
) -> None:
    path = (
        tmp_path
        / "subjects/A/registration_runs/baseline/registration_result.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(invalid_content, encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        discover_registration_run_inventory(batch_dir=tmp_path)
