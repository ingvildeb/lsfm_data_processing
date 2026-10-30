from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import tomllib

import pytest

from lsfm_data_processing.registration_and_transforms.project_workflow.baseline_generation import (
    apply_baseline_generation_plan,
    plan_baseline_generation,
)
from lsfm_data_processing.registration_and_transforms.project_workflow.generation import (
    HpcConfigSpec,
    RegistrationBatchSpec,
    RegistrationImageSpec,
)


def _registration_spec() -> RegistrationBatchSpec:
    return RegistrationBatchSpec(
        registration_presets=("tuned_syn_cc",),
        orientation_alignment="physical",
        write_input_images=True,
        output_subdir="registration_runs/baseline",
        images=(
            RegistrationImageSpec(
                "template",
                PurePosixPath("/templates/template.nii.gz"),
                "template",
                "LSP",
                20,
            ),
            RegistrationImageSpec(
                "A",
                PurePosixPath("/subjects/A/native.nii.gz"),
                "A",
                "LAS",
                20,
            ),
        ),
        image_to_template=(("A", "template"),),
        image_default_resolution_um=20,
        template_role="fixed",
        moving_segmentations_enabled=True,
        moving_segmentation_interpolation="genericLabel",
        moving_segmentations_write_intermediates=False,
    )


def _hpc_spec(*, dry_run: bool = True) -> HpcConfigSpec:
    return HpcConfigSpec(
        partition="compute",
        cpus_per_task=32,
        mem_gb=128,
        time="4-00:00:00",
        exclude_nodes=("bad-node",),
        conda_env="lsfm_data_processing",
        python_executable="python",
        dry_run=dry_run,
        job_name_prefix="test_base",
    )


def _plan(tmp_path: Path, *, dry_run: bool = True):
    return plan_baseline_generation(
        batch_id="batch001",
        configs_dir=tmp_path / "configs",
        helper_path=tmp_path / "submit_baseline_hpc.sh",
        registration_spec=_registration_spec(),
        hpc_spec=_hpc_spec(dry_run=dry_run),
    )


def test_baseline_plan_writes_provenance_and_resumes(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    assert {item.purpose for item in plan.files} == {
        "baseline registration config",
        "baseline provenance",
        "operational HPC config",
        "baseline HPC submission helper",
    }
    apply_baseline_generation_plan(plan)

    provenance = json.loads(
        (tmp_path / "configs" / "baseline.provenance.json").read_text()
    )
    assert provenance["batch_id"] == "batch001"
    assert provenance["registration_spec"]["images"][1]["path"].endswith(
        "/subjects/A/native.nii.gz"
    )
    assert provenance["registration_config_sha256"]
    resumed = _plan(tmp_path)
    assert all(item.state == "already_present" for item in resumed.files)


def test_operational_hpc_config_is_replaceable(tmp_path: Path) -> None:
    apply_baseline_generation_plan(_plan(tmp_path))
    changed = _plan(tmp_path, dry_run=False)
    states = {item.purpose: item.state for item in changed.files}
    assert states["operational HPC config"] == "replace"
    assert states["baseline registration config"] == "already_present"

    apply_baseline_generation_plan(changed)
    parsed = tomllib.loads((tmp_path / "configs" / "hpc.toml").read_text())
    assert parsed["workflow"]["dry_run"] is False


def test_semantically_equal_legacy_baseline_is_adopted_without_rewrite(
    tmp_path: Path,
) -> None:
    initial = _plan(tmp_path)
    baseline_artifact = next(
        item
        for item in initial.files
        if item.purpose == "baseline registration config"
    )
    baseline_artifact.destination.parent.mkdir(parents=True)
    legacy_content = b"# Legacy project renderer.\n" + baseline_artifact.content
    baseline_artifact.destination.write_bytes(legacy_content)

    adopted = _plan(tmp_path)
    baseline = next(
        item
        for item in adopted.files
        if item.purpose == "baseline registration config"
    )
    provenance = next(
        item for item in adopted.files if item.purpose == "baseline provenance"
    )
    assert baseline.state == "already_present"
    assert baseline.content == legacy_content
    assert json.loads(provenance.content)["adopted_existing_config"] is True

    apply_baseline_generation_plan(adopted)
    assert baseline.destination.read_bytes() == legacy_content


def test_changed_baseline_is_an_immutable_collision(tmp_path: Path) -> None:
    apply_baseline_generation_plan(_plan(tmp_path))
    baseline = tmp_path / "configs" / "baseline.toml"
    baseline.write_text("changed", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Immutable baseline artifact"):
        _plan(tmp_path)
