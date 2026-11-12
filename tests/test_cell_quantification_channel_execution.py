from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

pytest.importorskip("spatialsignal")

from lsfm_data_processing.cell_quantification.project_workflow import (
    ArtifactState,
    ChannelDefinition,
    DeduplicationSettings,
    ChannelObjectExecutionResult,
    ChannelObjectPlanExecutionError,
    PointCloudExecutionSettings,
    PointCloudExtractionSettings,
    SourceKind,
    SubjectWorkInput,
    apply_channel_object_plan,
    build_channel_object_plan,
    execute_channel_object_plan_item,
    resolve_channel_object_work_item,
    validate_channel_objects,
    validate_channel_pointcloud,
)


def _item(
    tmp_path: Path,
    source_kind: SourceKind,
    subject_id: str = "IEB1000",
):
    channel = ChannelDefinition(
        population_id="ch0_aldh",
        source_kind=source_kind,
        analysis_lane="cell_bodies",
        deduplication=DeduplicationSettings(
            max_plane_offset=1,
            max_xy_distance_um=3.0,
            max_n_planes=2,
            write_edge_table=False,
        ),
    )
    subject = SubjectWorkInput(
        subject_id,
        "P4",
        tmp_path,
        tmp_path / "_spatialsignal",
    )
    mask_dir = (
        subject.analysis_root
        / "segmentations"
        / channel.population_id
        / channel.analysis_lane
        / "masks"
    )
    mask_files = tuple(sorted(mask_dir.glob("masks_*.tif")))
    return resolve_channel_object_work_item(
        subject=subject,
        channel=channel,
        signal_id="aldh",
        channel_name="ALDH",
        input_type="mips20um",
        resolution_um=(20.0, 20.0, 20.0),
        mask_files=mask_files,
        pointcloud_settings=(
            PointCloudExtractionSettings(
                space_name="subject_native_masks",
                orientation="las",
                indexing="zero_based",
                mask_prefix="masks_",
            )
            if source_kind is SourceKind.INSTANCE_MASKS
            else None
        ),
    )


def _write_masks(tmp_path: Path) -> None:
    mask_dir = (
        tmp_path
        / "_spatialsignal"
        / "segmentations"
        / "ch0_aldh"
        / "cell_bodies"
        / "masks"
    )
    mask_dir.mkdir(parents=True)
    for plane in range(2):
        mask = np.zeros((8, 8), dtype=np.uint16)
        mask[2:4, 2:4] = 1
        tifffile.imwrite(mask_dir / f"masks_plane_{plane:04d}.tif", mask)


def _execute_initial_workflow(tmp_path: Path):
    _write_masks(tmp_path)
    item = _item(tmp_path, SourceKind.INSTANCE_MASKS)
    plan = build_channel_object_plan(work_items=(item,), policy="skip")
    return item, execute_channel_object_plan_item(
        plan.pending[0],
        extraction_settings=PointCloudExecutionSettings(
            max_workers=1,
            show_progress=False,
        ),
    )


def test_executes_mask_pointcloud_and_deduplication_with_post_validation(
    tmp_path: Path,
) -> None:
    item, result = _execute_initial_workflow(tmp_path)

    assert result.n_detections == 2
    assert result.n_objects == 1
    assert validate_channel_pointcloud(
        item, item.pointcloud_outputs
    ).state is ArtifactState.VALID
    assert validate_channel_objects(
        item, item.deduplication_outputs
    ).state is ArtifactState.VALID


def test_overwrite_preserves_existing_source_pointcloud(tmp_path: Path) -> None:
    generated, _ = _execute_initial_workflow(tmp_path)
    source = _item(tmp_path, SourceKind.EXISTING_POINTCLOUD)
    original = tuple(path.read_bytes() for path in generated.pointcloud_outputs.paths)

    plan = build_channel_object_plan(work_items=(source,), policy="overwrite")
    result = execute_channel_object_plan_item(
        plan.pending[0],
        extraction_settings=PointCloudExecutionSettings(show_progress=False),
    )

    assert result.n_objects == 1
    observed = tuple(
        path.read_bytes() for path in source.pointcloud_outputs.paths
    )
    assert observed == original


def test_post_write_validation_rejects_malformed_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated, _ = _execute_initial_workflow(tmp_path)
    source = _item(tmp_path, SourceKind.EXISTING_POINTCLOUD)
    for path in source.deduplication_outputs.paths:
        path.unlink()

    import spatialsignal.io

    original_writer = spatialsignal.io.save_deduplication_outputs

    def corrupting_writer(*args, **kwargs):
        outputs = original_writer(*args, **kwargs)
        pd.DataFrame({"wrong": [1]}).to_parquet(
            outputs.membership_table,
            index=False,
        )
        return outputs

    monkeypatch.setattr(
        spatialsignal.io,
        "save_deduplication_outputs",
        corrupting_writer,
    )
    plan = build_channel_object_plan(work_items=(source,), policy="repair")

    with pytest.raises(RuntimeError, match="post-write validation"):
        execute_channel_object_plan_item(
            plan.pending[0],
            extraction_settings=PointCloudExecutionSettings(show_progress=False),
        )


def test_apply_plan_excludes_already_complete_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item, _ = _execute_initial_workflow(tmp_path)
    plan = build_channel_object_plan(work_items=(item,), policy="skip")

    def unexpected_execution(*args, **kwargs):
        raise AssertionError("complete items must not enter the execution queue")

    monkeypatch.setattr(
        "lsfm_data_processing.cell_quantification.project_workflow."
        "channel_execution.execute_channel_object_plan_item",
        unexpected_execution,
    )
    result = apply_channel_object_plan(plan)

    assert result.results == ()
    assert result.failures == ()
    assert result.already_complete == ("IEB1000",)


def test_apply_plan_attempts_independent_subjects_and_raises_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_masks(first_root)
    _write_masks(second_root)
    first = _item(first_root, SourceKind.INSTANCE_MASKS, "IEB1000")
    second = _item(second_root, SourceKind.INSTANCE_MASKS, "IEB1001")
    plan = build_channel_object_plan(
        work_items=(first, second),
        policy="skip",
    )
    attempted: list[str] = []

    def controlled_execution(planned, **kwargs):
        subject_id = planned.work_item.subject_id
        attempted.append(subject_id)
        if subject_id == "IEB1000":
            raise ValueError("controlled subject failure")
        return ChannelObjectExecutionResult(
            subject_id=subject_id,
            pointcloud_action=planned.pointcloud.action,
            deduplication_action=planned.deduplication.action,
            n_detections=2,
            n_objects=1,
        )

    monkeypatch.setattr(
        "lsfm_data_processing.cell_quantification.project_workflow."
        "channel_execution.execute_channel_object_plan_item",
        controlled_execution,
    )
    messages: list[str] = []
    with pytest.raises(ChannelObjectPlanExecutionError) as caught:
        apply_channel_object_plan(plan, reporter=messages.append)

    assert attempted == ["IEB1000", "IEB1001"]
    assert tuple(item.subject_id for item in caught.value.result.results) == (
        "IEB1001",
    )
    assert len(caught.value.result.failures) == 1
    assert caught.value.result.failures[0].subject_id == "IEB1000"
    assert "IEB1000: ValueError: controlled subject failure" in str(caught.value)
    assert messages[-1].endswith(
        "1/2 pending subject(s) successful; 0 already complete."
    )
