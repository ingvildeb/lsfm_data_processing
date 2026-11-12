from pathlib import Path

from lsfm_data_processing.cell_quantification.project_workflow import (
    ArtifactInspection,
    ArtifactSet,
    ArtifactState,
    ChannelDefinition,
    DeduplicationSettings,
    PlannedAction,
    PointCloudExtractionSettings,
    SourceKind,
    SubjectWorkInput,
    build_channel_object_plan,
    resolve_channel_object_work_item,
    summarize_channel_object_plan,
)


def _channel(source_kind: SourceKind) -> ChannelDefinition:
    return ChannelDefinition(
        population_id="ch2_sytox",
        source_kind=source_kind,
        analysis_lane="cell_bodies",
        deduplication=DeduplicationSettings(
            max_plane_offset=1,
            max_xy_distance_um=3.0,
            max_n_planes=3,
        ),
    )


def _item(tmp_path: Path, source_kind: SourceKind):
    return resolve_channel_object_work_item(
        subject=SubjectWorkInput(
            "IEB0001", "P4", tmp_path, tmp_path / "_spatialsignal"
        ),
        channel=_channel(source_kind),
        signal_id="sytox",
        channel_name="Sytox",
        input_type="planes5um",
        resolution_um=(1.8, 1.8, 5.0),
        mask_files=(tmp_path / "mask_0001.tif",),
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


def _state(state: ArtifactState):
    def validate(_item, artifact: ArtifactSet) -> ArtifactInspection:
        return ArtifactInspection(artifact, state)

    return validate


def test_resolves_canonical_channel_paths(tmp_path: Path) -> None:
    item = _item(tmp_path, SourceKind.INSTANCE_MASKS)
    root = tmp_path / "_spatialsignal" / "segmentations" / "ch2_sytox" / "cell_bodies"
    assert item.cell_body_dir == root
    assert item.mask_dir == root / "masks"
    assert item.pointcloud_outputs.paths[0] == root / "IEB0001_pointcloud.parquet"
    assert item.deduplication_outputs.paths[0] == root / "IEB0001_objects.parquet"


def test_existing_pointcloud_is_a_source_even_under_overwrite(tmp_path: Path) -> None:
    item = _item(tmp_path, SourceKind.EXISTING_POINTCLOUD)
    for path in (*item.pointcloud_outputs.paths, *item.deduplication_outputs.paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("complete", encoding="utf-8")

    plan = build_channel_object_plan(
        work_items=(item,),
        policy="overwrite",
        pointcloud_validator=_state(ArtifactState.VALID),
        deduplication_validator=_state(ArtifactState.VALID),
    )

    planned = plan.items[0]
    assert planned.actions == {
        "point cloud": "source",
        "deduplication": "rebuild",
    }
    assert planned.pointcloud.action is PlannedAction.REUSE
    assert len(plan.pending) == 1


def test_complete_subject_is_excluded_from_pending_queue(tmp_path: Path) -> None:
    item = _item(tmp_path, SourceKind.EXISTING_POINTCLOUD)
    for path in (*item.pointcloud_outputs.paths, *item.deduplication_outputs.paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("complete", encoding="utf-8")

    plan = build_channel_object_plan(
        work_items=(item,),
        unavailable=("IEB0002: source unavailable",),
        policy="skip",
        pointcloud_validator=_state(ArtifactState.VALID),
        deduplication_validator=_state(ArtifactState.LEGACY),
    )

    assert plan.can_apply
    assert not plan.pending
    assert len(plan.complete) == 1
    summary = summarize_channel_object_plan(plan)
    assert "0 pending subject(s), 1 already complete, 1 unavailable" in summary
    assert "excluded from run queue" in summary


def test_generated_pointcloud_rebuild_invalidates_legacy_objects(tmp_path: Path) -> None:
    item = _item(tmp_path, SourceKind.INSTANCE_MASKS)
    for path in item.deduplication_outputs.paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("legacy", encoding="utf-8")

    plan = build_channel_object_plan(
        work_items=(item,),
        policy="repair",
        pointcloud_validator=_state(ArtifactState.VALID),
        deduplication_validator=_state(ArtifactState.LEGACY),
    )

    planned = plan.items[0]
    assert planned.pointcloud.action is PlannedAction.WRITE
    assert planned.deduplication is not None
    assert planned.deduplication.inspection.state is ArtifactState.STALE
    assert planned.deduplication.action is PlannedAction.REBUILD


def test_invalid_source_blocks_apply(tmp_path: Path) -> None:
    item = _item(tmp_path, SourceKind.EXISTING_POINTCLOUD)
    for path in item.pointcloud_outputs.paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("invalid", encoding="utf-8")

    plan = build_channel_object_plan(
        work_items=(item,),
        policy="repair",
        pointcloud_validator=_state(ArtifactState.INVALID),
        deduplication_validator=_state(ArtifactState.VALID),
    )

    assert not plan.can_apply
    assert plan.items[0].pointcloud.action is PlannedAction.STOP
    assert "required source point cloud is invalid" in plan.blocking_reasons[0]
