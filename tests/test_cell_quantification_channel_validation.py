import os
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("spatialsignal")

from spatialsignal.models import (
    DataRepresentation,
    DatasetMetadata,
    PointCloudDataset,
    ProcessingProvenance,
    SpaceDefinition,
)

from lsfm_data_processing.cell_quantification.project_workflow import (
    ArtifactState,
    ChannelDefinition,
    DeduplicationSettings,
    PointCloudExtractionSettings,
    SourceKind,
    SubjectWorkInput,
    build_channel_object_plan,
    deduplication_parameters_for_work_item,
    pointcloud_parameters_for_work_item,
    resolve_channel_object_work_item,
    validate_channel_objects,
    validate_channel_pointcloud,
    validate_loaded_point_dataset,
    with_lineage_parameters,
)


def _work_item(tmp_path: Path, source_kind: SourceKind):
    mask = tmp_path / "masks_plane_0001.tif"
    mask.write_bytes(b"mask")
    channel = ChannelDefinition(
        population_id="ch0_aldh",
        source_kind=source_kind,
        analysis_lane="cell_bodies",
        deduplication=DeduplicationSettings(
            max_plane_offset=1,
            max_xy_distance_um=2.0,
            max_n_planes=2,
            write_edge_table=False,
        ),
    )
    return resolve_channel_object_work_item(
        subject=SubjectWorkInput(
            "IEB1000", "P4", tmp_path, tmp_path / "_spatialsignal"
        ),
        channel=channel,
        signal_id="aldh",
        channel_name="ALDH",
        input_type="mips20um",
        resolution_um=(20.0, 20.0, 20.0),
        mask_files=(mask,),
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


def _metadata(
    representation_type: str,
    processing: ProcessingProvenance | None = None,
) -> DatasetMetadata:
    return DatasetMetadata(
        schema_name="spatialsignal.dataset_metadata",
        schema_version="0.1.0",
        space=SpaceDefinition(
            space_name="subject_native_masks",
            orientation="las",
            axis_labels=["x", "y", "z"],
            indexing="zero_based",
            units="voxel",
            shape=[10, 12, 1],
            resolution_um=[20.0, 20.0, 20.0],
        ),
        representation=DataRepresentation(
            kind="point_cloud",
            representation_type=representation_type,
        ),
        processing=processing,
    )


def _write_pointcloud(item, *, include_detection_columns: bool = True) -> None:
    table, sidecar = item.pointcloud_outputs.paths
    table.parent.mkdir(parents=True, exist_ok=True)
    data = {"x": [2], "y": [3], "z": [0]}
    if include_detection_columns:
        data.update({"detection_id": [0], "seg_num": [1]})
    pd.DataFrame(data).to_parquet(table, index=False)
    processing = None
    if item.channel.source_kind is SourceKind.INSTANCE_MASKS:
        processing = ProcessingProvenance(
            stage="build_pointcloud_from_masks",
            parameters=with_lineage_parameters(
                pointcloud_parameters_for_work_item(item), item.mask_files
            ),
        )
    _metadata("point_centroids", processing).to_json(sidecar)


def _write_objects(item) -> None:
    table, sidecar, *additional = item.deduplication_outputs.paths
    table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "object_id": [1],
            "x": [2],
            "y": [3],
            "z": [0],
            "n_detections": [1],
            "n_planes": [1],
            "z_min": [0],
            "z_max": [0],
        }
    ).to_parquet(table, index=False)
    _metadata(
        "cleaned_objects",
        ProcessingProvenance(
            stage="deduplicate_across_planes",
            parameters=with_lineage_parameters(
                deduplication_parameters_for_work_item(item),
                item.pointcloud_outputs.paths,
            ),
        ),
    ).to_json(sidecar)
    pd.DataFrame({"object_id": [1], "detection_id": [0]}).to_parquet(
        additional[0], index=False
    )


def test_generated_pointcloud_and_objects_validate_without_project_callbacks(
    tmp_path: Path,
) -> None:
    item = _work_item(tmp_path, SourceKind.INSTANCE_MASKS)
    _write_pointcloud(item)
    _write_objects(item)

    assert validate_channel_pointcloud(
        item, item.pointcloud_outputs
    ).state is ArtifactState.NO_LINEAGE
    assert validate_channel_objects(
        item, item.deduplication_outputs
    ).state is ArtifactState.NO_LINEAGE

    plan = build_channel_object_plan(work_items=(item,), policy="skip")
    assert len(plan.complete) == 1
    assert not plan.pending


def test_pointcloud_footer_missing_detection_columns_is_invalid(
    tmp_path: Path,
) -> None:
    item = _work_item(tmp_path, SourceKind.EXISTING_POINTCLOUD)
    _write_pointcloud(item, include_detection_columns=False)

    inspection = validate_channel_pointcloud(item, item.pointcloud_outputs)
    assert inspection.state is ArtifactState.INVALID
    assert "detection_id" in inspection.reasons[0]


def test_pointcloud_modification_time_drift_does_not_mark_objects_stale(
    tmp_path: Path,
) -> None:
    item = _work_item(tmp_path, SourceKind.EXISTING_POINTCLOUD)
    _write_pointcloud(item)
    _write_objects(item)
    for path in item.pointcloud_outputs.paths:
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    inspection = validate_channel_objects(item, item.deduplication_outputs)
    assert inspection.state is ArtifactState.NO_LINEAGE


def test_deep_validation_rejects_nonfinite_coordinates(tmp_path: Path) -> None:
    dataset = PointCloudDataset(
        subject_name="IEB1000",
        points=pd.DataFrame(
            {
                "detection_id": [0],
                "seg_num": [1],
                "x": [float("nan")],
                "y": [3.0],
                "z": [0.0],
            }
        ),
        metadata=_metadata("point_centroids"),
    )
    with pytest.raises(ValueError):
        validate_loaded_point_dataset(dataset, require_detection_columns=True)
