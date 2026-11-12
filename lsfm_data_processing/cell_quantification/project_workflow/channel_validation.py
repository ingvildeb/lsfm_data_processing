"""Canonical shallow and deep validation for workflow-01 point datasets."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from .artifacts import ArtifactInspection, ArtifactSet, ArtifactState
from .lineage import validate_lineage_parameters
from .models import SourceKind

if TYPE_CHECKING:
    from .work_items import ChannelObjectWorkItem


def _spatialsignal_models() -> tuple[type[Any], type[Any]]:
    try:
        from spatialsignal.models import DatasetMetadata, PointCloudDataset
    except ImportError as exc:
        raise RuntimeError(
            "Point-dataset validation requires the lsfm-data-processing "
            "cell-quantification extra."
        ) from exc
    return DatasetMetadata, PointCloudDataset


def inspect_point_dataset_metadata(
    metadata_path: Path,
    *,
    expected_representation: str,
) -> Any:
    """Validate a SpatialSignal sidecar without loading Parquet rows."""

    DatasetMetadata, _ = _spatialsignal_models()
    metadata = DatasetMetadata.from_json(metadata_path)
    representation = metadata.representation
    if (
        representation.kind != "point_cloud"
        or representation.representation_type != expected_representation
    ):
        raise ValueError(
            "point-dataset representation differs: "
            f"{representation.kind}/{representation.representation_type}"
        )
    space = metadata.space
    if len(space.axis_labels) != 3 or len(set(space.axis_labels)) != 3:
        raise ValueError(f"point-dataset axis labels are invalid: {space.axis_labels}")
    shape = tuple(int(value) for value in space.shape)
    if len(shape) != 3 or any(value < 1 for value in shape):
        raise ValueError(f"point-dataset space shape is invalid: {shape}")
    resolution = tuple(float(value) for value in space.resolution_um)
    if (
        len(resolution) != 3
        or not all(np.isfinite(value) for value in resolution)
        or any(value <= 0 for value in resolution)
    ):
        raise ValueError(f"point-dataset resolution is invalid: {resolution}")
    if space.indexing not in {"zero_based", "one_based"}:
        raise ValueError(f"point-dataset indexing is invalid: {space.indexing!r}")
    return metadata


def inspect_point_dataset_footer(
    table_path: Path,
    metadata: Any,
    *,
    required_columns: Sequence[str] = (),
) -> int:
    """Validate Parquet schema and footer without loading point rows."""

    try:
        from pyarrow import parquet
    except ImportError as exc:
        raise RuntimeError(
            "Point-dataset validation requires the lsfm-data-processing "
            "cell-quantification extra."
        ) from exc
    if not table_path.is_file() or table_path.stat().st_size <= 0:
        raise ValueError(f"point-dataset Parquet is missing or empty: {table_path}")
    parquet_file = parquet.ParquetFile(table_path)
    observed = set(parquet_file.schema_arrow.names)
    required = {*metadata.space.axis_labels, *required_columns}
    missing = sorted(required.difference(observed))
    if missing:
        raise ValueError(f"point-dataset Parquet is missing columns: {missing}")
    return int(parquet_file.metadata.num_rows)


def inspect_parquet_footer(
    table_path: Path,
    *,
    required_columns: Sequence[str] = (),
) -> int:
    """Validate a non-spatial Parquet artifact without loading its rows."""

    try:
        from pyarrow import parquet
    except ImportError as exc:
        raise RuntimeError(
            "Parquet validation requires the lsfm-data-processing "
            "cell-quantification extra."
        ) from exc
    if not table_path.is_file() or table_path.stat().st_size <= 0:
        raise ValueError(f"Parquet artifact is missing or empty: {table_path}")
    parquet_file = parquet.ParquetFile(table_path)
    observed = set(parquet_file.schema_arrow.names)
    missing = sorted(set(required_columns).difference(observed))
    if missing:
        raise ValueError(f"Parquet artifact is missing columns: {missing}")
    return int(parquet_file.metadata.num_rows)


def inspect_point_dataset(
    table_path: Path,
    metadata_path: Path,
    *,
    expected_representation: str,
    required_columns: Sequence[str] = (),
    inspect_footer: bool = True,
) -> Any:
    """Perform plan-safe sidecar and optional Parquet-footer validation."""

    metadata = inspect_point_dataset_metadata(
        metadata_path,
        expected_representation=expected_representation,
    )
    if inspect_footer:
        inspect_point_dataset_footer(
            table_path,
            metadata,
            required_columns=required_columns,
        )
    return metadata


def validate_loaded_point_dataset(
    dataset: Any,
    *,
    require_detection_columns: bool,
) -> None:
    """Deeply validate coordinates after rows are loaded for apply mode."""

    _, PointCloudDataset = _spatialsignal_models()
    if not isinstance(dataset, PointCloudDataset):
        raise TypeError("dataset must be a SpatialSignal PointCloudDataset")
    if require_detection_columns:
        dataset.validate()
    else:
        dataset.validate_spatial_points()
    for axis_label in dataset.space.axis_labels:
        values = dataset.points[axis_label].to_numpy(copy=False)
        try:
            finite = np.isfinite(values)
        except TypeError as exc:
            raise ValueError(
                f"point coordinate column {axis_label!r} is not numeric"
            ) from exc
        if not bool(finite.all()):
            raise ValueError(
                f"point coordinate column {axis_label!r} contains non-finite values"
            )


def pointcloud_parameters_for_work_item(
    item: ChannelObjectWorkItem,
) -> dict[str, Any]:
    """Return lineage-bearing scientific parameters for point-cloud extraction."""

    settings = item.pointcloud_settings
    if settings is None:
        raise ValueError("mask-derived point clouds require extraction settings")
    return {
        "signal_id": item.signal_id,
        "input_type": item.input_type,
        "resolution_um": [float(value) for value in item.resolution_um],
        "space_name": settings.space_name,
        "orientation": settings.orientation,
        "indexing": settings.indexing,
        "mask_prefix": settings.mask_prefix,
    }


def deduplication_parameters_for_work_item(
    item: ChannelObjectWorkItem,
) -> dict[str, Any]:
    """Return lineage-bearing scientific parameters for deduplication."""

    settings = item.channel.deduplication
    return {
        "signal_id": item.signal_id,
        "input_type": item.input_type,
        "resolution_um": [float(value) for value in item.resolution_um],
        "max_plane_offset": int(settings.max_plane_offset),
        "max_xy_distance_um": float(settings.max_xy_distance_um),
        "max_n_planes": (
            int(settings.max_n_planes)
            if settings.max_n_planes is not None
            else None
        ),
    }


def validate_deduplicated_object_artifact(
    *,
    artifact_set: ArtifactSet,
    metadata_path: Path,
    direct_inputs: Sequence[Path],
    expected_parameters: Mapping[str, Any],
) -> ArtifactInspection:
    """Validate one complete cleaned-object artifact and its lineage."""

    if len(artifact_set.paths) not in {3, 4}:
        return ArtifactInspection(
            artifact_set,
            ArtifactState.INVALID,
            ("deduplication artifact set must contain three or four files",),
        )

    try:
        metadata = inspect_point_dataset_metadata(
            metadata_path,
            expected_representation="cleaned_objects",
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return ArtifactInspection(
            artifact_set,
            ArtifactState.INVALID,
            (f"deduplication metadata cannot be read: {exc}",),
        )
    processing = metadata.processing
    if processing is None or processing.stage != "deduplicate_across_planes":
        actual: object = None if processing is None else processing.stage
        return ArtifactInspection(
            artifact_set,
            ArtifactState.INVALID,
            (
                "complete deduplication output has incompatible processing "
                f"metadata: stage={actual!r}; "
                "expected='deduplicate_across_planes'",
            ),
        )
    try:
        inspect_point_dataset_footer(
            artifact_set.paths[0],
            metadata,
            required_columns=(
                "object_id",
                "n_detections",
                "n_planes",
                "z_min",
                "z_max",
            ),
        )
        inspect_parquet_footer(
            artifact_set.paths[2],
            required_columns=("object_id", "detection_id"),
        )
        if len(artifact_set.paths) > 3:
            inspect_parquet_footer(
                artifact_set.paths[3],
                required_columns=(
                    "source_detection_id",
                    "target_detection_id",
                ),
            )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return ArtifactInspection(
            artifact_set,
            ArtifactState.INVALID,
            (f"deduplication Parquet footer is invalid: {exc}",),
        )
    return validate_lineage_parameters(
        artifact_set,
        processing.parameters,
        expected_parameters,
        direct_inputs,
    )


def validate_channel_pointcloud(
    item: ChannelObjectWorkItem,
    artifact: ArtifactSet,
) -> ArtifactInspection:
    """Plan-safe canonical validator for a workflow-01 point cloud."""

    try:
        metadata = inspect_point_dataset(
            artifact.paths[0],
            artifact.paths[1],
            expected_representation="point_centroids",
            required_columns=("detection_id", "seg_num"),
        )
        observed_resolution = tuple(
            float(value) for value in metadata.space.resolution_um
        )
        if observed_resolution != item.resolution_um:
            raise ValueError(
                "point-cloud resolution differs: "
                f"{observed_resolution} != {item.resolution_um}"
            )
        if item.channel.source_kind is SourceKind.INSTANCE_MASKS:
            settings = item.pointcloud_settings
            if settings is None:
                raise ValueError("mask-derived point cloud has no extraction settings")
            observed_space = (
                str(metadata.space.space_name),
                str(metadata.space.orientation),
                str(metadata.space.indexing),
            )
            expected_space = (
                settings.space_name,
                settings.orientation,
                settings.indexing,
            )
            if observed_space != expected_space:
                raise ValueError(
                    f"point-cloud space differs: {observed_space} != {expected_space}"
                )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return ArtifactInspection(
            artifact,
            ArtifactState.INVALID,
            (f"invalid point cloud: {exc}",),
        )
    if item.channel.source_kind is SourceKind.EXISTING_POINTCLOUD:
        return ArtifactInspection(artifact, ArtifactState.VALID)
    return validate_lineage_parameters(
        artifact,
        None if metadata.processing is None else metadata.processing.parameters,
        pointcloud_parameters_for_work_item(item),
        item.mask_files,
    )


def validate_channel_objects(
    item: ChannelObjectWorkItem,
    artifact: ArtifactSet,
) -> ArtifactInspection:
    """Plan-safe canonical validator for workflow-01 deduplicated objects."""

    return validate_deduplicated_object_artifact(
        artifact_set=artifact,
        metadata_path=item.deduplication_outputs.paths[1],
        direct_inputs=item.pointcloud_outputs.paths,
        expected_parameters=deduplication_parameters_for_work_item(item),
    )
