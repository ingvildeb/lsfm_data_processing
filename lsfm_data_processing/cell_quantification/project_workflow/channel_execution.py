"""Shared execution for canonical workflow-01 channel artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .artifacts import (
    ArtifactInspection,
    ArtifactState,
    PlannedAction,
    inspect_artifact_set,
)
from .channel_validation import (
    deduplication_parameters_for_work_item,
    pointcloud_parameters_for_work_item,
    validate_channel_objects,
    validate_channel_pointcloud,
    validate_loaded_point_dataset,
)
from .lineage import with_lineage_parameters
from .models import SourceKind
from .work_items import (
    ChannelObjectPlan,
    ChannelObjectPlanItem,
    ChannelObjectWorkItem,
)


StatusReporter = Callable[[str], None]


@dataclass(frozen=True)
class PointCloudExecutionSettings:
    """Operational settings for mask-centroid extraction."""

    max_workers: int = 1
    show_progress: bool = True
    progress_interval: int = 25
    retry_serial_on_worker_failure: bool = True

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be positive")
        if self.progress_interval < 1:
            raise ValueError("progress_interval must be positive")


@dataclass(frozen=True)
class ChannelObjectExecutionResult:
    """Compact result from executing one planned workflow-01 item."""

    subject_id: str
    pointcloud_action: PlannedAction
    deduplication_action: PlannedAction | None
    n_detections: int
    n_objects: int | None


@dataclass(frozen=True)
class ChannelObjectExecutionFailure:
    """One subject-level failure collected while applying a cohort plan."""

    subject_id: str
    error_type: str
    message: str

    def format(self) -> str:
        return f"{self.subject_id}: {self.error_type}: {self.message}"


@dataclass(frozen=True)
class ChannelObjectPlanExecutionResult:
    """Cohort-level outcome after all independent pending items were attempted."""

    results: tuple[ChannelObjectExecutionResult, ...]
    failures: tuple[ChannelObjectExecutionFailure, ...]
    already_complete: tuple[str, ...]

    @property
    def successful_count(self) -> int:
        return len(self.results)


class ChannelObjectPlanExecutionError(RuntimeError):
    """Aggregated workflow-01 failure with the completed cohort outcome."""

    def __init__(self, result: ChannelObjectPlanExecutionResult) -> None:
        self.result = result
        details = "\n".join(failure.format() for failure in result.failures)
        super().__init__(f"Point-cloud failures:\n{details}")


def _runtime() -> tuple[type[Any], type[Any], Any, Any, Any]:
    try:
        from spatialsignal.io import save_deduplication_outputs
        from spatialsignal.models import PointCloudDataset, ProcessingProvenance
        from spatialsignal.pointcloud import (
            build_pointcloud_from_masks,
            deduplicate_across_planes,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Channel-object execution requires the lsfm-data-processing "
            "cell-quantification extra."
        ) from exc
    return (
        PointCloudDataset,
        ProcessingProvenance,
        build_pointcloud_from_masks,
        deduplicate_across_planes,
        save_deduplication_outputs,
    )


def _report(reporter: StatusReporter | None, message: str) -> None:
    if reporter is not None:
        reporter(message)


def _require_reusable(inspection: ArtifactInspection, stage: str) -> None:
    if not inspection.reusable:
        details = "; ".join(inspection.reasons)
        suffix = f": {details}" if details else ""
        raise RuntimeError(
            f"{stage} failed validation as {inspection.state.value}{suffix}"
        )


def _require_newly_reusable(inspection: ArtifactInspection, stage: str) -> None:
    if inspection.state not in {
        ArtifactState.VALID,
        ArtifactState.NO_LINEAGE,
    }:
        details = "; ".join(inspection.reasons)
        suffix = f": {details}" if details else ""
        raise RuntimeError(
            f"newly written {stage} failed post-write validation as "
            f"{inspection.state.value}{suffix}"
        )


def _load_pointcloud(item: ChannelObjectWorkItem) -> Any:
    PointCloudDataset, _, _, _, _ = _runtime()
    table_path, metadata_path = item.pointcloud_outputs.paths
    dataset = PointCloudDataset.from_files(
        table_path,
        metadata_path,
        subject_name=item.subject_id,
    )
    validate_loaded_point_dataset(dataset, require_detection_columns=True)
    inspection = validate_channel_pointcloud(item, item.pointcloud_outputs)
    _require_reusable(inspection, "point cloud")
    return dataset


def _write_pointcloud(
    item: ChannelObjectWorkItem,
    settings: PointCloudExecutionSettings,
    reporter: StatusReporter | None,
) -> Any:
    if item.channel.source_kind is not SourceKind.INSTANCE_MASKS:
        raise RuntimeError("an existing source point cloud cannot be rewritten")
    extraction = item.pointcloud_settings
    if extraction is None:
        raise RuntimeError("mask-derived point cloud has no extraction settings")
    (
        PointCloudDataset,
        ProcessingProvenance,
        build_pointcloud_from_masks,
        _,
        _,
    ) = _runtime()

    def extract(max_workers: int) -> None:
        build_pointcloud_from_masks(
            mask_dir=item.mask_dir,
            out_dir=item.cell_body_dir,
            subject_name=item.subject_id,
            space_name=extraction.space_name,
            orientation=extraction.orientation,
            resolution_um=list(item.resolution_um),
            indexing=extraction.indexing,
            extensions=(".tif", ".tiff"),
            prefix=extraction.mask_prefix,
            max_workers=max_workers,
            show_progress=settings.show_progress,
            progress_interval=settings.progress_interval,
        )

    try:
        extract(settings.max_workers)
    except RuntimeError as exc:
        retryable = "Parallel centroid extraction failed" in str(exc)
        if (
            settings.max_workers <= 1
            or not settings.retry_serial_on_worker_failure
            or not retryable
        ):
            raise
        _report(reporter, "    Parallel extraction failed; retrying with one worker")
        extract(1)

    table_path, metadata_path = item.pointcloud_outputs.paths
    dataset = PointCloudDataset.from_files(
        table_path,
        metadata_path,
        subject_name=item.subject_id,
    )
    validate_loaded_point_dataset(dataset, require_detection_columns=True)
    dataset.metadata.processing = ProcessingProvenance(
        stage="build_pointcloud_from_masks",
        source_name=item.mask_dir.name,
        parameters=with_lineage_parameters(
            pointcloud_parameters_for_work_item(item),
            item.mask_files,
        ),
        summary={"n_detections": int(len(dataset.points))},
    )
    dataset.metadata.to_json(metadata_path)
    inspection = inspect_artifact_set(
        item.pointcloud_outputs,
        validator=lambda artifact: validate_channel_pointcloud(item, artifact),
    )
    _require_newly_reusable(inspection, "point cloud")
    return dataset


def _load_objects(item: ChannelObjectWorkItem) -> Any:
    PointCloudDataset, _, _, _, _ = _runtime()
    table_path, metadata_path = item.deduplication_outputs.paths[:2]
    dataset = PointCloudDataset.from_files(
        table_path,
        metadata_path,
        subject_name=item.subject_id,
    )
    validate_loaded_point_dataset(dataset, require_detection_columns=False)
    inspection = validate_channel_objects(item, item.deduplication_outputs)
    _require_reusable(inspection, "deduplicated objects")
    return dataset


def _write_objects(item: ChannelObjectWorkItem, pointcloud: Any) -> Any:
    if item.channel.morphology_filter is not None:
        raise NotImplementedError(
            "workflow-01 execution does not yet implement morphology filters"
        )
    PointCloudDataset, _, _, deduplicate_across_planes, save_outputs = _runtime()
    settings = item.channel.deduplication
    result = deduplicate_across_planes(
        pointcloud.points,
        pointcloud.space,
        max_plane_offset=settings.max_plane_offset,
        max_xy_distance_um=settings.max_xy_distance_um,
        max_n_planes=settings.max_n_planes,
    )
    save_outputs(
        pointcloud,
        result,
        item.cell_body_dir,
        source_name=item.pointcloud_outputs.paths[0].name,
        parameters=with_lineage_parameters(
            deduplication_parameters_for_work_item(item),
            item.pointcloud_outputs.paths,
        ),
        write_edge_table=settings.write_edge_table,
    )
    table_path, metadata_path = item.deduplication_outputs.paths[:2]
    objects = PointCloudDataset.from_files(
        table_path,
        metadata_path,
        subject_name=item.subject_id,
    )
    validate_loaded_point_dataset(objects, require_detection_columns=False)
    inspection = inspect_artifact_set(
        item.deduplication_outputs,
        validator=lambda artifact: validate_channel_objects(item, artifact),
    )
    _require_newly_reusable(inspection, "deduplicated objects")
    return objects


def execute_channel_object_plan_item(
    planned: ChannelObjectPlanItem,
    *,
    extraction_settings: PointCloudExecutionSettings | None = None,
    reporter: StatusReporter | None = None,
) -> ChannelObjectExecutionResult:
    """Execute one nonblocking workflow-01 plan item and validate its outputs."""

    if not planned.can_apply:
        raise RuntimeError("cannot execute a blocking channel-object plan item")
    settings = extraction_settings or PointCloudExecutionSettings()
    item = planned.work_item
    pointcloud_action = planned.pointcloud.action
    if pointcloud_action is PlannedAction.REUSE:
        pointcloud = _load_pointcloud(item)
        label = (
            "source"
            if planned.pointcloud_is_source
            else f"reused ({planned.pointcloud.inspection.state.value})"
        )
        _report(
            reporter,
            f"    Point cloud: {label} ({len(pointcloud.points):,} detections)",
        )
    elif pointcloud_action in {PlannedAction.WRITE, PlannedAction.REBUILD}:
        pointcloud = _write_pointcloud(item, settings, reporter)
        _report(
            reporter,
            f"    Point cloud complete: {len(pointcloud.points):,} detections",
        )
    else:
        raise RuntimeError(f"unsupported point-cloud action: {pointcloud_action.value}")

    object_action = (
        None if planned.deduplication is None else planned.deduplication.action
    )
    objects = None
    if object_action is PlannedAction.REUSE:
        objects = _load_objects(item)
        _report(
            reporter,
            f"    Deduplication: reused ({len(objects.points):,} objects; "
            f"{planned.deduplication.inspection.state.value})",
        )
    elif object_action in {PlannedAction.WRITE, PlannedAction.REBUILD}:
        objects = _write_objects(item, pointcloud)
        _report(
            reporter,
            f"    Deduplication complete: {len(pointcloud.points):,} detections -> "
            f"{len(objects.points):,} objects",
        )
    elif object_action is PlannedAction.STOP:
        raise RuntimeError("cannot execute a blocking deduplication action")

    return ChannelObjectExecutionResult(
        subject_id=item.subject_id,
        pointcloud_action=pointcloud_action,
        deduplication_action=object_action,
        n_detections=int(len(pointcloud.points)),
        n_objects=None if objects is None else int(len(objects.points)),
    )


def apply_channel_object_plan(
    plan: ChannelObjectPlan,
    *,
    extraction_settings: PointCloudExecutionSettings | None = None,
    reporter: StatusReporter | None = None,
) -> ChannelObjectPlanExecutionResult:
    """Apply every pending workflow-01 item and aggregate subject failures.

    A plan containing a blocking policy decision is rejected before any writes.
    Runtime failure of one otherwise independent subject does not prevent later
    pending subjects from being attempted.
    """

    if not plan.can_apply:
        details = "\n".join(plan.blocking_reasons)
        suffix = f"\n{details}" if details else ""
        raise RuntimeError(
            "Channel-object plan contains blocking items." + suffix
        )

    pending = plan.pending
    results: list[ChannelObjectExecutionResult] = []
    failures: list[ChannelObjectExecutionFailure] = []
    for index, planned in enumerate(pending, start=1):
        item = planned.work_item
        _report(
            reporter,
            f"\n[{index}/{len(pending)}] {item.subject_id} / {item.channel_id}",
        )
        try:
            result = execute_channel_object_plan_item(
                planned,
                extraction_settings=extraction_settings,
                reporter=reporter,
            )
            results.append(result)
            if planned.deduplication is None and not item.deduplication_enabled:
                _report(
                    reporter,
                    f"    Deduplication disabled for {item.signal_id}",
                )
        except Exception as exc:
            failure = ChannelObjectExecutionFailure(
                subject_id=item.subject_id,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            failures.append(failure)
            _report(reporter, f"    FAILED: {failure.format()}")

    result = ChannelObjectPlanExecutionResult(
        results=tuple(results),
        failures=tuple(failures),
        already_complete=tuple(
            item.work_item.subject_id for item in plan.complete
        ),
    )
    _report(
        reporter,
        f"\nPoint-cloud run complete: "
        f"{result.successful_count}/{len(pending)} pending subject(s) successful; "
        f"{len(result.already_complete)} already complete.",
    )
    if failures:
        raise ChannelObjectPlanExecutionError(result)
    return result
