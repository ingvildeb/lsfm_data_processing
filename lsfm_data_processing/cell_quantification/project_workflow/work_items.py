"""Resolved channel-object work items and read-only stage planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .artifacts import (
    ArtifactInspection,
    ArtifactSet,
    ArtifactState,
    CompleteSetValidator,
    ExistingOutputPolicy,
    PlannedAction,
    PolicyDecision,
    decide_existing_output,
    inspect_artifact_set,
)
from .channel_validation import (
    validate_channel_objects,
    validate_channel_pointcloud,
)
from .models import (
    ChannelDefinition,
    PointCloudExtractionSettings,
    SourceKind,
)
from .paths import CanonicalLayout


@dataclass(frozen=True)
class SubjectWorkInput:
    """Project-supplied identity and roots for one selected subject."""

    subject_id: str
    age: str
    subject_root: Path
    analysis_root: Path

    def __post_init__(self) -> None:
        if not str(self.subject_id).strip():
            raise ValueError("subject_id must not be empty")
        if not str(self.age).strip():
            raise ValueError("age must not be empty")
        object.__setattr__(self, "subject_root", Path(self.subject_root))
        object.__setattr__(self, "analysis_root", Path(self.analysis_root))


@dataclass(frozen=True)
class ChannelObjectWorkItem:
    """Project-neutral inputs and canonical outputs for workflow 01."""

    subject: SubjectWorkInput
    channel: ChannelDefinition
    signal_id: str
    channel_name: str
    input_type: str
    resolution_um: tuple[float, float, float]
    mask_files: tuple[Path, ...]
    cell_body_dir: Path
    mask_dir: Path
    pointcloud_outputs: ArtifactSet
    deduplication_outputs: ArtifactSet
    pointcloud_settings: PointCloudExtractionSettings | None = None
    deduplication_enabled: bool = True

    def __post_init__(self) -> None:
        if not str(self.signal_id).strip():
            raise ValueError("signal_id must not be empty")
        if not str(self.channel_name).strip():
            raise ValueError("channel_name must not be empty")
        if not str(self.input_type).strip():
            raise ValueError("input_type must not be empty")
        resolution = tuple(float(value) for value in self.resolution_um)
        if len(resolution) != 3 or any(value <= 0 for value in resolution):
            raise ValueError("resolution_um must contain three positive values")
        object.__setattr__(self, "resolution_um", resolution)
        object.__setattr__(
            self, "mask_files", tuple(Path(path) for path in self.mask_files)
        )
        object.__setattr__(self, "cell_body_dir", Path(self.cell_body_dir))
        object.__setattr__(self, "mask_dir", Path(self.mask_dir))
        if (
            self.channel.source_kind is SourceKind.INSTANCE_MASKS
            and self.pointcloud_settings is None
        ):
            raise ValueError(
                "instance-mask work items require pointcloud_settings"
            )

    @property
    def subject_id(self) -> str:
        return self.subject.subject_id

    @property
    def age(self) -> str:
        return self.subject.age

    @property
    def subject_root(self) -> Path:
        return self.subject.subject_root

    @property
    def channel_id(self) -> str:
        return self.channel.population_id

    @property
    def source_kind(self) -> str:
        return self.channel.source_kind.value

    @property
    def analysis_lane(self) -> str:
        return self.channel.analysis_lane


def resolve_channel_object_work_item(
    *,
    subject: SubjectWorkInput,
    channel: ChannelDefinition,
    signal_id: str,
    channel_name: str,
    input_type: str,
    resolution_um: tuple[float, float, float],
    mask_files: Sequence[Path] = (),
    pointcloud_settings: PointCloudExtractionSettings | None = None,
    deduplication_enabled: bool = True,
) -> ChannelObjectWorkItem:
    """Resolve canonical workflow-01 paths after project input validation."""

    layout = CanonicalLayout(subject.analysis_root)
    cell_body_dir = layout.channel_dir(
        channel.population_id, channel.analysis_lane
    )
    return ChannelObjectWorkItem(
        subject=subject,
        channel=channel,
        signal_id=str(signal_id).strip().casefold(),
        channel_name=str(channel_name).strip(),
        input_type=str(input_type).strip(),
        resolution_um=resolution_um,
        mask_files=tuple(mask_files),
        cell_body_dir=cell_body_dir,
        mask_dir=layout.masks_dir(
            channel.population_id, channel.analysis_lane
        ),
        pointcloud_outputs=layout.pointcloud(
            subject.subject_id,
            channel.population_id,
            channel.analysis_lane,
        ),
        deduplication_outputs=layout.objects(
            subject.subject_id,
            channel.population_id,
            channel.analysis_lane,
            write_edges=channel.deduplication.write_edge_table,
        ),
        pointcloud_settings=pointcloud_settings,
        deduplication_enabled=bool(deduplication_enabled),
    )


WorkItemValidator = Callable[[ChannelObjectWorkItem, ArtifactSet], ArtifactInspection]


@dataclass(frozen=True)
class ChannelObjectPlanItem:
    work_item: ChannelObjectWorkItem
    pointcloud: PolicyDecision
    deduplication: PolicyDecision | None
    pointcloud_is_source: bool

    @property
    def actions(self) -> dict[str, str]:
        actions = {
            "point cloud": (
                "source" if self.pointcloud_is_source else self.pointcloud.action.value
            )
        }
        if self.deduplication is not None:
            actions["deduplication"] = self.deduplication.action.value
        return actions

    @property
    def can_apply(self) -> bool:
        return self.pointcloud.action is not PlannedAction.STOP and (
            self.deduplication is None
            or self.deduplication.action is not PlannedAction.STOP
        )

    @property
    def already_complete(self) -> bool:
        return self.can_apply and self.pointcloud.action is PlannedAction.REUSE and (
            self.deduplication is None
            or self.deduplication.action is PlannedAction.REUSE
        )


@dataclass(frozen=True)
class ChannelObjectPlan:
    items: tuple[ChannelObjectPlanItem, ...]
    unavailable: tuple[str, ...] = ()

    @property
    def pending(self) -> tuple[ChannelObjectPlanItem, ...]:
        return tuple(item for item in self.items if not item.already_complete)

    @property
    def complete(self) -> tuple[ChannelObjectPlanItem, ...]:
        return tuple(item for item in self.items if item.already_complete)

    @property
    def can_apply(self) -> bool:
        return all(item.can_apply for item in self.items)

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        for item in self.items:
            for stage, decision in (
                ("point cloud", item.pointcloud),
                ("deduplication", item.deduplication),
            ):
                if decision is not None and decision.action is PlannedAction.STOP:
                    details = "; ".join(decision.inspection.reasons)
                    suffix = f": {details}" if details else ""
                    reasons.append(
                        f"{item.work_item.subject_id} {stage}: "
                        f"{decision.reason}{suffix}"
                    )
        return tuple(reasons)


def _source_decision(
    inspection: ArtifactInspection,
    policy: ExistingOutputPolicy,
) -> PolicyDecision:
    if inspection.reusable:
        return PolicyDecision(
            inspection,
            policy,
            PlannedAction.REUSE,
            "required source point cloud is validated",
        )
    return PolicyDecision(
        inspection,
        policy,
        PlannedAction.STOP,
        f"required source point cloud is {inspection.state.value}",
    )


def build_channel_object_plan(
    *,
    work_items: Sequence[ChannelObjectWorkItem],
    unavailable: Sequence[str] = (),
    policy: ExistingOutputPolicy | str,
    pointcloud_validator: WorkItemValidator = validate_channel_pointcloud,
    deduplication_validator: WorkItemValidator = validate_channel_objects,
    run_deduplication: bool = True,
) -> ChannelObjectPlan:
    """Plan workflow 01 without loading point rows or writing outputs."""

    policy = ExistingOutputPolicy(policy)
    planned: list[ChannelObjectPlanItem] = []
    for item in work_items:
        pointcloud_inspection = inspect_artifact_set(
            item.pointcloud_outputs,
            validator=lambda artifact, current=item: pointcloud_validator(
                current, artifact
            ),
        )
        pointcloud_is_source = (
            item.channel.source_kind is SourceKind.EXISTING_POINTCLOUD
        )
        if pointcloud_is_source:
            pointcloud_decision = _source_decision(pointcloud_inspection, policy)
        else:
            pointcloud_decision = decide_existing_output(
                pointcloud_inspection, policy
            )

        deduplication_decision: PolicyDecision | None = None
        if run_deduplication and item.deduplication_enabled:
            deduplication_inspection = inspect_artifact_set(
                item.deduplication_outputs,
                validator=lambda artifact, current=item: deduplication_validator(
                    current, artifact
                ),
            )
            deduplication_decision = decide_existing_output(
                deduplication_inspection,
                policy,
                upstream_rebuilt=(
                    not pointcloud_is_source
                    and pointcloud_decision.action
                    in {PlannedAction.WRITE, PlannedAction.REBUILD}
                ),
            )
        planned.append(
            ChannelObjectPlanItem(
                work_item=item,
                pointcloud=pointcloud_decision,
                deduplication=deduplication_decision,
                pointcloud_is_source=pointcloud_is_source,
            )
        )
    return ChannelObjectPlan(tuple(planned), tuple(str(value) for value in unavailable))


def summarize_channel_object_plan(plan: ChannelObjectPlan) -> str:
    """Format one read-only workflow-01 plan for a project entry point."""

    lines = [
        f"Point-cloud preflight: {len(plan.pending)} pending subject(s), "
        f"{len(plan.complete)} already complete, "
        f"{len(plan.unavailable)} unavailable."
    ]
    lines.extend(f"  unavailable: {message}" for message in plan.unavailable)
    if plan.complete:
        lines.append(
            "  already complete (validated; excluded from run queue): "
            + ", ".join(item.work_item.subject_id for item in plan.complete)
        )
    for planned in plan.pending:
        item = planned.work_item
        mask_summary = (
            "masks=n/a"
            if item.channel.source_kind is SourceKind.EXISTING_POINTCLOUD
            else f"masks={len(item.mask_files):,}"
        )
        lines.append(
            f"  pending: {item.subject_id} {item.age}; input={item.input_type}; "
            f"resolution={item.resolution_um}; {mask_summary}; "
            + ", ".join(
                f"{stage}={action}" for stage, action in planned.actions.items()
            )
        )
    return "\n".join(lines)
