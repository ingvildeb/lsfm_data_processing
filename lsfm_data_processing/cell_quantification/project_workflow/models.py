"""Small, format-neutral configuration model for shared workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Mapping


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _identifier(value: str, field_name: str) -> str:
    value = str(value).strip()
    if not _IDENTIFIER.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{field_name} must be a stable safe identifier")
    return value


class SourceKind(str, Enum):
    INSTANCE_MASKS = "instance_masks"
    EXISTING_POINTCLOUD = "existing_pointcloud"


@dataclass(frozen=True)
class PointCloudExtractionSettings:
    """Scientific settings that define a mask-derived point cloud."""

    space_name: str
    orientation: str
    indexing: str
    mask_prefix: str

    def __post_init__(self) -> None:
        if not str(self.space_name).strip():
            raise ValueError("space_name must not be empty")
        if not str(self.orientation).strip():
            raise ValueError("orientation must not be empty")
        if self.indexing not in {"zero_based", "one_based"}:
            raise ValueError("indexing must be zero_based or one_based")
        if not str(self.mask_prefix):
            raise ValueError("mask_prefix must not be empty")


@dataclass(frozen=True)
class DeduplicationSettings:
    max_plane_offset: int
    max_xy_distance_um: float
    max_n_planes: int | None = None
    write_edge_table: bool = False

    def __post_init__(self) -> None:
        if self.max_plane_offset < 1:
            raise ValueError("max_plane_offset must be positive")
        if self.max_xy_distance_um <= 0:
            raise ValueError("max_xy_distance_um must be positive")
        if self.max_n_planes is not None and self.max_n_planes < 1:
            raise ValueError("max_n_planes must be positive when provided")


@dataclass(frozen=True)
class ChannelDefinition:
    population_id: str
    source_kind: SourceKind | str
    analysis_lane: str
    deduplication: DeduplicationSettings
    enabled: bool = True
    morphology_filter: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "population_id", _identifier(self.population_id, "population_id")
        )
        object.__setattr__(
            self, "analysis_lane", _identifier(self.analysis_lane, "analysis_lane")
        )
        object.__setattr__(self, "source_kind", SourceKind(self.source_kind))
        if self.morphology_filter is not None:
            object.__setattr__(self, "morphology_filter", dict(self.morphology_filter))


@dataclass(frozen=True)
class ColocalizationDefinition:
    population_id: str
    channel_a: str
    channel_b: str
    analysis_lane: str
    max_distance_um: float
    matching_rule: str
    component_rule: str
    coordinate_rule: str
    enabled: bool = True

    def __post_init__(self) -> None:
        for name in ("population_id", "channel_a", "channel_b", "analysis_lane"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.channel_a == self.channel_b:
            raise ValueError("a colocalization requires two different channels")
        if self.max_distance_um <= 0:
            raise ValueError("max_distance_um must be positive")
        for name in ("matching_rule", "component_rule", "coordinate_rule"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True)
class QuantificationSettings:
    annotation_name: str = "labels"
    hemisphere_name: str = "hemispheres"
    atlas_preset: str = "allen_ccfv3"
    region_id_space: str = "kimlab16bit"
    hemispheres_required: bool = True
    include_background: bool = False
    include_out_of_bounds: bool = False

    def __post_init__(self) -> None:
        for name in (
            "annotation_name", "hemisphere_name", "atlas_preset", "region_id_space"
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        if not self.hemispheres_required:
            raise ValueError("hemispheres_required must be true for this workflow")


@dataclass(frozen=True)
class VoxelizationSettings:
    subject_map_enabled: bool = True
    reference_map_enabled: bool = True
    reference_modulation: str = "none"
    point_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.reference_modulation != "none":
            raise ValueError("routine reference count maps must be unmodulated")
        if self.point_weight != 1.0:
            raise ValueError("routine count maps require weight 1 per object")


@dataclass(frozen=True)
class WorkflowConfiguration:
    channels: tuple[ChannelDefinition, ...]
    colocalizations: tuple[ColocalizationDefinition, ...] = ()
    quantification: QuantificationSettings = field(
        default_factory=QuantificationSettings
    )
    voxelization: VoxelizationSettings = field(default_factory=VoxelizationSettings)

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("at least one channel must be configured")
        channel_ids = [item.population_id for item in self.channels]
        coloc_ids = [item.population_id for item in self.colocalizations]
        all_ids = channel_ids + coloc_ids
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("channel and colocalized population IDs must be unique")
        known_channels = set(channel_ids)
        for item in self.colocalizations:
            unknown = {item.channel_a, item.channel_b}.difference(known_channels)
            if unknown:
                raise ValueError(
                    f"colocalization {item.population_id!r} references unknown "
                    f"channels: {sorted(unknown)}"
                )

    @property
    def enabled_population_ids(self) -> tuple[str, ...]:
        return tuple(
            [item.population_id for item in self.channels if item.enabled]
            + [item.population_id for item in self.colocalizations if item.enabled]
        )

