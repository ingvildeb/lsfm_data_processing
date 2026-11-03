"""Canonical relative paths and stage artifact sets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .artifacts import ArtifactSet


_PATH_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _token(value: str, field: str) -> str:
    value = str(value).strip()
    if not _PATH_TOKEN.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{field} must be one safe path token, got {value!r}")
    return value


def canonical_subject_stem(subject_id: str) -> str:
    """Match SpatialSignal's filesystem-safe subject stem convention."""

    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", str(subject_id).strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ValueError("subject_id must contain an alphanumeric character")
    return normalized


@dataclass(frozen=True)
class CanonicalLayout:
    """Path builder rooted at one subject's configured analysis directory."""

    analysis_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "analysis_root", Path(self.analysis_root))

    @property
    def registration_dir(self) -> Path:
        return self.analysis_root / "registration"

    def detector_input_dir(self, channel_id: str, detector_input_type: str) -> Path:
        return (
            self.analysis_root / "segmentations"
            / _token(channel_id, "channel_id")
            / _token(detector_input_type, "detector_input_type")
        )

    def channel_dir(self, channel_id: str, analysis_lane: str) -> Path:
        return (
            self.analysis_root / "segmentations"
            / _token(channel_id, "channel_id")
            / _token(analysis_lane, "analysis_lane")
        )

    def masks_dir(self, channel_id: str, analysis_lane: str) -> Path:
        return self.channel_dir(channel_id, analysis_lane) / "masks"

    def pointcloud(
        self, subject_id: str, channel_id: str, analysis_lane: str
    ) -> ArtifactSet:
        stem = canonical_subject_stem(subject_id)
        root = self.channel_dir(channel_id, analysis_lane)
        return ArtifactSet(
            "pointcloud",
            (root / f"{stem}_pointcloud.parquet", root / f"{stem}_pointcloud_space.json"),
        )

    def objects(
        self, subject_id: str, channel_id: str, analysis_lane: str, *,
        write_edges: bool = False,
    ) -> ArtifactSet:
        stem = canonical_subject_stem(subject_id)
        root = self.channel_dir(channel_id, analysis_lane)
        paths = [
            root / f"{stem}_objects.parquet",
            root / f"{stem}_objects_space.json",
            root / f"{stem}_object_membership.parquet",
        ]
        if write_edges:
            paths.append(root / f"{stem}_object_edges.parquet")
        return ArtifactSet("objects", tuple(paths))

    def colocalization_dir(self, population_id: str, analysis_lane: str) -> Path:
        return (
            self.analysis_root / "colocalization"
            / _token(population_id, "population_id")
            / _token(analysis_lane, "analysis_lane")
        )

    def colocalized_objects(
        self, subject_id: str, population_id: str, analysis_lane: str
    ) -> ArtifactSet:
        stem = canonical_subject_stem(subject_id)
        root = self.colocalization_dir(population_id, analysis_lane)
        return ArtifactSet(
            "colocalized_objects",
            (root / f"{stem}_objects.parquet", root / f"{stem}_objects_space.json"),
        )

    def colocalization_evidence(
        self, subject_id: str, population_id: str, analysis_lane: str
    ) -> ArtifactSet:
        stem = canonical_subject_stem(subject_id)
        root = self.colocalization_dir(population_id, analysis_lane) / "evidence"
        return ArtifactSet(
            "colocalization_evidence",
            (
                root / f"{stem}_detection_matches.parquet",
                root / f"{stem}_object_relationships.parquet",
                root / f"{stem}_colocalization_metadata.json",
            ),
        )

    def colocalization_qc(
        self, subject_id: str, population_id: str, analysis_lane: str
    ) -> ArtifactSet:
        stem = canonical_subject_stem(subject_id)
        root = self.colocalization_dir(population_id, analysis_lane) / "qc"
        return ArtifactSet(
            "colocalization_qc", (root / f"{stem}_colocalization_qc.xlsx",)
        )

    def raw_quantification_dir(self, population_id: str, analysis_lane: str) -> Path:
        return (
            self.analysis_root / "quantification"
            / _token(population_id, "population_id")
            / _token(analysis_lane, "analysis_lane")
            / "raw"
        )

    def raw_quantification(
        self, subject_id: str, population_id: str, analysis_lane: str
    ) -> ArtifactSet:
        stem = canonical_subject_stem(subject_id)
        root = self.raw_quantification_dir(population_id, analysis_lane)
        return ArtifactSet(
            "raw_quantification",
            (
                root / f"{stem}_objects_with_regions.parquet",
                root / f"{stem}_objects_with_regions_space.json",
                root / f"{stem}_region_summary.csv",
            ),
        )

    def quantification_qc(
        self, subject_id: str, population_id: str, analysis_lane: str
    ) -> ArtifactSet:
        stem = canonical_subject_stem(subject_id)
        root = self.raw_quantification_dir(population_id, analysis_lane)
        return ArtifactSet(
            "quantification_qc", (root / f"{stem}_quantification_qc.png",)
        )

    def voxel_map_dir(self, population_id: str, analysis_lane: str) -> Path:
        return (
            self.analysis_root / "voxel_maps"
            / _token(population_id, "population_id")
            / _token(analysis_lane, "analysis_lane")
        )

    def subject_count_map(
        self, subject_id: str, population_id: str, analysis_lane: str
    ) -> ArtifactSet:
        stem = canonical_subject_stem(subject_id)
        root = self.voxel_map_dir(population_id, analysis_lane)
        return ArtifactSet(
            "subject_count_map",
            (root / f"{stem}_count_map.nii.gz", root / f"{stem}_count_map_space.json"),
        )

    def reference_count_map(
        self, subject_id: str, population_id: str, analysis_lane: str
    ) -> ArtifactSet:
        stem = canonical_subject_stem(subject_id)
        root = self.voxel_map_dir(population_id, analysis_lane)
        return ArtifactSet(
            "reference_count_map",
            (
                root / f"{stem}_reference_count_map.nii.gz",
                root / f"{stem}_reference_count_map_space.json",
            ),
        )
