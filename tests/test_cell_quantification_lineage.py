import json
from pathlib import Path

from lsfm_data_processing.cell_quantification.project_workflow import (
    ArtifactSet,
    ArtifactState,
    DIRECT_INPUT_SIGNATURES_KEY,
    validate_lineage_parameters,
    validate_spatial_sidecar_lineage,
    with_lineage_parameters,
)


def _artifact(tmp_path: Path) -> ArtifactSet:
    return ArtifactSet("objects", (tmp_path / "objects.parquet",))


def test_matching_parameters_and_direct_inputs_are_valid(tmp_path: Path) -> None:
    source = tmp_path / "pointcloud.parquet"
    source.write_bytes(b"points")
    scientific = {"max_plane_offset": 1, "resolution_um": (20.0, 20.0, 20.0)}
    recorded = with_lineage_parameters(scientific, (source,))

    assert set(recorded[DIRECT_INPUT_SIGNATURES_KEY][0]) == {
        "name", "size", "mtime_ns"
    }
    inspection = validate_lineage_parameters(
        _artifact(tmp_path), recorded, scientific, (source,)
    )
    assert inspection.state is ArtifactState.VALID


def test_changed_input_or_scientific_parameter_is_stale(tmp_path: Path) -> None:
    source = tmp_path / "pointcloud.parquet"
    source.write_bytes(b"points")
    recorded = with_lineage_parameters({"distance_um": 3.0}, (source,))

    changed_parameter = validate_lineage_parameters(
        _artifact(tmp_path), recorded, {"distance_um": 4.0}, (source,)
    )
    assert changed_parameter.state is ArtifactState.STALE

    source.write_bytes(b"changed points")
    changed_input = validate_lineage_parameters(
        _artifact(tmp_path), recorded, {"distance_um": 3.0}, (source,)
    )
    assert changed_input.state is ArtifactState.STALE


def test_unsigned_output_is_legacy_but_recorded_disagreement_is_stale(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pointcloud.parquet"
    source.write_bytes(b"points")

    legacy = validate_lineage_parameters(
        _artifact(tmp_path), {"distance_um": 3.0}, {"distance_um": 3.0}, (source,)
    )
    assert legacy.state is ArtifactState.LEGACY

    disagreement = validate_lineage_parameters(
        _artifact(tmp_path), {"distance_um": 2.0}, {"distance_um": 3.0}, (source,)
    )
    assert disagreement.state is ArtifactState.STALE


def test_malformed_signed_lineage_is_invalid(tmp_path: Path) -> None:
    source = tmp_path / "pointcloud.parquet"
    source.write_bytes(b"points")
    recorded = {"distance_um": 3.0, DIRECT_INPUT_SIGNATURES_KEY: "not-a-list"}

    inspection = validate_lineage_parameters(
        _artifact(tmp_path), recorded, {"distance_um": 3.0}, (source,)
    )
    assert inspection.state is ArtifactState.INVALID


def test_spatialsignal_sidecar_parameters_are_read_without_schema_changes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pointcloud.parquet"
    source.write_bytes(b"points")
    parameters = with_lineage_parameters({"distance_um": 3.0}, (source,))
    sidecar = tmp_path / "objects_space.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_name": "spatialsignal.dataset_metadata",
                "processing": {
                    "stage": "deduplicate_across_planes",
                    "parameters": parameters,
                },
            }
        ),
        encoding="utf-8",
    )

    inspection = validate_spatial_sidecar_lineage(
        _artifact(tmp_path), sidecar, {"distance_um": 3.0}, (source,)
    )
    assert inspection.state is ArtifactState.VALID


def test_non_object_sidecar_is_invalid(tmp_path: Path) -> None:
    sidecar = tmp_path / "objects_space.json"
    sidecar.write_text("[]", encoding="utf-8")
    inspection = validate_spatial_sidecar_lineage(
        _artifact(tmp_path), sidecar, {}, ()
    )
    assert inspection.state is ArtifactState.INVALID
