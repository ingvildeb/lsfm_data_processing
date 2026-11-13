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


def test_new_parameters_do_not_write_custom_file_signatures(tmp_path: Path) -> None:
    scientific = {"max_plane_offset": 1, "resolution_um": (20.0, 20.0, 20.0)}

    # A missing direct input is intentional: parameter construction must no
    # longer stat inputs or depend on filesystem metadata.
    recorded = with_lineage_parameters(
        scientific,
        (tmp_path / "input-that-does-not-exist.parquet",),
    )

    assert recorded == {
        "max_plane_offset": 1,
        "resolution_um": [20.0, 20.0, 20.0],
    }
    assert DIRECT_INPUT_SIGNATURES_KEY not in recorded


def test_matching_parameters_without_formal_inputs_are_no_lineage(
    tmp_path: Path,
) -> None:
    inspection = validate_lineage_parameters(
        _artifact(tmp_path),
        {"distance_um": 3.0},
        {"distance_um": 3.0},
        (tmp_path / "pointcloud.parquet",),
    )

    assert inspection.state is ArtifactState.NO_LINEAGE
    assert inspection.reusable


def test_recorded_scientific_parameter_disagreement_is_stale(
    tmp_path: Path,
) -> None:
    inspection = validate_lineage_parameters(
        _artifact(tmp_path),
        {"distance_um": 2.0},
        {"distance_um": 3.0},
        (),
    )

    assert inspection.state is ArtifactState.STALE


def test_historical_file_signatures_are_ignored(tmp_path: Path) -> None:
    source = tmp_path / "pointcloud.parquet"
    source.write_bytes(b"replacement with different size and timestamp")
    recorded = {
        "distance_um": 3.0,
        DIRECT_INPUT_SIGNATURES_KEY: [
            {"name": source.name, "size": 6, "mtime_ns": 1}
        ],
    }

    inspection = validate_lineage_parameters(
        _artifact(tmp_path), recorded, {"distance_um": 3.0}, (source,)
    )

    assert inspection.state is ArtifactState.NO_LINEAGE


def test_malformed_historical_signature_does_not_invalidate_artifact(
    tmp_path: Path,
) -> None:
    recorded = {
        "distance_um": 3.0,
        DIRECT_INPUT_SIGNATURES_KEY: "historical custom value",
    }

    inspection = validate_lineage_parameters(
        _artifact(tmp_path), recorded, {"distance_um": 3.0}, ()
    )

    assert inspection.state is ArtifactState.NO_LINEAGE


def test_spatialsignal_sidecar_parameters_are_read_without_schema_changes(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "objects_space.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_name": "spatialsignal.dataset_metadata",
                "processing": {
                    "stage": "deduplicate_across_planes",
                    "parameters": {"distance_um": 3.0},
                },
            }
        ),
        encoding="utf-8",
    )

    inspection = validate_spatial_sidecar_lineage(
        _artifact(tmp_path), sidecar, {"distance_um": 3.0}, ()
    )
    assert inspection.state is ArtifactState.NO_LINEAGE


def test_non_object_sidecar_is_invalid(tmp_path: Path) -> None:
    sidecar = tmp_path / "objects_space.json"
    sidecar.write_text("[]", encoding="utf-8")
    inspection = validate_spatial_sidecar_lineage(
        _artifact(tmp_path), sidecar, {}, ()
    )
    assert inspection.state is ArtifactState.INVALID
