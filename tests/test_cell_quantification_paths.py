from pathlib import Path

import pytest

from lsfm_data_processing.cell_quantification.project_workflow import (
    CanonicalLayout,
    canonical_subject_stem,
)


def test_canonical_channel_artifact_paths() -> None:
    layout = CanonicalLayout(Path("subject") / "_spatialsignal")

    assert layout.registration_dir == Path("subject/_spatialsignal/registration")
    assert layout.detector_input_dir("ch0_aldh", "planes5um") == Path(
        "subject/_spatialsignal/segmentations/ch0_aldh/planes5um"
    )
    assert layout.masks_dir("ch0_aldh", "cell_bodies") == Path(
        "subject/_spatialsignal/segmentations/ch0_aldh/cell_bodies/masks"
    )
    assert layout.pointcloud("IEB-001", "ch0_aldh", "cell_bodies").paths == (
        Path(
            "subject/_spatialsignal/segmentations/ch0_aldh/cell_bodies/"
            "IEB_001_pointcloud.parquet"
        ),
        Path(
            "subject/_spatialsignal/segmentations/ch0_aldh/cell_bodies/"
            "IEB_001_pointcloud_space.json"
        ),
    )
    assert len(
        layout.objects(
            "IEB-001", "ch0_aldh", "cell_bodies", write_edges=True
        ).paths
    ) == 4


def test_canonical_colocalization_quantification_and_map_paths() -> None:
    layout = CanonicalLayout(Path("_sing"))

    evidence = layout.colocalization_evidence(
        "S1", "aldh_sytox", "default"
    ).paths
    assert evidence[0] == Path(
        "_sing/colocalization/aldh_sytox/default/evidence/"
        "S1_detection_matches.parquet"
    )
    assert layout.raw_quantification(
        "S1", "aldh_sytox", "default"
    ).paths[0] == Path(
        "_sing/quantification/aldh_sytox/default/raw/"
        "S1_objects_with_regions.parquet"
    )
    assert layout.subject_count_map("S1", "aldh_sytox", "default").paths[0] == Path(
        "_sing/voxel_maps/aldh_sytox/default/S1_count_map.nii.gz"
    )
    assert layout.reference_count_map(
        "S1", "aldh_sytox", "default"
    ).paths[0] == Path(
        "_sing/voxel_maps/aldh_sytox/default/S1_reference_count_map.nii.gz"
    )


def test_path_tokens_cannot_escape_the_analysis_root() -> None:
    layout = CanonicalLayout(Path("_spatialsignal"))

    with pytest.raises(ValueError, match="safe path token"):
        layout.channel_dir("../outside", "default")


def test_subject_stem_matches_spatialsignal_convention() -> None:
    assert canonical_subject_stem(" IEB-001 / repeat ") == "IEB_001_repeat"
    with pytest.raises(ValueError, match="alphanumeric"):
        canonical_subject_stem("---")

