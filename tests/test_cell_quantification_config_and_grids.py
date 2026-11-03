from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from lsfm_data_processing.cell_quantification.project_workflow import (
    ArtifactSet,
    ArtifactState,
    ChannelDefinition,
    ColocalizationDefinition,
    DeduplicationSettings,
    NiftiGrid,
    QuantificationSettings,
    SourceKind,
    VoxelizationSettings,
    WorkflowConfiguration,
    grids_match,
    validate_nifti_grid,
)


def _channel(population_id: str) -> ChannelDefinition:
    return ChannelDefinition(
        population_id=population_id,
        source_kind=SourceKind.EXISTING_POINTCLOUD,
        analysis_lane="mips20um",
        deduplication=DeduplicationSettings(
            max_plane_offset=1, max_xy_distance_um=3.0, max_n_planes=3
        ),
    )


def test_configuration_supports_channels_and_colocalized_populations() -> None:
    configuration = WorkflowConfiguration(
        channels=(_channel("ch2_sytox"), _channel("ch0_aldh")),
        colocalizations=(
            ColocalizationDefinition(
                population_id="aldh_sytox",
                channel_a="ch0_aldh",
                channel_b="ch2_sytox",
                analysis_lane="default",
                max_distance_um=5.0,
                matching_rule="one_to_one",
                component_rule="both_channels",
                coordinate_rule="midpoint",
            ),
        ),
    )

    assert configuration.enabled_population_ids == (
        "ch2_sytox", "ch0_aldh", "aldh_sytox"
    )
    assert configuration.quantification.hemispheres_required
    assert configuration.voxelization.reference_modulation == "none"


def test_colocalization_rejects_unknown_channels() -> None:
    with pytest.raises(ValueError, match="unknown channels"):
        WorkflowConfiguration(
            channels=(_channel("ch2_sytox"),),
            colocalizations=(
                ColocalizationDefinition(
                    population_id="aldh_sytox",
                    channel_a="ch0_aldh",
                    channel_b="ch2_sytox",
                    analysis_lane="default",
                    max_distance_um=5.0,
                    matching_rule="one_to_one",
                    component_rule="both_channels",
                    coordinate_rule="midpoint",
                ),
            ),
        )


def test_quantification_requires_hemispheres() -> None:
    with pytest.raises(ValueError, match="hemispheres_required"):
        QuantificationSettings(hemispheres_required=False)


def test_routine_reference_maps_are_unmodulated_unit_counts() -> None:
    with pytest.raises(ValueError, match="unmodulated"):
        VoxelizationSettings(reference_modulation="jacobian")
    with pytest.raises(ValueError, match="weight 1"):
        VoxelizationSettings(point_weight=0.5)


def test_grid_comparison_requires_shape_and_affine_agreement() -> None:
    identity = tuple(tuple(float(value) for value in row) for row in np.eye(4))
    shifted_array = np.eye(4)
    shifted_array[0, 3] = 0.01
    shifted = tuple(tuple(float(value) for value in row) for row in shifted_array)

    reference = NiftiGrid((4, 5, 6), identity)
    assert grids_match(reference, NiftiGrid((4, 5, 6), identity))
    assert not grids_match(reference, NiftiGrid((5, 5, 6), identity))
    assert not grids_match(reference, NiftiGrid((4, 5, 6), shifted))


def test_nifti_output_is_validated_against_exact_annotation_grid(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "labels.nii.gz"
    output_path = tmp_path / "count_map.nii.gz"
    sidecar_path = tmp_path / "count_map_space.json"
    affine = np.diag([0.02, 0.02, 0.02, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6)), affine), reference_path)
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6)), affine), output_path)
    artifact_set = ArtifactSet("subject_count_map", (output_path, sidecar_path))

    valid = validate_nifti_grid(artifact_set, output_path, reference_path)
    assert valid.state is ArtifactState.VALID

    shifted = affine.copy()
    shifted[0, 3] = 0.02
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6)), shifted), output_path)
    invalid = validate_nifti_grid(artifact_set, output_path, reference_path)
    assert invalid.state is ArtifactState.INVALID
