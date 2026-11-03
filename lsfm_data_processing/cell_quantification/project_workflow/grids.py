"""Exact NIfTI-grid validation for canonical subject and reference maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.filebasedimages import ImageFileError

from .artifacts import ArtifactInspection, ArtifactSet, ArtifactState


@dataclass(frozen=True)
class NiftiGrid:
    shape: tuple[int, int, int]
    affine: tuple[tuple[float, float, float, float], ...]

    @classmethod
    def from_path(cls, path: Path) -> "NiftiGrid":
        image = nib.load(str(path))
        if len(image.shape) != 3:
            raise ValueError(f"expected a 3D NIfTI grid, got shape {image.shape}")
        affine = tuple(tuple(float(value) for value in row) for row in image.affine)
        return cls(tuple(int(value) for value in image.shape), affine)


def grids_match(
    actual: NiftiGrid, reference: NiftiGrid, *, affine_atol: float = 1e-6
) -> bool:
    """Return whether two volumes occupy the same voxel grid."""

    return actual.shape == reference.shape and bool(
        np.allclose(
            np.asarray(actual.affine),
            np.asarray(reference.affine),
            rtol=0.0,
            atol=affine_atol,
        )
    )


def validate_nifti_grid(
    artifact_set: ArtifactSet,
    output_path: Path,
    reference_path: Path,
    *,
    affine_atol: float = 1e-6,
) -> ArtifactInspection:
    """Validate an output map against its declared annotation grid."""

    try:
        actual = NiftiGrid.from_path(output_path)
        reference = NiftiGrid.from_path(reference_path)
    except (ImageFileError, OSError, ValueError) as exc:
        return ArtifactInspection(
            artifact_set, ArtifactState.INVALID, (f"invalid NIfTI grid: {exc}",)
        )
    if not grids_match(actual, reference, affine_atol=affine_atol):
        return ArtifactInspection(
            artifact_set,
            ArtifactState.INVALID,
            (
                "output grid differs from its declared annotation grid: "
                f"output_shape={actual.shape}; reference_shape={reference.shape}",
            ),
        )
    return ArtifactInspection(artifact_set, ArtifactState.VALID)

