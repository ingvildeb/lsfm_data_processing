"""Create atlas hemisphere sources and backfill completed registrations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np

from atlasspace.config.space_models import SpaceDefinition
from atlasspace.image import (
    assert_nifti_header_matches_declared_space,
    build_hemisphere_map,
)
from atlasspace.registration import transform_segmentation_from_registration
from atlasspace.registration.result_manifest import load_registration_result_manifest


class HemisphereAction(str, Enum):
    CREATE_SOURCE = "create_source"
    TRANSFORM = "transform"
    REUSE = "reuse"
    STOP = "stop"


@dataclass(frozen=True)
class HemisphereSourceSpec:
    image_id: str
    reference_image: Path
    output_path: Path
    orientation: str
    resolution_um: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_image", Path(self.reference_image))
        object.__setattr__(self, "output_path", Path(self.output_path))
        if not self.image_id.strip():
            raise ValueError("image_id must not be empty")
        if len(self.resolution_um) != 3 or any(x <= 0 for x in self.resolution_um):
            raise ValueError("resolution_um must contain three positive values")


@dataclass(frozen=True)
class HemisphereSourcePlan:
    spec: HemisphereSourceSpec
    action: HemisphereAction
    reason: str


@dataclass(frozen=True)
class HemisphereBackfillItem:
    registration_dir: Path
    moving_image_id: str
    source_path: Path | None
    annotation_path: Path | None
    output_path: Path | None
    action: HemisphereAction
    reason: str
    replace: bool = False


@dataclass(frozen=True)
class HemisphereBackfillPlan:
    items: tuple[HemisphereBackfillItem, ...]

    @property
    def can_apply(self) -> bool:
        return all(item.action is not HemisphereAction.STOP for item in self.items)


def _resolve(registration_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else registration_dir / path


def _same_path(first: str | Path, second: str | Path) -> bool:
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(
        os.path.abspath(second)
    )


def validate_hemisphere_nifti(
    hemisphere_path: Path,
    reference_path: Path,
    *,
    annotation_path: Path | None = None,
    affine_atol: float = 1e-6,
) -> None:
    """Validate grid and BrainGlobe hemisphere values."""

    hemisphere_image = nib.load(str(hemisphere_path))
    reference_image = nib.load(str(reference_path))
    if tuple(hemisphere_image.shape) != tuple(reference_image.shape):
        raise ValueError(
            f"hemisphere shape differs from reference: {hemisphere_image.shape} "
            f"vs {reference_image.shape}"
        )
    if not np.allclose(
        hemisphere_image.affine, reference_image.affine, rtol=0.0, atol=affine_atol
    ):
        raise ValueError("hemisphere affine differs from reference")
    data = np.asanyarray(hemisphere_image.dataobj)
    if annotation_path is None:
        values = set(int(value) for value in np.unique(data))
    else:
        annotation_image = nib.load(str(annotation_path))
        if tuple(annotation_image.shape) != tuple(reference_image.shape) or not np.allclose(
            annotation_image.affine,
            reference_image.affine,
            rtol=0.0,
            atol=affine_atol,
        ):
            raise ValueError("annotation grid differs from hemisphere reference")
        annotation = np.asanyarray(annotation_image.dataobj)
        values = set(int(value) for value in np.unique(data[annotation != 0]))
    if values != {1, 2}:
        location = "annotated voxels" if annotation_path is not None else "source map"
        raise ValueError(
            f"hemisphere values over {location} must be exactly {{1, 2}}, "
            f"found {sorted(values)}"
        )


def plan_hemisphere_source(
    spec: HemisphereSourceSpec, *, overwrite_invalid: bool = False
) -> HemisphereSourcePlan:
    if not spec.reference_image.is_file():
        return HemisphereSourcePlan(
            spec, HemisphereAction.STOP,
            f"reference image is missing: {spec.reference_image}",
        )
    try:
        reference = nib.load(str(spec.reference_image))
        if len(reference.shape) != 3:
            raise ValueError(f"reference must be 3D, got {reference.shape}")
        assert_nifti_header_matches_declared_space(
            spec.reference_image, spec.orientation, spec.resolution_um
        )
    except Exception as exc:
        return HemisphereSourcePlan(
            spec, HemisphereAction.STOP, f"invalid reference image: {exc}"
        )
    if not spec.output_path.exists():
        return HemisphereSourcePlan(
            spec, HemisphereAction.CREATE_SOURCE, "hemisphere source is absent"
        )
    try:
        validate_hemisphere_nifti(spec.output_path, spec.reference_image)
    except Exception as exc:
        if overwrite_invalid:
            return HemisphereSourcePlan(
                spec, HemisphereAction.CREATE_SOURCE,
                f"existing hemisphere source is invalid and will be replaced: {exc}",
            )
        return HemisphereSourcePlan(
            spec, HemisphereAction.STOP,
            f"existing hemisphere source is invalid: {exc}",
        )
    return HemisphereSourcePlan(
        spec, HemisphereAction.REUSE, "hemisphere source is valid"
    )


def apply_hemisphere_source_plan(plan: HemisphereSourcePlan) -> Path:
    if plan.action is HemisphereAction.STOP:
        raise RuntimeError(plan.reason)
    if plan.action is HemisphereAction.REUSE:
        return plan.spec.output_path
    spec = plan.spec
    reference = nib.load(str(spec.reference_image))
    space = SpaceDefinition(
        space_name=spec.image_id,
        orientation=spec.orientation,
        resolution_um=spec.resolution_um,
        shape=tuple(int(value) for value in reference.shape),
    )
    data = build_hemisphere_map(space)
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1.0, 0.0)
    output = nib.Nifti1Image(data, reference.affine, header=header)
    output.set_qform(reference.get_qform(), code=int(reference.header["qform_code"]))
    output.set_sform(reference.get_sform(), code=int(reference.header["sform_code"]))
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    name = spec.output_path.name
    stem = name[:-7] if name.lower().endswith(".nii.gz") else spec.output_path.stem
    temporary = spec.output_path.with_name(f".{stem}.partial.nii.gz")
    temporary.unlink(missing_ok=True)
    try:
        nib.save(output, str(temporary))
        validate_hemisphere_nifti(temporary, spec.reference_image)
        os.replace(temporary, spec.output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return spec.output_path


def build_hemisphere_backfill_plan(
    registration_dirs: Sequence[Path],
    hemisphere_sources: Mapping[str, Path],
    *,
    annotation_name: str = "labels",
    hemisphere_name: str = "hemispheres",
    repair_invalid: bool = False,
) -> HemisphereBackfillPlan:
    items: list[HemisphereBackfillItem] = []
    for value in registration_dirs:
        registration_dir = Path(value)
        image_id = "unknown"
        source: Path | None = None
        annotation: Path | None = None
        try:
            manifest = load_registration_result_manifest(registration_dir)
            if not manifest.success:
                raise ValueError("registration manifest is not successful")
            image_id = manifest.moving_image.image_id
            source = Path(hemisphere_sources[image_id])
            if not source.is_file():
                raise FileNotFoundError(f"hemisphere source is missing: {source}")
            annotation_value = manifest.transformed_segmentations[annotation_name]
            annotation = _resolve(registration_dir, annotation_value)
            if not annotation.is_file():
                raise FileNotFoundError(f"transformed annotation is missing: {annotation}")
        except Exception as exc:
            items.append(
                HemisphereBackfillItem(
                    registration_dir,
                    image_id,
                    source,
                    annotation,
                    None,
                    HemisphereAction.STOP,
                    str(exc),
                )
            )
            continue

        recorded_source = manifest.moving_image.segmentations.get(hemisphere_name)
        output_value = manifest.transformed_segmentations.get(hemisphere_name)
        output = (
            _resolve(registration_dir, output_value)
            if output_value is not None
            else registration_dir / f"{hemisphere_name}_WarpedSegmentation.nii.gz"
        )
        if recorded_source is not None and not _same_path(recorded_source, source):
            items.append(
                HemisphereBackfillItem(
                    registration_dir, image_id, source, annotation, output,
                    HemisphereAction.STOP,
                    f"manifest associates {hemisphere_name!r} with another source",
                )
            )
            continue
        if output_value is not None and output.is_file():
            try:
                validate_hemisphere_nifti(
                    output, annotation, annotation_path=annotation
                )
            except Exception as exc:
                action = (
                    HemisphereAction.TRANSFORM
                    if repair_invalid
                    else HemisphereAction.STOP
                )
                items.append(
                    HemisphereBackfillItem(
                        registration_dir, image_id, source, annotation, output, action,
                        f"transformed hemisphere map is invalid: {exc}",
                        replace=repair_invalid,
                    )
                )
            else:
                items.append(
                    HemisphereBackfillItem(
                        registration_dir, image_id, source, annotation, output,
                        HemisphereAction.REUSE, "transformed hemisphere map is valid",
                    )
                )
            continue
        items.append(
            HemisphereBackfillItem(
                registration_dir, image_id, source, annotation, output,
                HemisphereAction.TRANSFORM, "hemisphere transform is absent",
            )
        )
    return HemisphereBackfillPlan(tuple(items))


def apply_hemisphere_backfill_plan(
    plan: HemisphereBackfillPlan, *, hemisphere_name: str = "hemispheres"
) -> tuple[Path, ...]:
    if not plan.can_apply:
        failures = [
            f"{item.registration_dir}: {item.reason}"
            for item in plan.items
            if item.action is HemisphereAction.STOP
        ]
        raise RuntimeError("Hemisphere backfill preflight failed:\n" + "\n".join(failures))
    outputs: list[Path] = []
    for item in plan.items:
        if item.output_path is None or item.annotation_path is None:
            raise RuntimeError("backfill plan lacks validated output paths")
        if item.action is HemisphereAction.REUSE:
            outputs.append(item.output_path)
            continue
        assert item.source_path is not None
        output = transform_segmentation_from_registration(
            item.registration_dir, hemisphere_name, item.source_path,
            replace=item.replace,
        )
        validate_hemisphere_nifti(
            output, item.annotation_path, annotation_path=item.annotation_path
        )
        outputs.append(output)
    return tuple(outputs)
