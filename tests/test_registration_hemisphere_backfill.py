from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from atlasspace.config.space_models import SpaceDefinition
from atlasspace.io.nifti import write_nifti_from_array
from atlasspace.registration.result_manifest import (
    RegistrationManifestImage,
    RegistrationResultManifest,
)
from lsfm_data_processing.registration_and_transforms.project_workflow import (
    apply_hemisphere_backfill_plan,
    apply_hemisphere_source_plan,
    build_hemisphere_backfill_plan,
    plan_hemisphere_source,
)
from lsfm_data_processing.registration_and_transforms.project_workflow.hemisphere_backfill import (
    HemisphereAction,
    HemisphereSourceSpec,
    validate_hemisphere_nifti,
)


def _space(name: str = "template") -> SpaceDefinition:
    return SpaceDefinition(
        space_name=name,
        orientation="lsp",
        resolution_um=(20.0, 20.0, 20.0),
        shape=(5, 4, 3),
    )


def _write(path: Path, data: np.ndarray, space: SpaceDefinition, dtype=None) -> None:
    write_nifti_from_array(data, space, path, dtype=dtype)


def test_source_plan_creates_orientation_aware_exact_grid_map(tmp_path: Path) -> None:
    reference = tmp_path / "template.nii.gz"
    output = tmp_path / "hemispheres.nii.gz"
    space = _space()
    _write(reference, np.zeros(space.shape), space)
    spec = HemisphereSourceSpec(
        image_id="template_p6",
        reference_image=reference,
        output_path=output,
        orientation="lsp",
        resolution_um=(20.0, 20.0, 20.0),
    )

    plan = plan_hemisphere_source(spec)
    assert plan.action is HemisphereAction.CREATE_SOURCE
    apply_hemisphere_source_plan(plan)
    validate_hemisphere_nifti(output, reference)

    data = np.asanyarray(nib.load(str(output)).dataobj)
    assert set(np.unique(data)) == {1, 2}
    assert np.all(data[2, :, :] == 1)
    assert plan_hemisphere_source(spec).action is HemisphereAction.REUSE


def test_invalid_existing_source_is_never_silently_reused(tmp_path: Path) -> None:
    reference = tmp_path / "template.nii.gz"
    output = tmp_path / "hemispheres.nii.gz"
    space = _space()
    _write(reference, np.zeros(space.shape), space)
    _write(output, np.zeros(space.shape), space, dtype=np.uint8)
    spec = HemisphereSourceSpec(
        "template_p6", reference, output, "lsp", (20.0, 20.0, 20.0)
    )

    assert plan_hemisphere_source(spec).action is HemisphereAction.STOP
    assert plan_hemisphere_source(
        spec, overwrite_invalid=True
    ).action is HemisphereAction.CREATE_SOURCE


def _registration(
    registration_dir: Path, source: Path, *, include_hemisphere: bool
) -> None:
    registration_dir.mkdir()
    subject_space = _space("subject")
    template_space = _space("template")
    labels = registration_dir / "labels_WarpedSegmentation.nii.gz"
    _write(labels, np.ones(subject_space.shape), subject_space, dtype=np.uint16)
    transformed = {"labels": labels.name}
    moving_segmentations = {"labels": "template_labels.nii.gz"}
    if include_hemisphere:
        hemispheres = registration_dir / "hemispheres_WarpedSegmentation.nii.gz"
        data = np.ones(subject_space.shape, dtype=np.uint8)
        data[3:, :, :] = 2
        _write(hemispheres, data, subject_space, dtype=np.uint8)
        transformed["hemispheres"] = hemispheres.name
        moving_segmentations["hemispheres"] = str(source)
    manifest = RegistrationResultManifest(
        success=True,
        preset_name="baseline",
        orientation_alignment="none",
        fixed_image=RegistrationManifestImage(
            image_id="subject", image="subject.nii.gz", space=subject_space
        ),
        moving_image=RegistrationManifestImage(
            image_id="template_p6",
            image="template.nii.gz",
            space=template_space,
            segmentations=moving_segmentations,
        ),
        effective_fixed_space=subject_space,
        effective_moving_space=template_space,
        transformed_segmentations=transformed,
    )
    (registration_dir / "registration_result.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def test_backfill_plan_reuses_valid_map_and_plans_missing_transform(
    tmp_path: Path,
) -> None:
    source = tmp_path / "template_hemispheres.nii.gz"
    source_data = np.ones(_space().shape, dtype=np.uint8)
    source_data[3:, :, :] = 2
    _write(source, source_data, _space(), dtype=np.uint8)

    complete = tmp_path / "complete"
    missing = tmp_path / "missing"
    _registration(complete, source, include_hemisphere=True)
    _registration(missing, source, include_hemisphere=False)

    plan = build_hemisphere_backfill_plan(
        (complete, missing), {"template_p6": source}
    )

    assert [item.action for item in plan.items] == [
        HemisphereAction.REUSE,
        HemisphereAction.TRANSFORM,
    ]
    assert plan.can_apply


def test_unknown_template_stops_complete_backfill_before_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "template_hemispheres.nii.gz"
    source_data = np.ones(_space().shape, dtype=np.uint8)
    source_data[3:, :, :] = 2
    _write(source, source_data, _space(), dtype=np.uint8)
    registration = tmp_path / "registration"
    _registration(registration, source, include_hemisphere=False)

    plan = build_hemisphere_backfill_plan((registration,), {})
    assert plan.items[0].action is HemisphereAction.STOP
    assert not plan.can_apply
    with pytest.raises(RuntimeError, match="preflight failed"):
        apply_hemisphere_backfill_plan(plan)
