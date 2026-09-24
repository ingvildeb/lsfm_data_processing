"""Resumable preparation of subject-specific masks for registration rescues."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from typing import Iterable

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from atlasspace.config.image_models import ImageConfig
from atlasspace.config.space_models import SpaceDefinition
from atlasspace.image import resample_image_to_resolution, segmentation_to_binary_mask
from atlasspace.image.reorientation import reorient_array_to_match
from atlasspace.registration import load_registration_result_manifest

from .models import (
    RescueRequestPlan,
    RescueRequestKey,
    RescueRequestStatus,
    RescueStrategyKind,
    ResolvedRescueRequest,
)


@dataclass(frozen=True)
class MaskingSettings:
    review_resolution_um: float = 50.0
    source_segmentation_filename: str = "labels_WarpedSegmentation.nii.gz"
    source_fixed_filename: str = "fixed_normalized_for_registration.nii.gz"
    source_warped_filename: str = "ANTsPy_Warped.nii.gz"
    draft_mask_filename: str = "brain_mask_50um_draft.nii.gz"
    review_fixed_filename: str = "fixed_image_50um_reference.nii.gz"
    review_warped_filename: str = "warped_template_50um_reference.nii.gz"
    completed_mask_filename: str = "brain_mask_50um_complete.nii.gz"
    native_mask_filename: str = "brain_mask_native_applied.nii.gz"
    masked_fixed_filename: str = "fixed_image_native_masked.nii.gz"
    provenance_filename: str = "masking_provenance.json"

    def __post_init__(self) -> None:
        if self.review_resolution_um <= 0:
            raise ValueError("Mask review resolution must be positive")


@dataclass(frozen=True)
class MaskingPlan:
    request: ResolvedRescueRequest
    subject_dir: Path
    source_run_dir: Path
    source_segmentation: Path
    source_fixed_image: Path
    source_warped_image: Path
    original_fixed_image: Path
    output_dir: Path
    draft_mask: Path
    review_fixed_image: Path
    review_warped_image: Path
    completed_mask: Path
    native_mask: Path
    masked_fixed_image: Path
    provenance_path: Path
    state: RescueRequestStatus
    action: str

    @property
    def subject_id(self) -> str:
        return self.request.request.subject_id

    @property
    def source_run(self) -> str:
        return self.request.request.source_run


def build_masking_plans(
    request_plan: RescueRequestPlan,
    *,
    subjects_root: Path | str,
    original_fixed_filename: str,
    settings: MaskingSettings = MaskingSettings(),
) -> tuple[MaskingPlan, ...]:
    """Build masking actions without changing files."""

    root = Path(subjects_root)
    plans: list[MaskingPlan] = []
    errors: list[str] = []
    active_by_subject: dict[str, list[ResolvedRescueRequest]] = {}
    for item in request_plan.active:
        active_by_subject.setdefault(item.request.subject_id, []).append(item)

    terminal = {
        RescueRequestStatus.GENERATED,
        RescueRequestStatus.COMPLETED,
        RescueRequestStatus.FAILED,
    }
    for item in request_plan.active:
        if item.strategy is None or item.strategy.kind is not RescueStrategyKind.MASKING:
            continue
        if item.status in terminal:
            continue
        subject_id = item.request.subject_id
        unfinished_ordinary = [
            other
            for other in active_by_subject.get(subject_id, ())
            if other.strategy is not None
            and other.strategy.kind is not RescueStrategyKind.MASKING
            and other.status not in {
                RescueRequestStatus.COMPLETED,
                RescueRequestStatus.FAILED,
            }
        ]
        if unfinished_ordinary:
            plans.append(
                _placeholder_plan(
                    item,
                    root=root,
                    original_fixed_filename=original_fixed_filename,
                    settings=settings,
                    state=RescueRequestStatus.BLOCKED_PENDING_RESCUES,
                    action="blocked",
                )
            )
            continue
        try:
            plan = _build_subject_masking_plan(
                item,
                root=root,
                original_fixed_filename=original_fixed_filename,
                settings=settings,
            )
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{subject_id}: {exc}")
        else:
            plans.append(plan)
    if errors:
        raise ValueError("Invalid masking rescue requests:\n" + "\n".join(errors))
    return tuple(plans)


def apply_masking_plans(
    plans: Iterable[MaskingPlan],
    *,
    settings: MaskingSettings = MaskingSettings(),
) -> None:
    """Apply draft or completed-mask actions without replacing user files."""

    for plan in plans:
        if plan.action in {"blocked", "await"}:
            continue
        if plan.action == "create_review":
            _create_review_artifacts(plan, settings=settings)
        elif plan.action == "apply_completed":
            _apply_completed_mask(plan, settings=settings)
        elif plan.action != "ready":
            raise ValueError(f"Unsupported masking action: {plan.action!r}")


def masking_statuses(
    plans: Iterable[MaskingPlan],
) -> dict[RescueRequestKey, RescueRequestStatus]:
    return {plan.request.key: plan.state for plan in plans}


def summarize_masking_plans(plans: Iterable[MaskingPlan]) -> str:
    rows = tuple(plans)
    if not rows:
        return "Mask-assisted rescues: no active masking requests."
    lines = [f"Mask-assisted rescues: {len(rows)} subject(s)"]
    for plan in rows:
        lines.append(
            f"  {plan.subject_id}: {plan.source_run} -> {plan.output_dir} "
            f"[{plan.state.value}; {plan.action}]"
        )
    return "\n".join(lines)


def _subject_dir(root: Path, subject_id: str, source_run: str) -> Path:
    candidates = (root / subject_id, root / f"{subject_id}_transferred")
    matches = [path for path in candidates if (path / source_run).is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one subject directory containing {source_run!r}; "
            f"found {matches}"
        )
    return matches[0]


def _output_dir(subject_dir: Path, source_run: str) -> Path:
    source = Path(source_run)
    try:
        relative = source.relative_to("registration_runs")
    except ValueError as exc:
        raise ValueError(
            f"masking source run must be under registration_runs/: {source_run!r}"
        ) from exc
    return subject_dir / "registration_masks" / relative


def _placeholder_plan(
    item: ResolvedRescueRequest,
    *,
    root: Path,
    original_fixed_filename: str,
    settings: MaskingSettings,
    state: RescueRequestStatus,
    action: str,
) -> MaskingPlan:
    subject_dir = root / item.request.subject_id
    source_dir = subject_dir / item.request.source_run
    output = _output_dir(subject_dir, item.request.source_run)
    return _plan_paths(
        item,
        subject_dir,
        source_dir,
        subject_dir / original_fixed_filename,
        output,
        settings,
        state,
        action,
    )


def _build_subject_masking_plan(
    item: ResolvedRescueRequest,
    *,
    root: Path,
    original_fixed_filename: str,
    settings: MaskingSettings,
) -> MaskingPlan:
    subject_dir = _subject_dir(root, item.request.subject_id, item.request.source_run)
    source_dir = subject_dir / item.request.source_run
    source_segmentation = source_dir / settings.source_segmentation_filename
    source_fixed = source_dir / settings.source_fixed_filename
    original_fixed = subject_dir / original_fixed_filename
    for path, label in (
        (source_segmentation, "source segmentation"),
        (source_fixed, "normalized fixed image"),
        (original_fixed, "original fixed image"),
        (source_dir / "registration_result.json", "registration manifest"),
        (source_dir / "registration_parameters.yaml", "parameter snapshot"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")

    _validate_source_grid(source_segmentation, source_fixed)
    output = _output_dir(subject_dir, item.request.source_run)
    plan = _plan_paths(
        item,
        subject_dir,
        source_dir,
        original_fixed,
        output,
        settings,
        RescueRequestStatus.AWAITING_MANUAL_MASK,
        "create_review",
    )
    review_exists = plan.review_fixed_image.is_file()
    draft_exists = plan.draft_mask.is_file()
    if review_exists != draft_exists:
        raise ValueError(
            f"mask review artifacts are incomplete in {output}; expected both "
            f"{plan.draft_mask.name} and {plan.review_fixed_image.name}"
        )
    if plan.completed_mask.is_file():
        _validate_completed_mask(plan)
        derived = (plan.native_mask.is_file(), plan.masked_fixed_image.is_file())
        if derived[0] != derived[1]:
            raise ValueError(
                f"native masking outputs are incomplete in {output}; expected both "
                f"{plan.native_mask.name} and {plan.masked_fixed_image.name}"
            )
        return _replace_plan_state(
            plan,
            RescueRequestStatus.MASK_READY,
            "ready" if all(derived) else "apply_completed",
        )
    if review_exists:
        _validate_review_pair(plan)
        return _replace_plan_state(
            plan, RescueRequestStatus.AWAITING_MANUAL_MASK, "await"
        )
    _validate_warped_review_source(plan)
    return plan


def _plan_paths(
    item: ResolvedRescueRequest,
    subject_dir: Path,
    source_dir: Path,
    original_fixed: Path,
    output: Path,
    settings: MaskingSettings,
    state: RescueRequestStatus,
    action: str,
) -> MaskingPlan:
    return MaskingPlan(
        request=item,
        subject_dir=subject_dir,
        source_run_dir=source_dir,
        source_segmentation=source_dir / settings.source_segmentation_filename,
        source_fixed_image=source_dir / settings.source_fixed_filename,
        source_warped_image=source_dir / settings.source_warped_filename,
        original_fixed_image=original_fixed,
        output_dir=output,
        draft_mask=output / settings.draft_mask_filename,
        review_fixed_image=output / settings.review_fixed_filename,
        review_warped_image=output / settings.review_warped_filename,
        completed_mask=output / settings.completed_mask_filename,
        native_mask=output / settings.native_mask_filename,
        masked_fixed_image=output / settings.masked_fixed_filename,
        provenance_path=output / settings.provenance_filename,
        state=state,
        action=action,
    )


def _replace_plan_state(
    plan: MaskingPlan, state: RescueRequestStatus, action: str
) -> MaskingPlan:
    return replace(plan, state=state, action=action)


def _validate_source_grid(segmentation: Path, fixed: Path) -> None:
    segmentation_nifti = nib.load(str(segmentation))
    fixed_nifti = nib.load(str(fixed))
    if segmentation_nifti.shape != fixed_nifti.shape or not np.allclose(
        segmentation_nifti.affine, fixed_nifti.affine, atol=1e-5, rtol=1e-5
    ):
        raise ValueError(
            "warped labels and normalized fixed image are not on the same grid: "
            f"{segmentation_nifti.shape} vs {fixed_nifti.shape}"
        )


def _create_review_artifacts(plan: MaskingPlan, *, settings: MaskingSettings) -> None:
    _validate_warped_review_source(plan)
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_registration_result_manifest(plan.source_run_dir)
    effective_space = manifest.effective_fixed_space
    segmentation = ImageConfig(
        image_id=f"{plan.subject_id}_warped_labels",
        image=plan.source_segmentation,
        space=effective_space,
    )
    fixed = ImageConfig(
        image_id=f"{plan.subject_id}_fixed_normalized",
        image=plan.source_fixed_image,
        space=effective_space,
    )
    warped = ImageConfig(
        image_id=f"{plan.subject_id}_warped_template",
        image=plan.source_warped_image,
        space=effective_space,
    )
    with tempfile.TemporaryDirectory(dir=plan.output_dir) as temporary_dir:
        temporary = Path(temporary_dir)
        native_mask = temporary / "brain_mask_native.nii.gz"
        draft = temporary / settings.draft_mask_filename
        reference = temporary / settings.review_fixed_filename
        warped_reference = temporary / settings.review_warped_filename
        native_config = segmentation_to_binary_mask(segmentation, native_mask)
        target = (settings.review_resolution_um,) * 3
        resample_image_to_resolution(
            native_config,
            draft,
            target_resolution_um=target,
            interpolation="nearest",
        )
        resample_image_to_resolution(
            fixed,
            reference,
            target_resolution_um=target,
            interpolation="linear",
        )
        resample_image_to_resolution(
            warped,
            warped_reference,
            target_resolution_um=target,
            interpolation="linear",
        )
        _validate_binary_nonempty(draft, "draft mask")
        _validate_same_grid(draft, reference, "draft mask", "review fixed image")
        _validate_same_grid(
            warped_reference, reference, "warped template", "review fixed image"
        )
        _install_new_file(draft, plan.draft_mask)
        _install_new_file(reference, plan.review_fixed_image)
        _install_new_file(warped_reference, plan.review_warped_image)
    _write_provenance(plan, settings=settings, stage="awaiting_manual_mask")


def _validate_warped_review_source(plan: MaskingPlan) -> None:
    if not plan.source_warped_image.is_file():
        raise FileNotFoundError(f"warped template image is missing: {plan.source_warped_image}")
    _validate_source_grid(plan.source_warped_image, plan.source_fixed_image)


def _validate_review_pair(plan: MaskingPlan) -> None:
    _validate_binary_nonempty(plan.draft_mask, "draft mask")
    _validate_same_grid(
        plan.draft_mask,
        plan.review_fixed_image,
        "draft mask",
        "review fixed image",
    )


def _validate_completed_mask(plan: MaskingPlan) -> None:
    if not plan.review_fixed_image.is_file():
        raise ValueError(
            f"completed mask exists without review fixed image: {plan.review_fixed_image}"
        )
    _validate_binary_nonempty(plan.completed_mask, "completed mask")
    _validate_same_grid(
        plan.completed_mask,
        plan.review_fixed_image,
        "completed mask",
        "review fixed image",
    )


def _apply_completed_mask(plan: MaskingPlan, *, settings: MaskingSettings) -> None:
    _validate_completed_mask(plan)
    manifest = load_registration_result_manifest(plan.source_run_dir)
    completed = nib.load(str(plan.completed_mask))
    effective_fixed = nib.load(str(plan.source_fixed_image))
    resampled = resample_from_to(
        completed,
        (effective_fixed.shape, effective_fixed.affine),
        order=0,
    )
    effective_mask = (np.asanyarray(resampled.dataobj) > 0).astype(np.uint8)
    effective_space = manifest.effective_fixed_space.model_copy(
        update={"shape": tuple(int(value) for value in effective_mask.shape)}
    )
    original = nib.load(str(plan.original_fixed_image))
    declared_space = manifest.fixed_image.space.model_copy(
        update={"shape": tuple(int(value) for value in original.shape)}
    )
    native_mask, native_space = reorient_array_to_match(
        effective_mask,
        effective_space,
        declared_space,
    )
    if tuple(native_mask.shape) != tuple(original.shape):
        raise ValueError(
            "completed mask does not map back to the original fixed-image grid: "
            f"{native_mask.shape} vs {original.shape}"
        )
    native_mask = (native_mask > 0).astype(np.uint8)
    if not np.any(native_mask):
        raise ValueError("completed mask became empty on the original fixed-image grid")
    original_data = np.asanyarray(original.dataobj)
    masked = np.where(native_mask != 0, original_data, 0).astype(
        original.get_data_dtype(), copy=False
    )
    with tempfile.TemporaryDirectory(dir=plan.output_dir) as temporary_dir:
        temporary = Path(temporary_dir)
        native_path = temporary / settings.native_mask_filename
        masked_path = temporary / settings.masked_fixed_filename
        _save_like_original(native_mask, original, native_path, dtype=np.uint8)
        _save_like_original(masked, original, masked_path, dtype=original.get_data_dtype())
        _install_new_file(native_path, plan.native_mask)
        _install_new_file(masked_path, plan.masked_fixed_image)
    _write_provenance(
        plan,
        settings=settings,
        stage="mask_ready",
        extra={
            "completed_mask_sha256": _file_sha256(plan.completed_mask),
            "native_orientation": native_space.orientation,
            "native_nonzero_voxels": int(np.count_nonzero(native_mask)),
        },
    )


def _validate_binary_nonempty(path: Path, label: str) -> None:
    data = np.asanyarray(nib.load(str(path)).dataobj)
    values = set(np.unique(data).tolist())
    if not values.issubset({0, 1}) or not np.any(data):
        raise ValueError(f"{label} is not a nonempty binary mask: {sorted(values)}")


def _validate_same_grid(first: Path, second: Path, first_label: str, second_label: str) -> None:
    first_nifti = nib.load(str(first))
    second_nifti = nib.load(str(second))
    if first_nifti.shape != second_nifti.shape or not np.allclose(
        first_nifti.affine, second_nifti.affine, atol=1e-5, rtol=1e-5
    ):
        raise ValueError(
            f"{first_label} and {second_label} are not on the same grid: "
            f"{first_nifti.shape} vs {second_nifti.shape}"
        )


def _save_like_original(array, original, destination: Path, *, dtype) -> None:
    header = original.header.copy()
    header.set_data_dtype(dtype)
    nib.save(nib.Nifti1Image(array, original.affine, header), str(destination))


def _install_new_file(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Masking output already exists and was not replaced: {destination}")
    source.replace(destination)


def _write_provenance(
    plan: MaskingPlan,
    *,
    settings: MaskingSettings,
    stage: str,
    extra: dict[str, object] | None = None,
) -> None:
    content = {
        "schema_version": 1,
        "operation": "mask_assisted_registration_rescue",
        "stage": stage,
        "subject_id": plan.subject_id,
        "source_run": plan.source_run,
        "source_segmentation": str(plan.source_segmentation),
        "source_fixed_image": str(plan.source_fixed_image),
        "original_fixed_image": str(plan.original_fixed_image),
        "review_resolution_um": settings.review_resolution_um,
        "draft_mask": str(plan.draft_mask),
        "review_fixed_image": str(plan.review_fixed_image),
        "completed_mask": str(plan.completed_mask),
        "native_mask": str(plan.native_mask),
        "masked_fixed_image": str(plan.masked_fixed_image),
        **(extra or {}),
    }
    encoded = (json.dumps(content, indent=2, sort_keys=True) + "\n").encode("utf-8")
    plan.provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=plan.provenance_path.parent,
        prefix=f".{plan.provenance_path.stem}_",
        suffix=plan.provenance_path.suffix,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
    temporary.replace(plan.provenance_path)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
