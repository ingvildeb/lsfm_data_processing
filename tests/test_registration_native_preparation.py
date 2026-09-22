from __future__ import annotations

import json
from pathlib import Path

import pytest

from lsfm_data_processing.registration_and_transforms.project_workflow.native_preparation import (
    NativeImageSpec,
    apply_native_preparation_plan,
    build_native_preparation_plan,
)


def _spec(tmp_path: Path, *, staged: bool = True) -> NativeImageSpec:
    source = tmp_path / "source.nii.gz"
    source.write_bytes(b"image")
    native_dir = tmp_path / "native_20um"
    return NativeImageSpec(
        subject_id="A",
        channel="ch1",
        source_path=source,
        native_path=native_dir / "ch1_native20um.nii.gz",
        orientation="LAS",
        resolution_um=20,
        provenance_path=native_dir / "ch1_provenance.json",
        header_validation_path=native_dir / "ch1_header_validation.json",
        staged_path=tmp_path / "batch" / "A" / "ch1_native20um.nii.gz"
        if staged
        else None,
        metadata=(("recorded_age", "P7"),),
    )


def _patch_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    def check(*args: object, **kwargs: object) -> dict[str, object]:
        return {"matches": True}

    def rewrite(*, input_path: Path, output_path: Path, **kwargs: object) -> None:
        output_path.write_bytes(input_path.read_bytes())

    monkeypatch.setattr(
        "lsfm_data_processing.registration_and_transforms.project_workflow."
        "native_preparation._load_header_functions",
        lambda: (check, rewrite),
    )


def _plan(spec: NativeImageSpec):
    return build_native_preparation_plan(
        batch_id="batch001",
        subjects=(("A", (spec,), (("recorded_age", "P7"),)),),
    )


def test_new_native_image_is_atomic_staged_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_headers(monkeypatch)
    spec = _spec(tmp_path)
    plan = _plan(spec)
    assert plan.subjects[0].images[0].native_state == "write"

    apply_native_preparation_plan(
        plan, confirmation=plan.required_confirmation
    )

    assert spec.native_path.read_bytes() == b"image"
    assert spec.staged_path is not None
    assert spec.staged_path.read_bytes() == b"image"
    provenance = json.loads(spec.provenance_path.read_text())
    assert provenance["schema_version"] == 2
    assert provenance["derivation_status"] == "generated"
    assert provenance["native"]["sha256"]
    resumed = _plan(spec)
    image = resumed.subjects[0].images[0]
    assert image.native_state == "already_present"
    assert image.staged_state == "already_present"
    assert image.provenance_state == "already_present"


def test_existing_legacy_image_is_adopted_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_headers(monkeypatch)
    spec = _spec(tmp_path, staged=False)
    spec.native_path.parent.mkdir(parents=True)
    spec.native_path.write_bytes(b"image")
    spec.provenance_path.write_text(
        json.dumps({"schema_version": 1, "source_path": str(spec.source_path)})
    )
    spec.header_validation_path.write_text(json.dumps({"matches": True}))

    plan = _plan(spec)
    assert plan.subjects[0].images[0].provenance_state == "adopt_existing"
    apply_native_preparation_plan(plan, confirmation=plan.required_confirmation)

    provenance = json.loads(spec.provenance_path.read_text())
    assert provenance["schema_version"] == 2
    assert (
        provenance["derivation_status"]
        == "adopted_existing_unverified_source_link"
    )
    assert _plan(spec).subjects[0].images[0].provenance_state == "already_present"


def test_source_change_conflicts_with_canonical_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_headers(monkeypatch)
    spec = _spec(tmp_path, staged=False)
    plan = _plan(spec)
    apply_native_preparation_plan(plan, confirmation=plan.required_confirmation)
    spec.source_path.write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="provenance source differs"):
        _plan(spec)


def test_unknown_provenance_schema_is_not_treated_as_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_headers(monkeypatch)
    spec = _spec(tmp_path, staged=False)
    spec.native_path.parent.mkdir(parents=True)
    spec.native_path.write_bytes(b"image")
    spec.provenance_path.write_text(json.dumps({"schema_version": 99}))

    with pytest.raises(RuntimeError, match="Unsupported native provenance schema"):
        _plan(spec)
