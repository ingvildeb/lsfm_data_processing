"""Shared native-image preparation, staging, and provenance contracts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any


NATIVE_PROVENANCE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class NativeImageSpec:
    subject_id: str
    channel: str
    source_path: Path
    native_path: Path
    orientation: str
    resolution_um: float
    provenance_path: Path
    header_validation_path: Path
    staged_path: Path | None = None
    metadata: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class NativeImagePreparation:
    spec: NativeImageSpec
    native_state: str
    staged_state: str | None
    provenance_state: str
    validation: object | None = None

    @property
    def channel(self) -> str:
        return self.spec.channel

    @property
    def source_path(self) -> Path:
        return self.spec.source_path

    @property
    def native_path(self) -> Path:
        return self.spec.native_path

    @property
    def staged_path(self) -> Path | None:
        return self.spec.staged_path


@dataclass(frozen=True)
class NativeSubjectPreparation:
    subject_id: str
    images: tuple[NativeImagePreparation, ...]
    metadata: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class NativePreparationPlan:
    batch_id: str
    subjects: tuple[NativeSubjectPreparation, ...]

    @property
    def required_confirmation(self) -> str:
        return f"PREPARE REGISTRATION {self.batch_id}"


def _load_header_functions():
    try:
        from atlasspace.image.space_validation import (
            check_nifti_header_matches_declared_space,
        )
        from atlasspace.io.nifti import rewrite_nifti_header_to_declared_space
    except ModuleNotFoundError as exc:
        if exc.name != "atlasspace":
            raise
        raise ModuleNotFoundError(
            "atlasspace is required for native registration-image preparation."
        ) from exc
    return (
        check_nifti_header_matches_declared_space,
        rewrite_nifti_header_to_declared_space,
    )


def _header_is_valid(result: object) -> bool:
    if isinstance(result, bool):
        return result
    if isinstance(result, dict):
        for key in ("matches", "valid", "is_valid"):
            if key in result:
                return bool(result[key])
        return not bool(result.get("errors"))
    return bool(result)


def _validate_header(path: Path, spec: NativeImageSpec) -> object:
    check, _ = _load_header_functions()
    result = check(
        path,
        orientation=spec.orientation,
        resolution_um=spec.resolution_um,
    )
    if not _header_is_valid(result):
        raise ValueError(
            f"Image does not match declared {spec.orientation} "
            f"{spec.resolution_um:g} um space: {path}"
        )
    return result


def _fingerprint_source(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_time_ns": stat.st_mtime_ns,
    }


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_provenance(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise FileExistsError(f"Provenance destination is not a file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid native-image provenance: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Invalid native-image provenance object: {path}")
    return value


def _validate_canonical_provenance(
    provenance: dict[str, Any], spec: NativeImageSpec
) -> None:
    if provenance.get("source") != _fingerprint_source(spec.source_path):
        raise ValueError(
            "Native provenance source differs from the current source: "
            f"{spec.provenance_path}"
        )
    native = provenance.get("native")
    if not isinstance(native, dict) or native.get("path") != str(spec.native_path):
        raise ValueError(f"Native provenance output differs: {spec.provenance_path}")
    if native.get("sha256") != _sha256(spec.native_path):
        raise ValueError(f"Native image fingerprint differs: {spec.native_path}")
    if provenance.get("declared_orientation") != spec.orientation:
        raise ValueError(f"Native provenance orientation differs: {spec.provenance_path}")
    if float(provenance.get("declared_resolution_um", -1)) != spec.resolution_um:
        raise ValueError(f"Native provenance resolution differs: {spec.provenance_path}")


def _header_payload(spec: NativeImageSpec, validation: object) -> dict[str, object]:
    return {
        "schema_version": 2,
        "subject_id": spec.subject_id,
        "channel": spec.channel,
        "declared_orientation": spec.orientation,
        "declared_resolution_um": spec.resolution_um,
        "native_path": str(spec.native_path),
        "validation": validation,
    }


def _json_bytes(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def build_native_preparation_plan(
    *,
    batch_id: str,
    subjects: tuple[
        tuple[
            str,
            tuple[NativeImageSpec, ...],
            tuple[tuple[str, object], ...],
        ],
        ...,
    ],
) -> NativePreparationPlan:
    """Preflight all required images without writing any output."""

    errors: list[str] = []
    planned_subjects: list[NativeSubjectPreparation] = []
    for subject_id, image_specs, subject_metadata in subjects:
        images: list[NativeImagePreparation] = []
        for spec in image_specs:
            if spec.subject_id != subject_id:
                errors.append(
                    f"{subject_id}: image spec subject mismatch for {spec.channel}"
                )
                continue
            if not spec.source_path.is_file():
                errors.append(
                    f"{subject_id} {spec.channel}: source image is missing: "
                    f"{spec.source_path}"
                )
                continue
            if (
                spec.header_validation_path.exists()
                and not spec.header_validation_path.is_file()
            ):
                errors.append(
                    f"{subject_id} {spec.channel}: header-validation destination "
                    f"is not a file: {spec.header_validation_path}"
                )
                continue
            native_state = "write"
            provenance_state = "write"
            validation: object | None = None
            if spec.native_path.exists():
                if not spec.native_path.is_file():
                    errors.append(
                        f"{subject_id} {spec.channel}: native destination is not a file: "
                        f"{spec.native_path}"
                    )
                    continue
                try:
                    validation = _validate_header(spec.native_path, spec)
                    provenance = _read_provenance(spec.provenance_path)
                    if provenance is None or provenance.get("schema_version") == 1:
                        provenance_state = "adopt_existing"
                    elif provenance.get("schema_version") == 2:
                        _validate_canonical_provenance(provenance, spec)
                        provenance_state = "already_present"
                        if spec.header_validation_path.exists():
                            if not spec.header_validation_path.is_file():
                                raise FileExistsError(
                                    "Header-validation destination is not a file: "
                                    f"{spec.header_validation_path}"
                                )
                            expected = _json_bytes(
                                _header_payload(spec, validation)
                            )
                            if spec.header_validation_path.read_bytes() != expected:
                                raise ValueError(
                                    "Canonical header validation differs: "
                                    f"{spec.header_validation_path}"
                                )
                    else:
                        raise ValueError(
                            "Unsupported native provenance schema "
                            f"{provenance.get('schema_version')!r}: "
                            f"{spec.provenance_path}"
                        )
                    native_state = "already_present"
                except (FileExistsError, ValueError) as exc:
                    errors.append(f"{subject_id} {spec.channel}: {exc}")
                    continue
            elif spec.provenance_path.exists():
                errors.append(
                    f"{subject_id} {spec.channel}: provenance exists without native image: "
                    f"{spec.provenance_path}"
                )
                continue
            elif spec.header_validation_path.exists():
                errors.append(
                    f"{subject_id} {spec.channel}: header validation exists without "
                    f"native image: {spec.header_validation_path}"
                )
                continue

            staged_state: str | None = None
            if spec.staged_path is not None:
                staged_state = "write"
                if spec.staged_path.exists():
                    if not spec.staged_path.is_file():
                        errors.append(
                            f"{subject_id}: staged destination is not a file: "
                            f"{spec.staged_path}"
                        )
                        continue
                    if native_state == "already_present":
                        if _sha256(spec.native_path) != _sha256(spec.staged_path):
                            errors.append(
                                f"{subject_id}: staged image differs from canonical native "
                                f"image: {spec.staged_path}"
                            )
                            continue
                        staged_state = "already_present"
                    else:
                        staged_state = "verify_after_native"
            images.append(
                NativeImagePreparation(
                    spec=spec,
                    native_state=native_state,
                    staged_state=staged_state,
                    provenance_state=provenance_state,
                    validation=validation,
                )
            )
        planned_subjects.append(
            NativeSubjectPreparation(subject_id, tuple(images), subject_metadata)
        )
    if errors:
        raise RuntimeError(
            "Registration preparation preflight failed. No files were written.\n"
            + "\n".join(errors)
        )
    return NativePreparationPlan(batch_id, tuple(planned_subjects))


def summarize_native_preparation_plan(plan: NativePreparationPlan) -> str:
    images = [image for subject in plan.subjects for image in subject.images]
    native_writes = sum(image.native_state == "write" for image in images)
    adopted = sum(image.provenance_state == "adopt_existing" for image in images)
    stage_writes = sum(
        image.staged_state not in {None, "already_present"} for image in images
    )
    return "\n".join(
        [
            f"Registration preparation plan: {plan.batch_id}",
            f"Subjects: {len(plan.subjects)}",
            f"Native images to write: {native_writes}",
            f"Legacy native images to adopt: {adopted}",
            f"Staged images to write or verify: {stage_writes}",
        ]
    )


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json_atomic(
    path: Path, payload: dict[str, object], *, replace: bool
) -> None:
    content = _json_bytes(payload)
    if path.exists() and path.is_file() and path.read_bytes() == content:
        return
    if path.exists() and not replace:
        raise FileExistsError(f"Provenance exists and differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_atomic(source: Path, destination: Path) -> str:
    source_hash = _sha256(source)
    if destination.exists():
        if destination.is_file() and _sha256(destination) == source_hash:
            return "already_present"
        raise FileExistsError(
            f"Destination exists and differs; it will not be overwritten: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    try:
        with source.open("rb") as source_handle, temporary.open("xb") as output:
            shutil.copyfileobj(source_handle, output)
        shutil.copystat(source, temporary)
        if _sha256(temporary) != source_hash:
            raise IOError(f"Staged copy failed verification: {destination}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


def apply_native_preparation_plan(
    plan: NativePreparationPlan, *, confirmation: str
) -> None:
    if confirmation != plan.required_confirmation:
        raise ValueError(f"Required confirmation: {plan.required_confirmation!r}")
    check_header, rewrite_header = _load_header_functions()
    images = [image for subject in plan.subjects for image in subject.images]
    for index, item in enumerate(images, start=1):
        spec = item.spec
        print(f"[{index}/{len(images)}] {spec.subject_id} {spec.channel}")
        spec.native_path.parent.mkdir(parents=True, exist_ok=True)
        if item.native_state == "write":
            if spec.native_path.exists():
                raise FileExistsError(
                    f"Native destination appeared after preflight: {spec.native_path}"
                )
            temporary = spec.native_path.with_name(
                f".{spec.native_path.name}.partial.nii.gz"
            )
            temporary.unlink(missing_ok=True)
            try:
                rewrite_header(
                    input_path=spec.source_path,
                    output_path=temporary,
                    orientation=spec.orientation,
                    resolution_um=spec.resolution_um,
                )
                validation = check_header(
                    temporary,
                    orientation=spec.orientation,
                    resolution_um=spec.resolution_um,
                )
                if not _header_is_valid(validation):
                    raise ValueError(
                        f"Prepared native image failed validation: {temporary}"
                    )
                temporary.replace(spec.native_path)
            finally:
                temporary.unlink(missing_ok=True)
            derivation_status = "generated"
        else:
            validation = check_header(
                spec.native_path,
                orientation=spec.orientation,
                resolution_um=spec.resolution_um,
            )
            if not _header_is_valid(validation):
                raise ValueError(
                    f"Prepared native image failed validation: {spec.native_path}"
                )
            existing = _read_provenance(spec.provenance_path)
            derivation_status = (
                str(existing.get("derivation_status"))
                if existing and existing.get("schema_version") == 2
                else "adopted_existing_unverified_source_link"
            )

        staged_status = (
            _copy_atomic(spec.native_path, spec.staged_path)
            if spec.staged_path is not None
            else "not_staged"
        )
        header_payload = _header_payload(spec, validation)
        provenance_payload = {
            "schema_version": NATIVE_PROVENANCE_SCHEMA_VERSION,
            "subject_id": spec.subject_id,
            "channel": spec.channel,
            "source": _fingerprint_source(spec.source_path),
            "native": {
                "path": str(spec.native_path),
                "sha256": _sha256(spec.native_path),
            },
            "declared_orientation": spec.orientation,
            "declared_resolution_um": spec.resolution_um,
            "operation": "metadata_only_header_rewrite",
            "preserve_voxel_array": True,
            "derivation_status": derivation_status,
            "staged_path": str(spec.staged_path) if spec.staged_path else None,
            "metadata": dict(spec.metadata),
        }
        replace_legacy = item.provenance_state == "adopt_existing"
        _write_json_atomic(
            spec.header_validation_path, header_payload, replace=replace_legacy
        )
        _write_json_atomic(
            spec.provenance_path, provenance_payload, replace=replace_legacy
        )
        print(
            f"    native={item.native_state}; staged={staged_status}; "
            f"provenance={item.provenance_state}"
        )
