"""Lightweight direct-input signatures stored in SpatialSignal parameters."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import ArtifactInspection, ArtifactSet, ArtifactState


DIRECT_INPUT_SIGNATURES_KEY = "direct_input_signatures"


@dataclass(frozen=True)
class FileSignature:
    name: str
    size: int
    mtime_ns: int

    @classmethod
    def from_path(cls, path: Path) -> "FileSignature":
        path = Path(path)
        stat = path.stat()
        if not path.is_file():
            raise FileNotFoundError(f"direct input is not a file: {path}")
        return cls(path.name, stat.st_size, stat.st_mtime_ns)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FileSignature":
        if set(value) != {"name", "size", "mtime_ns"}:
            raise ValueError("file signature must contain name, size, and mtime_ns")
        signature = cls(str(value["name"]), int(value["size"]), int(value["mtime_ns"]))
        if not signature.name or signature.size < 0 or signature.mtime_ns < 0:
            raise ValueError("file signature contains invalid values")
        return signature

    def to_dict(self) -> dict[str, int | str]:
        return {"name": self.name, "size": self.size, "mtime_ns": self.mtime_ns}


def _json_value(value: Any) -> Any:
    """Normalize a value to the representation persisted in JSON."""

    return json.loads(json.dumps(value, sort_keys=True))


def direct_input_signatures(paths: Sequence[Path]) -> list[dict[str, int | str]]:
    return [FileSignature.from_path(path).to_dict() for path in paths]


def with_lineage_parameters(
    scientific_parameters: Mapping[str, Any], direct_inputs: Sequence[Path]
) -> dict[str, Any]:
    """Return SpatialSignal processing parameters with lightweight lineage."""

    if DIRECT_INPUT_SIGNATURES_KEY in scientific_parameters:
        raise ValueError(f"{DIRECT_INPUT_SIGNATURES_KEY!r} is reserved")
    parameters = _json_value(dict(scientific_parameters))
    parameters[DIRECT_INPUT_SIGNATURES_KEY] = direct_input_signatures(direct_inputs)
    return parameters


def validate_lineage_parameters(
    artifact_set: ArtifactSet,
    recorded_parameters: Mapping[str, Any] | None,
    expected_scientific_parameters: Mapping[str, Any],
    direct_inputs: Sequence[Path],
) -> ArtifactInspection:
    """Validate scientific parameters and direct inputs for one complete set."""

    recorded = dict(recorded_parameters or {})
    expected = _json_value(dict(expected_scientific_parameters))
    signed = DIRECT_INPUT_SIGNATURES_KEY in recorded

    for key, expected_value in expected.items():
        if key not in recorded:
            if signed:
                return ArtifactInspection(
                    artifact_set, ArtifactState.STALE,
                    (f"signed output is missing scientific parameter {key!r}",),
                )
            continue
        if _json_value(recorded[key]) != expected_value:
            return ArtifactInspection(
                artifact_set, ArtifactState.STALE,
                (f"scientific parameter {key!r} differs",),
            )

    if not signed:
        return ArtifactInspection(
            artifact_set, ArtifactState.LEGACY,
            ("direct-input signatures are not recorded",),
        )

    try:
        raw_signatures = recorded[DIRECT_INPUT_SIGNATURES_KEY]
        if not isinstance(raw_signatures, list):
            raise ValueError("direct_input_signatures must be a list")
        actual = [FileSignature.from_mapping(item) for item in raw_signatures]
        current = [FileSignature.from_path(path) for path in direct_inputs]
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        return ArtifactInspection(
            artifact_set, ArtifactState.INVALID, (f"invalid lineage: {exc}",)
        )
    if actual != current:
        return ArtifactInspection(
            artifact_set, ArtifactState.STALE,
            ("one or more direct-input signatures differ",),
        )
    return ArtifactInspection(artifact_set, ArtifactState.VALID)


def validate_spatial_sidecar_lineage(
    artifact_set: ArtifactSet,
    sidecar_path: Path,
    expected_scientific_parameters: Mapping[str, Any],
    direct_inputs: Sequence[Path],
) -> ArtifactInspection:
    """Read processing.parameters from an existing SpatialSignal sidecar."""

    try:
        payload = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("sidecar root must be an object")
        processing = payload.get("processing")
        if processing is not None and not isinstance(processing, dict):
            raise ValueError("processing must be an object")
        parameters = None if processing is None else processing.get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            raise ValueError("processing.parameters must be an object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return ArtifactInspection(
            artifact_set, ArtifactState.INVALID, (f"invalid sidecar: {exc}",)
        )
    return validate_lineage_parameters(
        artifact_set, parameters, expected_scientific_parameters, direct_inputs
    )
