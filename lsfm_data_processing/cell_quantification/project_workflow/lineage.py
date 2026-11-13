"""Scientific-parameter validation with optional formal lineage.

The early Glia pilot stored file names, sizes, and modification times under a
custom ``direct_input_signatures`` processing parameter. Modification times are
not portable across the filesystems used by the projects, so that extension is
now historical metadata only. New outputs record scientific parameters without
custom file signatures and remain valid, reusable ``no_lineage`` artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import ArtifactInspection, ArtifactSet, ArtifactState


DIRECT_INPUT_SIGNATURES_KEY = "direct_input_signatures"


def _json_value(value: Any) -> Any:
    """Normalize a value to the representation persisted in JSON."""

    return json.loads(json.dumps(value, sort_keys=True))


def with_lineage_parameters(
    scientific_parameters: Mapping[str, Any], direct_inputs: Sequence[Path]
) -> dict[str, Any]:
    """Return normalized scientific parameters without custom file lineage.

    ``direct_inputs`` remains in the signature so projects pinned to the first
    shared workflow-01 API do not need a simultaneous code change. It is not
    inspected. Formal SpatialSignal ``processing.inputs`` references may be
    added independently when upstream artifacts expose stable IDs.
    """

    if DIRECT_INPUT_SIGNATURES_KEY in scientific_parameters:
        raise ValueError(f"{DIRECT_INPUT_SIGNATURES_KEY!r} is reserved")
    del direct_inputs
    return _json_value(dict(scientific_parameters))


def validate_lineage_parameters(
    artifact_set: ArtifactSet,
    recorded_parameters: Mapping[str, Any] | None,
    expected_scientific_parameters: Mapping[str, Any],
    direct_inputs: Sequence[Path],
) -> ArtifactInspection:
    """Validate scientific parameters and classify absent formal lineage.

    Historical ``direct_input_signatures`` are deliberately ignored. They are
    neither formal SpatialSignal lineage nor reliable evidence of staleness
    after transfer between filesystems.
    """

    recorded = dict(recorded_parameters or {})
    expected = _json_value(dict(expected_scientific_parameters))
    del direct_inputs

    for key, expected_value in expected.items():
        if key not in recorded:
            continue
        if _json_value(recorded[key]) != expected_value:
            return ArtifactInspection(
                artifact_set, ArtifactState.STALE,
                (f"scientific parameter {key!r} differs",),
            )

    return ArtifactInspection(
        artifact_set,
        ArtifactState.NO_LINEAGE,
        ("formal input lineage is not recorded",),
    )


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
