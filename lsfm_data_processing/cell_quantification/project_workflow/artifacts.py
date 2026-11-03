"""Artifact-set state inspection and existing-output policy decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


class ArtifactState(str, Enum):
    ABSENT = "absent"
    PARTIAL = "partial"
    VALID = "valid"
    LEGACY = "legacy"
    STALE = "stale"
    INVALID = "invalid"


class ExistingOutputPolicy(str, Enum):
    ERROR = "error"
    SKIP = "skip"
    REPAIR = "repair"
    OVERWRITE = "overwrite"


class PlannedAction(str, Enum):
    WRITE = "write"
    REUSE = "reuse"
    REBUILD = "rebuild"
    STOP = "stop"


@dataclass(frozen=True)
class ArtifactSet:
    """The complete set of files representing one stage output."""

    stage: str
    paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        if not self.stage.strip():
            raise ValueError("artifact stage must not be empty")
        paths = tuple(Path(path) for path in self.paths)
        if not paths:
            raise ValueError("an artifact set must contain at least one path")
        if len(paths) != len(set(paths)):
            raise ValueError("an artifact set must not contain duplicate paths")
        object.__setattr__(self, "paths", paths)


@dataclass(frozen=True)
class ArtifactInspection:
    artifact_set: ArtifactSet
    state: ArtifactState
    reasons: tuple[str, ...] = ()

    @property
    def reusable(self) -> bool:
        return self.state in {ArtifactState.VALID, ArtifactState.LEGACY}


@dataclass(frozen=True)
class PolicyDecision:
    inspection: ArtifactInspection
    policy: ExistingOutputPolicy
    action: PlannedAction
    reason: str


CompleteSetValidator = Callable[[ArtifactSet], ArtifactInspection]


def inspect_artifact_set(
    artifact_set: ArtifactSet, *, validator: CompleteSetValidator | None
) -> ArtifactInspection:
    """Inspect existence and require semantic validation when complete."""

    present = tuple(path.is_file() for path in artifact_set.paths)
    if not any(present):
        return ArtifactInspection(artifact_set, ArtifactState.ABSENT)
    if not all(present):
        missing = [
            str(path)
            for path, exists in zip(artifact_set.paths, present, strict=True)
            if not exists
        ]
        return ArtifactInspection(
            artifact_set,
            ArtifactState.PARTIAL,
            ("missing required files: " + ", ".join(missing),),
        )
    if validator is None:
        raise ValueError(
            f"complete artifact set {artifact_set.stage!r} requires a validator"
        )
    inspection = validator(artifact_set)
    if inspection.artifact_set != artifact_set:
        raise ValueError("artifact validator returned an inspection for another set")
    if inspection.state in {ArtifactState.ABSENT, ArtifactState.PARTIAL}:
        raise ValueError("complete-set validator returned an existence-only state")
    return inspection


def decide_existing_output(
    inspection: ArtifactInspection,
    policy: ExistingOutputPolicy | str,
    *,
    upstream_rebuilt: bool = False,
) -> PolicyDecision:
    """Apply the agreed output-policy matrix to a validated state."""

    policy = ExistingOutputPolicy(policy)
    state = inspection.state
    if state is ArtifactState.LEGACY and upstream_rebuilt:
        state = ArtifactState.STALE
        inspection = ArtifactInspection(
            inspection.artifact_set,
            state,
            inspection.reasons
            + ("unsigned legacy output has a rebuilt upstream input",),
        )

    if state is ArtifactState.ABSENT:
        action = (
            PlannedAction.REBUILD
            if policy is ExistingOutputPolicy.OVERWRITE
            else PlannedAction.WRITE
        )
        return PolicyDecision(inspection, policy, action, "artifact set is absent")

    if state in {ArtifactState.VALID, ArtifactState.LEGACY}:
        if policy is ExistingOutputPolicy.ERROR:
            return PolicyDecision(
                inspection, policy, PlannedAction.STOP,
                "existing artifact is forbidden by the error policy",
            )
        if policy is ExistingOutputPolicy.OVERWRITE:
            return PolicyDecision(
                inspection, policy, PlannedAction.REBUILD,
                "overwrite policy rebuilds selected artifact sets",
            )
        return PolicyDecision(
            inspection, policy, PlannedAction.REUSE,
            "artifact set is validated and reusable",
        )

    if policy in {ExistingOutputPolicy.REPAIR, ExistingOutputPolicy.OVERWRITE}:
        return PolicyDecision(
            inspection, policy, PlannedAction.REBUILD,
            f"{policy.value} policy rebuilds {state.value} artifact sets",
        )
    return PolicyDecision(
        inspection, policy, PlannedAction.STOP,
        f"{policy.value} policy cannot reuse a {state.value} artifact set",
    )
