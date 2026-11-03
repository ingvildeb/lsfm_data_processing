"""Read-only planning across dependent artifact sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .artifacts import (
    ArtifactSet,
    CompleteSetValidator,
    ExistingOutputPolicy,
    PlannedAction,
    PolicyDecision,
    decide_existing_output,
    inspect_artifact_set,
)


@dataclass(frozen=True)
class ArtifactRequest:
    """One planned artifact set and its already-declared dependencies."""

    key: str
    artifact_set: ArtifactSet
    validator: CompleteSetValidator
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("artifact request key must not be empty")
        if self.key in self.depends_on:
            raise ValueError("an artifact request cannot depend on itself")


@dataclass(frozen=True)
class WorkflowPlan:
    """Ordered, read-only decisions for a workflow invocation."""

    decisions: tuple[tuple[str, PolicyDecision], ...]

    @property
    def can_apply(self) -> bool:
        return all(
            decision.action is not PlannedAction.STOP
            for _, decision in self.decisions
        )

    @property
    def by_key(self) -> dict[str, PolicyDecision]:
        return dict(self.decisions)

    def summary_lines(self) -> tuple[str, ...]:
        return tuple(
            f"{key}: {decision.inspection.state.value} -> "
            f"{decision.action.value} ({decision.reason})"
            for key, decision in self.decisions
        )


def build_workflow_plan(
    requests: Sequence[ArtifactRequest],
    policy: ExistingOutputPolicy | str,
) -> WorkflowPlan:
    """Validate and plan artifact sets in dependency order without writing."""

    policy = ExistingOutputPolicy(policy)
    decisions: list[tuple[str, PolicyDecision]] = []
    by_key: dict[str, PolicyDecision] = {}
    for request in requests:
        if request.key in by_key:
            raise ValueError(f"duplicate artifact request key: {request.key!r}")
        missing = [key for key in request.depends_on if key not in by_key]
        if missing:
            raise ValueError(
                f"artifact request {request.key!r} has dependencies that were not "
                f"planned first: {missing}"
            )
        upstream_rebuilt = any(
            by_key[key].action in {PlannedAction.WRITE, PlannedAction.REBUILD}
            for key in request.depends_on
        )
        inspection = inspect_artifact_set(
            request.artifact_set, validator=request.validator
        )
        decision = decide_existing_output(
            inspection, policy, upstream_rebuilt=upstream_rebuilt
        )
        by_key[request.key] = decision
        decisions.append((request.key, decision))
    return WorkflowPlan(tuple(decisions))
