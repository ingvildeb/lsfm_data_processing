"""Pure final-decision and rescue-request resolution."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping

from .models import (
    DuplicateRescueRequest,
    FinalDecision,
    FinalDecisionKind,
    RescueRequest,
    RescueRequestKey,
    RescueRequestPlan,
    RescueRequestStatus,
    RescueStrategy,
    RescueStrategyKind,
    ResolvedRescueRequest,
    RunEvaluation,
    SourceRunRequirement,
    normalize_run_path,
    normalize_subject_id,
    normalize_text,
)


class WorkflowContractError(ValueError):
    """Raised when normalized workflow records violate the shared contract."""


def resolve_final_decisions(
    evaluations: Iterable[RunEvaluation],
    *,
    subject_ids: Iterable[str] = (),
) -> dict[str, FinalDecision]:
    """Resolve selected, excluded, and unresolved subjects without filesystem I/O."""

    grouped: dict[str, list[RunEvaluation]] = defaultdict(list)
    for evaluation in evaluations:
        grouped[evaluation.subject_id].append(evaluation)
    for subject_id in subject_ids:
        grouped.setdefault(normalize_subject_id(subject_id), [])

    decisions: dict[str, FinalDecision] = {}
    errors: list[str] = []
    for subject_id, rows in grouped.items():
        selected = [row for row in rows if row.selected]
        if len(selected) > 1:
            errors.append(
                f"{subject_id}: has multiple selected final runs: "
                + ", ".join(row.run_path or "<blank>" for row in selected)
            )
            continue
        if not selected:
            decisions[subject_id] = FinalDecision(
                subject_id=subject_id,
                kind=FinalDecisionKind.UNRESOLVED,
            )
            continue

        row = selected[0]
        if not row.run_path:
            errors.append(f"{subject_id}: selected row has no run path")
            continue
        if row.result.casefold() == "exclude":
            decisions[subject_id] = FinalDecision(
                subject_id=subject_id,
                kind=FinalDecisionKind.EXCLUDED,
                run_path=row.run_path,
                evaluation=row,
            )
            continue
        if row.manifest_status != "success":
            status = row.manifest_status or "<blank>"
            errors.append(
                f"{subject_id}: selected run {row.run_path!r} does not have a "
                f"successful manifest (status={status!r})"
            )
            continue
        decisions[subject_id] = FinalDecision(
            subject_id=subject_id,
            kind=FinalDecisionKind.SELECTED,
            run_path=row.run_path,
            evaluation=row,
        )

    if errors:
        raise WorkflowContractError(
            "Invalid final registration decisions:\n" + "\n".join(errors)
        )
    return decisions


def known_runs_from_evaluations(
    evaluations: Iterable[RunEvaluation],
) -> dict[str, frozenset[str]]:
    runs: dict[str, set[str]] = defaultdict(set)
    for evaluation in evaluations:
        if evaluation.run_path:
            runs[evaluation.subject_id].add(evaluation.run_path)
    return {subject_id: frozenset(paths) for subject_id, paths in runs.items()}


def rescue_strategy_lookup(
    strategies: Iterable[RescueStrategy],
) -> dict[str, RescueStrategy]:
    """Index configured strategy IDs, labels, and aliases case-insensitively."""

    lookup: dict[str, RescueStrategy] = {}
    for strategy in strategies:
        for name in (strategy.strategy_id, strategy.workbook_label, *strategy.aliases):
            key = name.casefold()
            previous = lookup.get(key)
            if previous is not None and previous != strategy:
                raise WorkflowContractError(
                    f"Rescue strategy name {name!r} refers to more than one strategy"
                )
            lookup[key] = strategy
    return lookup


def make_rescue_request_key(
    request: RescueRequest,
    strategy: RescueStrategy,
) -> RescueRequestKey:
    required_source = (
        request.source_run
        if strategy.source_run_requirement is SourceRunRequirement.REQUIRED
        else ""
    )
    return RescueRequestKey(
        subject_id=request.subject_id,
        strategy_id=strategy.strategy_id,
        required_source_run=required_source,
        variant_name=(
            request.effective_variant_name
            if strategy.kind is RescueStrategyKind.CUSTOM
            else ""
        ),
    )


def resolve_rescue_requests(
    requests: Iterable[RescueRequest],
    *,
    strategies: Iterable[RescueStrategy],
    decisions: Mapping[str, FinalDecision],
    known_runs: Mapping[str, Collection[str]] | None = None,
    existing_statuses: Mapping[RescueRequestKey, RescueRequestStatus | str] | None = None,
) -> RescueRequestPlan:
    """Resolve active rescue requests while retaining finalized historical rows."""

    lookup = rescue_strategy_lookup(strategies)
    validate_known_runs = known_runs is not None
    normalized_known_runs = _normalize_known_runs(known_runs or {})
    existing_statuses = existing_statuses or {}
    active_by_key: dict[RescueRequestKey, ResolvedRescueRequest] = {}
    occurrences: Counter[RescueRequestKey] = Counter()
    ignored: list[ResolvedRescueRequest] = []
    errors: list[str] = []
    warnings: list[str] = []
    custom_parameter_sets: dict[tuple[str, tuple[tuple[str, float], ...]], str] = {}

    for request in requests:
        decision = decisions.get(request.subject_id)
        if decision is not None and decision.finalized:
            ignored.append(
                ResolvedRescueRequest(
                    request=request,
                    key=RescueRequestKey(
                        request.subject_id,
                        normalize_text(request.strategy).casefold(),
                        request.source_run,
                    ),
                    status=RescueRequestStatus.IGNORED_FINALIZED,
                    reason=f"subject is finalized as {decision.kind.value}",
                )
            )
            continue

        strategy = lookup.get(request.strategy.casefold())
        if strategy is None:
            errors.append(
                f"{request.subject_id}: unknown rescue strategy {request.strategy!r}"
            )
            continue
        if strategy.kind is RescueStrategyKind.CUSTOM:
            if not request.custom_parameters:
                errors.append(
                    f"{request.subject_id}: custom rescue requires at least one parameter override"
                )
                continue
            if not request.notes:
                errors.append(
                    f"{request.subject_id}: custom rescue requires a rationale in Notes"
                )
                continue
            fingerprint = (request.subject_id, request.custom_parameters)
            previous = custom_parameter_sets.get(fingerprint)
            if previous is not None:
                errors.append(
                    f"{request.subject_id}: duplicate custom parameter combination "
                    f"for variants {previous!r} and {request.effective_variant_name!r}"
                )
                continue
            custom_parameter_sets[fingerprint] = request.effective_variant_name
        elif request.variant_name or request.custom_parameters:
            warnings.append(
                f"{request.subject_id}: custom variant/parameter values are ignored for "
                f"predefined rescue {strategy.workbook_label!r}"
            )
        if (
            strategy.source_run_requirement is SourceRunRequirement.REQUIRED
            and not request.source_run
        ):
            errors.append(
                f"{request.subject_id}: rescue strategy {strategy.workbook_label!r} "
                "requires a source run"
            )
            continue
        if request.source_run and validate_known_runs:
            subject_runs = normalized_known_runs.get(request.subject_id, frozenset())
            if request.source_run not in subject_runs:
                errors.append(
                    f"{request.subject_id}: source run {request.source_run!r} is not "
                    "a known run for this subject"
                )
                continue

        key = make_rescue_request_key(request, strategy)
        occurrences[key] += 1
        if key in active_by_key:
            continue
        try:
            status = RescueRequestStatus(
                existing_statuses.get(key, RescueRequestStatus.PENDING)
            )
        except ValueError:
            errors.append(
                f"{request.subject_id}: unsupported existing status for {key.strategy_id!r}: "
                f"{existing_statuses[key]!r}"
            )
            continue
        active_by_key[key] = ResolvedRescueRequest(
            request=request,
            key=key,
            status=status,
            strategy=strategy,
        )

    if errors:
        raise WorkflowContractError(
            "Invalid registration rescue requests:\n" + "\n".join(errors)
        )

    duplicates = tuple(
        DuplicateRescueRequest(key=key, occurrences=count)
        for key, count in occurrences.items()
        if count > 1
    )
    return RescueRequestPlan(
        active=tuple(active_by_key.values()),
        ignored=tuple(ignored),
        duplicates=duplicates,
        warnings=tuple(warnings),
    )


def _normalize_known_runs(
    known_runs: Mapping[str, Collection[str]],
) -> dict[str, frozenset[str]]:
    return {
        normalize_subject_id(subject_id): frozenset(
            path for value in paths if (path := normalize_run_path(value))
        )
        for subject_id, paths in known_runs.items()
    }
