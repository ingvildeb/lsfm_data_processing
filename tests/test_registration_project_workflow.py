import pytest

from lsfm_data_processing.registration_and_transforms.project_workflow.decisions import (
    FinalDecisionKind,
    WorkflowContractError,
    known_runs_from_evaluations,
    resolve_final_decisions,
    resolve_rescue_requests,
)
from lsfm_data_processing.registration_and_transforms.project_workflow.models import (
    RescueRequest,
    RescueRequestKey,
    RescueRequestStatus,
    RescueStrategy,
    RescueStrategyKind,
    RunEvaluation,
    SourceRunRequirement,
    selected_marker,
)


def _strategies() -> tuple[RescueStrategy, ...]:
    return (
        RescueStrategy("lower_gradient", "Lower gradient step"),
        RescueStrategy("higher_gradient", "Higher gradient step"),
        RescueStrategy(
            "exact_retry",
            "High-memory retry",
            SourceRunRequirement.REQUIRED,
            RescueStrategyKind.EXACT_RETRY,
        ),
        RescueStrategy(
            "masking",
            "Masking",
            SourceRunRequirement.REQUIRED,
            RescueStrategyKind.MASKING,
        ),
        RescueStrategy("custom", "custom", kind=RescueStrategyKind.CUSTOM),
    )


def test_selected_markers_match_existing_workbooks() -> None:
    for value in (True, 1, 1.0, "1", "YES", " selected ", "true"):
        assert selected_marker(value)
    for value in (False, 0, None, "", "no", "accepted"):
        assert not selected_marker(value)


def test_final_decisions_resolve_selected_excluded_and_unresolved() -> None:
    evaluations = (
        RunEvaluation("A", "registration_runs\\baseline", selected="yes", manifest_status="success"),
        RunEvaluation("B", "registration_runs/baseline", result="Exclude", selected=1, manifest_status="failed"),
        RunEvaluation("C", "registration_runs/baseline", manifest_status="success"),
    )

    decisions = resolve_final_decisions(evaluations, subject_ids=("D",))

    assert decisions["A"].kind is FinalDecisionKind.SELECTED
    assert decisions["A"].run_path == "registration_runs/baseline"
    assert decisions["B"].kind is FinalDecisionKind.EXCLUDED
    assert decisions["C"].kind is FinalDecisionKind.UNRESOLVED
    assert decisions["D"].kind is FinalDecisionKind.UNRESOLVED


def test_multiple_selected_runs_are_invalid() -> None:
    evaluations = (
        RunEvaluation("A", "baseline", selected=True, manifest_status="success"),
        RunEvaluation("A", "rescue/gs0p01", selected=True, manifest_status="success"),
    )

    with pytest.raises(WorkflowContractError, match="multiple selected final runs"):
        resolve_final_decisions(evaluations)


def test_selected_non_excluded_run_requires_successful_manifest() -> None:
    with pytest.raises(WorkflowContractError, match="does not have a successful manifest"):
        resolve_final_decisions(
            (RunEvaluation("A", "baseline", selected=True, manifest_status="failed"),)
        )


def test_multiple_distinct_rescues_are_active() -> None:
    decisions = resolve_final_decisions((), subject_ids=("A",))
    requests = (
        RescueRequest("A", "Lower gradient step"),
        RescueRequest("A", "Higher gradient step", source_run="baseline"),
    )

    plan = resolve_rescue_requests(
        requests,
        strategies=_strategies(),
        decisions=decisions,
        known_runs={"A": {"baseline"}},
    )

    assert [item.key.strategy_id for item in plan.active] == [
        "lower_gradient",
        "higher_gradient",
    ]
    assert all(item.status is RescueRequestStatus.PENDING for item in plan.active)


def test_duplicate_ordinary_requests_are_deduplicated_even_if_source_differs() -> None:
    decisions = resolve_final_decisions((), subject_ids=("A",))
    requests = (
        RescueRequest("A", "Lower gradient step", source_run="baseline"),
        RescueRequest("A", "lower_gradient", source_run="rescue/old"),
    )

    plan = resolve_rescue_requests(
        requests,
        strategies=_strategies(),
        decisions=decisions,
        known_runs={"A": {"baseline", "rescue/old"}},
    )

    assert len(plan.active) == 1
    assert plan.active[0].key == RescueRequestKey("A", "lower_gradient")
    assert plan.duplicates[0].occurrences == 2


def test_finalized_subject_requests_are_retained_but_ignored() -> None:
    decisions = resolve_final_decisions(
        (RunEvaluation("A", "baseline", selected=True, manifest_status="success"),)
    )

    plan = resolve_rescue_requests(
        (RescueRequest("A", "Old option no longer configured"),),
        strategies=_strategies(),
        decisions=decisions,
    )

    assert not plan.active
    assert plan.ignored[0].status is RescueRequestStatus.IGNORED_FINALIZED


@pytest.mark.parametrize("strategy", ("High-memory retry", "Masking"))
def test_exact_retry_and_masking_require_source_run(strategy: str) -> None:
    decisions = resolve_final_decisions((), subject_ids=("A",))

    with pytest.raises(WorkflowContractError, match="requires a source run"):
        resolve_rescue_requests(
            (RescueRequest("A", strategy),),
            strategies=_strategies(),
            decisions=decisions,
        )


def test_source_run_must_belong_to_subject() -> None:
    decisions = resolve_final_decisions((), subject_ids=("A",))

    with pytest.raises(WorkflowContractError, match="is not a known run"):
        resolve_rescue_requests(
            (RescueRequest("A", "Masking", source_run="B/baseline"),),
            strategies=_strategies(),
            decisions=decisions,
            known_runs={"A": {"A/baseline"}, "B": {"B/baseline"}},
        )


def test_source_run_is_rejected_when_explicit_known_run_index_is_empty() -> None:
    decisions = resolve_final_decisions((), subject_ids=("A",))

    with pytest.raises(WorkflowContractError, match="is not a known run"):
        resolve_rescue_requests(
            (RescueRequest("A", "Masking", source_run="baseline"),),
            strategies=_strategies(),
            decisions=decisions,
            known_runs={},
        )


@pytest.mark.parametrize(
    "existing_status",
    (RescueRequestStatus.GENERATED, RescueRequestStatus.COMPLETED, RescueRequestStatus.FAILED),
)
def test_existing_request_state_is_preserved(existing_status: RescueRequestStatus) -> None:
    decisions = resolve_final_decisions((), subject_ids=("A",))
    key = RescueRequestKey("A", "lower_gradient")

    plan = resolve_rescue_requests(
        (RescueRequest("A", "Lower gradient step"),),
        strategies=_strategies(),
        decisions=decisions,
        existing_statuses={key: existing_status},
    )

    assert plan.active[0].status is existing_status


def test_known_runs_are_normalized_from_evaluations() -> None:
    known = known_runs_from_evaluations(
        (RunEvaluation("A", "registration_runs\\baseline"),)
    )

    assert known == {"A": frozenset({"registration_runs/baseline"})}


def test_rescue_request_keys_normalize_external_artifact_identity() -> None:
    assert RescueRequestKey(" A ", "lower_gradient", r"rescues\old") == (
        RescueRequestKey("A", "lower_gradient", "rescues/old")
    )


def test_custom_rows_support_multiple_variants_and_generated_names() -> None:
    decisions = resolve_final_decisions((), subject_ids=("A",))
    requests = (
        RescueRequest(
            "A",
            "custom",
            gradient_step=0.08,
            notes="higher step",
        ),
        RescueRequest(
            "A",
            "custom",
            variant_name="coarse",
            working_resolution_um=40,
            notes="coarser registration",
        ),
    )

    plan = resolve_rescue_requests(
        requests,
        strategies=_strategies(),
        decisions=decisions,
    )

    assert [item.key.variant_name for item in plan.active] == ["gs0p08", "coarse"]


def test_custom_rescue_does_not_require_notes() -> None:
    decisions = resolve_final_decisions((), subject_ids=("A",))
    plan = resolve_rescue_requests(
        (RescueRequest("A", "custom", gradient_step=0.08),),
        strategies=_strategies(),
        decisions=decisions,
    )

    assert len(plan.active) == 1
    assert not plan.active[0].request.notes


def test_predefined_rescue_parameter_values_are_ignored_with_warning() -> None:
    decisions = resolve_final_decisions((), subject_ids=("A",))
    plan = resolve_rescue_requests(
        (RescueRequest("A", "Lower gradient step", gradient_step=0.08),),
        strategies=_strategies(),
        decisions=decisions,
    )

    assert len(plan.active) == 1
    assert "ignored" in plan.warnings[0]
