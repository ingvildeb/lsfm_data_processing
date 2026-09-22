from lsfm_data_processing.registration_and_transforms import project_workflow


EXPECTED_PUBLIC_API = {
    "apply_baseline_generation_plan",
    "apply_batch_membership_plan",
    "apply_evaluation_workbook_plan",
    "apply_masking_plans",
    "apply_native_preparation_plan",
    "apply_registration_distribution_plan",
    "apply_rescue_generation_plan",
    "build_evaluation_workbook_plan",
    "build_masking_plans",
    "build_native_preparation_plan",
    "build_registration_distribution_plan",
    "discover_registration_run_inventory",
    "plan_baseline_generation",
    "plan_batch_membership",
    "plan_rescue_generation",
    "resolve_final_decisions",
    "resolve_rescue_requests",
}


def test_project_workflow_facade_is_intentionally_small() -> None:
    assert set(project_workflow.__all__) == EXPECTED_PUBLIC_API
    assert not hasattr(project_workflow, "RunEvaluation")
    assert not hasattr(project_workflow, "render_registration_batch_config")
