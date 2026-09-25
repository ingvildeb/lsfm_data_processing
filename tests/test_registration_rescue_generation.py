import tomllib
import json
from pathlib import Path

import pytest
import yaml

from lsfm_data_processing.registration_and_transforms.project_workflow.decisions import (
    resolve_final_decisions,
    resolve_rescue_requests,
)
from lsfm_data_processing.registration_and_transforms.project_workflow.generation import (
    HpcSubmissionEntry,
    RegistrationBatchSpec,
    RegistrationImageSpec,
    RegistrationSegmentationSpec,
    render_hpc_submission_helper,
    render_registration_batch_config,
)
from lsfm_data_processing.registration_and_transforms.project_workflow.models import RescueRequest
from lsfm_data_processing.registration_and_transforms.project_workflow.rescue_catalog import (
    ExpandedRescueJob,
    canonical_rescue_strategies,
    expand_rescue_request_plan,
)
from lsfm_data_processing.registration_and_transforms.project_workflow.rescue_generation import (
    RescueGenerationRequest,
    apply_rescue_generation_plan,
    plan_rescue_generation,
)
def _base_preset() -> dict[str, object]:
    return {
        "name": "tuned_syn_cc",
        "description": "baseline",
        "preprocessing": {"fixed_padding_um": None},
        "registration": {
            "working_resolution_um": 20,
            "syn_gradient_step": 0.05,
        },
    }


def _registration_spec(subject_id: str) -> RegistrationBatchSpec:
    return RegistrationBatchSpec(
        registration_presets=("placeholder",),
        orientation_alignment="direction_only",
        write_input_images=True,
        output_subdir="placeholder",
        images=(
            RegistrationImageSpec(
                "template", "/templates/template.nii.gz", "template", "LSP", 20
            ),
            RegistrationImageSpec(
                subject_id,
                f"/subjects/{subject_id}/native.nii.gz",
                subject_id,
                "LAS",
                20,
            ),
        ),
        image_to_template=((subject_id, "template"),),
        image_default_resolution_um=20,
        template_role="fixed",
        moving_segmentations_enabled=True,
        moving_segmentation_interpolation="genericLabel",
        moving_segmentations_write_intermediates=False,
    )


def _resolved(*requests: RescueRequest):
    decisions = resolve_final_decisions(
        (), subject_ids=tuple(request.subject_id for request in requests)
    )
    return resolve_rescue_requests(
        requests,
        strategies=canonical_rescue_strategies(),
        decisions=decisions,
        known_runs={
            request.subject_id: {"registration_runs/baseline"}
            for request in requests
        },
    )


def test_canonical_menu_expands_each_subject_independently() -> None:
    plan = _resolved(
        RescueRequest("A", "lower gradient step"),
        RescueRequest("B", "lower gradient step"),
    )

    jobs = expand_rescue_request_plan(plan)

    assert len(jobs) == 6
    assert {(job.subject_id, job.variant) for job in jobs} == {
        (subject, variant)
        for subject in ("A", "B")
        for variant in ("gs0p01", "gs0p02", "gs0p025")
    }
    assert all(job.hpc_mem_gb is None for job in jobs)


def test_high_memory_retry_keeps_source_identity_and_changes_resources_only() -> None:
    plan = _resolved(
        RescueRequest(
            "A",
            "high memory retry",
            source_run="registration_runs/baseline",
        )
    )

    jobs = expand_rescue_request_plan(plan)

    assert len(jobs) == 1
    assert jobs[0].source_run == "registration_runs/baseline"
    assert jobs[0].variant == "from_baseline_mem256gb"
    assert jobs[0].override_mapping == {}
    assert jobs[0].hpc_mem_gb == 256


def test_custom_rescue_expands_to_one_subject_job() -> None:
    plan = _resolved(
        RescueRequest(
            "A",
            "custom",
            variant_name="test_combo",
            gradient_step=0.09,
            padding_um=500,
            notes="Investigate ambiguous fit",
        )
    )

    job = expand_rescue_request_plan(plan)[0]

    assert job.subject_id == "A"
    assert job.variant == "test_combo"
    assert job.override_mapping == {
        "syn_gradient_step": 0.09,
        "fixed_padding_um": 500.0,
    }


def test_registration_renderer_uses_only_typed_project_inputs() -> None:
    text = render_registration_batch_config(
        RegistrationBatchSpec(
            registration_presets=("registration_presets/rescue.yaml",),
            orientation_alignment="direction_only",
            write_input_images=True,
            output_subdir="registration_runs/rescue/gs0p06",
            images=(
                RegistrationImageSpec(
                    "template",
                    "/templates/template.nii.gz",
                    "template_space",
                    "LSP",
                    20,
                    (
                        RegistrationSegmentationSpec(
                            "labels", "/templates/labels.nii.gz"
                        ),
                    ),
                ),
                RegistrationImageSpec(
                    "A", "/subjects/A/ch1_native20um.nii.gz", "A", "LAS", 20
                ),
            ),
            image_to_template=(("A", "template"),),
            image_default_resolution_um=20,
            template_role="fixed",
            moving_segmentations_enabled=True,
            moving_segmentation_interpolation="genericLabel",
            moving_segmentations_write_intermediates=False,
        )
    )

    parsed = tomllib.loads(text)
    assert parsed["images"]["A"]["orientation"] == "LAS"
    assert parsed["images"]["template"]["segmentations"]["labels"].endswith(
        "labels.nii.gz"
    )
    assert parsed["batch"]["image_to_template"] == {"A": "template"}


def test_hpc_helper_carries_per_job_memory_without_hardcoded_project_root() -> None:
    text = render_hpc_submission_helper(
        batch_id="batch001",
        entries=(
            HpcSubmissionEntry("ordinary.toml"),
            HpcSubmissionEntry("high_memory.toml", 256),
        ),
        helper_filename="submit_rescues_hpc.sh",
        job_name_prefix="project_batch001_rescue",
        log_dir="logs/rescue",
    )

    assert '"ordinary.toml|"' in text
    assert '"high_memory.toml|256"' in text
    assert "runtime_contract" in text
    assert "shared_registration/code" not in text


def test_end_to_end_plan_reuses_presets_but_keeps_jobs_subject_specific(
    tmp_path: Path,
) -> None:
    requests = tuple(
        RescueGenerationRequest(
            job=ExpandedRescueJob(
                subject_id=subject,
                strategy_id="higher_gradient_step",
                output_group="higher_gradient_step_rescue",
                variant="gs0p06",
                overrides=(("syn_gradient_step", 0.06),),
                rationale="higher step",
            ),
            registration_spec=_registration_spec(subject),
            base_preset=_base_preset(),
        )
        for subject in ("A", "B")
    )
    plan = plan_rescue_generation(
        batch_id="batch001",
        evaluation_workbook=tmp_path / "evaluation.xlsx",
        configs_dir=tmp_path / "configs",
        helper_path=tmp_path / "submit_rescues_hpc.sh",
        requests=requests,
        job_name_prefix="test_rescue",
    )

    presets = [
        item for item in plan.files if item.purpose == "generated registration preset"
    ]
    configs = [item for item in plan.files if item.purpose == "registration config"]
    job_provenance = [
        item for item in plan.files if item.purpose == "registration job provenance"
    ]
    assert len(presets) == 1
    assert len(configs) == 2
    assert len(job_provenance) == 2
    assert {item.destination.name for item in configs} == {
        "higher_gradient_step_a_gs0p06.toml",
        "higher_gradient_step_b_gs0p06.toml",
    }
    generated = yaml.safe_load(presets[0].content)
    assert generated["registration"]["syn_gradient_step"] == 0.06
    assert generated["registration"]["working_resolution_um"] == 20
    first_job_provenance = json.loads(job_provenance[0].content)
    assert first_job_provenance["subject_image"].endswith("/native.nii.gz")
    assert first_job_provenance["config_sha256"]

    apply_rescue_generation_plan(plan)
    resumed = plan_rescue_generation(
        batch_id="batch001",
        evaluation_workbook=tmp_path / "evaluation.xlsx",
        configs_dir=tmp_path / "configs",
        helper_path=tmp_path / "submit_rescues_hpc.sh",
        requests=requests,
        job_name_prefix="test_rescue",
    )
    assert all(item.state == "already_present" for item in resumed.files)


def test_content_addressed_source_presets_are_shared_by_scientific_content(
    tmp_path: Path,
) -> None:
    requests = tuple(
        RescueGenerationRequest(
            job=ExpandedRescueJob(
                subject_id=subject,
                strategy_id="masking",
                output_group="masking_rescue",
                variant="from_baseline",
                overrides=(),
                rationale="masked retry",
                source_run="registration_runs/baseline",
            ),
            registration_spec=_registration_spec(subject),
            base_preset=_base_preset(),
            content_addressed_preset=True,
        )
        for subject in ("A", "B")
    )

    plan = plan_rescue_generation(
        batch_id="batch001",
        evaluation_workbook=tmp_path / "evaluation.xlsx",
        configs_dir=tmp_path / "configs",
        helper_path=tmp_path / "submit_rescues_hpc.sh",
        requests=requests,
        job_name_prefix="test_rescue",
    )

    presets = [
        item for item in plan.files if item.purpose == "generated registration preset"
    ]
    assert len(presets) == 1
    assert presets[0].destination.name.startswith("registration_")


def test_content_addressed_provenance_reuses_equivalent_scientific_preset(
    tmp_path: Path,
) -> None:
    first = RescueGenerationRequest(
        job=ExpandedRescueJob(
            subject_id="A",
            strategy_id="higher_gradient_step",
            output_group="higher_gradient_step_rescue",
            variant="gs0p06",
            overrides=(("syn_gradient_step", 0.06),),
            rationale="higher step",
        ),
        registration_spec=_registration_spec("A"),
        base_preset=_base_preset(),
        content_addressed_preset=True,
    )
    updated_base = _base_preset()
    updated_base["registration"]["syn_gradient_step"] = 0.06
    second = RescueGenerationRequest(
        job=ExpandedRescueJob(
            subject_id="B",
            strategy_id="masking",
            output_group="masking_rescue",
            variant="from_baseline",
            overrides=(),
            rationale="masked retry",
            source_run="registration_runs/baseline",
        ),
        registration_spec=_registration_spec("B"),
        base_preset=updated_base,
        content_addressed_preset=True,
    )
    kwargs = dict(
        batch_id="batch001",
        evaluation_workbook=tmp_path / "evaluation.xlsx",
        configs_dir=tmp_path / "configs",
        helper_path=tmp_path / "submit_rescues_hpc.sh",
        job_name_prefix="test_rescue",
    )

    combined = plan_rescue_generation(**kwargs, requests=(first, second))
    assert len([item for item in combined.files if item.purpose == "preset provenance"]) == 1
    first_plan = plan_rescue_generation(**kwargs, requests=(first,))
    apply_rescue_generation_plan(first_plan)
    second_plan = plan_rescue_generation(**kwargs, requests=(second,))
    provenance = next(
        item for item in second_plan.files if item.purpose == "preset provenance"
    )
    assert provenance.state == "already_present"
    apply_rescue_generation_plan(second_plan)


def test_content_addressed_provenance_rejects_mismatched_scientific_hash(
    tmp_path: Path,
) -> None:
    request = RescueGenerationRequest(
        job=ExpandedRescueJob(
            subject_id="A",
            strategy_id="higher_gradient_step",
            output_group="higher_gradient_step_rescue",
            variant="gs0p06",
            overrides=(("syn_gradient_step", 0.06),),
            rationale="higher step",
        ),
        registration_spec=_registration_spec("A"),
        base_preset=_base_preset(),
        content_addressed_preset=True,
    )
    kwargs = dict(
        batch_id="batch001",
        evaluation_workbook=tmp_path / "evaluation.xlsx",
        configs_dir=tmp_path / "configs",
        helper_path=tmp_path / "submit_rescues_hpc.sh",
        requests=(request,),
        job_name_prefix="test_rescue",
    )
    plan = plan_rescue_generation(**kwargs)
    apply_rescue_generation_plan(plan)
    provenance = next(
        item.destination for item in plan.files if item.purpose == "preset provenance"
    )
    changed = json.loads(provenance.read_text(encoding="utf-8"))
    changed["scientific_sha256"] = "not-the-same-preset"
    provenance.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(FileExistsError, match="scientific preset"):
        plan_rescue_generation(**kwargs)


def test_immutable_job_collision_stops_planning(tmp_path: Path) -> None:
    request = RescueGenerationRequest(
        job=ExpandedRescueJob(
            subject_id="A",
            strategy_id="padding",
            output_group="padding_rescue",
            variant="pad500um",
            overrides=(("fixed_padding_um", 500),),
            rationale="padding",
        ),
        registration_spec=_registration_spec("A"),
        base_preset=_base_preset(),
    )
    kwargs = dict(
        batch_id="batch001",
        evaluation_workbook=tmp_path / "evaluation.xlsx",
        configs_dir=tmp_path / "configs",
        helper_path=tmp_path / "submit_rescues_hpc.sh",
        requests=(request,),
        job_name_prefix="test_rescue",
    )
    plan = plan_rescue_generation(**kwargs)
    apply_rescue_generation_plan(plan)
    config_path = next(
        item.destination for item in plan.files if item.purpose == "registration config"
    )
    config_path.write_text("changed", encoding="utf-8")

    with pytest.raises(FileExistsError, match="exists and differs"):
        plan_rescue_generation(**kwargs)
