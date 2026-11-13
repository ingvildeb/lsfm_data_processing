from pathlib import Path

import pytest

from lsfm_data_processing.cell_quantification.project_workflow import (
    ArtifactInspection,
    ArtifactRequest,
    ArtifactSet,
    ArtifactState,
    PlannedAction,
    build_workflow_plan,
)


def _inspection(state: ArtifactState):
    def validate(artifact_set: ArtifactSet) -> ArtifactInspection:
        return ArtifactInspection(artifact_set, state)

    return validate


def test_plan_is_read_only_and_propagates_upstream_rebuild_to_no_lineage(
    tmp_path: Path,
) -> None:
    pointcloud = ArtifactSet("pointcloud", (tmp_path / "pointcloud.parquet",))
    quantification = ArtifactSet(
        "quantification", (tmp_path / "region_summary.csv",)
    )
    quantification.paths[0].write_text("no lineage", encoding="utf-8")

    plan = build_workflow_plan(
        (
            ArtifactRequest("pointcloud", pointcloud, _inspection(ArtifactState.VALID)),
            ArtifactRequest(
                "quantification",
                quantification,
                _inspection(ArtifactState.NO_LINEAGE),
                depends_on=("pointcloud",),
            ),
        ),
        "repair",
    )

    assert not pointcloud.paths[0].exists()
    assert plan.by_key["pointcloud"].action is PlannedAction.WRITE
    assert plan.by_key["quantification"].inspection.state is ArtifactState.STALE
    assert plan.by_key["quantification"].action is PlannedAction.REBUILD
    assert plan.can_apply


def test_plan_stops_when_policy_cannot_handle_stale_output(tmp_path: Path) -> None:
    output = ArtifactSet("objects", (tmp_path / "objects.parquet",))
    output.paths[0].write_text("stale", encoding="utf-8")

    plan = build_workflow_plan(
        (ArtifactRequest("objects", output, _inspection(ArtifactState.STALE)),),
        "skip",
    )

    assert plan.by_key["objects"].action is PlannedAction.STOP
    assert not plan.can_apply
    assert "stale -> stop" in plan.summary_lines()[0]


def test_dependencies_must_be_declared_before_their_consumers(tmp_path: Path) -> None:
    output = ArtifactSet("objects", (tmp_path / "objects.parquet",))
    with pytest.raises(ValueError, match="not planned first"):
        build_workflow_plan(
            (
                ArtifactRequest(
                    "objects", output, _inspection(ArtifactState.VALID),
                    depends_on=("pointcloud",),
                ),
            ),
            "skip",
        )
