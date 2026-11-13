from pathlib import Path

import pytest

from lsfm_data_processing.cell_quantification.project_workflow import (
    ArtifactInspection,
    ArtifactSet,
    ArtifactState,
    ExistingOutputPolicy,
    PlannedAction,
    decide_existing_output,
    inspect_artifact_set,
)


def _valid(artifact_set: ArtifactSet) -> ArtifactInspection:
    return ArtifactInspection(artifact_set, ArtifactState.VALID)


def test_artifact_set_distinguishes_absent_partial_and_valid(tmp_path: Path) -> None:
    artifact_set = ArtifactSet(
        "objects", (tmp_path / "objects.parquet", tmp_path / "objects.json")
    )
    assert inspect_artifact_set(artifact_set, validator=_valid).state is ArtifactState.ABSENT

    artifact_set.paths[0].write_text("table", encoding="utf-8")
    partial = inspect_artifact_set(artifact_set, validator=_valid)
    assert partial.state is ArtifactState.PARTIAL
    assert "objects.json" in partial.reasons[0]

    artifact_set.paths[1].write_text("{}", encoding="utf-8")
    assert inspect_artifact_set(artifact_set, validator=_valid).state is ArtifactState.VALID


def test_complete_artifact_set_cannot_be_reused_without_validation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "complete.parquet"
    path.write_text("table", encoding="utf-8")
    with pytest.raises(ValueError, match="requires a validator"):
        inspect_artifact_set(ArtifactSet("objects", (path,)), validator=None)


@pytest.mark.parametrize(
    ("policy", "valid_action", "bad_action"),
    [
        (ExistingOutputPolicy.ERROR, PlannedAction.STOP, PlannedAction.STOP),
        (ExistingOutputPolicy.SKIP, PlannedAction.REUSE, PlannedAction.STOP),
        (ExistingOutputPolicy.REPAIR, PlannedAction.REUSE, PlannedAction.REBUILD),
        (ExistingOutputPolicy.OVERWRITE, PlannedAction.REBUILD, PlannedAction.REBUILD),
    ],
)
def test_output_policy_matrix(
    tmp_path: Path,
    policy: ExistingOutputPolicy,
    valid_action: PlannedAction,
    bad_action: PlannedAction,
) -> None:
    artifact_set = ArtifactSet("objects", (tmp_path / "objects.parquet",))
    valid = ArtifactInspection(artifact_set, ArtifactState.VALID)
    stale = ArtifactInspection(artifact_set, ArtifactState.STALE)

    assert decide_existing_output(valid, policy).action is valid_action
    assert decide_existing_output(stale, policy).action is bad_action


def test_absent_sets_are_written_except_overwrite_uses_rebuild(tmp_path: Path) -> None:
    inspection = ArtifactInspection(
        ArtifactSet("objects", (tmp_path / "objects.parquet",)),
        ArtifactState.ABSENT,
    )
    assert decide_existing_output(inspection, "skip").action is PlannedAction.WRITE
    assert decide_existing_output(inspection, "overwrite").action is PlannedAction.REBUILD


def test_no_lineage_output_cannot_survive_an_upstream_rebuild(
    tmp_path: Path,
) -> None:
    inspection = ArtifactInspection(
        ArtifactSet("quantification", (tmp_path / "summary.csv",)),
        ArtifactState.NO_LINEAGE,
    )
    assert decide_existing_output(inspection, "skip").action is PlannedAction.REUSE
    decision = decide_existing_output(inspection, "skip", upstream_rebuilt=True)
    assert decision.inspection.state is ArtifactState.STALE
    assert decision.action is PlannedAction.STOP


def test_legacy_state_value_remains_reusable_for_api_compatibility(
    tmp_path: Path,
) -> None:
    inspection = ArtifactInspection(
        ArtifactSet("objects", (tmp_path / "objects.parquet",)),
        ArtifactState("legacy"),
    )

    assert inspection.state is ArtifactState.LEGACY
    assert inspection.reusable

