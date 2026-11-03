"""Public contract for reusable cell-quantification project orchestration."""

from .artifacts import (
    ArtifactInspection,
    ArtifactSet,
    ArtifactState,
    ExistingOutputPolicy,
    PlannedAction,
    PolicyDecision,
    decide_existing_output,
    inspect_artifact_set,
)
from .grids import NiftiGrid, grids_match, validate_nifti_grid
from .lineage import (
    DIRECT_INPUT_SIGNATURES_KEY,
    FileSignature,
    direct_input_signatures,
    validate_lineage_parameters,
    validate_spatial_sidecar_lineage,
    with_lineage_parameters,
)
from .models import (
    ChannelDefinition,
    ColocalizationDefinition,
    DeduplicationSettings,
    QuantificationSettings,
    SourceKind,
    VoxelizationSettings,
    WorkflowConfiguration,
)
from .paths import CanonicalLayout, canonical_subject_stem
from .planning import ArtifactRequest, WorkflowPlan, build_workflow_plan

__all__ = [
    "ArtifactInspection",
    "ArtifactRequest",
    "ArtifactSet",
    "ArtifactState",
    "CanonicalLayout",
    "ChannelDefinition",
    "ColocalizationDefinition",
    "DIRECT_INPUT_SIGNATURES_KEY",
    "DeduplicationSettings",
    "ExistingOutputPolicy",
    "FileSignature",
    "NiftiGrid",
    "PlannedAction",
    "PolicyDecision",
    "QuantificationSettings",
    "SourceKind",
    "VoxelizationSettings",
    "WorkflowConfiguration",
    "WorkflowPlan",
    "build_workflow_plan",
    "canonical_subject_stem",
    "decide_existing_output",
    "direct_input_signatures",
    "grids_match",
    "inspect_artifact_set",
    "validate_lineage_parameters",
    "validate_nifti_grid",
    "validate_spatial_sidecar_lineage",
    "with_lineage_parameters",
]
