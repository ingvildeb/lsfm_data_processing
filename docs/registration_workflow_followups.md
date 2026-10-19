# Registration Workflow Follow-Ups

This ledger distinguishes shared behavior already implemented on `main` for
the `0.3.0` release from migration work that still requires project-level
validation. Active runtime requirements belong in
`registration_workflow_contract.md`; temporary compatibility belongs in
`registration_workflow_migration.md`.

## Deferred Until After 0.3.0

### Replace The `BatchSubject` Compatibility Adapter After 0.3.0

**Decision:** defer this internal cleanup until after the 0.3.0 release.

`BatchSubject` is project-local and is not part of the shared public API, so
retaining it does not constrain 0.3.0 users. Removing it now would introduce
cross-project churn after workflow validation without changing runtime
behavior. The current adapters also carry details that are deliberately absent
from canonical membership records: Glia tracks the source spreadsheet row,
project code converts portable string paths into local `Path` objects, and
rescue generation sometimes constructs a transient subject description rather
than a new membership record.

Revisit the adapter after 0.3.0. Remove it only when shared interfaces can
accept `BatchMember` without forcing tracker-only or rescue-only state into the
membership schema, all four projects have migrated to those interfaces, and
their intended-behavior suites pass. A small project-specific selection record
may remain where tracker provenance is genuinely local.

## Remaining Release Work

### Complete Project Adapter Validation

Glia Mapping and SING require final commits and plan-only checks against
`main`. Chandelier Cell requires final adapter alignment and updated tests.
Triple Transgenic still requires migration. The detailed state and temporary
compatibility rules are recorded in `registration_workflow_migration.md`.

## Implemented On Main For 0.3.0

### Complete Staged Masking Dry Runs

The shared mask plan reports derived outputs and exposes completed-mask plans
that become rescue-ready after application. Adapter tests in Glia Mapping,
SING, Chandelier Cell, and Triple Transgenic verify that plan mode reports the
future mask-derived writes without applying them and that apply mode performs
the mask step, replans, and generates the rescue in the same invocation.

### Source-Derived Automatic Rescues

A blank source run uses the project baseline. A populated source run loads that
run's `registration_parameters.yaml`, applies only the requested strategy
overrides, and preserves the remaining parameter and input-image context.
Source-specific variants and output paths prevent collisions with equivalent
baseline-derived rescues. Follow-ups to mask-assisted runs inherit the masked
input, and an accepted mask can be reused with parameters from another run
without creating a second mask workspace.

The canonical menu uses explicit value-sweep labels so its behavior remains
clear when the source run is non-baseline.

### Intentional In-Place Mask Replacement

The confirmation-gated replacement action removes only generated mask-derived
artifacts, the corresponding mask-rescue output, and its human evaluation. It
preserves the edited completed mask and review files, resets the existing
request, and regenerates the same canonical paths. File edits alone never
trigger replacement.

### Human-Facing Run Labels

Content-addressed preset IDs remain in TOMLs, manifests, provenance, and the
evaluation workbook's hidden internal column. Evaluators identify runs by the
visible, readable run path rather than names such as
`registration_0f48f693ed5a`.

### Consolidated Rescue Provenance

One batch-level `configs/rescue_generation_manifest.json` replaces per-TOML
provenance sidecars. It records generated jobs, source runs, effective
parameters, rationale, resources, and subject-image provenance. Masked inputs
include byte size and SHA-256; submission preflight and job startup reject a
missing or stale HPC copy.

### Reconstructed Submission Helpers

Submission helpers are disposable derived artifacts. They are rebuilt from
the consolidated generation manifest and current run results, include
generated jobs without a terminal result, and omit both successful and failed
terminal jobs.

### Concise HPC Submission Output

Dry runs print one human-readable readiness line per subject with input,
template, working resolution, SyN gradient step, padding, memory, and masked
input verification when applicable. This output was validated on the HPC with
SING batch013. Submission reports are stored under `logs/submission/` rather
than cluttering the batch root.

## Future Ideas

### Registration Workflow Workspace Generator

A future scaffolding command could generate a small project workspace with
numbered registration entry points, configuration templates, a README, and
thin adapters for project paths, channels, templates, orientations, and age
mapping. Shared workflow behavior would remain in `lsfm_data_processing`
rather than being copied into each generated repository.

This is a realistic extension once the `0.3.0` adapter API is stable, but it is
not part of the `0.3.0` release scope.
