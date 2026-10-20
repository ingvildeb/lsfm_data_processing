# Registration Workflow Follow-Ups

This ledger distinguishes shared behavior implemented for the `0.3.0` release
from post-release project rollout and future cleanup. Active runtime
requirements belong in
`registration_workflow_contract.md`; temporary compatibility belongs in
`registration_workflow_migration.md`.

## Post-Release Rollout

### Complete Tagged-Release Validation

All four project adapters and local environments are pinned to `v0.3.0`; the
shared HPC environment also passes the tagged runtime contract. Finish batches
started before the release and validate the next Triple Transgenic batch before
retiring historical compatibility. Temporary rules remain recorded in
`registration_workflow_migration.md`, and the manifest backfill, audit, and
`v0.4.0` removal gate are defined in
`registration_compatibility_retirement.md`.

## Implemented On Main For 0.3.0

### Canonical Batch Membership In Project Adapters

Glia Mapping, SING, Chandelier Cell, and Triple Transgenic now pass shared
`BatchMember` records directly through membership, preparation, and baseline
generation. Local filesystem paths are converted to `Path` objects only at the
preparation boundary. Glia's tracker source row remains in its selection audit
table but is not treated as stable batch identity or written into native-image
provenance.

Rescue generation uses an explicit subject/template specification derived from
the immutable baseline mapping. It no longer constructs placeholder membership
records with values such as `age="baseline"`.

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
