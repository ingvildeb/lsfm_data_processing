# Registration Compatibility Retirement Checkpoint

This document defines the finite checkpoint for removing pre-`v0.3.0`
registration compatibility from `lsfm_data_processing`. Compatibility remains
available only to protect batches started before the shared workflow release;
it is not a permanent alternative workflow.

## Trigger

Perform this checkpoint after every registration batch started before
`lsfm_data_processing v0.3.0` is finalized, including the current Chandelier
Cell work, and before releasing the compatibility-free workflow.

Completed historical batches remain read-only. Reopening one after this
checkpoint requires an explicit canonical conversion; routine numbered scripts
must not silently adopt it.

## Schema Names

Schema numbers are local to each JSON contract:

- `registration_result.json` schema 1 is the `atlasspace` registration-result
  contract reported by the runtime preflight.
- Native-image provenance schema 2 is the local preparation sidecar containing
  source and native-image fingerprints, orientation, resolution, and paths.

The result contract remaining at schema 1 does not imply that native-image
provenance is using its legacy schema.

## Historical Membership Backfill

Cross-batch duplicate checking must become manifest-only before legacy workbook
support is removed.

1. Inventory every batch root used by Glia Mapping, SING, Chandelier Cell, and
   Triple Transgenic.
2. Identify every historical batch that participates in duplicate checking but
   lacks `batch_manifest.json`.
3. Select the authoritative frozen membership source for each batch. If legacy
   membership files disagree, stop and resolve the discrepancy explicitly.
4. Generate canonical `batchNNN.xlsx` and `batch_manifest.json` metadata without
   changing native images, registration outputs, evaluations, or transferred
   results.
5. Compare subject IDs, subject count, normalized ages, template assignments,
   and source paths with the authoritative source.
6. Audit all resulting manifests together and fail on any cross-batch duplicate.
7. Preserve a machine-readable audit report recording each source file, its
   checksum, the generated manifest checksum, and the validation result.
8. Change cross-batch discovery to require manifests and reject legacy workbook
   pairs with a clear historical-conversion error.

The backfill utility is temporary. Remove it after the audit is accepted; do
not add it to routine project workflows.

## Remaining Compatibility Audits

Before removing compatibility branches, confirm:

- Every active prepared image has native-provenance schema 2 and matching
  source/native fingerprints.
- No active preparation depends on replacement of legacy header-validation or
  provenance sidecars.
- No active baseline depends on adopting a project-rendered TOML.
- Every active evaluation workbook uses canonical `Run evaluations` and
  `Rescue requests` sheets without legacy proposal columns or custom-request
  sheets.
- No project storage root contains an unfinished legacy registration
  distribution partial needed for resume.
- No active distribution depends on archiving a pre-workflow canonical
  registration folder.

## Removal Map

Remove implementation and focused compatibility tests together:

- Native-image adoption and sidecar replacement from
  `project_workflow/native_preparation.py`.
- Legacy membership workbook parsing and conversion from
  `project_workflow/batch_membership.py`.
- Existing project-rendered baseline adoption from
  `project_workflow/baseline_generation.py`.
- Legacy rescue-proposal and custom-request migration from
  `project_workflow/workbook_reader.py`.
- Legacy distribution partial promotion and pre-workflow canonical archival
  from `project_workflow/distribution.py`.

Thin project adapters for paths, trackers, channels, templates, orientations,
age mappings, and project-specific distribution finalization remain. They are
not compatibility code.

## Acceptance And Release

Compatibility retirement is complete when:

- Historical duplicate checking uses canonical manifests only.
- All active-batch audits above pass.
- Compatibility tests are replaced by strict canonical-contract tests.
- All shared and project suites pass without legacy fixtures.
- Plan-only preparation, evaluation, rescue, and distribution checks pass in
  all four projects.
- The compatibility ledger contains no unresolved row.

Release the removal as `lsfm_data_processing v0.4.0` because historical inputs
that were previously adopted will instead require explicit conversion.
