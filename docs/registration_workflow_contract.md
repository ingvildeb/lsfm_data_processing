# Registration Workflow Contract

## Purpose

This document defines the shared behavioral contract for project registration
workflows built on `atlasspace` and `lsfm_data_processing`. It is the reference
for extracting duplicated workflow code from project repositories without
changing established scientific or operational behavior.

The contract covers final run decisions, rescue requests, mask-assisted
rescues, generated artifact collisions, resumability, distribution, and adding
or replacing transformed segmentations after registration.

## Ownership Boundaries

- `atlasspace` owns spatial definitions, transforms, registration results, and
  the `registration_result.json` contract.
- `lsfm_data_processing` owns reusable batch workflow behavior: evaluation
  workbooks, rescue planning, mask-assisted rescue preparation, HPC submission,
  distribution, and batch-level orchestration.
- Project repositories own tracker selection, project paths, templates,
  age/template assignments, and decisions about when shared operations run.
  `lsfm_data_processing` owns the canonical rescue menu and parameter values.

## Common Terms

- **Run**: one registration attempt with one subject, input set, and parameter
  set, represented by its output directory and registration result manifest.
- **Final decision**: either one selected run or an explicit exclusion for a
  subject.
- **Rescue request**: a request to generate one predefined rescue strategy for
  one subject. A strategy may expand to multiple variants.
- **Source run**: an evaluated run that motivated a request. It is required
  when the new operation must reproduce or derive data from that exact run.
- **Artifact identity**: the subject and scientific settings that determine a
  generated file or output directory.

## 1. Final Run Decisions

1. A subject may have zero or one final-decision row.
2. Multiple selected rows for one subject are invalid and stop preflight.
3. A selected, non-excluded run must exist and have a successful, valid
   `registration_result.json`.
4. `Result = exclude` together with a selected marker is a finalized exclusion.
   Available runs are archived, but no canonical `registration/` directory is
   created.
5. A subject without a final decision remains unresolved. Distribution skips
   it without failing the rest of the batch.
6. Preflight must prominently summarize selected runs, exclusions, and
   unresolved subjects before changes are applied.

### Registration Run Inventory

Evaluation refresh discovers runs from generated registration TOMLs and
`registration_result.json` manifests before synchronizing the workbook.

1. Generated TOMLs define planned subject/run identities.
2. A valid manifest defines an executed run and is authoritative for its
   preset name and `success` or `failed` status.
3. A planned run with an existing output directory but no manifest is
   `missing_manifest`; one without an output directory is `planned`.
4. Invalid JSON, an unsupported manifest schema, or a manifest that fails the
   AtlasSpace result model stops the complete workbook refresh.
5. Runs remain discoverable from either `<subject>` or
   `<subject>_transferred`.
6. The simultaneous presence of both directory forms is a conflicting state
   and stops refresh. Reopening transferred subjects is not currently
   supported.
7. Workbook synchronization preserves human-entered evaluations and rescue
   requests while updating discovered run metadata.

## 2. Rescue Requests

The standard rescue menu is identical for every project workflow and is not
filtered or overridden by project configuration. It contains padding, lower
gradient step, padding plus lower gradient step, higher gradient step, higher
gradient step plus padding, bidirectional gradient step, resolution sweep,
padding plus resolution sweep, high-memory retry, masking, and custom.
Unusual parameter combinations use `custom` rather than adding a
project-specific predefined strategy.

Scientific parameter changes and HPC resources are represented separately.
High-memory retry therefore reproduces its required source run unchanged while
requesting 256 GB for the submitted job.

### Workbook Structure

The evaluation workbook has two conceptual tables:

- **Run evaluations**: one row per discovered registration run.
- **Rescue requests**: one row per requested rescue strategy.

The rescue table supports more than one request for the same subject. Ordinary
Excel data validation is used; VBA and macro-based multi-select controls are
not part of the workflow.

The source-run field follows strategy-specific requirements:

- Optional but recommended for predefined parameter sweeps.
- Required for an exact retry of an existing run.
- Required for masking.

The workbook generator should provide subject and source-run validation lists.
When a subject is chosen, the available source runs must be limited to runs
known for that subject.

### Request Behavior

1. A final decision closes all rescue requests for that subject. Historical
   requests remain visible but are reported as `ignored_finalized`.
2. Multiple distinct rescue requests may be active for an unresolved subject.
3. Exact duplicate requests do not produce duplicate configurations or jobs.
4. A stable request identity includes the subject, strategy, expanded variant,
   and any required source-run or mask revision information.
5. Previously generated, completed, or failed variants are discovered from
   their artifacts and are not generated again automatically.
6. A failed variant requires an explicit retry request. It is not silently
   resubmitted.
7. Request status is derived by synchronization code, not manually entered.

Supported request states include `pending`, `generated`, `completed`, `failed`,
`awaiting_manual_mask`, `mask_ready`, `blocked_pending_rescues`,
`ignored_finalized`, and `conflict`.

## 3. Mask-Assisted Rescue

Masking is the one shared manual rescue workflow. It is a final-stage rescue,
not a generic manual-action extension point.

### Sequencing

1. Ordinary parameter rescues may run in parallel.
2. Their outputs are evaluated.
3. If masking is still required, the user chooses the best existing run as the
   masking source.
4. A masking request is blocked while ungenerated ordinary requests remain for
   the subject.
5. The masked rerun reproduces the source run's registration parameters and
   changes only the fixed-image input.

### Review Artifacts

Mask preparation writes subject-specific files beneath the registration batch,
for example under `registration_masks/<source-run>/`:

```text
brain_mask_50um_draft.nii.gz
fixed_image_50um_reference.nii.gz
brain_mask_50um_complete.nii.gz
brain_mask_native_applied.nii.gz
fixed_image_native_masked.nii.gz
masking_provenance.json
```

Behavior:

1. The draft binary mask is generated from the source run's transformed atlas
   segmentation.
2. The fixed reference and draft mask are written at 50 um for manual review.
3. Generated drafts never overwrite a completed mask.
4. The workflow waits until `brain_mask_50um_complete.nii.gz` exists.
5. The completed mask must be nonempty, binary, and on the expected review
   grid.
6. It is resampled with nearest-neighbor interpolation onto the original fixed
   image grid.
7. The native-grid mask is applied to the original fixed image and written as
   a separate masked image. The original fixed image is never modified.
8. The rescue TOML points only the masking rescue to the masked fixed image.
9. The source run's parameter snapshot is reused for the masked registration.
10. Mask provenance records the source run, source files, derived files, and
    completed-mask content identity used by the registration.

## 4. Artifact Identity And Collisions

Generated files are divided into three categories.

### Shared Presets

A preset filename identifies one parameter combination. An identical existing
preset is reused. Different content at the same semantic preset name is a
conflict.

Standard rescue presets use readable names derived from the baseline preset
and scientific overrides. Source-derived presets used by masking and exact
retries are named by scientific-content hash. They are never named for a
subject, and equivalent source parameter snapshots reuse the same preset.

Preset provenance records only scientific origin and overrides. Subject,
request, source-run, input-image, output, config hash, and HPC-resource
provenance are stored beside each subject-specific registration TOML.

### Registration Configurations

Baseline batch configurations may contain multiple subjects because batch
membership is frozen before generation.

Rescue configurations are subject- and variant-specific. Adding another
subject to an existing rescue strategy creates another TOML rather than
changing an earlier TOML. A representative name is:

```text
102371__resolution_rescue__wr40um.toml
```

An identical existing configuration is reused. Different content for the same
artifact identity is a conflict.

### Derived Submission Helpers

Submission helpers are generated indexes of pending configurations. They may
be regenerated atomically when the request set changes, provided they are
clearly marked as generated and are not treated as user-editable scientific
records.

### Registration Outputs

Registration output directories are not silently overwritten. A successful
matching manifest is complete; a failure manifest is failed; a directory with
no valid manifest is incomplete and requires explicit recovery or retry.

## 5. Resumability

Resumability means orchestration can be rerun safely after interruption or
after adding work. It does not mean ANTs can continue a registration midway.

All mutating workflows follow:

```text
build plan -> summarize -> confirm -> apply
```

Planning performs no writes and classifies operations using a shared
vocabulary:

- `pending`: required output is absent.
- `in_progress`: a recognized temporary artifact or progress marker exists.
- `complete`: output exists, validates, and matches its expected identity.
- `failed`: an explicit failure manifest exists.
- `blocked`: a prerequisite is unavailable.
- `conflict`: an existing artifact differs from the expected artifact.

Rules:

1. Complete operations are validated and skipped.
2. Missing operations are created.
3. Safe file-by-file operations resume after interruption.
4. Differing partial or completed files are conflicts rather than overwrite
   targets.
5. Temporary files and atomic replacement are used where practical.
6. Completion markers are written only after output validation.
7. Evaluation synchronization preserves notes, decisions, and historical
   requests while appending newly discovered runs.
8. Submitted HPC job identifiers are recorded when available so rerunning a
   submit helper does not accidentally duplicate known submissions.
9. ANTs failures require an explicit retry identity or attempt. They are
   restartable, not resumable.
10. Distribution verifies identical existing files, copies missing files, and
    stops on differing files.
11. A source subject folder receives `_transferred` only after all planned
    files and decision metadata have been verified.

## 6. Post-Registration Segmentation Transformation

AtlasSpace will expose a registration-specific wrapper with the public name:

```python
transform_segmentation_from_registration(
    registration_dir,
    segmentation_name,
    segmentation_path,
    *,
    replace=False,
)
```

The wrapper reuses the existing `transform_segmentation()` implementation. It
does not introduce another transformation algorithm.

### Input Contract

1. The segmentation array must already occupy the registration moving
   template's voxel grid: the same voxel ordering, orientation, resolution,
   and shape.
2. The moving space in `registration_result.json` is authoritative. The source
   NIfTI header is not used to infer a different orientation.
3. The loaded array shape is validated against the declared moving shape.
4. AtlasSpace prepares a temporary NIfTI with spatial metadata reconstructed
   from the declared moving space before applying the registration transforms.
5. A segmentation in another real grid or orientation must first be explicitly
   reoriented or transformed onto the moving-template grid.

### Naming And Manifest Contract

`segmentation_name` is both the manifest key and output filename prefix:

```text
labels      -> labels_WarpedSegmentation.nii.gz
hemispheres -> hemispheres_WarpedSegmentation.nii.gz
```

After transformation, both mappings are updated:

```text
moving_image.segmentations[segmentation_name] = segmentation_path
transformed_segmentations[segmentation_name] = transformed_output_path
```

The transformed output must use label-preserving interpolation and match the
registration fixed/subject reference grid.

### Add And Replace

- **Add** is idempotent. A matching source entry with a valid transformed
  output is a no-op. Missing output from the same source is recreated. An
  unmanifested canonical output left by an interrupted add is recreated and
  attached. An existing name associated with another source is a conflict.
- **Replace** is explicit through `replace=True`. It transforms the requested
  source to a temporary output, replaces the canonical transformed file, and
  replaces both manifest entries. It does not retain a backup or history copy.

The manifest is written through a temporary file after the transformed output
has been validated.

## Behavioral Test Matrix

The scenarios below are acceptance criteria. They should become automated
tests as the shared implementation is extracted.

### Final Decisions

| Scenario | Expected behavior |
| --- | --- |
| No selected row | Subject remains unresolved and distribution skips it. |
| One successful selected run | Run is accepted as the canonical selection. |
| Two selected rows | Preflight fails before writes. |
| Selected run is absent | Preflight fails before writes. |
| Selected run manifest is unsuccessful | Preflight fails before writes. |
| Selected exclusion row | Runs are archived and no canonical registration is created. |

### Rescue Requests

| Scenario | Expected behavior |
| --- | --- |
| Two different requests for one subject | Both are planned. |
| Exact duplicate request rows | One request identity is planned and duplication is reported. |
| Selected subject retains old requests | Requests are reported as `ignored_finalized`. |
| Ordinary request omits source run | Request is valid when the strategy fully defines its parameters. |
| Mask or exact retry omits source run | Preflight fails. |
| Completed request remains in workbook | No new config or job is generated. |
| Failed request remains in workbook | It remains failed and is not automatically retried. |
| New request is added in a later round | Only the new request artifacts are generated. |

### Run Inventory

| Scenario | Expected behavior |
| --- | --- |
| Config exists but no output directory exists | Record the run as `planned`. |
| Output directory exists without a manifest | Record the run as `missing_manifest`. |
| Valid success or failure manifest exists | Manifest status and preset are authoritative. |
| Manifest is invalid or uses an unsupported schema | Stop the complete workbook refresh. |
| Only `_transferred` subject directory exists | Continue discovering its historical runs. |
| Active and `_transferred` directories both exist | Stop and require explicit state resolution. |

### Mask-Assisted Rescue

| Scenario | Expected behavior |
| --- | --- |
| Mask requested while ordinary requests are pending | State is `blocked_pending_rescues`. |
| No draft exists | Draft mask and 50 um reference are generated. |
| Draft exists but completed mask is absent | State is `awaiting_manual_mask`; no HPC config is generated. |
| Completed mask is empty or nonbinary | Preflight fails. |
| Completed mask grid differs | Preflight fails. |
| Completed mask is valid | Native mask and separate masked fixed image are generated. |
| Original fixed image exists | It remains byte-for-byte unchanged. |
| Masked rescue config is generated | It points to the masked fixed image and reproduces source parameters. |
| Preparation is rerun | Existing valid review and derived artifacts are reused. |

### Collision Handling

| Scenario | Expected behavior |
| --- | --- |
| Shared preset is absent | Write it. |
| Shared preset is identical | Reuse it. |
| Shared preset name exists with different content | Report a conflict. |
| Rescue is later requested for another subject | Create a new subject-specific TOML. |
| Subject-specific TOML is identical | Reuse it. |
| Subject-specific TOML differs | Report a conflict. |
| Submission helper request set changes | Regenerate the helper atomically. |
| Successful matching output manifest exists | Mark the registration complete. |
| Failed output manifest exists | Mark it failed and require an explicit retry. |
| Output directory lacks a valid manifest | Mark it incomplete; do not treat it as success. |

### Baseline Artifact Generation

Projects provide a typed registration batch specification and their operational
HPC defaults. The shared workflow generates four canonical artifacts:

- `configs/baseline.toml` is an immutable scientific/job definition.
- `configs/baseline.provenance.json` is immutable and records the typed inputs
  plus the baseline TOML SHA-256 hash.
- `configs/hpc.toml` is operational state and may be atomically refreshed.
- `submit_baseline_hpc.sh` is a derived operational helper and may be atomically
  refreshed.

Planning preflights the complete bundle before any write. Identical files are
resumed, differing immutable files stop the operation, and only the two
explicitly operational files can be replaced.

### Native Image Preparation

Projects define required channels, source and canonical filenames, the declared
subject space, and which single channel is staged for registration. The shared
workflow validates every required source before applying changes, rewrites only
NIfTI spatial metadata, validates the prepared header, and stages the selected
registration channel through a verified atomic copy.

Canonical schema-2 provenance records the source path/size/modification time,
the prepared-image SHA-256 hash, declared orientation and resolution, operation,
and project metadata. Matching outputs are resumed; source, output, or space
disagreement is a hard conflict. Temporary handling of older provenance is
tracked in `registration_workflow_migration.md`.

### Batch Membership

Every batch has exactly one canonical specification workbook named
`batchXXX.xlsx` and one package-managed `batch_manifest.json`. The workbook
contains `ID`, `age`, and `path`; its location is derived from the batch ID and
is not separately configured.

The first applied preparation freezes normalized subject IDs, recorded ages,
template assignments, and session paths in the manifest. Later formatting or
row-order edits are accepted, but adding, removing, or changing a subject stops
preflight with a difference report. Additional subjects require a new batch.

### Resumability And Distribution

| Scenario | Expected behavior |
| --- | --- |
| Plan mode is run | No files change. |
| Apply confirmation is wrong | No files change. |
| Apply stops after some independent files | Rerun validates completed files and continues missing files. |
| Existing destination is identical | Skip it as complete. |
| Existing destination differs | Stop with a conflict. |
| Workbook synchronization discovers a new run | Append it without changing existing evaluations. |
| Distribution stops after a partial copy | Rerun verifies copied files and continues. |
| Distribution has not fully verified a subject | Do not append `_transferred`. |
| Distribution finishes and verifies metadata | Append `_transferred` last. |
| ANTs job terminates midway | Require an explicit retry/attempt; do not claim continuation. |

### Segmentation Transformation

| Scenario | Expected behavior |
| --- | --- |
| New `hemispheres` source matches moving grid | Write `hemispheres_WarpedSegmentation.nii.gz` and both manifest entries. |
| Source shape differs from moving grid | Fail before changing the manifest. |
| Source header is faulty but array matches declared moving grid | Prepare metadata from the manifest and transform successfully. |
| Add is rerun with the same source and valid output | Return as a no-op. |
| Manifest entry exists but same-source output is missing | Recreate the output. |
| Canonical output is unmanifested after interruption | Recreate it and complete the manifest entry. |
| Add uses an existing name with another source | Fail and require explicit replacement. |
| Replace is requested | Replace the canonical output and both manifest entries. |
| Replace is requested and transformation fails | Preserve the existing canonical output and manifest. |
| Replace succeeds | Do not create a backup or history file. |

## Implementation Sequence

1. Turn the relevant matrix rows into tests around the existing project
   behavior and new shared interfaces.
2. Add `transform_segmentation_from_registration()` to AtlasSpace.
3. Extract shared evaluation, rescue-request, and artifact identity models into
   `lsfm_data_processing`.
4. Implement subject-specific rescue configs and shared masking state handling.
5. Implement common resumability and distribution primitives. *(Implemented.)*
6. Pilot in Chandelier Cell, then Triple Transgenic, Glia Mapping, and SING.
   *(Chandelier Cell pilot implemented; remaining migrations pending.)*
7. Remove project copies only after parity tests pass.
