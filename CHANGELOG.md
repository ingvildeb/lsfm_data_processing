# Changelog

## [Unreleased]

## [0.5.1] - 2026-11-12

### Added

- Cohort-level workflow-01 application with complete-item exclusion,
  independent subject execution, structured outcomes, and aggregated failures.

### Changed

- Project adapters can now delegate workflow-01 cohort iteration, progress
  reporting, and failure aggregation to the shared layer.

## [0.5.0] - 2026-11-12

### Added

- Canonical workflow-01 work items and read-only planning for channel point
  clouds and deduplicated objects.
- Plan-safe SpatialSignal sidecar and Parquet-footer validation, deep apply-time
  coordinate validation, and lineage-aware object validation.
- Shared mask-centroid extraction and cross-plane deduplication execution with
  mandatory post-write validation and immutable source point clouds.
- A `cell-quantification` dependency extra pinned to `spatialsignal v0.2.0`.
- Acceptance tests for Sytox source reuse, ALDH mask extraction, restart-safe
  reuse, output policies, malformed outputs, and stale lineage.

### Changed

- Workflow-01 planning now uses the canonical point-cloud and object validators
  by default.
- Deduplication artifact validation now includes membership and optional edge
  table schemas.

## [0.4.1] - 2026-11-09

### Added

- Restart-safe selected-registration distribution that creates and validates a
  transformed hemisphere map only for the selected run before backtransfer.

## [0.4.0] - 2026-11-03

### Added

- Reusable cell-quantification contracts for canonical paths, artifact sets,
  output policies, exact-grid validation, lineage validation, configuration,
  and stage planning.
- Developmental hemisphere-source creation and resumable backfill planning for
  existing registration results.
- Acceptance tests covering the shared cell-quantification and hemisphere-map
  contracts used by project repositories.

## [0.3.2] - 2026-10-22

### Fixed

- Registration installations now use `atlasspace v0.2.3`, which records all
  declared and effective image shapes for new registrations and supports
  post-registration segmentation transformation for existing manifests with
  omitted shape metadata.

## [0.3.1] - 2026-10-21

### Fixed

- Registration installations now use `atlasspace v0.2.2`, allowing
  post-registration segmentation transformation when a valid result manifest
  omits the optional declared moving-image shape.

## [0.3.0] - 2026-10-20

### Added

- A reusable `project_workflow` package for building project-specific
  registration pipelines around shared preparation, evaluation, rescue, and
  distribution behavior.
- Immutable batch membership manifests with cross-batch subject checks and
  controlled adoption of existing project batches.
- Resumable native-image preparation with header validation, deterministic
  provenance, and verified staging for HPC registration.
- Canonical evaluation workbooks that synchronize discovered runs while
  preserving human decisions, support multiple rescue requests per subject,
  and warn about unresolved subjects without an active next step.
- A shared rescue catalog with standard parameter sweeps, custom rescues,
  high-memory retries, and source-derived rescues that inherit parameters and
  masked or unmasked inputs from an existing run.
- A mask-assisted rescue workflow with 50 um review files, completed-mask
  validation, masked native-image generation, provenance, and explicit
  replacement of a revised mask rescue.
- Deterministic rescue configuration generation with consolidated batch-level
  provenance and reconstruction of submission helpers for unfinished jobs.
- Atomic, retry-safe distribution of selected registration results, including
  resumable copies and archival handling for alternate runs and exclusions.
- Concise batch-registration HPC preflight and submission summaries with
  missing-input checks, masked-input SHA-256 verification, and submission
  reports stored under `logs/submission/`.

## [0.2.0] - 2026-08-28

### Added

- Canonical batch and sweep registration workflows powered by `atlasspace`.
- Installable registration and HPC configuration templates.
- HPC submission support with per-registration Slurm jobs.
- Registration runtime preflight and `registration_runtime.json` provenance.
- Batch and sweep registration documentation.

### Changed

- Registration now uses tagged `atlasspace v0.2.0` through the optional
  `registration` dependency group.
- HPC workflows execute installed Python modules rather than shared source
  checkouts.
- Registration modules no longer inject a neighboring `atlasspace/src`
  checkout into `sys.path`.
