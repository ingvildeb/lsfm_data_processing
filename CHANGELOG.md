# Changelog

## [0.2.0] - 8/28/2026

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
