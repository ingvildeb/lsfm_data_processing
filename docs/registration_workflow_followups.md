# Registration Workflow Follow-Ups

This list records improvements identified while validating the shared
registration workflow across projects. These items are deliberately separate
from active runtime contracts and should be addressed after the current
registration batches have completed.

## Open Items

### Replace the `BatchSubject` compatibility adapter

The project adapters currently convert shared `BatchMember` records into a
project-local `BatchSubject` type for preparation, generation, and rescue
workflow calls. This keeps the current projects stable during migration, but
duplicates a small amount of membership data modeling.

Proposed direction: update the shared preparation, generation, and rescue
interfaces to accept `BatchMember` directly, then remove the project-local
`BatchSubject` adapters once Chandelier, Glia Mapping, Triple Transgenic, and
SING have passed parity validation.

Do not remove the adapters while any project still depends on them.

### Consolidate rescue provenance

The current rescue-generation workflow writes a provenance JSON sidecar for
each generated rescue TOML. This is useful for auditability, but produces a
large amount of repetitive configuration clutter.

Proposed direction: replace per-TOML provenance sidecars with one
batch-level `configs/rescue_generation_manifest.json` that records each
generated TOML, subject, strategy, source run, effective parameter overrides,
workbook rationale, generation timestamp, and package/runtime provenance. Keep
the effective parameters in each TOML and runtime provenance in the
registration output.

Do not remove the current sidecars until the manifest schema and migration
path have been designed and the active project workflows have been checked.

### Simplify HPC dry-run output

The HPC submission dry-run currently prints long shell commands containing
full paths and implementation details. This is difficult to review manually.

Proposed direction: print a compact human-readable summary per submission,
including batch, subject, strategy/config name, resources, exclusion settings,
and the output/log locations. Optionally provide a separate `--verbose` mode
for the exact `sbatch` command when debugging is needed.

The compact output must still make it possible to confirm the number of jobs,
their subjects, resource requests, and excluded nodes without reading the
generated shell command.
