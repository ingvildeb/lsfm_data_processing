# Registration Workflow Follow-Ups

This list records improvements identified while validating the shared
registration workflow across projects. These items are deliberately separate
from active runtime contracts and should be addressed after the current
registration batches have completed.

## Open Items

### Support source-derived automatic rescues

Automatic rescue strategies currently record an optional source run for
provenance, but generate their parameter preset from the project baseline.
Only masking and high-memory retries currently reproduce source-run
parameters. This makes a populated `Source run` field misleading for standard
gradient, resolution, and padding strategies.

Proposed direction: make automatic rescues source-derived when `Source run` is
provided. A blank source run means the project baseline; a populated source
run means load that run's `registration_parameters.yaml`, then apply the
strategy's explicit parameter values while preserving all other parameters.
Rename relative-sounding menu entries to explicit value sweeps, for example
`gradient sweep: 0.01, 0.02, 0.025`, so their meaning remains clear even when
the source run has a non-baseline gradient step.

Source-derived jobs must use source-specific output variants, TOML filenames,
and expected-run status matching to avoid colliding with an equivalent
baseline-derived rescue for the same subject. For example, a 0.06 gradient
step derived from `custom_rescue/pad1000um` should be represented as
`from_custom_rescue_pad1000um_gs0p06`, not merely `gs0p06`.

Do not retrofit this behavior to active or completed batches. Use `custom`
for any required iterative combination until the shared behavior is released.

### Separate human-facing run labels from content-addressed preset IDs

Generated rescue presets currently use content-addressed names such as
`registration_0f48f693ed5a`. These names are deliberately based on the full
scientific preset, rather than a short label such as `pad500um`: the same
readable override can be applied to different inherited parameter sets, and
historical batches can contain different presets with the same readable name.
The hash therefore prevents silent reuse of scientifically different files.

The technical preset ID should not be presented as an evaluator-facing label.
The human-readable identity of a run is its run path, strategy, source run,
and explicit variant/parameter values. Proposed direction: retain the preset
ID in TOMLs, manifests, and provenance, but remove or hide `Preset name` from
the active evaluation view and rely on `Run path` (with an optional
human-readable parameter summary) instead. Preserve backwards-compatible
reading of existing workbooks while introducing the finalized workbook view.

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
