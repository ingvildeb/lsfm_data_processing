# Registration Workflow Follow-Ups

This list records improvements identified while validating the shared
registration workflow across projects. These items are deliberately separate
from active runtime contracts and should be addressed after the current
registration batches have completed.

## Open Items

### Support source-derived automatic rescues

Implementation status: implemented on the development branch; awaiting
cross-project candidate validation.

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

Source-derived rescues must inherit the full execution context, not only the
parameter snapshot. In particular, a follow-up to a mask-assisted run must
reuse that run's masked subject-image input as well as its registration
parameters. Store and resolve this source input from the generated job
provenance or registration configuration, rather than assuming the project's
unmasked native image.

The workflow must also support an alternate parameter source with an already
accepted mask. This is distinct from requesting a new masking operation: it
reuses the existing subject-level masked native image and must not create a
second mask-review workspace, reapply a mask, or overwrite that shared masked
image. For example, a subject with an accepted mask may be rerun using the
parameter snapshot from `padding_rescue/wr20um_pad500um`; the output should
remain source-derived and distinct while the mask input is reused unchanged.

### Support intentional replacement of a mask rescue in place

Implementation status: implemented on the development branch; awaiting
adapter wiring and cross-project candidate validation.

When post-registration review shows that a completed mask was not optimal,
the practical workflow is often to replace that mask rescue rather than retain
an inferior run and create a parallel revision workspace. The current workflow
does not support this: editing `brain_mask_50um_complete.nii.gz` leaves the
previous native mask and masked subject image in place, and the terminal rescue
status prevents regeneration.

Proposed direction: provide an explicit, confirmation-gated `replace mask
rescue` action for a source run. It deliberately removes only the generated
mask-rescue output, generated native mask, generated masked subject image, and
the corresponding human evaluation for that output run; it never deletes the
user-edited completed mask or review files. It then resets the existing masking
request to pending and regenerates the same paths from the edited completed
mask. This keeps the ordinary correction workflow simple while ensuring the
replacement is never triggered implicitly by a file edit.

Do not infer replacement merely because a completed-mask file was edited. The
explicit action prevents accidental resubmission and makes the intentional
replacement visible in the workflow.

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

Implementation status: implemented on the development branch; awaiting
project-adapter wiring and candidate validation.

The candidate workflow replaces per-TOML provenance sidecars with one
batch-level `configs/rescue_generation_manifest.json`. It records each
generated TOML, subject, strategy, source run, effective parameter overrides,
workbook rationale, generation timestamp, and package provenance. Effective
parameters remain in each TOML, and registration outputs retain runtime
provenance.

The consolidated manifest must also record the size and SHA256 digest of every
subject-image input, especially generated masked images. Submission preflight
and job startup must verify the transferred HPC input against that record so a
revised local mask cannot accidentally be paired with a stale remote masked
image. The compact successful check should read, for example,
`IEB0158 input: ch2_native20um_masked.nii.gz. verified sha256: yes`. A mismatch
must stop submission with a direct instruction to transfer the updated file.

### Reconstruct submission helpers from unfinished generated requests

Implementation status: implemented on the development branch; awaiting
project-adapter wiring and candidate validation.

A rescue request marked `generated` may outlive, or become disconnected from,
its batch submission helper. The candidate workflow reconstructs the helper
from the consolidated generation manifest and current run manifests without
regenerating scientifically immutable TOMLs or asking the user to reset the
workbook status manually.

Include generated requests that have no terminal registration result. Exclude
requests whose expected runs are complete or failed. Helper generation must be
deterministic and resumable when only a subset of expanded jobs has finished.

### Make masking dry-runs describe both preparation stages

Implementation status: the shared mask plan now reports its derived outputs
and exposes completed-mask plans that become rescue-ready after application.
Project adapters must use that staged readiness when building their plan-only
rescue artifact plan; this remains to be verified during project migration.

Mask-assisted rescue preparation has two local stages: applying a completed
review mask, then generating the registration rescue that consumes the masked
image. Plan-only output currently reports the first action while sometimes
claiming that the second stage writes no files because those artifacts are
planned only after application.

The dry-run must present the complete staged outcome without writing either
stage. It should distinguish files produced by mask application from rescue
artifacts that become valid once those planned mask outputs exist.

### Simplify HPC dry-run output

Implementation status: implemented on the development branch; awaiting
candidate validation on HPC.

The candidate workflow prints one concise line per subject. It reports whether
required inputs were found, the subject input filename, rescue-input SHA-256
verification when applicable, template identity, working resolution, SyN
gradient step, fixed padding, and requested memory. Missing inputs are reported
as errors before any job is submitted, existing outputs are visibly skipped,
and successful submissions receive a short confirmation. Runtime details,
full paths for successful inputs, and raw `sbatch` commands are suppressed.
