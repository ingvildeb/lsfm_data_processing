# Cell Quantification Workflow Contract

## Status and purpose

This document records the agreed contract for a small reusable cell-quantification
layer in `lsfm_data_processing`. The first implementation will be tested in Glia
Mapping for routine Sytox and ALDH processing, with later adoption by SING,
Chandelier Cell, and Triple Transgenic.

The initial scope includes point-cloud extraction, deduplication, object
filtering, colocalization, raw regional quantification, subject-space count maps,
reference-space count maps, and the spatial inputs required for hemisphere-aware
quantification. Biological screening is a later workflow and must never overwrite
raw quantification.

## Ownership boundaries

- `spatialsignal` owns scientific primitives and artifact models: point clouds,
  objects and membership, matching, regional quantification, voxelization,
  spatial sidecars, and their validation.
- `lsfm_data_processing` owns reusable orchestration: canonical paths, workflow
  planning, artifact-set validation, reuse and output policies, optional-lineage
  compatibility, and shared project adapters.
- Project repositories own subject discovery, project roots, channel/population
  definitions, scientific parameter values, numbered entry-point scripts, and
  genuinely project-specific extensions.
- AtlasSpace and the registration workflow own atlas transformations and the
  registration manifest, including transforming hemisphere segmentations into
  each subject's registered annotation grid.

The shared layer is not a project tracker, a replacement for `spatialsignal`, or
a general workflow engine.

## Populations and canonical stages

A **channel population** comes from one semantic imaging channel. Its configured
source is either instance masks or an existing point cloud. A **colocalized
population** is a named, materialized population derived from two channel
populations and retains the evidence needed to audit how it was formed.

The internal stages are:

1. point-cloud extraction, when the source is instance masks;
2. deduplication and object filtering;
3. colocalization and materialization of derived populations;
4. raw regional and hemisphere quantification;
5. subject-space count-map generation; and
6. reference-space count-map generation.

These stages define dependencies and validation. They are not an arbitrary public
`STAGES` setting in project scripts.

## Canonical relative layout

All paths are relative to a project-configured analysis root. Existing names for
that root may remain (`_sing` in SING and `_spatialsignal` elsewhere).
`<analysis_lane>` distinguishes scientifically different processing variants.

```text
<analysis_root>/
  registration/
  segmentations/<channel_id>/<detector_input_type>/...
  segmentations/<channel_id>/<analysis_lane>/
    masks/...
    <ID>_pointcloud.parquet
    <ID>_pointcloud_space.json
    <ID>_objects.parquet
    <ID>_objects_space.json
    <ID>_object_membership.parquet
  colocalization/<colocalized_population_id>/<analysis_lane>/
    <ID>_objects.parquet
    <ID>_objects_space.json
    evidence/
      <ID>_detection_matches.parquet
      <ID>_object_relationships.parquet
      <ID>_colocalization_metadata.json
    qc/
      <ID>_colocalization_qc.xlsx
      planes/*.png
  quantification/<population_id>/<analysis_lane>/raw/
    <ID>_objects_with_regions.parquet
    <ID>_objects_with_regions_space.json
    <ID>_region_summary.csv
    <ID>_quantification_qc.png
  quantification/<population_id>/<analysis_lane>/qc_screened/...
  voxel_maps/<population_id>/<analysis_lane>/
    <ID>_count_map.nii.gz
    <ID>_count_map_space.json
    <ID>_reference_count_map.nii.gz
    <ID>_reference_count_map_space.json
```

Rejected detections, object-edge tables, and method-specific QC are optional
members of their respective artifact sets. When enabled, they are validated and
reused as part of the complete set.

The shared production runner reads only this layout. Historical Chandelier Cell
and Triple Transgenic layouts require separate targeted inventory and migration;
they do not become permanent fallback paths.

## Numbered project workflows

Projects retain the applicable fixed, user-facing numbered scripts:

1. `01_build_channel_objects.py` builds channel artifacts. ALDH runs masks to
   point cloud to deduplicated objects; Sytox runs an existing point cloud to
   deduplicated objects.
2. `02_run_colocalization.py` is present only in projects with configured
   colocalized populations. It writes detection matches, object relationships,
   materialized colocalized populations, and configured method QC.
3. `03_quantify_populations.py` writes hemisphere-aware raw regional
   quantification, quantification QC, subject maps, and reference maps. Its
   internal artifact sets remain independently valid and reusable.
4. `04_screen_quantification.py` is reserved for later biological screening.
   It reads raw results and writes only beneath `qc_screened/`.

An unnumbered `audit_cell_quantification.py` performs a read-only project audit
using the same validators as the workflows.

Each numbered script exposes a small consistent operational surface:

- `RUN_MODE`: `plan` or `apply`;
- `EXISTING_OUTPUT_POLICY`: `error`, `skip`, `repair`, or `overwrite`; and
- identifiers appropriate to the script, such as `CHANNEL_IDS`,
  `COLOCALIZATION_IDS`, or `POPULATION_IDS`.

Subject discovery may come from a live tracker, batch workbook, or local settings
through a project adapter. There is no required central cross-project subject
manifest.

## Configuration model

Configuration is format-neutral and intentionally small. JSON, TOML, or typed
Python adapters may provide the same logical fields.

- A channel defines its stable ID, enabled state, source kind
  (`instance_masks` or `existing_pointcloud`), analysis lane, deduplication
  parameters, and optional morphology filter.
- A colocalization defines its stable ID, two input population IDs, enabled
  state, maximum distance, matching rule, component rule, and coordinate rule.
- Quantification identifies registered annotation and hemisphere maps, requires
  hemispheres, declares the atlas/region ID space, and states how background and
  out-of-bounds objects are handled.
- Voxelization enables subject and reference maps. Routine reference maps are
  unmodulated count maps with weight 1 per object.

Project scripts select configured entities; they do not redefine a dependency
graph or select arbitrary internal stages.

## Spatial and hemisphere contract

Hemisphere maps are required for every new regional quantification. Their labels
are `1 = left` and `2 = right`; any midline convention must be explicit and
tested.

Developmental atlas hemisphere sources must be generated on the exact moving
template grid using its affine/orientation. The anatomical midline may be at the
center of a pose-standardized volume, but code must resolve the relevant array
axis from spatial metadata.

For an existing registration, the registration workflow must:

1. transform the hemisphere source with label-preserving interpolation through
   `transform_segmentation_from_registration()`;
2. record `hemispheres` in `registration_result.json` under transformed
   segmentations; and
3. verify that labels and hemispheres occupy the exact same subject grid and
   contain only expected hemisphere values over annotated voxels.

Subject count maps use the exact registered subject annotation grid. Reference
count maps use the exact moving-template grid declared by the registration.
Approximate grids derived only from bounds or nominal resolution are not
canonical outputs.

## Artifact states and output policies

A stage artifact is a complete expected file set, not one representative file.
Validation classifies it as `absent`, `partial`, `valid`, `no_lineage`, `stale`,
or `invalid`. Artifacts without formal input lineage are reported as
`no_lineage` rather than being treated as old or automatically called stale.

Reuse validation is mandatory. Policies behave as follows:

| Policy | Absent | Valid and compatible | Partial, stale, or invalid |
| --- | --- | --- | --- |
| `error` | write | stop | stop |
| `skip` | write | reuse | stop |
| `repair` | write | reuse | rebuild selected artifact set |
| `overwrite` | rebuild | rebuild | rebuild |

Planning reports classifications and proposed actions without writing. Applying
executes only the planned actions and validates completed outputs.

For channel-object workflow planning, projects provide selected subject roots
and validate project-specific detector evidence. The shared layer resolves the
canonical point-cloud and object artifact sets into `ChannelObjectWorkItem`
instances, produces a read-only `ChannelObjectPlan`, and formats that plan for
the numbered entry point. Plan summarization is a presentation helper, not a
third run mode.

Canonical workflow-01 reuse validation is shared. Planning reads SpatialSignal
sidecars and Parquet schemas/footers, but does not scan every coordinate row.
Apply mode performs deep SpatialSignal point validation after loading an input
or newly written dataset. Deduplicated-object reuse additionally validates its
processing stage and configured scientific parameters. Formal direct
point-cloud lineage remains optional; workflow 01 does not invent input
identities when the upstream artifacts do not already provide stable IDs.

Workflow-01 scientific execution is shared as well: mask-centroid extraction,
point-cloud provenance, cross-plane deduplication, canonical SpatialSignal
writers, and mandatory post-write validation. Projects map their operational
worker/progress settings into the shared executor. The shared cohort operation
excludes validated complete items, attempts every independent pending item,
returns structured successes and failures, and raises one aggregate error after
all pending items have been attempted. Existing source point clouds are
read-only under every output policy.

## Optional lineage and scientific-parameter contract

No workflow-level manifest, UUID graph, compatibility hash, or new sidecar schema
is introduced. Existing SpatialSignal sidecars remain authoritative.

New artifacts record scientific parameters that affect their result in the
existing `processing.parameters` field. They do not record custom file names,
sizes, or modification times as lineage: those properties are not stable across
the filesystems used by the projects.

Formal `artifact_id` and `processing.inputs` lineage is optional. Until stable
input IDs are available, an otherwise valid artifact is classified as
`no_lineage` and remains reusable. Validation still checks the complete file
set, SpatialSignal schema and representation, coordinate spaces and grids,
identifiers, membership and relationship integrity, and every recorded
scientific parameter relevant to the requested operation.
The SpatialSignal v0.2 model can read and round-trip formal IDs, but workflow
01 does not yet generate them or migrate existing sidecars. That adoption is a
separate future change and is not a prerequisite for routine processing.

The early Glia pilot wrote a custom `direct_input_signatures` parameter. It is
accepted as historical metadata but is not compared with current filesystem
timestamps and is not written to new artifacts. Those sidecars are not rewritten
merely to remove the field.

One safety rule remains strict: a no-lineage downstream artifact cannot be
reused when the current plan rebuilds its upstream input. It must be regenerated
in the same workflow plan. A future formal lineage mismatch will likewise be
classified as stale, but formal lineage will not be required for reuse.

The read-only project audit uses these same validators. A separate audit-scope
control is unnecessary because reuse validation already occurs at every stage.

## Legacy data and migration

SING and Glia Mapping are close to the canonical structure. New Glia regional
quantification adds the `raw/` boundary. Existing pre-hemisphere quantification
in Glia Mapping, Chandelier Cell, and Triple Transgenic must be archived and
rerun rather than silently adopted as current output.

Archive and migration utilities must be plan-first, collision-safe, preserve old
relative paths, and emit an inventory of moved artifacts. They remain separate
from routine production workflows.

Chandelier Cell and Triple Transgenic require later targeted cleanup to reconcile
historical folder generations. Their maintained colocalization artifact model
informs this contract, but old layouts do not become aliases in the shared runner.

## Initial exclusions

The first implementation excludes biological screening, tracker mutation,
cross-project tracker standardization, project-specific participation metrics,
automatic cleanup of Chandelier Cell or Triple Transgenic, and retrofitting
formal lineage into historical sidecars.

## Implementation sequence

1. **Contract and tests:** implement path, artifact-set, policy, grid, and
   lineage tests alongside the shared core, before replacing a project's
   existing entry points with shared orchestration.
2. **Shared core:** add typed configuration, path builders, artifact validators,
   planning, output-policy handling, and optional-lineage compatibility under
   `lsfm_data_processing.cell_quantification.project_workflow`.
3. **Hemisphere infrastructure:** generate developmental-atlas hemisphere
   sources, add registration backfill, validate exact subject grids, and backfill
   Glia registrations before quantification.
4. **Glia pilot:** add fixed numbered scripts and a project adapter; validate
   Sytox from existing point clouds and ALDH from masks through raw regional
   quantification and exact-grid subject/reference maps.
5. **Glia transition:** inventory and archive pre-hemisphere quantification,
   exercise plan/apply and all output policies, then run the new workflow.
6. **Audit:** add the read-only audit and compatibility validators needed for
   safe reuse of eligible historical upstream artifacts.
7. **Later adoption:** clean up and migrate Chandelier Cell and Triple
   Transgenic explicitly. Align SING after the Glia pilot demonstrates that the
   shared contract covers its established behavior.

## Initial acceptance criteria

The Glia pilot is ready for routine use when:

- plan mode is read-only and accurately reports actions;
- valid complete artifact sets are reused and partial sets are never mistaken
  for complete work;
- changing a recorded scientific parameter prevents stale reuse;
- no-lineage reuse is impossible downstream of an upstream rebuild in the same
  plan;
- Sytox begins at its existing point cloud and ALDH at instance masks;
- when a project configures colocalized populations, they preserve matching and
  relationship evidence;
- every regional result has the required hemisphere interpretation;
- subject and reference maps exactly match their declared annotation grids;
- future biological screening cannot overwrite raw quantification; and
- the same validators drive workflow reuse and the read-only audit.
