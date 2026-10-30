# Registration Config Setup

This guide explains the batch and sweep registration TOML files used by `atlasspace` and `lsfm_data_processing`.
It is independent of where the workflow is launched. For Slurm/HPC deployment and submission commands, see
`docs/registration_hpc_setup.md`.

## Project Layout

Registration configs usually live in a small project folder:

```text
my_registration_project/
  subjects/
  configs/
  logs/
```

Subject images are typically organized by subject:

```text
my_registration_project/
  subjects/
    subject_001/
      ch1_iso20um.nii.gz
    subject_002/
      ch1_iso20um.nii.gz
```

## Batch Registration

Batch registration runs one preset across many image/template pairs.

### `[run]`

Typical fields:

- `registration_presets`
- `output_subdir`
- `orientation_alignment`
- `write_input_images`

Batch registration should define exactly one preset:

```toml
[run]
registration_presets = ["tuned_syn_cc"]
output_subdir = "registration"
orientation_alignment = "moving_to_fixed"
write_input_images = false
```

`output_subdir` is written under the parent folder of each run image. If a run image is:

```text
/path/to/my_registration_project/subjects/subject_001/ch1_iso20um.nii.gz
```

and `output_subdir = "registration"`, outputs go to:

```text
/path/to/my_registration_project/subjects/subject_001/registration/
```

### `[image_defaults]`

Use this for shared image metadata:

```toml
[image_defaults]
orientation = "las"
resolution_um = 20.0
```

Individual image entries can override these defaults.

### `[moving_segmentations]`

Use this section when segmentations attached to the moving image should be transformed after registration:

```toml
[moving_segmentations]
enabled = true
interpolation = "genericLabel"
write_intermediates = false
```

Interpolation should usually be:

- `genericLabel` for masks and label volumes
- `nearestNeighbor` only when you explicitly want nearest-neighbor resampling

If `output_subdir` is omitted under `[moving_segmentations]`, transformed segmentations are written into the same
registration output folder.

### `[images.<name>]`

Define every image used by the batch:

- all run images
- all template images

Example:

```toml
[images.subject_001]
image = "/path/to/my_registration_project/subjects/subject_001/ch1_iso20um.nii.gz"
space_name = "subject_001"
orientation = "las"
resolution_um = 20.0

[images.template_p56]
image = "/path/to/shared_registration/templates/template_p56_20um.nii.gz"
space_name = "template_p56"
orientation = "lsp"
resolution_um = 20.0
```

You can optionally attach segmentations to whichever image may be moving:

```toml
[images.template_p56.segmentations]
labels = "/path/to/shared_registration/templates/template_p56_labels_20um.nii.gz"
mask = "/path/to/shared_registration/templates/template_p56_mask_20um.nii.gz"
```

### `[batch]`

This section defines how run images are paired to template images:

```toml
[batch]
template_role = "moving"

[batch.image_to_template]
"subject_001" = "template_p56"
"subject_002" = "template_p56"
```

Interpretation:

- keys are run-image ids
- values are template-image ids
- `template_role` decides whether the template is treated as fixed or moving

With `template_role = "moving"`, segmentations attached to the template are moving-image segmentations and can be
propagated to each subject when `[moving_segmentations].enabled = true`.

## Sweep Registration

Sweep registration tests one or more presets across one shared image and multiple run images.

### `[run]`

Typical fields:

- `registration_presets`
- `output_root`
- `orientation_alignment`
- `write_input_images`

Example:

```toml
[run]
registration_presets = ["baseline_syn_kimlab", "tuned_syn_cc"]
output_root = "../outputs/sweep_registration"
orientation_alignment = "moving_to_fixed"
write_input_images = false
```

`output_root` is interpreted relative to the location of `sweep_register.toml`. If your config lives under `configs/`,
using:

```toml
output_root = "../outputs/sweep_registration"
```

places outputs under the project-root `outputs/` folder.

### `[image_defaults]`

Use this for shared image metadata:

```toml
[image_defaults]
orientation = "las"
resolution_um = 20.0
```

### `[moving_segmentations]`

Optional support for propagating one or more segmentation volumes from the moving image:

```toml
[moving_segmentations]
enabled = true
interpolation = "genericLabel"
write_intermediates = false
```

### `[run].registration_presets`

List the preset names to test. These can be:

- built-in `atlasspace` preset names like `baseline_syn_kimlab` or `tuned_syn_cc`
- absolute paths to custom preset YAML files

For custom preset details, see `docs/registration_presets.md`.

### `[images.<name>]`

Define every image available to the sweep, including the shared image and all run images.

Example:

```toml
[images.subject_001]
image = "/path/to/my_registration_project/subjects/subject_001/ch1_iso20um.nii.gz"
space_name = "subject_001"
orientation = "las"
segmentations = { brain_mask = "/path/to/my_registration_project/subjects/subject_001/brain_mask.nii.gz" }

[images.template_p56]
image = "/path/to/shared_registration/templates/template_p56_20um.nii.gz"
space_name = "template_p56"
orientation = "lsp"
resolution_um = 20.0
```

### `[sweep]`

This section defines the shared image for the sweep and which side of the registration it should occupy:

```toml
[sweep]
shared_image = "template_p56"
shared_image_role = "fixed"
run_images = ["subject_001", "subject_002"]
```

Interpretation:

- `shared_image` is the image id reused for every registration pair
- `run_images` are the image ids paired against that shared image
- `shared_image_role` decides whether the shared image is fixed or moving

## Output Locations

Batch outputs usually go under each subject folder:

```text
subjects/subject_001/registration/
```

Sweep outputs go under the sweep `output_root`, for example:

```text
outputs/
  sweep_registration/
    template_p56__subject_001/
      baseline_syn_kimlab/
```
