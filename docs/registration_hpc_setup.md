# HPC Registration Setup

This guide is for Kim lab users who want to run `lsfm_data_processing` registration workflows on HPC.

## Installation

### Shared folder layout

Shared scripts and templates are found under:

```text
/gpfs/Labs/Kim/shared_registration/
  code/
    atlasspace/
    lsfm_data_processing/
  templates/
    ...
```

The `code/` folder is for the shared source repos.
The `templates/` folder is for lab-shared template and label sets.


### Create your conda environment

Each user should create and use their own environment.

Example:

```bash
conda create --name lsfm_data_processing python=3.11
conda activate lsfm_data_processing
```


### Install the packages into your environment

Install from the shared source folders.

Recommended stable install:

```bash
pip install /gpfs/Labs/Kim/shared_registration/code/atlasspace
pip install /gpfs/Labs/Kim/shared_registration/code/lsfm_data_processing
```

This is a non-editable local install:

- the package is installed from a local folder, not from PyPI
- later edits to the source repo do not affect your environment until you reinstall

You can verify imports with:

```bash
python -c "import atlasspace; import lsfm_data_processing; print('Imports OK')"
```


## Batch Registration

### Create a registration project folder

Each project should be a small, clean folder with this structure:

```text
my_registration_project/
  subjects/
  configs/
  logs/
```

Example:

```bash
mkdir -p /gpfs/Labs/Kim/Ingvild/my_registration_project/subjects
mkdir -p /gpfs/Labs/Kim/Ingvild/my_registration_project/configs
mkdir -p /gpfs/Labs/Kim/Ingvild/my_registration_project/logs
```

Put your subject folders inside `subjects/`, for example:

```text
my_registration_project/
  subjects/
    IEB0001/
      ch1_iso20um.nii.gz
    IEB0002/
      ch1_iso20um.nii.gz
```


### Copy the config templates into the project

```bash
cp /gpfs/Labs/Kim/shared_registration/code/lsfm_data_processing/lsfm_data_processing/registration_and_transforms/batch_registration/config_templates/batch_register.toml \
  /gpfs/Labs/Kim/Ingvild/my_registration_project/configs/batch_register.toml

cp /gpfs/Labs/Kim/shared_registration/code/lsfm_data_processing/lsfm_data_processing/registration_and_transforms/batch_registration/config_templates/hpc.toml \
  /gpfs/Labs/Kim/Ingvild/my_registration_project/configs/hpc.toml
```


### Edit `batch_register.toml`

Typical fields to check:

#### `[run]`

- `registration_preset`
- `template_role`
- `output_subdir`
- `orientation_alignment`
- `write_input_images`

#### `[subject_defaults]`

- `subjects_dir = "subjects"`
- `image_name`
- `orientation`
- `resolution_um`
- `underscores_to_id`

If your subject folders are just named by subject ID, for example `IEB0001`, set:

```toml
underscores_to_id = 0
```

#### `[segmentation_transform]`

Set:

```toml
enabled = true
```

if you want template segmentations transformed after registration.

If you do not have any segmentations, set:

```toml
enabled = false
```

and remove any `[templates.<name>.segmentations]` section.

#### `[templates.<name>]`

Define one or more templates here, including:

- `image`
- `space_name`
- `orientation`
- `resolution_um`

These can point into the shared lab template folder, for example:

```toml
[templates.lsfm-neun-v1-p56]
image = "/gpfs/Labs/Kim/shared_registration/templates/neun_p56_v1/T_P56_NeuN_v1_20um.nii.gz"
space_name = "lsfm-neun-v1-p56"
orientation = "lsp"
resolution_um = 20.0
```

#### `[templates.<name>.segmentations]`

Optional template segmentation files that should be transformed after registration.

Example:

```toml
[templates.lsfm-neun-v1-p56.segmentations]
labels = "/gpfs/Labs/Kim/shared_registration/templates/neun_p56_v1/L_P56_NeuN_v1_20um.nii.gz"
mask = "/gpfs/Labs/Kim/shared_registration/templates/neun_p56_v1/L_P56_NeuN_v1_20um_mask.nii.gz"
```

#### `[subject_to_template]`

This maps each subject ID to the template name it should use.

Example:

```toml
[subject_to_template]
"IEB0001" = "lsfm-neun-v1-p56"
"IEB0002" = "lsfm-neun-v1-p56"
```


### Edit `hpc.toml`

Typical settings:

```toml
[cluster]
partition = "compute"
cpus_per_task = 12
mem_gb = 128
time = "24:00:00"
conda_env = "lsfm_data_processing"
python_executable = "python"
```

For batch, the workflow section should usually be:

```toml
[workflow]
registration_config = "configs/batch_register.toml"
skip_if_output_exists = true
dry_run = false
job_name_prefix = "reg"
```

Logs usually go to:

```toml
[logging]
log_dir = "logs/registration"
```


### Launch batch registration on HPC

Move into the project root:

```bash
cd /gpfs/Labs/Kim/Ingvild/my_registration_project
```

Activate your environment:

```bash
conda activate lsfm_data_processing
```

Run the batch submit script:

```bash
bash /gpfs/Labs/Kim/shared_registration/code/lsfm_data_processing/lsfm_data_processing/registration_and_transforms/batch_registration/hpc/submit_batch_register.sh
```

This script:

- reads `configs/hpc.toml`
- reads `configs/batch_register.toml`
- finds the subjects to run
- submits one Slurm job per subject


### Typical batch output locations

Outputs usually go under each subject folder:

```text
subjects/IEB0001/_01_registration/
```

If segmentation transform is enabled, template segmentations usually go to:

```text
subjects/IEB0001/_02_template_segmentations/
```


## Sweep Registration

### Copy the config templates into the project

```bash
cp /gpfs/Labs/Kim/shared_registration/code/lsfm_data_processing/lsfm_data_processing/registration_and_transforms/sweep_registration/config_templates/sweep_register.toml \
  /gpfs/Labs/Kim/Ingvild/my_registration_project/configs/sweep_register.toml

cp /gpfs/Labs/Kim/shared_registration/code/lsfm_data_processing/lsfm_data_processing/registration_and_transforms/sweep_registration/config_templates/hpc.toml \
  /gpfs/Labs/Kim/Ingvild/my_registration_project/configs/hpc.toml
```


### Edit `sweep_register.toml`

Typical fields to check:

#### `[run]`

- `output_root`
- `orientation_alignment`
- `write_input_images`

#### `[presets]`

List the preset names to test.

These can be:

- built-in `atlasspace` preset names like `baseline_syn_kimlab` or `tuned_syn_cc`
- absolute paths to custom preset YAML files

#### `[templates.<name>]`

Define the template images available for the sweep.

#### `[images.<name>]`

Define the images to register.

#### `[image_to_template]`

Map each image to one or more template names.

Example:

```toml
[image_to_template]
subject_101422 = ["ccfv3", "neun_v1"]
subject_101425 = "ccfv3"
```


### Edit `hpc.toml`

For sweep, the workflow section should usually be:

```toml
[workflow]
registration_config = "configs/sweep_register.toml"
skip_if_output_exists = true
dry_run = false
job_name_prefix = "sweep"
```

Sweep logs usually go to:

```toml
[logging]
log_dir = "logs/sweep_registration"
```


### Launch sweep registration on HPC

Move into the project root:

```bash
cd /gpfs/Labs/Kim/Ingvild/my_registration_project
```

Activate your environment:

```bash
conda activate lsfm_data_processing
```

Run the sweep submit script:

```bash
bash /gpfs/Labs/Kim/shared_registration/code/lsfm_data_processing/lsfm_data_processing/registration_and_transforms/sweep_registration/hpc/submit_sweep_register.sh
```

This script:

- reads `configs/hpc.toml`
- reads `configs/sweep_register.toml`
- expands all image/template/preset combinations
- submits one Slurm job per sweep job


### Typical sweep output locations

Outputs go under the sweep `output_root`, for example:

```text
sweep_outputs/
  subject_101422/
    ccfv3/
      baseline_syn_kimlab/
```


## General Tips

### Dry run first

Before submitting real jobs, it is often a good idea to set:

```toml
dry_run = true
```

in `configs/hpc.toml`.

Then rerun the submit command. This prints the `sbatch` commands without actually submitting jobs.


### Check job logs

After submission, logs go to the folder specified in `configs/hpc.toml`, for example:

```text
logs/registration/
logs/sweep_registration/
```

Each job will generate `.out` and `.err` files there.


### Quick checklist

Before launch, make sure:

- your conda environment is activated
- you are standing in the project root


## Shared Code Maintenance

This section is more developer-facing. Most users only need the sections above.

### One-time shared code setup

Choose a shared code location on HPC:

```bash
mkdir -p /gpfs/Labs/Kim/shared_registration/code
cd /gpfs/Labs/Kim/shared_registration/code
git clone https://github.com/ingvildeb/atlasspace.git
git clone https://github.com/ingvildeb/lsfm_data_processing.git
cd lsfm_data_processing
git switch registration-beta
```

At the moment, `lsfm_data_processing` registration work lives on the `registration-beta` branch, so switch to that branch before installing.

Once that branch is merged, these instructions can be simplified back to the default branch.


### Update after code changes

If `atlasspace` or `lsfm_data_processing` changes, update the shared checkout and reinstall:

```bash
cd /gpfs/Labs/Kim/shared_registration/code/atlasspace
git pull

cd /gpfs/Labs/Kim/shared_registration/code/lsfm_data_processing
git fetch
git switch registration-beta
git pull

conda activate lsfm_data_processing
pip install /gpfs/Labs/Kim/shared_registration/code/atlasspace
pip install /gpfs/Labs/Kim/shared_registration/code/lsfm_data_processing
```


## Current Workflow Design

The current registration setup is intentionally split like this:

- local workstation use:
  - repo-top scripts under `registration_and_transforms/`
  - `_template.toml` copied to `_local.toml`
- HPC use:
  - installed package workflow code
  - project-root `configs/`
  - shell launchers under the installed package

This is deliberate:

- it keeps local use consistent with the rest of `lsfm_data_processing`
- it keeps HPC projects cleaner and easier to reuse
