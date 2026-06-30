# HPC Registration Setup

This guide is for Kim lab users who want to run `lsfm_data_processing` registration workflows on HPC.

It covers:

- where to keep the code on HPC
- how to install `atlasspace` and `lsfm_data_processing` into a conda environment
- how to set up a registration project folder
- how to launch batch and sweep registrations with exact commands

The current recommendation is:

- keep one shared source checkout of `atlasspace`
- keep one shared source checkout of `lsfm_data_processing`
- install both into your own conda environment from those shared paths
- keep each registration run in a separate project folder containing only data, configs, and logs


## 1. One-Time Shared Code Setup

These steps only need to be done once per shared code location.

Choose a shared code location on HPC, for example:

```bash
/gpfs/Labs/Kim/shared_code
```

Clone the repos there:

```bash
cd /gpfs/Labs/Kim/shared_code
git clone https://github.com/ingvildeb/atlasspace.git
git clone https://github.com/ingvildeb/lsfm_data_processing.git
```

If the repos are already present, update them instead:

```bash
cd /gpfs/Labs/Kim/shared_code/atlasspace
git pull

cd /gpfs/Labs/Kim/shared_code/lsfm_data_processing
git pull
```


## 2. Create Your Conda Environment

Each user should create and use their own environment.

Example:

```bash
conda create --name lsfm_data_processing python=3.11
conda activate lsfm_data_processing
```


## 3. Install the Packages into Your Environment

Install from the shared source folders.

Recommended stable install:

```bash
pip install /gpfs/Labs/Kim/shared_code/atlasspace
pip install /gpfs/Labs/Kim/shared_code/lsfm_data_processing
```

This is a non-editable local install:

- the package is installed from a local folder, not from PyPI
- later edits to the source repo do not affect your environment until you reinstall

If the shared source is updated later and you want the new version, rerun the same two commands.

You can verify imports with:

```bash
python -c "import atlasspace; import lsfm_data_processing; print('Imports OK')"
```


## 4. Create a Registration Project Folder

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


## 5. Copy the Config Templates into the Project

### Batch registration

Copy these two files:

```bash
cp /gpfs/Labs/Kim/shared_code/lsfm_data_processing/lsfm_data_processing/registration_and_transforms/batch_registration/config_templates/batch_register.toml \
  /gpfs/Labs/Kim/Ingvild/my_registration_project/configs/batch_register.toml

cp /gpfs/Labs/Kim/shared_code/lsfm_data_processing/lsfm_data_processing/registration_and_transforms/batch_registration/config_templates/hpc.toml \
  /gpfs/Labs/Kim/Ingvild/my_registration_project/configs/hpc.toml
```

### Sweep registration

Copy these two files:

```bash
cp /gpfs/Labs/Kim/shared_code/lsfm_data_processing/lsfm_data_processing/registration_and_transforms/sweep_registration/config_templates/sweep_register.toml \
  /gpfs/Labs/Kim/Ingvild/my_registration_project/configs/sweep_register.toml

cp /gpfs/Labs/Kim/shared_code/lsfm_data_processing/lsfm_data_processing/registration_and_transforms/sweep_registration/config_templates/hpc.toml \
  /gpfs/Labs/Kim/Ingvild/my_registration_project/configs/hpc.toml
```

You only need one workflow config at a time:

- `configs/batch_register.toml` for subject batch runs
- `configs/sweep_register.toml` for template/preset sweep runs

Both workflows use the same:

- `configs/hpc.toml`


## 6. Edit `batch_register.toml`

Typical fields to check:

### `[run]`

- `registration_preset`
- `template_role`
- `output_subdir`
- `orientation_alignment`
- `write_input_images`

### `[subject_defaults]`

- `subjects_dir = "subjects"`
- `image_name`
- `orientation`
- `resolution_um`
- `underscores_to_id`

If your subject folders are just named by subject ID, for example `IEB0001`, set:

```toml
underscores_to_id = 0
```

### `[segmentation_transform]`

Set:

```toml
enabled = true
```

if you want template segmentations transformed after registration.

### `[templates.<name>]`

Define one or more templates here, including:

- `image`
- `space_name`
- `orientation`
- `resolution_um`

### `[templates.<name>.segmentations]`

Optional template segmentation files that should be transformed after registration.

### `[subject_to_template]`

This maps each subject ID to the template name it should use.

Example:

```toml
[subject_to_template]
"IEB0001" = "lsfm-neun-v1-p56"
"IEB0002" = "lsfm-neun-v1-p56"
```


## 7. Edit `sweep_register.toml`

Typical fields to check:

### `[run]`

- `output_root`
- `orientation_alignment`
- `write_input_images`

### `[presets]`

List the preset names to test.

These can be:

- built-in `atlasspace` preset names like `baseline_syn_kimlab` or `tuned_syn_cc`
- absolute paths to custom preset YAML files

### `[templates.<name>]`

Define the template images available for the sweep.

### `[images.<name>]`

Define the images to register.

### `[image_to_template]`

Map each image to one or more template names.

Example:

```toml
[image_to_template]
subject_101422 = ["ccfv3", "neun_v1"]
subject_101425 = "ccfv3"
```


## 8. Edit `hpc.toml`

This controls Slurm submission settings.

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

The workflow section should usually point to the config in the current project:

### For batch

```toml
[workflow]
registration_config = "configs/batch_register.toml"
skip_if_output_exists = true
dry_run = false
job_name_prefix = "reg"
```

### For sweep

```toml
[workflow]
registration_config = "configs/sweep_register.toml"
skip_if_output_exists = true
dry_run = false
job_name_prefix = "sweep"
```

Logs usually go to:

```toml
[logging]
log_dir = "logs/registration"
```

or for sweep:

```toml
[logging]
log_dir = "logs/sweep_registration"
```


## 9. Launch Batch Registration on HPC

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
bash /gpfs/Labs/Kim/shared_code/lsfm_data_processing/lsfm_data_processing/registration_and_transforms/batch_registration/hpc/submit_batch_register.sh
```

This script:

- reads `configs/hpc.toml`
- reads `configs/batch_register.toml`
- finds the subjects to run
- submits one Slurm job per subject


## 10. Launch Sweep Registration on HPC

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
bash /gpfs/Labs/Kim/shared_code/lsfm_data_processing/lsfm_data_processing/registration_and_transforms/sweep_registration/hpc/submit_sweep_register.sh
```

This script:

- reads `configs/hpc.toml`
- reads `configs/sweep_register.toml`
- expands all image/template/preset combinations
- submits one Slurm job per sweep job


## 11. Dry Run First

Before submitting real jobs, it is often a good idea to set:

```toml
dry_run = true
```

in `configs/hpc.toml`.

Then rerun the submit command. This prints the `sbatch` commands without actually submitting jobs.


## 12. Check Job Logs

After submission, logs go to the folder specified in `configs/hpc.toml`, for example:

```text
logs/registration/
logs/sweep_registration/
```

Each job will generate `.out` and `.err` files there.


## 13. Typical Output Locations

### Batch registration

Outputs usually go under each subject folder:

```text
subjects/IEB0001/_01_registration/
```

If segmentation transform is enabled, template segmentations usually go to:

```text
subjects/IEB0001/_02_template_segmentations/
```

### Sweep registration

Outputs go under the sweep `output_root`, for example:

```text
sweep_outputs/
  subject_101422/
    ccfv3/
      baseline_syn_kimlab/
```


## 14. Update After Code Changes

If `atlasspace` or `lsfm_data_processing` changes, update the shared checkout and reinstall:

```bash
cd /gpfs/Labs/Kim/shared_code/atlasspace
git pull

cd /gpfs/Labs/Kim/shared_code/lsfm_data_processing
git pull

conda activate lsfm_data_processing
pip install /gpfs/Labs/Kim/shared_code/atlasspace
pip install /gpfs/Labs/Kim/shared_code/lsfm_data_processing
```


## 15. Quick Checklist

Before launch, make sure:

- your conda environment is activated
- `atlasspace` is installed in that environment
- `lsfm_data_processing` is installed in that environment
- you are standing in the project root
- `subjects/` exists
- `configs/hpc.toml` exists
- `configs/batch_register.toml` or `configs/sweep_register.toml` exists
- all image/template paths in the config files are correct


## 16. Current Workflow Design

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
