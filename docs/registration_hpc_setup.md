# HPC Registration Setup

This guide describes how to deploy and launch `lsfm_data_processing` registration workflows on a Slurm-based HPC
system. The paths below are examples. Replace `/path/to/...` with the paths used on your cluster.

For the batch and sweep registration TOML fields themselves, see `docs/registration_config_setup.md`.

## Shared Folder Layout

If your group maintains a shared checkout of the registration code and templates, a useful layout is:

```text
/path/to/shared_registration/
  code/
    atlasspace/
    lsfm_data_processing/
  templates/
    ...
```

The `code/` folder contains source checkouts. The `templates/` folder contains shared template images, label volumes,
and masks.

## Create A Conda Environment

Each user should create and use their own environment:

```bash
conda create --name lsfm_data_processing python=3.11
conda activate lsfm_data_processing
```

## Install The Packages

If your HPC has shared source checkouts, install from those folders:

```bash
pip install /path/to/shared_registration/code/atlasspace
pip install /path/to/shared_registration/code/lsfm_data_processing
```

This is a non-editable local install:

- the package is installed from a local folder, not from PyPI
- later edits to the source checkout do not affect your environment until you reinstall

Verify imports with:

```bash
python -c "import atlasspace; import lsfm_data_processing; print('Imports OK')"
```

## Create A Registration Project

Each project should be a small, clean folder with configs, logs, and subject inputs:

```text
my_registration_project/
  subjects/
  configs/
  logs/
```

Example:

```bash
mkdir -p /path/to/my_registration_project/subjects
mkdir -p /path/to/my_registration_project/configs
mkdir -p /path/to/my_registration_project/logs
```

Put subject folders inside `subjects/`, for example:

```text
my_registration_project/
  subjects/
    subject_001/
      ch1_iso20um.nii.gz
    subject_002/
      ch1_iso20um.nii.gz
```

## Copy Config Templates

For batch registration:

```bash
cp /path/to/shared_registration/code/atlasspace/src/atlasspace/config_templates/registration_batch_template.toml \
  /path/to/my_registration_project/configs/batch_register.toml

cp /path/to/shared_registration/code/lsfm_data_processing/lsfm_data_processing/registration_and_transforms/batch_registration/config_templates/hpc.toml \
  /path/to/my_registration_project/configs/hpc.toml
```

For sweep registration:

```bash
cp /path/to/shared_registration/code/atlasspace/src/atlasspace/config_templates/registration_sweep_template.toml \
  /path/to/my_registration_project/configs/sweep_register.toml

cp /path/to/shared_registration/code/lsfm_data_processing/lsfm_data_processing/registration_and_transforms/sweep_registration/config_templates/hpc.toml \
  /path/to/my_registration_project/configs/hpc.toml
```

## Edit `hpc.toml`

Example cluster settings:

```toml
[cluster]
partition = "your_partition"
cpus_per_task = 12
mem_gb = 128
time = "24:00:00"
conda_env = "lsfm_data_processing"
python_executable = "python"
```

For batch registration:

```toml
[workflow]
registration_config = "configs/batch_register.toml"
skip_if_output_exists = true
dry_run = true
job_name_prefix = "reg"

[logging]
log_dir = "logs/registration"
```

For sweep registration:

```toml
[workflow]
registration_config = "configs/sweep_register.toml"
skip_if_output_exists = true
dry_run = true
job_name_prefix = "sweep"

[logging]
log_dir = "logs/sweep_registration"
```

Start with `dry_run = true`. After the printed `sbatch` commands look correct, set `dry_run = false` and rerun the
submit command.

## Launch Batch Registration

Move into the project root:

```bash
cd /path/to/my_registration_project
```

Activate your environment:

```bash
conda activate lsfm_data_processing
```

Run the batch submit script:

```bash
bash /path/to/shared_registration/code/lsfm_data_processing/lsfm_data_processing/registration_and_transforms/batch_registration/hpc/submit_batch_register.sh
```

The script reads `configs/hpc.toml`, reads the registration config listed in `[workflow].registration_config`, expands
the batch jobs, and submits one Slurm job per registration.

## Launch Sweep Registration

Move into the project root:

```bash
cd /path/to/my_registration_project
```

Activate your environment:

```bash
conda activate lsfm_data_processing
```

Run the sweep submit script:

```bash
bash /path/to/shared_registration/code/lsfm_data_processing/lsfm_data_processing/registration_and_transforms/sweep_registration/hpc/submit_sweep_register.sh
```

The script reads `configs/hpc.toml`, reads the registration config listed in `[workflow].registration_config`, expands
the sweep jobs, and submits one Slurm job per registration.

## Check Logs

After submission, logs go to the folder specified in `configs/hpc.toml`, for example:

```text
logs/registration/
logs/sweep_registration/
```

Each job writes `.out` and `.err` files there.

## Quick Checklist

Before launching real jobs, make sure:

- your conda environment is activated
- you are standing in the project root
- `configs/hpc.toml` points to the intended registration config
- `dry_run = true` has been tested at least once
