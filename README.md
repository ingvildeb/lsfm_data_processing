# lsfm_data_processing

Utilities and pipelines for LSFM preprocessing, chunk generation, atlas alignment support, and dataset management.

## Repository layout

- `preprocess_for_cellpose/`: pre-process data for segmentation and build Cellpose training datasets from stitched TIFF images
- `preprocess_for_ants/`: build and apply NIfTI brain masks (ANTs-oriented prep)
- `registration_and_transforms/`: legacy top-level wrappers around the installed registration workflows built on `atlasspace`
- `data_eval_and_management/`: one-off scripts for normalization tuning and batch visual QC
- `utils/`: shared helpers used by multiple scripts
- `archived_and_test/`: older/testing utilities

# Get started
1. Create and activate a conda environment:

```powershell
conda create --name lsfm_data_processing python=3.11
conda activate lsfm_data_processing
```

2. Install this repo's Python requirements from the repo root:

```powershell
pip install -r requirements.txt
```

3. If you are using the registration workflows, install `atlasspace` separately:

```powershell
cd C:\path\to\atlasspace
pip install .
```

4. For any script you want to use, make a copy of the corresponding config template and edit it for your dataset.
6. Run the corresponding script using your preferred software (e.g. VSCode) or in the terminal with Python from repo root, e.g.:

```powershell
python preprocess_for_cellpose/1_preprocess_data.py
python preprocess_for_cellpose/2_select_representative_sections.py
```

## Registration workflow setup
For the current registration and transform workflows, the recommended model is:

- install `lsfm_data_processing` and `atlasspace` into the environment
- keep each registration project as a small folder containing `subjects/`, `configs/`, and `logs/`
- use the installed workflow code from the environment rather than copying script bundles into each project

For local interactive use inside this repo, the familiar top-level scripts are still available:

- `registration_and_transforms/1_batch_register.py`
- `registration_and_transforms/2_sweep_register.py`

These local scripts follow the same pattern as the rest of the repo:
- on first run, the script creates a gitignored `*_local.toml`
- that local config is bootstrapped from the canonical package template
- bootstrapped local configs can warn later if the canonical template revision changes
- edit the `_local` file
- rerun the script from the repo

Suggested project layout:

```text
my_registration_project/
  subjects/
    IEB0001/
    IEB0002/
  configs/
    batch_register.toml
    hpc.toml
    registration_presets/
      my_custom.yaml
  logs/
    registration/
```

The installed workflow code now lives under:

- `lsfm_data_processing/registration_and_transforms/batch_registration/`
- `lsfm_data_processing/registration_and_transforms/sweep_registration/`

Each workflow is separated into:

- `local/`: workstation/local execution entrypoints
- `hpc/`: Slurm submission and per-job runners
- `config_templates/`: workflow-specific starter files such as `hpc.toml`

1. Clone `lsfm_data_processing` somewhere local and check out the working branch you intend to use:

```powershell
cd C:\Users\YourName\Documents\GitHub
git clone https://github.com/ingvildeb/lsfm_data_processing.git
cd lsfm_data_processing
# Example:
git switch registration-beta
```

2. Clone `atlasspace` somewhere local as a separate repo:

```powershell
cd C:\Users\YourName\Documents\GitHub
git clone https://github.com/ingvildeb/atlasspace.git
```

3. Create and activate the `lsfm_data_processing` conda environment.
4. Run `pip install -r requirements.txt` from this repo.
5. Run `pip install .` from this repo.
6. Run `pip install .` from your `atlasspace` repo.

This keeps `lsfm_data_processing` as the lab-facing workflow repo while letting the registration workflows import the installed `atlasspace` code directly.

For local batch registration from an installed package project root:

```powershell
python -m lsfm_data_processing.registration_and_transforms.batch_registration.local.run_batch_register
```

For HPC batch submission from a project root on the cluster:

```bash
bash /path/to/lsfm_data_processing/lsfm_data_processing/registration_and_transforms/batch_registration/hpc/submit_batch_register.sh
```

For local sweep registration from an installed package project root:

```powershell
python -m lsfm_data_processing.registration_and_transforms.sweep_registration.local.run_sweep_register
```

For HPC sweep submission:

```bash
bash /path/to/lsfm_data_processing/lsfm_data_processing/registration_and_transforms/sweep_registration/hpc/submit_sweep_register.sh
```

Starter configs are currently split like this:

- `atlasspace/src/atlasspace/config_templates/registration_batch_template.toml`
- `lsfm_data_processing/registration_and_transforms/batch_registration/config_templates/hpc.toml`
- `atlasspace/src/atlasspace/config_templates/registration_sweep_template.toml`
- `lsfm_data_processing/registration_and_transforms/sweep_registration/config_templates/hpc.toml`

In `batch_register.toml`:

- `[run].registration_presets` should contain exactly one preset for batch mode
- `[run].output_subdir` controls the registration folder created under each run image's parent folder
- `orientation_alignment` controls whether registration inputs are reoriented before ANTs is run
- `[batch].template_role` controls whether the shared template is treated as the moving or fixed image
- `[moving_segmentations].enabled = true` propagates segmentations attached to whichever image is moving in each pair
- image/template mappings are defined under `[images.<image_id>]` plus `[batch].image_to_template`

In `hpc.toml`:

- `registration_config` should usually point to `configs/batch_register.toml` or `configs/sweep_register.toml`
- the submitters assume you launch from the project root, so no separate `project_dir` field is needed

## Important note about file naming
Many of the scripts expect specific filename token positions (underscore-delimited naming), for example to extract z levels, subject id, etcetera. Indexing settings in template configs are according to Kim lab naming conventions, but can always be modified in the config files to match your patterns as long as you use an underscore-separated file naming convention. Feel free to open an issue if you have any questions about making these scripts work for your own data!

# Overview of script functionalities

## Cellpose data pipeline (`preprocess_for_cellpose`)

Recommended sequence:
1. `1_preprocess_data.py`
2. `2_select_representative_sections.py`
3. Optional: `2a_get_selected_atlas_sections.py`
4. `3_chunk_data.py`
5. `4_filter_black_chunks.py`
6. Either `5a_select_random_chunks.py` or `5b_select_representative_chunks.py`
7. Optional utility: `6_recreate_chunk_selection.py`

### `1_preprocess_data.py`
- Inputs: one or more sample folders with stitched channel images
- Main functions:
  - creates MIPs at a target thickness (`create_MIPs=true`)
  - optionally normalizes images by percentile clipping
  - optional conversion to 8-bit output
- Handles both old/new folder naming conventions and custom folder formats.
- Config template: `preprocess_for_cellpose/configs/1_preprocess_data_config_template.toml`

### `2_select_representative_sections.py`
- Inputs: one or more folders of TIFF images (commonly MIP outputs)
- Main functions:
  - selects evenly spaced sections with deterministic per-sample shuffling
  - removes first/last sampled slices to avoid edge artifacts
  - can copy selected sections or generate small z-stacks around each section
- Config template: `preprocess_for_cellpose/configs/2_select_representative_sections_template.toml`

### `2a_get_selected_atlas_sections.py`
- Inputs:
  - sample folders containing `_01_registration/ANTs_TransformedImage.nii.gz`
  - folder of selected images from step 2
- Main functions:
  - maps each selected image to a corresponding atlas slice from registered volume
  - resizes/rotates atlas slice to image dimensions
  - optionally shows preview overlay for visual validation
  - saves `*_atlas_slice.tif` alongside selected images
- Config template: `preprocess_for_cellpose/configs/2a_get_selected_atlas_sections_template.toml`

### `3_chunk_data.py`
- Inputs: folder containing TIFF images (2D images or 3D z-stacks)
- Main functions:
  - cuts each image/stack into spatial chunks of fixed size
  - writes outputs under `chunked_images_<size>by<size>/<source_image_stem>/`
- Config template: `preprocess_for_cellpose/configs/3_chunk_data_template.toml`

### `4_filter_black_chunks.py`
- Inputs: parent folder of chunked image folders
- Main functions:
  - computes per-chunk average intensity
  - copies only chunks above a threshold into `filtered_image_chunks/`
  - optional atlas-paired mode: also copies matching atlas chunks into `filtered_atlas_chunks/`
- Config template: `preprocess_for_cellpose/configs/4_filter_black_chunks_template.toml`

### `5a_select_random_chunks.py`
- Use when you only have image chunks (no atlas pairing).
- Inputs: `filtered_image_chunks/`
- Main functions:
  - selects approximately evenly spaced chunks across the dataset
  - shuffles selected set and copies to `out_dir` with prefixed names
- Config template: `preprocess_for_cellpose/configs/5a_select_random_chunks_template.toml`

### `5b_select_representative_chunks.py`
- Use when atlas chunk pairs are available.
- Inputs:
  - filtered image chunks
  - filtered atlas chunks
- Main functions:
  - greedily selects chunk pairs to maximize atlas region coverage
  - fills remaining quota randomly if needed
  - writes paired outputs to `selected_image_chunks/` and `selected_atlas_chunks/`
- Config template: `preprocess_for_cellpose/configs/5b_select_representative_chunks_template.toml`

### `6_recreate_chunk_selection.py`
- Utility script to recreate an old chunk selection from newly preprocessed source images.
- Inputs:
  - existing selected chunk folder
  - mapping from subject IDs to new source image folders
- Main functions:
  - parses old chunk filenames to recover chunk coordinates
  - finds matching new source image
  - re-extracts same chunk window
  - optional copy of matching `*_seg.npy` annotation files
- Config template: `preprocess_for_cellpose/configs/6_recreate_chunk_selection_template.toml`

## ANTs/masking pipeline (`preprocess_for_ants`)

Recommended sequence:
1. `1_nii_to_2D_files.py`
2. Manual/ilastik segmentation outside this repo
3. `2_2D_to_nii_mask.py`
4. `3_dilate_and_fill_mask.py`
5. `4_apply_mask.py`

### `1_nii_to_2D_files.py`
- Converts a 3D NIfTI volume into 2D coronal TIFF slices for annotation.
- Config template: `preprocess_for_ants/configs/1_nii_to_2D_files_template.toml`

### `2_2D_to_nii_mask.py`
- Rebuilds a 3D binary NIfTI mask from segmented 2D images (for example ilastik outputs).
- Uses `foreground_label` to binarize segmentations.
- Config template: `preprocess_for_ants/configs/2_2D_to_nii_mask_template.toml`

### `3_dilate_and_fill_mask.py`
- Post-processes binary mask with dilation, hole filling, Gaussian smoothing, and thresholding.
- Config template: `preprocess_for_ants/configs/3_dilate_and_fill_mask_template.toml`

### `4_apply_mask.py`
- Applies mask volume to raw volume and optionally clips by slice index range.
- Saves masked NIfTI volume for downstream registration/preprocessing.
- Config template: `preprocess_for_ants/configs/4_apply_mask_template.toml`

## Data evaluation and management scripts

These are currently one-off scripts with parameters set directly in the file (not TOML-driven):

### `data_eval_and_management/determine_norm_params.py`
- Applies several normalization percentile settings to a set of test images.
- Intended for quickly comparing clipping ranges.

### `data_eval_and_management/lfsm_batch_eval.py`
- Builds a collage from middle sections across multiple LSFM samples.
- Useful for batch-level QC snapshots.

## Shared utilities

### `utils/io_helpers.py`
- Path normalization and strict path validation helpers
- Standardized config loading with local/template fallback

### `utils/utils.py`
- Image normalization helpers
- MIP creation
- 2D and 3D chunking
- atlas-slice extraction and preview relabeling
- z-stack assembly helpers
