# lsfm_data_processing

Utilities and pipelines for LSFM preprocessing, chunk generation, atlas alignment support, and dataset management.

## Repository layout

- `preprocess_for_cellpose/`: pre-process data for segmentation and build Cellpose training datasets from stitched TIFF images
- `preprocess_for_ants/`: build and apply NIfTI brain masks (ANTs-oriented prep)
- `data_eval_and_management/`: one-off scripts for normalization tuning and batch visual QC
- `utils/`: shared helpers used by multiple scripts
- `archived_and_test/`: older/testing utilities

## Typical usage pattern

1. Copy the relevant `*_template.toml` to `*_local.toml`.
2. Edit `*_local.toml` paths/parameters for your dataset.
3. Run the corresponding script with Python from repo root, e.g.:

```powershell
python preprocess_for_cellpose/1_preprocess_data.py
python preprocess_for_cellpose/2_select_representative_sections.py
```

## Important note about file naming
Many of the scripts expect specific filename token positions (underscore-delimited naming), for example to extract z levels, subject id, etcetera. Indexing settings in template configs are according to Kim lab naming conventions. 
However, underscore index settings can always be modified in the config files to match your patterns as long as you use an underscore-separated file naming convention. Feel free to open an issue if you have any questions about making these scripts work for your own data!

## At-a-glance script table

| ID | Use | Main input(s) | Main output(s) |
|---|---|---|---|
| `C1` | MIPs + normalization | stitched TIFF folders | MIP folders<br>normalized image folders |
| `C2` | Representative section sampling | MIP/image folders | selected TIFFs<br>or z-stack TIFFs |
| `C2a` | Atlas slice extraction for selected sections | registered atlas NIfTI<br>selected TIFFs | `*_atlas_slice.tif` files |
| `C3` | Chunking 2D/3D images | TIFF images<br>or z-stacks | `chunked_images_<size>by<size>/...` |
| `C4` | Low-signal chunk filtering | chunked image folders | `filtered_image_chunks/`<br>optional `filtered_atlas_chunks/` |
| `C5a` | Random chunk subset | `filtered_image_chunks/` | selected chunk subset |
| `C5b` | Coverage-based paired chunk selection | filtered image chunks<br>filtered atlas chunks | `selected_image_chunks/`<br>`selected_atlas_chunks/` |
| `C6` | Recreate previous chunk selection | existing chunks<br>new source image mapping | recreated chunks<br>optional copied `*_seg.npy` |
| `A1` | NIfTI to 2D slices | raw NIfTI | slice TIFF folder |
| `A2` | 2D segmentations to 3D mask | segmentation image folder | binary mask NIfTI |
| `A3` | Mask dilation/fill/smooth | binary mask NIfTI | processed mask NIfTI |
| `A4` | Apply mask to raw volume | raw NIfTI<br>mask NIfTI | masked NIfTI |
| `D1` | Normalization parameter comparison | TIFF set<br>(path set in script) | normalized TIFF variants |
| `D2` | Batch QC collage | sample folder list<br>(set in script) | collage PNG |

Script key:
- `C1`: `preprocess_for_cellpose/1_preprocess_data.py`
- `C2`: `preprocess_for_cellpose/2_select_representative_sections.py`
- `C2a`: `preprocess_for_cellpose/2a_get_selected_atlas_sections.py`
- `C3`: `preprocess_for_cellpose/3_chunk_data.py`
- `C4`: `preprocess_for_cellpose/4_filter_black_chunks.py`
- `C5a`: `preprocess_for_cellpose/5a_select_random_chunks.py`
- `C5b`: `preprocess_for_cellpose/5b_select_representative_chunks.py`
- `C6`: `preprocess_for_cellpose/6_recreate_chunk_selection.py`
- `A1`: `preprocess_for_ants/1_nii_to_2D_files.py`
- `A2`: `preprocess_for_ants/2_2D_to_nii_mask.py`
- `A3`: `preprocess_for_ants/3_dilate_and_fill_mask.py`
- `A4`: `preprocess_for_ants/4_apply_mask.py`
- `D1`: `data_eval_and_management/determine_norm_params.py`
- `D2`: `data_eval_and_management/lfsm_batch_eval.py`

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
