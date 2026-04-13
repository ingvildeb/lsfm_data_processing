"""
Create a representative validation set from full MIP images + prediction-mask TIFFs.

User-facing flow:
1) configure subject parent folders + standardized subfolder names,
2) script selects representative sections per subject,
3) optional: saves selected full sections + selected predictions for QC,
4) chunks selected sections and chooses a representative subset:
   - atlas-aware coverage when registration is enabled,
   - random selection when registration is disabled,
5) writes TIFF chunks + Cellpose-style *_seg.npy files for easy correction in Cellpose.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import random
import shutil
import sys

import nibabel as nib
import numpy as np
import tifffile

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from lsfm_data_processing.utils.atlas import atlas_slice_for_mip  # noqa: E402
from lsfm_data_processing.utils.cellpose_io import (  # noqa: E402
    build_prediction_index,
    create_cellpose_npy_dict,
    load_prediction_masks,
    match_prediction_for_mip,
)
from lsfm_data_processing.utils.io_helpers import (  # noqa: E402
    list_tiff_files,
    load_script_config,
    normalize_user_path,
    require_dir,
    require_file,
    require_subpath,
)
from lsfm_data_processing.utils.naming import get_underscore_int, get_underscore_token  # noqa: E402
from lsfm_data_processing.utils.selection import (  # noqa: E402
    balanced_random_seed_selection,
    greedy_region_coverage_select,
    random_fill_selection,
    select_sections_evenly,
)


def split_preselected_mip_name(mip_path: Path) -> tuple[str, str]:
    """
    Split a pre-selected MIP stem into `(sample_id, original_mip_stem)`.

    Expected format:
      <sample_id>_<original MIP stem>
    Example:
      100644_MIP_014650_014690.tif -> ("100644", "MIP_014650_014690")
    """
    if "_" not in mip_path.stem:
        raise RuntimeError(
            f"Pre-selected image filename must start with '<sample_id>_':\n{mip_path.name}"
        )

    sample_id, original_stem = mip_path.stem.split("_", 1)
    if not sample_id or not original_stem:
        raise RuntimeError(
            f"Pre-selected image filename must start with '<sample_id>_':\n{mip_path.name}"
        )

    return sample_id, original_stem


# -------------------------
# CONFIG LOADING
# -------------------------
test_mode = False
cfg = load_script_config(Path(__file__), "7_predictions_to_validation_set", test_mode=test_mode)

# -------------------------
# CONFIG PARAMETERS
# -------------------------
subject_parent_dirs = [
    require_dir(normalize_user_path(p), "Subject parent folder")
    for p in cfg["subject_parent_dirs"]
]

mip_subfolder = cfg["mip_subfolder"]
prediction_subfolder = cfg["prediction_subfolder"]
atlas_subfolder = cfg["atlas_subfolder"]
underscores_to_id = cfg.get("underscores_to_id", 5)
prediction_required_prefix = cfg.get("prediction_required_prefix", "masks_")
use_atlas_registration = cfg.get("use_atlas_registration", True)

output_dir = normalize_user_path(cfg["output_dir"])
sections_per_subject = cfg["sections_per_subject"]
save_selected_sections = cfg["save_selected_sections"]
preselected_images_dir_cfg = cfg.get("preselected_images_dir", "")
preselected_images_dir = (
    require_dir(normalize_user_path(preselected_images_dir_cfg), "Pre-selected images folder")
    if preselected_images_dir_cfg
    else None
)

chunk_size = cfg["chunk_size"]
number_of_chunks = cfg["number_of_chunks"]
min_chunks_per_sample = cfg["min_chunks_per_sample"]
require_nonzero_prediction = cfg["require_nonzero_prediction"]
save_atlas_chunks = cfg["save_atlas_chunks"]
random_seed = cfg["random_seed"]

underscores_to_index = cfg["underscores_to_index"]
file_number_increment = cfg["file_number_increment"]
all_images_subfolder = cfg.get("all_images_subfolder", "Ex_561_Ch1_stitched")

if not use_atlas_registration and save_atlas_chunks:
    save_atlas_chunks = False
    print("Warning: use_atlas_registration=false, forcing save_atlas_chunks=false.")

# -------------------------
# PER-SUBJECT INDEX + SECTION SELECTION
# -------------------------
subject_meta: dict[str, dict] = {}
jobs: list[dict] = []

output_dir.mkdir(parents=True, exist_ok=True)
selected_sections_root = output_dir / "selected_sections"
if save_selected_sections:
    selected_sections_root.mkdir(parents=True, exist_ok=True)

for subject_parent in subject_parent_dirs:
    sample_id = get_underscore_token(subject_parent.stem, underscores_to_id, "sample_id")
    subject_meta[sample_id] = {}
    mip_dir = require_dir(subject_parent / mip_subfolder, f"MIP folder for {sample_id}")
    prediction_dir = require_dir(subject_parent / prediction_subfolder, f"Prediction TIFF folder for {sample_id}")

    if use_atlas_registration:
        atlas_subject_dir = require_dir(subject_parent / atlas_subfolder, f"Atlas folder for {sample_id}")

        direct_reg_vol = atlas_subject_dir / "ANTs_TransformedImage.nii.gz"
        if direct_reg_vol.exists():
            reg_vol_path = require_file(direct_reg_vol, "registered atlas volume")
        else:
            registration_dir = require_subpath(atlas_subject_dir, "_01_registration", "registration folder")
            reg_vol_path = require_file(registration_dir / "ANTs_TransformedImage.nii.gz", "registered atlas volume")
        all_images_path = require_dir(subject_parent / all_images_subfolder, "full stitched image folder")

        no_images = len(list_tiff_files(all_images_path))
        if no_images == 0:
            raise RuntimeError(f"No TIFF files found in all_images_subfolder:\n{all_images_path}")

        reg_data = np.asanyarray(nib.load(reg_vol_path).dataobj)
        if reg_data.ndim != 3:
            raise RuntimeError(f"Registered atlas volume must be 3D. Found shape {reg_data.shape}:\n{reg_vol_path}")

        subject_meta[sample_id].update({"reg_data": reg_data, "no_images": no_images})

    mip_files = list_tiff_files(mip_dir)
    if not mip_files:
        raise RuntimeError(f"No TIFF MIP files found for sample {sample_id} in:\n{mip_dir}")

    pred_index = build_prediction_index(prediction_dir, prediction_required_prefix)
    subject_meta[sample_id].update(
        {
            "mip_dir": mip_dir,
            "prediction_dir": prediction_dir,
            "pred_index": pred_index,
            "mip_files": mip_files,
        }
    )

if preselected_images_dir is None:
    for sample_id, meta in subject_meta.items():
        selected_mips = select_sections_evenly(
            files=meta["mip_files"],
            sample_id=sample_id,
            sample_size=sections_per_subject,
            drop_edges=False,
        )

        if save_selected_sections:
            sample_img_out = selected_sections_root / sample_id / "images"
            sample_pred_out = selected_sections_root / sample_id / "predictions"
            sample_img_out.mkdir(parents=True, exist_ok=True)
            sample_pred_out.mkdir(parents=True, exist_ok=True)

        for mip_path in selected_mips:
            prediction_path = match_prediction_for_mip(mip_path, meta["pred_index"], prediction_required_prefix)
            jobs.append({"sample_id": sample_id, "mip_path": mip_path, "prediction_path": prediction_path})

            if save_selected_sections:
                shutil.copy2(mip_path, (selected_sections_root / sample_id / "images" / mip_path.name))
                shutil.copy2(prediction_path, (selected_sections_root / sample_id / "predictions" / prediction_path.name))

        print(f"Selected {len(selected_mips)} sections for sample {sample_id}")
else:
    preselected_mips = list_tiff_files(preselected_images_dir)
    if not preselected_mips:
        raise RuntimeError(f"No TIFF files found in pre-selected images folder:\n{preselected_images_dir}")

    counts_by_sample: dict[str, int] = defaultdict(int)
    for selected_mip in preselected_mips:
        sample_id, original_mip_stem = split_preselected_mip_name(selected_mip)
        if sample_id not in subject_meta:
            raise RuntimeError(
                f"Pre-selected image sample_id does not match any provided subject_parent_dirs:\n{selected_mip.name}"
            )

        match_stub = Path(f"{original_mip_stem}{selected_mip.suffix}")
        prediction_path = match_prediction_for_mip(
            match_stub,
            subject_meta[sample_id]["pred_index"],
            prediction_required_prefix,
        )
        jobs.append(
            {
                "sample_id": sample_id,
                "mip_path": selected_mip,
                "prediction_path": prediction_path,
                "source_mip_stem": original_mip_stem,
            }
        )
        counts_by_sample[sample_id] += 1

        if save_selected_sections:
            sample_img_out = selected_sections_root / sample_id / "images"
            sample_pred_out = selected_sections_root / sample_id / "predictions"
            sample_img_out.mkdir(parents=True, exist_ok=True)
            sample_pred_out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected_mip, (sample_img_out / selected_mip.name))
            shutil.copy2(prediction_path, (sample_pred_out / prediction_path.name))

    for sample_id in sorted(counts_by_sample):
        print(f"Using {counts_by_sample[sample_id]} pre-selected sections for sample {sample_id}")

if not jobs:
    raise RuntimeError("No selected section jobs found.")

# -------------------------
# BUILD CANDIDATE CHUNKS
# -------------------------
candidates: list[dict] = []
by_sample: dict[str, list[int]] = defaultdict(list)

for job in jobs:
    sample_id = job["sample_id"]
    mip_path = job["mip_path"]
    prediction_path = job["prediction_path"]
    source_mip_stem = job.get("source_mip_stem", mip_path.stem)

    mip_img = tifffile.TiffFile(mip_path).asarray()
    if mip_img.ndim != 2:
        raise RuntimeError(f"MIP image must be 2D. Found shape {mip_img.shape}:\n{mip_path}")

    pred_masks = load_prediction_masks(prediction_path)
    if pred_masks.shape != mip_img.shape:
        raise RuntimeError(
            f"Prediction masks shape does not match MIP image shape:\n"
            f"MIP {mip_path.name}: {mip_img.shape}\n"
            f"Prediction {prediction_path.name}: {pred_masks.shape}"
        )

    atlas_img = None
    if use_atlas_registration:
        section_number = get_underscore_int(source_mip_stem, underscores_to_index, "section number")
        atlas_img = atlas_slice_for_mip(
            reg_volume_data=subject_meta[sample_id]["reg_data"],
            no_images=subject_meta[sample_id]["no_images"],
            section_number=section_number,
            file_number_increment=file_number_increment,
            target_h=mip_img.shape[0],
            target_w=mip_img.shape[1],
        )

    for y in range(0, mip_img.shape[0], chunk_size):
        for x in range(0, mip_img.shape[1], chunk_size):
            if y + chunk_size > mip_img.shape[0] or x + chunk_size > mip_img.shape[1]:
                continue

            mask_chunk = pred_masks[y : y + chunk_size, x : x + chunk_size]
            mask_nonzero = int(np.count_nonzero(mask_chunk))
            if require_nonzero_prediction and mask_nonzero == 0:
                continue

            regions: set[int] = set()
            if use_atlas_registration and atlas_img is not None:
                atlas_chunk = atlas_img[y : y + chunk_size, x : x + chunk_size]
                regions = set(np.unique(atlas_chunk))
                regions.discard(0)

            idx = len(candidates)
            candidates.append(
                {
                    "sample_id": sample_id,
                    "mip_path": mip_path,
                    "prediction_path": prediction_path,
                    "y": y,
                    "x": x,
                    "size": chunk_size,
                    "mask_nonzero": mask_nonzero,
                    "regions": regions,
                    "region_count": len(regions),
                    "source_mip_stem": source_mip_stem,
                }
            )
            by_sample[sample_id].append(idx)

if not candidates:
    raise RuntimeError("No chunk candidates found after filtering. Adjust thresholds/config.")

if number_of_chunks <= 0:
    raise RuntimeError("number_of_chunks must be > 0.")

if min_chunks_per_sample < 0:
    raise RuntimeError("min_chunks_per_sample must be >= 0.")

# -------------------------
# REPRESENTATIVE CHUNK SELECTION
# -------------------------
rng = random.Random(random_seed)
all_regions = set()
for c in candidates:
    all_regions.update(c["regions"])

selected: set[int] = set()
covered_regions: set[int] = set()
target_total = min(number_of_chunks, len(candidates))

if use_atlas_registration:
    regions_by_id = {cid: candidates[cid]["regions"] for cid in range(len(candidates))}
    secondary_score_by_id = {
        cid: (int(candidates[cid]["mask_nonzero"]), int(candidates[cid]["region_count"]))
        for cid in range(len(candidates))
    }
    for sample_id in sorted(by_sample.keys()):
        sample_candidate_ids = by_sample[sample_id]
        target_for_sample = min(min_chunks_per_sample, len(sample_candidate_ids))
        selected, covered_regions = greedy_region_coverage_select(
            candidate_ids=sample_candidate_ids,
            regions_by_id=regions_by_id,
            limit=len(selected) + target_for_sample,
            selected=selected,
            covered_regions=covered_regions,
            secondary_score_by_id=secondary_score_by_id,
        )

    all_candidate_ids = list(range(len(candidates)))
    selected, covered_regions = greedy_region_coverage_select(
        candidate_ids=all_candidate_ids,
        regions_by_id=regions_by_id,
        limit=target_total,
        selected=selected,
        covered_regions=covered_regions,
        secondary_score_by_id=secondary_score_by_id,
    )
else:
    selected = balanced_random_seed_selection(
        group_to_candidate_ids=by_sample,
        min_per_group=min_chunks_per_sample,
        target_total=target_total,
        rng=rng,
    )

all_candidate_ids = list(range(len(candidates)))
selected = random_fill_selection(
    selected=selected,
    all_ids=all_candidate_ids,
    target_total=target_total,
    rng=rng,
)

selected_ids = sorted(selected)

# -------------------------
# OUTPUT WRITING
# -------------------------
out_chunk_dir = output_dir / "validation_image_chunks"
out_chunk_dir.mkdir(parents=True, exist_ok=True)

out_atlas_dir = output_dir / "validation_atlas_chunks"
if save_atlas_chunks and use_atlas_registration:
    out_atlas_dir.mkdir(parents=True, exist_ok=True)

metadata_csv = output_dir / "validation_chunk_metadata.csv"

mip_cache: dict[Path, np.ndarray] = {}
pred_cache: dict[Path, np.ndarray] = {}
atlas_cache: dict[tuple[str, Path], np.ndarray] = {}
metadata_rows: list[dict[str, object]] = []

for idx in selected_ids:
    c = candidates[idx]
    sample_id = c["sample_id"]
    mip_path = c["mip_path"]
    prediction_path = c["prediction_path"]
    source_mip_stem = str(c.get("source_mip_stem", mip_path.stem))
    y = int(c["y"])
    x = int(c["x"])
    size = int(c["size"])

    if mip_path not in mip_cache:
        mip_cache[mip_path] = tifffile.TiffFile(mip_path).asarray()
    if prediction_path not in pred_cache:
        pred_cache[prediction_path] = load_prediction_masks(prediction_path)

    atlas_chunk = None
    if use_atlas_registration:
        atlas_key = (sample_id, mip_path)
        if atlas_key not in atlas_cache:
            section_number = get_underscore_int(source_mip_stem, underscores_to_index, "section number")
            atlas_cache[atlas_key] = atlas_slice_for_mip(
                reg_volume_data=subject_meta[sample_id]["reg_data"],
                no_images=subject_meta[sample_id]["no_images"],
                section_number=section_number,
                file_number_increment=file_number_increment,
                target_h=mip_cache[mip_path].shape[0],
                target_w=mip_cache[mip_path].shape[1],
            )
        atlas_chunk = atlas_cache[atlas_key][y : y + size, x : x + size]

    full_img = mip_cache[mip_path]
    masks = pred_cache[prediction_path]
    image_chunk = full_img[y : y + size, x : x + size]
    mask_chunk = masks[y : y + size, x : x + size]

    chunk_stem = f"{sample_id}_{source_mip_stem}_chunk_{y}_{x}"
    out_tif = out_chunk_dir / f"{chunk_stem}.tif"
    out_seg = out_chunk_dir / f"{chunk_stem}_seg.npy"

    if out_tif.exists() or out_seg.exists():
        raise RuntimeError(f"Refusing to overwrite existing output files:\n{out_tif}\n{out_seg}")

    tifffile.imwrite(out_tif, image_chunk)
    np.save(out_seg, create_cellpose_npy_dict(mask_chunk, out_tif), allow_pickle=True)

    if save_atlas_chunks and use_atlas_registration and atlas_chunk is not None:
        out_atlas = out_atlas_dir / f"{chunk_stem}_atlas.tif"
        if out_atlas.exists():
            raise RuntimeError(f"Refusing to overwrite existing output file:\n{out_atlas}")
        tifffile.imwrite(out_atlas, atlas_chunk)

    metadata_rows.append(
        {
            "chunk_stem": chunk_stem,
            "sample_id": sample_id,
            "mip_file": mip_path.name,
            "prediction_file": prediction_path.name,
            "chunk_y": y,
            "chunk_x": x,
            "chunk_size": size,
            "prediction_nonzero_pixels": c["mask_nonzero"],
            "atlas_region_count": c["region_count"],
        }
    )

with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "chunk_stem",
            "sample_id",
            "mip_file",
            "prediction_file",
            "chunk_y",
            "chunk_x",
            "chunk_size",
            "prediction_nonzero_pixels",
            "atlas_region_count",
        ],
    )
    writer.writeheader()
    writer.writerows(metadata_rows)

print("Validation-set generation complete.")
print(f"Selected sections processed: {len(jobs)}")
print(f"Candidate chunks: {len(candidates)}")
print(f"Selected chunks written: {len(selected_ids)}")
if use_atlas_registration:
    print(f"Atlas regions covered: {len(covered_regions)} / {len(all_regions)}")
else:
    print("Selection mode: random (atlas registration disabled)")
print(f"Chunk output directory: {out_chunk_dir}")
if save_selected_sections:
    print(f"Selected full sections directory: {selected_sections_root}")
if save_atlas_chunks and use_atlas_registration:
    print(f"Atlas chunk output directory: {out_atlas_dir}")
print(f"Metadata CSV: {metadata_csv}")
