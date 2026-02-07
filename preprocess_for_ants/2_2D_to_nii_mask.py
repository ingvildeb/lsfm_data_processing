import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.io_helpers import load_script_config, normalize_user_path, require_dir

# -------------------------
# CONFIG LOADING
# -------------------------

cfg = load_script_config(
    Path(__file__),
    "2_2D_to_nii_mask"
)

# -------------------------
# CONFIG PARAMETERS
# -------------------------

segmentation_dir = require_dir(
    normalize_user_path(cfg["segmentation_dir"]),
    "Segmentation folder"
)

output_nifti_name = cfg["output_nifti_name"]
slice_prefix = cfg["slice_prefix"]
segmentation_suffix = cfg["segmentation_suffix"]
foreground_label = cfg["foreground_label"]

# -------------------------
# MAIN CODE
# -------------------------

output_nifti_file = segmentation_dir / output_nifti_name
slice_indices: list[int] = []

# Collect all slice indices from the segmented images
pattern = f"*{segmentation_suffix}"
for seg_file in segmentation_dir.glob(pattern):
    name_wo_suffix = seg_file.name.removesuffix(segmentation_suffix)
    if not name_wo_suffix.startswith(f"{slice_prefix}_"):
        continue
    index_str = name_wo_suffix.split("_")[1]
    slice_indices.append(int(index_str))

# Sort indices to ensure they are in correct order
slice_indices.sort()

if not slice_indices:
    raise RuntimeError(
        f"No segmented files matching '*{segmentation_suffix}' found in:\n{segmentation_dir}"
    )

# Initialize an empty list to hold the slices
slices = []

# Load each segmented image into the array
for i in slice_indices:
    slice_filename = f"{slice_prefix}_{i:03d}{segmentation_suffix}"
    slice_path = segmentation_dir / slice_filename

    if slice_path.exists():
        img = Image.open(slice_path)
        img_array = np.array(img)

        # Set 1 where label matches foreground_label, else 0
        made_binary = np.where(img_array == foreground_label, 1, 0)
        slices.append(made_binary)

    else:
        print(f"Warning: {slice_path} does not exist.")

if not slices:
    raise RuntimeError(
        "No slices were loaded. Please confirm segmentation file names and config values."
    )

# Stack the slices into a 3D NumPy array
volume = np.stack(slices, axis=1).astype("uint8")

# Create and save NIfTI image
nifti_image = nib.Nifti1Image(volume, np.eye(4))
nib.save(nifti_image, output_nifti_file)

print(f"Successfully saved the segmented volume to {output_nifti_file}")
