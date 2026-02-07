import nibabel as nib
from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.io_helpers import load_script_config, normalize_user_path, require_file

# -------------------------
# CONFIG LOADING
# -------------------------

cfg = load_script_config(
    Path(__file__),
    "4_apply_mask"
)

# -------------------------
# CONFIG PARAMETERS
# -------------------------

raw_volume = require_file(
    normalize_user_path(cfg["raw_volume"]),
    "Raw volume"
)

segmentation_volume = require_file(
    normalize_user_path(cfg["segmentation_volume"]),
    "Segmentation volume"
)

mask_by_slices = cfg["mask_by_slices"]
start_slice = cfg["start_slice"]
end_slice = cfg["end_slice"]
output_name = cfg["output_name"]

# -------------------------
# MAIN CODE
# -------------------------

# Load the original image
original_image = nib.load(raw_volume)
image_data = original_image.get_fdata()

# Load mask
mask_nii = nib.load(segmentation_volume)
mask_data = mask_nii.get_fdata()

# Apply mask
masked_image_data = image_data * mask_data

if mask_by_slices:
    masked_image_data[:, :start_slice, :] = 0
    masked_image_data[:, end_slice + 1:, :] = 0

masked_image = nib.Nifti1Image(masked_image_data.astype(float), original_image.affine, original_image.header)

# Save the masked image
output_path = segmentation_volume.parent.parent / output_name
nib.save(masked_image, output_path)

print(f"Saved masked volume to {output_path}")
