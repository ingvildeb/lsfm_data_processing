import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, binary_fill_holes
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
    "3_dilate_and_fill_mask"
)

# -------------------------
# CONFIG PARAMETERS
# -------------------------

segmentation_volume = require_file(
    normalize_user_path(cfg["segmentation_volume"]),
    "Input segmentation volume"
)

output_name = cfg["output_name"]
dilation_structure_size = cfg["dilation_structure_size"]
gaussian_sigma = cfg["gaussian_sigma"]
threshold = cfg["threshold"]

# -------------------------
# MAIN CODE
# -------------------------

raw_mask = nib.load(segmentation_volume)
data = raw_mask.get_fdata()

# Expand the mask using binary dilation
structure = np.ones(tuple(dilation_structure_size))
dilated_mask = binary_dilation(data, structure=structure)

# Fill holes in the dilated mask
filled_mask = binary_fill_holes(dilated_mask)

# Smooth and threshold back to binary
smoothed_mask = gaussian_filter(filled_mask.astype(float), sigma=gaussian_sigma)
thresholded_mask = (smoothed_mask > threshold).astype(float)

# Create and save a new NIfTI image
expanded_mask = nib.Nifti1Image(thresholded_mask, raw_mask.affine, raw_mask.header)
nib.save(expanded_mask, segmentation_volume.parent / output_name)

print(f"Saved dilated mask to {segmentation_volume.parent / output_name}")
