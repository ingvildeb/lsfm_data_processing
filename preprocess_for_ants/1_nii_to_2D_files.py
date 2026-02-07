import nibabel as nib
import numpy as np
from PIL import Image
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
    "1_nii_to_2D_files"
)

# -------------------------
# CONFIG PARAMETERS
# -------------------------

nifti_file = require_file(
    normalize_user_path(cfg["nifti_file"]),
    "Input NIfTI volume"
)

output_folder_name = cfg["output_folder_name"]
output_prefix = cfg["output_prefix"]

# -------------------------
# MAIN CODE
# -------------------------

nii = nib.load(nifti_file)
data = nii.get_fdata()

# Create output directory
output_dir = nifti_file.parent / output_folder_name
output_dir.mkdir(exist_ok=True)

# Loop through slices
for i in range(data.shape[1]):
    slice_data = data[:, i, :]

    # Normalize safely
    min_val = slice_data.min()
    max_val = slice_data.max()

    if max_val > min_val:
        slice_data_normalized = (
            (slice_data - min_val) / (max_val - min_val) * 255
        ).astype(np.uint8)
    else:
        slice_data_normalized = np.zeros_like(slice_data, dtype=np.uint8)

    img = Image.fromarray(slice_data_normalized)

    # Pathlib-based save
    out_path = output_dir / f"{output_prefix}_{i:03d}.tif"
    img.save(out_path)

print(f"Successfully saved TIFF files to {output_dir}")
