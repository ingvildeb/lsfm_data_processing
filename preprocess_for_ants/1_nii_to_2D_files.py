import nibabel as nib
import numpy as np
from PIL import Image
from pathlib import Path

"""
Written by: Ingvild Bjerke
Last modified: 2/2/2026

Purpose: Create 2D coronal slices from a 20 um downsampled LSFM volume. The slices can be used to train
a machine learning classifier to mask the brain.

"""
#### USER SETTINGS
# Give the path to your raw nifti volume file
nifti_file = Path(r"Z:\LSFM\2025\2025_12\2025_12_12\20251212_15_40_45_EH_EH0032_F_P10_MOBP_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE\ch2_iso20um.nii.gz")



#### MAIN CODE, do not edit

nii = nib.load(nifti_file)
data = nii.get_fdata()

# Create output directory
output_dir = nifti_file.parent / "2D_for_mask_generation"
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
    out_path = output_dir / f"slice_{i:03d}.tif"
    img.save(out_path)

print(f"Successfully saved TIFF files to {output_dir}")