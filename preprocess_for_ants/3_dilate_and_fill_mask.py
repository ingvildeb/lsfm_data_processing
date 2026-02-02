import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, binary_fill_holes
from pathlib import Path

"""
Written by: Ingvild Bjerke
Last modified: 2/2/2026

Purpose: Dilate and fill a binary brain mask to get rid of any small holes in the mask are filled
and ensure no edges are removed.

"""
#### USER SETTINGS
# Give the path to your segmentation file
segmentation_volume = Path(rf"Z:\LSFM\2025\2025_12\2025_12_12\20251212_15_40_45_EH_EH0032_F_P10_MOBP_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE\2D_for_mask_generation\segmented_volume.nii.gz")


##### MAIN CODE, do not edit 
# Load the existing mask
raw_mask = nib.load(segmentation_volume)

data = raw_mask.get_fdata()

# Expand the mask using binary dilation
# You can define the structure to control the dilation
dilated_mask = binary_dilation(data, structure=np.ones((3, 3, 3)))

# Fill holes in the dilated mask
filled_mask = binary_fill_holes(dilated_mask)

# Smoothen the filled mask using Gaussian filtering
smoothed_mask = gaussian_filter(filled_mask.astype(float), sigma=3)  # Adjust sigma to control smoothing

# Threshold the smoothed mask to create a binary mask again
thresholded_mask = (smoothed_mask > 0.5).astype(float)  # Using 0.5 as threshold for binary conversion

# Create a new NIfTI image with the smoothed and expanded mask
expanded_mask = nib.Nifti1Image(thresholded_mask, raw_mask.affine, raw_mask.header)

# Save the expanded and smoothed mask
nib.save(expanded_mask, segmentation_volume.parent / "segmented_volume_dilated.nii.gz")
