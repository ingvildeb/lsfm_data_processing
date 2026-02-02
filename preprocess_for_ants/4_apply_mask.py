import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

"""
Written by: Ingvild Bjerke
Last modified: 2/2/2026

Purpose: Apply a binary mask and optional start / end slices in the coronal plane to mask everything outside the brain in a volume.
Useful as a pre-processing step before registration.

"""
#### USER SETTINGS
# Give the path to your raw 20 um volume file
raw_volume = Path(r"Z:\LSFM\2025\2025_12\2025_12_12\20251212_15_40_45_EH_EH0032_F_P10_MOBP_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE\ch2_iso20um.nii.gz")

# Give the path to your mask file
segmentation_volume = Path(rf"Z:\LSFM\2025\2025_12\2025_12_12\20251212_15_40_45_EH_EH0032_F_P10_MOBP_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE\2D_for_mask_generation\segmented_volume_dilated.nii.gz")

# Define the start and end slices (make sure to adjust these to your needs)
# You can use this to make sure anything before and after these coronal planes are masked out (regardless of how your mask file looks in that area)
# Useful because we often see artifacts segmented as brain in these slices
# To define the slices, open your raw volume in ITK snap and look at the slice numbers in the lower right corner of the coronal view
# If you do not want to give start / end slices, just set mask_by_slices to False

mask_by_slices = True
start_slice = 9  # Replace with your desired start slice index. Everything prior to this slice will be removed.
end_slice = 606    # Replace with your desired end slice index. Everything after this slice will be removed.


##### MAIN CODE, do not edit 

# Load the original image
original_image = nib.load(raw_volume)
image_data = original_image.get_fdata()

# Load the inverted mask
mask_path = nib.load(segmentation_volume)
mask_data = mask_path.get_fdata()

# Apply the inverted mask to the image
masked_image_data = image_data * mask_data  # Multiply to apply the mask

plt.imshow(masked_image_data[:, 200, :], cmap='gray')

if mask_by_slices:
    # Set all voxels outside the range from start_slice to end_slice in dimension 2 to zero
    masked_image_data[:, :start_slice, :] = 0  # Zero out before start_slice
    masked_image_data[:, end_slice + 1:, :] = 0  # Zero out after end_slice

masked_image = nib.Nifti1Image(masked_image_data.astype(np.float64), original_image.affine, original_image.header)

# Save the masked image
nib.save(masked_image, segmentation_volume.parent.parent / f"{raw_volume.stem.split(".")[0]}_masked.nii.gz")
