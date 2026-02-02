import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path

"""
Written by: Ingvild Bjerke
Last modified: 2/2/2026

Purpose: Create a 3D volume from Simple Segmentation files generated in ilastik.
The script is intended to be used after using 1_nii_to_2D_files and ilastik to create and segment 2D slices.

"""
#### USER SETTINGS
# Give the path to the folder containing your segmented images
segmentation_dir = Path(r"Z:\LSFM\2025\2025_12\2025_12_12\20251212_15_40_45_EH_EH0032_F_P10_MOBP_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE\2D_for_mask_generation")



#### MAIN CODE, do not edit

output_nifti_file = segmentation_dir / 'segmented_volume.nii.gz'
slice_indices = []

# Collect all slice indices from the segmented images
for png_file in segmentation_dir.glob('*_Simple Segmentation.png'):
    index_str = png_file.stem.split("_")[1]
    slice_indices.append(int(index_str))

# Sort indices to ensure they are in correct order
slice_indices.sort()

# Initialize an empty list to hold the slices
slices = []

# Load each segmented image into the array
for i in slice_indices:
    slice_filename = f'slice_{i:03d}_Simple Segmentation.png'
    slice_path = segmentation_dir / slice_filename
    
    if slice_path.exists():
        # Read the image
        img = Image.open(slice_path)
        img_array = np.array(img)

        # Apply the value mapping
        # Set 1 where img_array is 1, and 0 where img_array is 2
        made_binary = np.where(img_array == 1, 1, 0)  # Sets pixels to 1 where original value was 1 or else 0
        slices.append(made_binary)
        
    else:
        print(f"Warning: {slice_path} does not exist.")

# Stack the slices into a 3D NumPy array
volume = np.stack(slices, axis=1).astype('uint8')

# Create a NIfTI image
nifti_image = nib.Nifti1Image(volume, np.eye(4))  # Identity matrix for affine

# Save to a NIfTI file
nib.save(nifti_image, output_nifti_file)

print(f"Successfully saved the segmented volume to {output_nifti_file}")