import pandas as pd
import numpy as np
import shutil
from pathlib import Path

##################
# USER PARAMETERS

## Specify the paths (any number of paths) to LSFM data 
folder_paths = [Path(r"M:\SmartSPIM_Data\2025\2025_04\2025_04_24\20250424_11_45_45_IEB_IEB0031_F_P14_Aldh1_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE\Ex_488_Ch0_stitched_MIP20um_min0_max99.9\\"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_06\2025_06_02\20250602_14_50_20_LJS_IEB0035_M_P56_Aldh1_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE\Ex_488_Ch0_stitched_MIP20um_min0_max99.9\\"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_05\2025_05_23\20250523_10_33_49_IEB_IEB0046_F_P10_Aldh1_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE\Ex_488_Ch0_stitched_MIP20um_min0_max99.9\\"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_05\2025_05_23\20250523_12_15_52_IEB_IEB0066_F_P8_Aldh1_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE\Ex_488_Ch0_stitched_MIP20um_min0_max99.9\\")
                 ]

## Specify the channel to select images from and the sample size (number of selected images per sample)
sample_size = 5

## Specify where you want your selected images to be saved
out_path = Path(r"Z:\Labmembers\Ingvild\Cellpose\Aldh_model\training_sections\\")
out_path.mkdir(exist_ok=True)

##################

# Adding 2 to sample size so first and last (likely black) images can be removed after sampling
sample_size = sample_size + 2

for path in folder_paths:
    folder_parent = path.parent.name
    sample_id, sample_age, sample_geno = folder_parent.split("_")[5], folder_parent.split("_")[7], folder_parent.split("_")[8]

    # Using pathlib to glob files
    files = sorted(path.glob("*.tif"))

    # Ensure at least sample_size items exist in selected_images
    if len(files) < sample_size:
        print("Not enough images to sample.")
        regularly_spaced_samples = files
    else:
        # Calculate the step size using integer indices correctly
        step = len(files) // (sample_size - 1) if sample_size > 1 else len(files)

        # Initialize selected samples empty to collect 5 samples
        regularly_spaced_samples = []

        # Collect samples using computed step
        for i in range(sample_size):
            idx = min(i * step, len(files) - 1)  # Ensure indices are valid
            regularly_spaced_samples.append(files[idx])

    regularly_spaced_samples = sorted(regularly_spaced_samples)

    del regularly_spaced_samples[0]
    del regularly_spaced_samples[-1]

    # Copy each file from selected_files to the new directory
    for file in regularly_spaced_samples:
        # Define the destination path for each file
        destination_path = out_path / f"{sample_id}_{file.name}"
        print(f"Copying {file} to {destination_path}")
        shutil.copy2(str(file), str(destination_path))

    print(f"All selected files from {sample_id} have been copied to {destination_path}")
