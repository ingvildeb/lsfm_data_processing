import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from utils import tifs_to_zstack

##################
# USER PARAMETERS

## Specify the paths (any number of paths) to LSFM data 
folder_paths = [Path(r"M:\SmartSPIM_Data\2025\2025_06\2025_06_11\20250611_09_52_20_LJS_IEB0041_M_P14_Aldh1_LAS_488Bg_561NeuN_640Iba1_4x_4umstep_Destripe_DONE\Ex_640_Ch2_stitched_MIP20um_min0_max99.7\\")
                 ]

## Specify the channel to select images from and the sample size (number of selected images per sample)
sample_size = 3

## Specify where you want your selected images to be saved
out_path = Path(r"Z:\Labmembers\Ingvild\Testing_CellPose\test_3d\test_iba1\\")

## Z stack options. Z stacks can be useful in training models for cells that have complex morphology,
# such as microglia or pericytes
# Set to False if you do not want z stacks
make_zstacks = True

# Specify the number of planes per z stack
z_stack_number = 5 

##################

# Main code, do not edit

out_path.mkdir(exist_ok=True)

# Adding 2 to sample size so first and last (likely black) images can be removed after sampling
sample_size = sample_size + 2

for path in folder_paths:
    folder_parent = path.parent.name
    sample_id = folder_parent.split("_")[5]

    # Using pathlib to glob files
    files = sorted(path.glob("*.tif"))

    # Ensure at least sample_size items exist in selected_images
    if len(files) < sample_size:
        print("Not enough images to sample.")
        regularly_spaced_samples = files
    else:
        # Calculate the step size using integer indices correctly
        step = len(files) // (sample_size - 1) if sample_size > 1 else len(files)

        regularly_spaced_samples = []

        # Collect samples using computed step
        for i in range(sample_size):
            idx = min(i * step, len(files) - 1)  # Ensure indices are valid
            regularly_spaced_samples.append(files[idx])

    regularly_spaced_samples = sorted(regularly_spaced_samples)

    del regularly_spaced_samples[0]
    del regularly_spaced_samples[-1]

    # Copy each file from selected_files to the new directory
    if make_zstacks:
        
        for file in regularly_spaced_samples:
            print(file)
            stacked_samples = []

            # Determine the index of the current sample
            sample_idx = files.index(file)

            # Collect the surrounding images for the Z-stack
            start_idx = max(sample_idx - (z_stack_number // 2), 0)
            end_idx = min(start_idx + z_stack_number, len(files))

            # Collect the images to form the Z-stack
            for idx in range(start_idx, end_idx):
                stacked_samples.append(files[idx])

            # Generate the Z-stack for the collected images
            tifs_to_zstack(stacked_samples, out_path, f"{sample_id}")

    else:
        for file in regularly_spaced_samples:
            # Define the destination path for each file
            destination_path = out_path / f"{sample_id}_{file.name}"
            print(f"Copying {file} to {destination_path}")
            #shutil.copy2(str(file), str(destination_path))
            print(f"All selected files from {sample_id} have been copied to {destination_path}")
