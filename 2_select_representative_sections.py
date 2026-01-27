import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from utils import tifs_to_zstack

"""
Written by: Ingvild Bjerke
Last modified: 1/27/2026

Purpose: Selecting representative samples from a folder of LSFM tif files.
Samples will be selected to be evenly spaced, with a small amount of shuffling between brains to avoid sampling similar levels from all brains.
I recommend "oversampling" a bit (e.g. if you want 5 sections in total, set sample_size to 7), and manually curate the selection afterwards; this increases
the likelihood of also getting some very dorsal or very ventral sections.

"""

##################
# USER PARAMETERS

## Specify the paths (any number of paths) to LSFM data 
## The path needs to be the SUBFOLDER where tifs are placed, not just the parent folder
folder_paths = [
    Path(r"Z:\LSFM\2025\2025_10\2025_10_08\20251008_18_03_51_NB_100356_F_P14_B6J_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE\Ex_561_Ch1_stitched_MIP20um_min0_max99.9"),
    Path(r"Z:\LSFM\2025\2025_11\2025_11_05\20251105_13_01_02_NB_100640_F_P14_Grn_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE\Ex_561_Ch1_stitched_MIP20um_min0_max99.9"),
    Path(r"Z:\LSFM\2025\2025_11\2025_11_10\20251110_19_19_16_NB_100434_F_P61_Shank3_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE\Ex_561_Ch1_stitched_MIP20um_min0_max99.9"),
    Path(r"Z:\LSFM\2025\2025_12\2025_12_19\20251219_10_45_56_NB_100642_M_P14_Grn_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE\Ex_561_Ch1_stitched_MIP20um_min0_max99.9"),
    Path(r"Z:\LSFM\2025\2025_12\2025_12_22\20251222_12_07_48_EH_100672_F_P14_Kcnd3_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE\Ex_561_Ch1_stitched_MIP20um_min0_max99.9"),
]

## Specify the sample size (number of selected images per sample)
sample_size = 7

## Specify where you want your selected images to be saved
out_path = Path(r"Z:\Labmembers\Ingvild\Cellpose\NeuN_model\1_training_data\model_256by256_val\\")

## Z STACK OPTIONS
# Optional creation of z stack. If relevant, set to true and indicate the number of images you want in your stack.
# If not, set make_zstacks to False

make_zstacks = False
z_stack_number = 5 

## Advanced options
flag_custom_format = True

# Give the number of underscores that precedes the sample id in the name of the parent folder
# For example, if the folder name is 20251222_12_07_48_EH_100672_F_P14_Kcnd3_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE,
# where 100672 is the sample ID, the number of underscores_to_id is 5
underscores_to_id = 5

##################
# MAIN CODE

out_path.mkdir(exist_ok=True)

# Adding two to sample size so first and / or last (usually black) slices can be removed
sample_size = sample_size + 2

for path in folder_paths:

    folder_parent = path.parent.name
    sample_id = folder_parent.split("_")[5]

    # Load files
    files = sorted(path.glob("*.tif*"))
    n = len(files)
    
    if n == 0:
        raise RuntimeError(f"Found no images in:\n{path}\n"
                           "Did you give the MIP folder as input path?")
    
    print(f"Selecting sections from {sample_id}...")
    
    if n < sample_size:
        print("Not enough images to sample. Selecting all sections.")
        regularly_spaced_samples = files

    else:
        
        # deterministic RNG per sample
        seed = hash(sample_id) % 2**32
        rng = np.random.default_rng(seed)

        # spacing
        step = n // (sample_size - 1)

        # bounded offset
        offset = rng.integers(0, step)

        # evenly spaced positions with offset
        positions = offset + np.arange(sample_size) * step

        # removing any out-of-bound positions
        positions = positions[positions < n]

        # permutation to break anatomical alignment
        indices = rng.permutation(n)

        regularly_spaced_samples = [
            files[indices[p]] for p in positions
        ]

    regularly_spaced_samples = sorted(regularly_spaced_samples)

    # drop first / last
    if len(positions) - sample_size == 0: 
        del regularly_spaced_samples[0]
        del regularly_spaced_samples[-1]

    elif len(positions) - sample_size == -1:
        del regularly_spaced_samples[0]
    
    else:
        continue

    # Copy each file from selected_files to the new directory
    if make_zstacks:
        
        for file in regularly_spaced_samples:

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
            shutil.copy2(str(file), str(destination_path))
    
    print(f"All selected files from {sample_id} have been copied to {destination_path}")
