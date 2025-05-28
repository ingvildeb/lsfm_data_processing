import pandas as pd
import numpy as np
import shutil
from pathlib import Path

##################
# USER PARAMETERS

## Specify the paths (any number of paths) to LSFM data 
folder_paths = [Path(r"M:\SmartSPIM_Data\2025\2025_03\2025_03_20\20250320_17_02_13_NB_CS0290_M_P533_C57_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE\\"),
                Path(r"M:\SmartSPIM_Data\2025\2025_03\2025_03_25\20250325_14_16_38_NB_CS0295_F_P417_Tg_SwDI_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE\\"),
                Path(r"M:\SmartSPIM_Data\2025\2025_03\2025_03_27\20250327_19_07_55_NB_CS0302_F_P428_C57_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE\\"),
                ]

## Specify the channel to select images from and the sample size (number of selected images per sample)
channel = 1
sample_size = 5

## Specify where you want your selected images to be saved
out_path = Path(r"Z:\Labmembers\Ingvild\RM1\protocol_testing\202503_LargeBatch_AgingCCFBrains\pilot_analysis\561Neun\training_with_MIPs\selected_sections\\")
out_path.mkdir(exist_ok=True)

##################

# Adding 2 to sample size so first and last (likely black) images can be removed after sampling
sample_size = sample_size + 2

for path in folder_paths:
    folder_name = path.name
    sample_id, sample_age, sample_geno = folder_name.split("_")[5], folder_name.split("_")[7], folder_name.split("_")[8]

    channel_wavelengths = {0: "488", 1: "561", 2: "640"}
    file_path = path / f"Ex_{channel_wavelengths.get(channel)}_Ch{channel}_stitched_MIP20um"

    # Using pathlib to glob files
    files = sorted(file_path.glob("*.tiff"))

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
