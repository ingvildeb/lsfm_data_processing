from pathlib import Path
import cv2
import numpy as np
from utils import create_mips_from_folder, normalize_image

# USER PARAMETERS

# Give your input folders 
# (the path should be to the stitched folder of the channel to create MIPs for)
# You can put as many as you'd like within the brackets, separated by comma
input_folders = [Path(r"M:\SmartSPIM_Data\2025\2025_04\2025_04_24\20250424_11_45_45_IEB_IEB0031_F_P14_Aldh1_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE\\"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_06\2025_06_02\20250602_14_50_20_LJS_IEB0035_M_P56_Aldh1_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE\\"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_05\2025_05_23\20250523_10_33_49_IEB_IEB0046_F_P10_Aldh1_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE\\"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_05\2025_05_23\20250523_12_15_52_IEB_IEB0066_F_P8_Aldh1_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE\\")
                 ]

# MIP parameters
z_step_size = 5  # Give the step size of your images
mip_thickness = 20  # The desired thickness for MIP in micrometers
channel = 0
 
# Normalization parameters
# Set to False if you do not want normalization
do_normalization = True 

# Optionally (if doing normalization), adjust the min and max clipping values. 
# These values work well for NeuN images.
min_val = 0
max_val = 99.9

## MAIN CODE, DO NOT EDIT
channel_wavelengths = {0:"488", 1:"561", 2:"640"}

for folder in input_folders:
    channel_folder = Path(folder / f"Ex_{channel_wavelengths.get(channel)}_Ch{channel}_stitched//")

    if do_normalization:
        MIP_output_folder = channel_folder.parent / f"{channel_folder.name}_MIP{mip_thickness}um_min{min_val}_max{max_val}"
        create_mips_from_folder(channel_folder, MIP_output_folder, z_step_size, mip_thickness)

        MIP_images = sorted(MIP_output_folder.glob("*.tif"))
        
        for image in MIP_images:
            normalized_image = normalize_image(image, min_val, max_val)
            cv2.imwrite(str(image), normalized_image)
    else:
        MIP_output_folder = channel_folder.parent / f"{channel_folder.name}_MIP{mip_thickness}um"
        create_mips_from_folder(channel_folder, MIP_output_folder, z_step_size, mip_thickness)

    


