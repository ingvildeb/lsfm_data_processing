from pathlib import Path
import cv2
import numpy as np
from utils import create_mips_from_folder, normalize_image

# USER PARAMETERS

# Give your input folders 
# (the path should be to the stitched folder of the channel to create MIPs for)
# You can put as many as you'd like within the brackets, separated by comma
input_folders = [Path(r"Z:\LSFM\2023\2023_12\20231211_JL_JL0810_M_P07_Fmr1_Npas4Ai75_LAS_488Bg_561TdTomato_4x_5umstep_LC_SUNY_A1\\"),
                 ]

# MIP parameters
z_step_size = 5  # Give the step size of your images
mip_thickness = 20  # The desired thickness for MIP in micrometers
channel = 1
 
# Normalization parameters
# Set to False if you do not want normalization
do_normalization = True 

# Optionally (if doing normalization), adjust the min and max clipping values. 
# These values work well for NeuN images.
min_val = 0
max_val = 99.9

# Flag if your files follow the old file / folder naming convention
flag_old_format = True

## MAIN CODE, DO NOT EDIT
for folder in input_folders:

    if flag_old_format:
        underscores_to_z_plane = 0
        channel_wavelengths = {0:"00", 1:"01", 2:"02"}
        channel_folder = Path(folder / f"stitched_{channel_wavelengths.get(channel)}//")
                              
    else:
        underscores_to_z_plane = 2
        channel_wavelengths = {0:"488", 1:"561", 2:"640"}
        channel_folder = Path(folder / f"Ex_{channel_wavelengths.get(channel)}_Ch{channel}_stitched//")
    

    if do_normalization:
        MIP_output_folder = channel_folder.parent / f"{channel_folder.name}_MIP{mip_thickness}um_min{min_val}_max{max_val}"
        create_mips_from_folder(channel_folder, MIP_output_folder, z_step_size, mip_thickness, underscores_to_z_plane)

        MIP_images = sorted(MIP_output_folder.glob("*.tif"))
        
        for image in MIP_images:
            normalized_image = normalize_image(image, min_val, max_val)
            cv2.imwrite(str(image), normalized_image)
    else:
        MIP_output_folder = channel_folder.parent / f"{channel_folder.name}_MIP{mip_thickness}um"
        create_mips_from_folder(channel_folder, MIP_output_folder, z_step_size, mip_thickness, underscores_to_z_plane)

    


