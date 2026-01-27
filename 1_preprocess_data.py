from pathlib import Path
import cv2
import numpy as np
import shutil
from utils import create_mips_from_folder, normalize_image
import json


"""
Written by: Ingvild Bjerke
Last modified: 1/27/2026

Purpose: Preprocess LSFM data for use in further analysis. The script allows for creation Maximum Intensity Projection (MIP) images and / or normalizing
images to improve visibility of signal.

The script can be used on one or more samples (listed in input_folders).

The script expects channel folders to be named in the LifeCanvas format:
Ex_488_Ch0_stitched for channel 0
Ex_561_Ch1_stitched for channel 1
Ex_640_Ch2_stitched for channel 2

If your data do not follow this format, use flag_custom_format and related settings under advanced parameters.

"""

# USER PARAMETERS

# Give your input folders 
# The path should be to the PARENT folder for the brain you want to create MIPs for
# You can put as many as you'd like within the brackets, separated by comma
input_folders = [Path(r"Z:\LSFM\2025\2025_12\2025_12_18\20251218_11_02_46_NB_100179_F_P14_B6J_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_12\2025_12_18\20251218_13_18_09_NB_100451_M_P14_Kcnd3_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_12\2025_12_18\20251218_16_45_25_NB_100639_F_P14_Grn_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_12\2025_12_18\20251218_19_09_23_NB_100641_M_P14_Grn_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_12\2025_12_19\20251219_10_45_56_NB_100642_M_P14_Grn_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_12\2025_12_19\20251219_13_29_48_NB_100643_M_P14_Grn_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_12\2025_12_19\20251219_16_40_25_EH_100644_M_P14_Grn_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_12\2025_12_22\20251222_12_07_48_EH_100672_F_P14_Kcnd3_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_12\2025_12_22\20251222_15_18_21_EH_100682_M_P14_C3_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_12\2025_12_23\20251223_12_52_05_EH_100671_F_P14_Kcnd3_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_12\2025_12_23\20251223_15_14_03_EH_100689_F_P14_C3_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE")

                ]
#Path(r"Z:\LSFM\2025\2025_12\2025_12_19\20251219_20_13_26_EH_100670_F_P14_Kcnd3_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),

# MIP parameters

# Set to False if you do not want to make MIPs
create_MIPs = True
mip_thickness = 20  # The desired thickness for MIP in micrometers
channel = 2
 
# Normalization parameters

# Set to False if you do not want normalization
do_normalization = True 

# Optionally (if doing normalization), adjust the min and max clipping values. 
# These values work well for NeuN images.
min_val = 0
max_val = 99.99


## ADVANCED PARAMETERS, not relevant for most users

# If you do not have a metadata.json file, set your z step manually
z_step_user = None

# Flag if your files follow the old file / folder naming convention
flag_old_format = False

# Flag if your files follow a custom fle / folder naming convention than expected by the script (see description)
flag_custom_format = False

# Set your subfolder name manually. This needs to be the same for all input_folders.
subfolder_name = ""

# Set the number of underscores occurs before your section indices in your tif file names 
# For example, if the file name is 609720_476140_000020_ch1.tif where 000020 represents the z level, set underscores_to_z_plane to 2
underscores_to_z_plane = 1

## MAIN CODE

for folder in input_folders:
    print(f"Creating MIPs for {(folder.name.split("_")[5])} ...")
    
    json_file = Path(folder) / "metadata.json"

    # Check if the JSON file exists
    if json_file.is_file():
        with open(json_file, 'r', encoding='cp1252') as file:
            json_data = json.load(file)
            z_step_size = int(float(json_data['session_config']['Z step (µm)']))
            print(f"Using z step of {z_step_size}")
    else:
        print(f"No {json_file} file found.")

        # Check if z_step_user is defined
        if z_step_user is not None:
            print(f"Using user defined z step, which is {z_step_user}.")
            z_step_size = z_step_user
        else:
            print("Z step is set to None. Please set your z step manually and try again.")

    params_dict = {"create_MIPs": create_MIPs,
                "mip_thickness": mip_thickness,
                "channel": channel,
                "do_normalization": do_normalization,
                "min_val": min_val,
                "max_val": max_val,
                "z_step_size": z_step_size}

    if flag_old_format:
        underscores_to_z_plane = 0
        channel_wavelengths = {0:"00", 1:"01", 2:"02"}
        channel_folder = Path(folder / f"stitched_{channel_wavelengths.get(channel)}//")
    
    elif flag_custom_format:
        underscores_to_z_plane = underscores_to_z_plane
        channel_folder = Path(folder / subfolder_name)
                              
    else:
        underscores_to_z_plane = 2
        channel_wavelengths = {0:"488", 1:"561", 2:"640"}
        channel_folder = Path(folder / f"Ex_{channel_wavelengths.get(channel)}_Ch{channel}_stitched//")

    if create_MIPs:
        if do_normalization:
            MIP_output_folder = channel_folder.parent / f"{channel_folder.name}_MIP{mip_thickness}um_min{min_val}_max{max_val}"
            
            create_mips_from_folder(channel_folder, MIP_output_folder, z_step_size, mip_thickness, underscores_to_z_plane)

            MIP_images = sorted(MIP_output_folder.glob("*.tif"))
            
            for image in MIP_images:
                normalized_image = normalize_image(image, min_val, max_val)
                cv2.imwrite(str(image), normalized_image)

            with open(Path(MIP_output_folder / "parameters.txt"), "w") as file:
                file.write(str(params_dict))

        else:
            MIP_output_folder = channel_folder.parent / f"{channel_folder.name}_MIP{mip_thickness}um"
            create_mips_from_folder(channel_folder, MIP_output_folder, z_step_size, mip_thickness, underscores_to_z_plane)
            
            with open(Path(MIP_output_folder / "parameters.txt"), "w") as file:
                file.write(str(params_dict))
    else:

        if do_normalization:
            img_output_folder = channel_folder.parent / f"{channel_folder.name}_normalized_min{min_val}_max{max_val}"
            shutil.copytree(channel_folder, img_output_folder)

            images = sorted(img_output_folder.glob("*.tif"))

            for image in images:
                normalized_image = normalize_image(image, min_val, max_val)
                cv2.imwrite(str(image), normalized_image)
            
            with open(Path(img_output_folder / "parameters.txt"), "w") as file:
                file.write(str(params_dict))
                
        else:
            print("MIP creation and normalization set to False. Nothing to do here...")

    print(f"Finished creating MIPs for {(folder.name.split("_")[5])}")





