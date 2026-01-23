from pathlib import Path
import cv2
import numpy as np
import shutil
from utils import create_mips_from_folder, normalize_image
import json

# USER PARAMETERS

# Give your input folders 
# (the path should be to the stitched folder of the channel to create MIPs for)
# You can put as many as you'd like within the brackets, separated by comma
input_folders = [Path(r"Z:\LSFM\2025\2025_10\2025_10_21\20251021_10_41_55_EH_IEB0130_M_P0_Aldh1_LAS_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_21\20251021_11_06_22_EH_IEB0132_M_P0_Aldh1_LAS_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_09\20251009_19_34_52_IEB_IEB0133_M_P4_Aldh1_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_07\20251007_18_51_33_IEB_IEB0134_M_P4_Aldh1_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_09\20251009_20_53_50_IEB_IEB0136_F_P4_Aldh1_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_07\20251007_20_01_06_IEB_IEB0137_M_P4_Aldh1_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_22\20251022_11_34_41_EH_IEB0138_M_P2_MOBP_LAS_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_22\20251022_13_30_43_EH_IEB0139_M_P2_MOBP_LAS_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_22\20251022_14_05_23_EH_IEB0140_F_P2_MOBP_LAS_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_13\20251013_10_58_46_IEB_IEB0142_F_P4_MOBP_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_23\20251023_10_08_22_EH_IEB0143_M_P4_MOBP_LAS_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_09\20251009_14_07_07_IEB_IEB0144_F_P4_MOBP_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_13\20251013_12_05_56_IEB_IEB0145_F_P4_MOBP_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_23\20251023_10_41_47_EH_IEB0146_F_P4_MOBP_LAS_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_23\20251023_11_48_35_EH_IEB0149_M_P8_Aldh1_LAS_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_24\20251024_14_11_14_EH_IEB0150_M_P4_MOBP_LAS_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_13\20251013_14_20_35_IEB_IEB0151_F_P4_MOBP_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_09\20251009_15_19_25_IEB_IEB0152_F_P4_MOBP_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_24\20251024_14_46_29_EH_IEB0153_F_P4_MOBP_LAS_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_23\20251023_12_41_35_EH_IEB0154_M_P6_MOBP_LAS_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_13\20251013_15_32_10_IEB_IEB0155_F_P6_MOBP_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_13\20251013_18_36_35_IEB_IEB0156_F_P6_MOBP_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_13\20251013_20_05_23_IEB_IEB0157_F_P6_MOBP_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_25\20251025_13_30_30_EH_IEB0158_M_P10_MOBP_LAS_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                 Path(r"Z:\LSFM\2025\2025_10\2025_10_25\20251025_14_44_10_EH_IEB0159_F_P10_MOBP_LAS_561Bg_640Sytox_4x_5umstep_Destripe_DONE"),
                ]

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
max_val = 99.9


## ADVANCED PARAMETERS, not relevant for most users

# If you do not have a metadata.json file, set your z step manually
z_step_user = None

# Flag if your files follow the old file / folder naming convention
flag_old_format = False

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




