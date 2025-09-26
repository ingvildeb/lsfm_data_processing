from pathlib import Path
import cv2
import numpy as np
import shutil
from utils import create_mips_from_folder, normalize_image

# USER PARAMETERS

# Give your input folders 
# (the path should be to the stitched folder of the channel to create MIPs for)
# You can put as many as you'd like within the brackets, separated by comma
input_folders = [Path(r"M:\SmartSPIM_Data\2025\2025_07\2025_07_30\20250730_10_05_08_NB_CS0318_M_P728_C57_LAS_488Bg_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_07\2025_07_30\20250730_13_00_06_NB_CS0319_M_P728_C57_LAS_488Bg_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_07\2025_07_31\20250731_10_39_04_NB_CS0324_F_P104_C57_LAS_488Bg_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_07\2025_07_31\20250731_13_55_49_NB_CS0325_F_P104_C57_LAS_488Bg_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_07\2025_07_29\20250729_13_56_01_NB_51120_M_P56_C57_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_06\2025_06_12\20250612_11_31_09_IEB_IEB0023_F_P56_Aldh1_LAS_488Bg_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_06\2025_06_11\20250611_12_32_08_LJS_IEB0059_F_P14_MOBP_LAS_488Bg_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_06\2025_06_06\20250606_11_00_53_IEB_IEB0001_M_P58_OxtR_LAS_488Bg_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_06\2025_06_17\20250617_15_36_41_LJS_IEB0002_M_P58_OxtR_LAS_488Bg_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_06\2025_06_17\20250617_18_19_21_LJS_IEB0003_M_P58_OxtR_LAS_488Bg_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_06\2025_06_06\20250606_13_48_56_IEB_IEB0029_F_P14_Aldh1_LAS_488Bg_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_06\2025_06_20\20250620_10_22_49_LJS_IEB0032_F_P14_Aldh1_LAS_488Bg_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_06\2025_06_13\20250613_10_26_40_LJS_IEB0038_M_P14_Aldh1_LAS_488Bg_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_06\2025_06_10\20250610_14_31_10_LJS_IEB0043_F_P14_Aldh1_LAS_488Bg_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_06\2025_06_12\20250612_14_38_54_IEB_IEB0055_F_P14_MOBP_LAS_488Bg_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_06\2025_06_20\20250620_12_59_29_LJS_IEB0037_M_P56_Aldh1_LAS_488Bg_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_06\2025_06_11\20250611_09_52_20_LJS_IEB0041_M_P14_Aldh1_LAS_488Bg_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_07\2025_07_29\20250729_10_56_23_NB_51119_M_P56_C57_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE")

                 ]


# MIP parameters

# Set to False if you do not want to make MIPs
create_MIPs = True
mip_thickness = 20  # The desired thickness for MIP in micrometers
channel = 1
 
# Normalization parameters

# Set to False if you do not want normalization
do_normalization = True 

# Optionally (if doing normalization), adjust the min and max clipping values. 
# These values work well for NeuN images.
min_val = 0
max_val = 99.9

# Flag for automatically finding z step size
# Only works if your folder names have _xumzstep_ in it, where x is the z step size
auto_detect_z_step = True

# Alternatively, set your z step manually
z_step_user = 4

# Flag if your files follow the old file / folder naming convention
flag_old_format = False

## MAIN CODE

if auto_detect_z_step:

    z_steps = [i.name.split('umstep')[0].split('_')[-1] for i in input_folders]
    
    if len(set(z_steps)) == 1:
        z_step_size = z_steps[0]
        try:
            z_step_size = int(z_step_size)
            print(f'Z step set to {z_step_size}')

        except ValueError:
            print(f"Error: The detected z step size, {z_step_size} cannot be converted to an integer. Try setting your z step manually instead.")
        
    else:
        raise ValueError(f"Different z step values found in file names! All z step values for folders must be the same.")

else:
    z_step_size = z_step_user

params_dict = {"create_MIPs": create_MIPs,
               "mip_thickness": mip_thickness,
               "channel": channel,
               "do_normalization": do_normalization,
               "min_val": min_val,
               "max_val": max_val,
               "z_step_size": z_step_size}

for folder in input_folders:

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



