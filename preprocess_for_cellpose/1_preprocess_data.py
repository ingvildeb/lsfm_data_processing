"""
Preprocess LSFM samples by creating MIPs and/or normalized images.

Expected default channel folder naming:
- Ex_488_Ch0_stitched for channel 0
- Ex_561_Ch1_stitched for channel 1
- Ex_640_Ch2_stitched for channel 2

Use `flag_custom_format` settings in config if your folder layout differs.
"""

from pathlib import Path
import sys
import json
import tifffile

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.utils import create_mips_from_folder, normalize_array, _raise_if_windows_path_too_long
from utils.io_helpers import (
    load_script_config,
    normalize_user_path,
    require_dir
)

# -------------------------
# CONFIG LOADING (shared helper)
# -------------------------
test_mode = False
cfg = load_script_config(
    Path(__file__),
    "1_preprocess_data_config",
    test_mode=test_mode,
)

# -------------------------
# CONFIG PARAMETERS
# -------------------------

input_folders = [
    require_dir(normalize_user_path(p), "Input sample folder")
    for p in cfg["input_folders"]
]

create_MIPs = cfg["create_MIPs"]
mip_thickness = cfg["mip_thickness"]
channel = cfg["channel"]

do_normalization = cfg["do_normalization"]
min_val = cfg["min_val"]
max_val = cfg["max_val"]
convert_to_8bit = cfg.get("convert_to_8bit", False)

# advanced
z_step_user = cfg.get("z_step_user")
flag_old_format = cfg["flag_old_format"]
flag_custom_format = cfg["flag_custom_format"]
subfolder_name = cfg["subfolder_name"]
underscores_to_z_plane_cfg = cfg["underscores_to_z_plane"]

# -------------------------
# MAIN CODE
# -------------------------

for folder in input_folders:

    params_dict = {"create_MIPs": create_MIPs,
                "mip_thickness": mip_thickness,
                "channel": channel,
                "do_normalization": do_normalization,
                "min_val": min_val,
                "max_val": max_val,
                "convert_to_8bit": convert_to_8bit}

    if flag_old_format:
        underscores_to_z_plane = 0
        channel_wavelengths = {0:"00", 1:"01", 2:"02"}
        channel_folder = Path(folder / f"stitched_{channel_wavelengths.get(channel)}//")
    
    elif flag_custom_format:
        underscores_to_z_plane = underscores_to_z_plane_cfg
        channel_folder = Path(folder / subfolder_name)
                              
    else:
        underscores_to_z_plane = 2
        channel_wavelengths = {0:"488", 1:"561", 2:"640"}
        channel_folder = Path(folder / f"Ex_{channel_wavelengths.get(channel)}_Ch{channel}_stitched//")

    if create_MIPs:
        print(f"Creating MIPs for {(folder.name.split('_')[5])} ...")
    
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

        params_dict["z_step_size"] = z_step_size

        if do_normalization:
            MIP_output_folder = channel_folder.parent / f"{channel_folder.name}_MIP{mip_thickness}um_min{min_val}_max{max_val}"
            
            create_mips_from_folder(
                channel_folder,
                MIP_output_folder,
                z_step_size,
                mip_thickness,
                underscores_to_z_plane,
                do_normalization=True,
                min_val=min_val,
                max_val=max_val,
                convert_to_8bit=convert_to_8bit,
            )

            params_file = Path(MIP_output_folder / "parameters.txt")
            if sys.platform.startswith("win"):
                _raise_if_windows_path_too_long(params_file)
            with open(params_file, "w") as file:
                file.write(str(params_dict))

        else:
            MIP_output_folder = channel_folder.parent / f"{channel_folder.name}_MIP{mip_thickness}um"
            create_mips_from_folder(
                channel_folder,
                MIP_output_folder,
                z_step_size,
                mip_thickness,
                underscores_to_z_plane,
                do_normalization=False,
                convert_to_8bit=convert_to_8bit,
            )
            
            params_file = Path(MIP_output_folder / "parameters.txt")
            if sys.platform.startswith("win"):
                _raise_if_windows_path_too_long(params_file)
            with open(params_file, "w") as file:
                file.write(str(params_dict))
    else:

        if do_normalization:
            print(f"Creating normalized images for {(folder.name.split('_')[5])} ...")
            img_output_folder = channel_folder.parent / f"{channel_folder.name}_norm_min{min_val}_max{max_val}"
            img_output_folder.mkdir(parents=True, exist_ok=True)

            images = sorted(channel_folder.glob("*.tif"))
            if sys.platform.startswith("win"):
                for image in images:
                    _raise_if_windows_path_too_long(img_output_folder / image.name)

            for image in images:
                image_array = tifffile.TiffFile(image).asarray()
                normalized_image = normalize_array(
                    image_array,
                    min_val=min_val,
                    max_val=max_val,
                    convert_to_8bit=convert_to_8bit,
                )
                tifffile.imwrite(img_output_folder / image.name, normalized_image)
            
            params_file = Path(img_output_folder / "parameters.txt")
            if sys.platform.startswith("win"):
                _raise_if_windows_path_too_long(params_file)
            with open(params_file, "w") as file:
                file.write(str(params_dict))
                
        else:
            print("MIP creation and normalization set to False. Nothing to do here...")

    print(f"Finished creating MIPs for {(folder.name.split('_')[5])}")





