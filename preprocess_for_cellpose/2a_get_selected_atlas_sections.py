"""
Extract atlas slices matching selected section images from step 2.

Requires each sample folder to contain:
- _01_registration/ANTs_TransformedImage.nii.gz
"""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.utils import extract_atlas_plate, relabel_sequential_for_preview
from utils.io_helpers import (
    load_script_config,
    normalize_user_path,
    require_dir,
    require_file,
    require_subpath,
)

# -------------------------
# CONFIG LOADING
# -------------------------
test_mode = False
cfg = load_script_config(Path(__file__), "2a_get_selected_atlas_sections", test_mode=test_mode)

# -------------------------
# CONFIG PARAMETERS
# -------------------------

sample_paths = [
    require_dir(normalize_user_path(p), "Sample folder")
    for p in cfg["sample_paths"]
]
selected_images_path = require_dir(
    normalize_user_path(cfg["selected_images_path"]),
    "Selected images folder"
)

underscores_to_index = cfg["underscores_to_index"]
file_number_increment = cfg["file_number_increment"]
flag_custom_format = cfg.get("flag_custom_format", False)
underscores_to_id_cfg = cfg.get("underscores_to_id", 5)
all_images_subfolder = cfg.get("all_images_subfolder", "Ex_561_Ch1_stitched")
show_preview = cfg.get("show_preview", True)

# -------------------------
# MAIN CODE
# -------------------------

for sample_path in sample_paths:
    
    registration_dir = require_subpath(
        sample_path,
        "_01_registration",
        "registration folder"
    )

    reg_vol_path = require_file(
        registration_dir / "ANTs_TransformedImage.nii.gz",
        "registered atlas volume"
    )

    sample_parts = sample_path.stem.split("_")

    if flag_custom_format:
        underscores_to_id = underscores_to_id_cfg
    else:
        underscores_to_id = 5

    if underscores_to_id >= len(sample_parts):
        raise RuntimeError(
            f"Cannot extract sample_id from folder name:\n{sample_path.stem}\n"
            f"Expected at least {underscores_to_id+1} underscore parts"
        )

    sample_id = sample_parts[underscores_to_id]
    print(f"Selecting atlas sections from sample {sample_id}")
    print("-------------")
    all_images_path = require_dir(
        sample_path / all_images_subfolder,
        "full stitched image folder"
    )

    subset_images = sorted(selected_images_path.glob(f"{sample_id}*.tif"))

    for image in subset_images:
        print(image)
        name, atlas_slice = extract_atlas_plate(reg_vol_path, image, all_images_path, underscores_to_index, file_number_increment)
        image_data = np.array(Image.open(image))
        print(name)

        # plot the slice using matplotlib for visual validation
        # Compute atlas boundaries for preview
        atlas_preview = relabel_sequential_for_preview(atlas_slice)

        if show_preview:
            plt.figure(figsize=(6, 6))
            plt.imshow(atlas_preview.T, cmap="tab20", origin="lower", alpha=0.9)
            plt.imshow(image_data.T, cmap="gray", origin="lower", alpha=0.3)
            plt.title(f"Horizontal Slice for {image.name}")
            plt.axis("off")
            plt.show()
            print("-------------")

        # save the atlas slice as a 16 bit tiff image

        out_file = selected_images_path / f"{name.split('.')[0]}_atlas_slice.tif"
        cv2.imwrite(str(out_file), atlas_slice)

        print("Image has been saved successfully.")

