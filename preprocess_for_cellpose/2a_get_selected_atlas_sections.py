"""
Extract atlas slices matching selected section images from step 2.

Requires each sample folder to contain:
- _01_registration/ANTs_TransformedImage.nii.gz
"""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import nibabel as nib
from PIL import Image
import numpy as np
import cv2

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from lsfm_data_processing.utils.atlas import atlas_slice_for_mip, relabel_sequential_for_preview
from lsfm_data_processing.utils.io_helpers import (
    load_script_config,
    normalize_user_path,
    require_dir,
    require_file,
    require_subpath,
)
from lsfm_data_processing.utils.naming import get_underscore_int, get_underscore_token

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
    reg_data = np.asanyarray(nib.load(reg_vol_path).dataobj)

    if flag_custom_format:
        underscores_to_id = underscores_to_id_cfg
    else:
        underscores_to_id = 5
    sample_id = get_underscore_token(sample_path.stem, underscores_to_id, "sample_id")
    print(f"Selecting atlas sections from sample {sample_id}")
    print("-------------")
    all_images_path = require_dir(
        sample_path / all_images_subfolder,
        "full stitched image folder"
    )
    all_images = [*all_images_path.glob("*.tif"), *all_images_path.glob("*.tiff")]
    no_images = len(all_images)
    if no_images == 0:
        raise RuntimeError(f"No TIFF files found in full stitched image folder:\n{all_images_path}")

    subset_images = sorted(selected_images_path.glob(f"{sample_id}*.tif"))

    for image in subset_images:
        print(image)
        name = image.stem
        number = get_underscore_int(name, underscores_to_index, "section number")
        print(f"Number extracted is {number}")
        image_data = np.array(Image.open(image))
        atlas_slice = atlas_slice_for_mip(
            reg_volume_data=reg_data,
            no_images=no_images,
            section_number=number,
            file_number_increment=file_number_increment,
            target_h=image_data.shape[0],
            target_w=image_data.shape[1],
        )
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

