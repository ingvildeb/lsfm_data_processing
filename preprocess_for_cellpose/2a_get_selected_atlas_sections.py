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
    require_dir,
    require_file,
    require_subpath,
)

# -------------------------
# CONFIG LOADING
# -------------------------

cfg = load_script_config(Path(__file__), "2a_get_selected_atlas_sections")

# -------------------------
# CONFIG PARAMETERS
# -------------------------

sample_paths = [require_dir(p, "Sample folder") for p in cfg["sample_paths"]]
selected_images_path = require_dir(cfg["selected_images_path"], "Selected images folder")

underscores_to_index = cfg["underscores_to_index"]
file_number_increment = cfg["file_number_increment"]

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

    sample_id = sample_path.stem.split("_")[5]
    print(f"Selecting atlas sections from sample {sample_id}")
    print("-------------")
    all_images_path = sample_path / "Ex_561_Ch1_stitched"

    subset_images = sorted(selected_images_path.glob(f"{sample_id}*.tif"))

    for image in subset_images:
        print(image)
        name, atlas_slice = extract_atlas_plate(reg_vol_path, image, all_images_path, underscores_to_index, file_number_increment)
        image_data = np.array(Image.open(image))
        print(name)

        # plot the slice using matplotlib for visual validation
        # Compute atlas boundaries for preview
        atlas_preview = relabel_sequential_for_preview(atlas_slice)

        plt.figure(figsize=(6, 6))
        plt.imshow(atlas_preview.T, cmap="tab20", origin="lower", alpha=0.9)
        plt.imshow(image_data.T, cmap="gray", origin="lower", alpha=0.3)
        plt.title(f"Horizontal Slice for {image.name}")
        plt.axis("off")
        plt.show()
        print("-------------")

        # save the atlas slice as a 16 bit tiff image

        cv2.imwrite(rf'{selected_images_path}\{name.split(".")[0]}_atlas_slice.tif', atlas_slice)

        print("Image has been saved successfully.")