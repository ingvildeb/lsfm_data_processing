"""
Filter out chunk images with low average intensity.

Optionally copies corresponding atlas chunks when atlas pairing is enabled.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.utils import get_avg_pixel_value
from utils.io_helpers import load_script_config, normalize_user_path, require_dir

# -------------------------
# CONFIG LOADING
# -------------------------
test_mode = False
cfg = load_script_config(Path(__file__), "4_filter_black_chunks", test_mode=test_mode)

# -------------------------
# CONFIG PARAMETERS
# -------------------------

data_path = require_dir(
    normalize_user_path(cfg["data_path"]),
    "Chunked image parent folder"
)

pixel_val_threshold = cfg["pixel_val_threshold"]
display_selected_chunks = cfg["display_selected_chunks"]
atlas_chunks_included = cfg["atlas_chunks_included"]

# -------------------------
# OUTPUT PATHS
# -------------------------

image_out_path = data_path.parent / "filtered_image_chunks"
image_out_path.mkdir(parents=True, exist_ok=False)

if atlas_chunks_included:
    atlas_out_path = data_path.parent / "filtered_atlas_chunks"
    atlas_out_path.mkdir(parents=True, exist_ok=False)

# -------------------------
# MAIN CODE
# -------------------------

# Get input paths using pathlib
# Define output paths and create directories if they don't exist
image_chunk_paths = sorted([p for p in data_path.glob("*") if not p.name.endswith("_atlas_slice")])

# Process each image chunk
for image_chunk_path in image_chunk_paths:
    # Glob for tif files using pathlib
    image_chunks = list(image_chunk_path.glob("*.tif"))
    
    for chunk_path in image_chunks:

        # Extract chunk name and number using pathlib
        chunk_name = chunk_path.stem.split("_chunk")[0] 
        chunk_number = chunk_path.stem.split("chunk_")[-1]

        # Calculate average pixel value
        average_pixel_value = get_avg_pixel_value(str(chunk_path))
            
        if average_pixel_value > pixel_val_threshold:
            # Display the chunk image
            chunk_img = np.array(Image.open(chunk_path))

            if display_selected_chunks:
                print(f"Displaying {chunk_path}, Shape: {chunk_img.shape}")
                plt.imshow(chunk_img)
                plt.axis('off')
                plt.show()
                plt.close()

            shutil.copy2(chunk_path, image_out_path / chunk_path.name) 

            if atlas_chunks_included:

                atlas_chunk_path = chunk_path.parent.with_name(f"{image_chunk_path.name}_atlas_slice") / f"{chunk_name}_atlas_slice_chunk_{chunk_number}.tif"

                if atlas_chunk_path.exists():
                    shutil.copy2(atlas_chunk_path, atlas_out_path / atlas_chunk_path.name)
                else:
                    print(f"Warning: atlas chunk missing for {chunk_path}")
        
    print(f"Finished copying filtered chunks for {image_chunk_path.name}")
    



            

