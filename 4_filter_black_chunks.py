from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil
from utils import get_avg_pixel_value

# Define base data path
base_data_path = Path(r"Z:\Labmembers\Ingvild\RM1\protocol_testing\202503_LargeBatch_AgingCCFBrains\pilot_analysis\561Neun\training_with_MIPs\selected_sections\chunked_images\\")

# Get input paths using pathlib
image_chunk_paths = base_data_path.glob("*")
#atlas_chunk_paths = base_data_path.glob("*atlas_slice")

# Define output paths and create directories if they don't exist
image_out_path = base_data_path / "filtered_image_chunks"
#atlas_out_path = base_data_path / "filtered_atlas_chunks"

image_out_path.mkdir(exist_ok=True)
#atlas_out_path.mkdir(exist_ok=True)

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
        average_pixel_value = int(average_pixel_value)
        # Build atlas chunk path using pathlib
        #atlas_chunk_path = chunk_path.parent.with_name(f"{image_chunk_path.name}_atlas_slice") / f"{chunk_name}_atlas_slice_chunk_{chunk_number}.tif"

        if average_pixel_value > 100:
            # Display the chunk image
            chunk_img = np.array(Image.open(chunk_path))
            print(f"Displaying {chunk_path}, Shape: {chunk_img.shape}")
            plt.imshow(chunk_img)
            plt.axis('off')
            plt.show()

            # Copy files using pathlib for path management
            shutil.copy2(chunk_path, image_out_path / chunk_path.name) 
            #shutil.copy2(atlas_chunk_path, atlas_out_path / atlas_chunk_path.name)
