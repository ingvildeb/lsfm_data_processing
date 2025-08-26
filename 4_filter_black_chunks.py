from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil
from utils import get_avg_pixel_value


##################
# USER PARAMETERS

# Define the path to your chunked images
data_path = Path(r"Z:\Labmembers\Ingvild\Cellpose\Iba1_model\training_sections\chunked_images\\")

# Define a pixel value threshold. Images with an average intensity below this threshold will be filtered out.
pixel_val_threshold = 10

# Set to True if you also have atlas chunks
atlas_chunks_included = False


# MAIN CODE, do not edit

# Get input paths using pathlib
# Define output paths and create directories if they don't exist
image_chunk_paths = data_path.glob("*")
image_out_path = data_path.parent / "filtered_image_chunks"
image_out_path.mkdir(exist_ok=True)

if atlas_chunks_included:
    atlas_chunk_paths = data_path.glob("*atlas_slice")
    atlas_out_path = data_path / "filtered_atlas_chunks"
    atlas_out_path.mkdir(exist_ok=True)

# Process each image chunk
for image_chunk_path in image_chunk_paths:
    #print(image_chunk_path)
    # Glob for tif files using pathlib
    image_chunks = list(image_chunk_path.glob("*.tif"))
    
    for chunk_path in image_chunks:
        #print(chunk_path)
        # Extract chunk name and number using pathlib
        chunk_name = chunk_path.stem.split("_chunk")[0] 
        chunk_number = chunk_path.stem.split("chunk_")[-1]

        # Calculate average pixel value
        average_pixel_value = get_avg_pixel_value(str(chunk_path))

        # Build atlas chunk path using pathlib
        if atlas_chunks_included:
            atlas_chunk_path = chunk_path.parent.with_name(f"{image_chunk_path.name}_atlas_slice") / f"{chunk_name}_atlas_slice_chunk_{chunk_number}.tif"

        if average_pixel_value > pixel_val_threshold:
            # Display the chunk image
            chunk_img = np.array(Image.open(chunk_path))
            print(f"Displaying {chunk_path}, Shape: {chunk_img.shape}")
            plt.imshow(chunk_img)
            plt.axis('off')
            plt.show()
            plt.close()


            shutil.copy2(chunk_path, image_out_path / chunk_path.name) 

            if atlas_chunks_included:
                shutil.copy2(atlas_chunk_path, atlas_out_path / atlas_chunk_path.name)
    
    print("Finished copying filtered chunks")

            
