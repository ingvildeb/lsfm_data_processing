from pathlib import Path
from utils import chunk_image, chunk_z_stack
import tifffile

"""
Written by: Ingvild Bjerke
Last modified: 1/27/2026

Purpose: Chunk images for use with cellpose. 
The script takes a folder of images and creates chunks of a user-defined size.
Works with single-plane images, MIPs or z stacks. Also works for atlas section images (which can be created in step 2a).

Chunk images will be saved to individual folders for each parent image and named after the row and column indices from the
parent image.

"""

# USER PARAMETERS

# Give the path to the files you would like to chunk
# The path can be to raw images or atlas slices
file_path = Path(r"Z:\Labmembers\Ingvild\Cellpose\Iba1_model\2_validation_images\\")

# Define the chunk size. 
# NB: The network expects a chunk size of 256. It is not recommended to use chunks below this size for training,
# as it can cause problems when applying the model to larger images.
# Larger chunks (e.g. 512) could be useful for medium to very sparse signals, but if 256 works for your signal it 
# is the safest option.
chunk_size = 256


# MAIN CODE, do not edit

# Glob for TIFF files using pathlib
files = file_path.glob("*.tif") 

# Process each file
for file in files:
    
    img = tifffile.TiffFile(file).asarray()
    shape = img.shape

    # Extract folder name using pathlib
    folder_name = file.stem

    # Define the output directory for chunked images
    image_outdir = file_path / f"chunked_images_{chunk_size}by{chunk_size}" / folder_name

    # Create the output directory if it doesn't exist
    image_outdir.mkdir(parents=True, exist_ok=True)
    
    # Call chunk_image function with path objects
    if len(shape) == 2:
        chunk_image(file, image_outdir, chunk_size=chunk_size)

    elif len(shape) == 3:
        chunk_z_stack(file, image_outdir, chunk_size=chunk_size)
    
