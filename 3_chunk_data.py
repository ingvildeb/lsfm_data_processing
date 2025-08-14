from pathlib import Path
from utils import chunk_image, chunk_z_stack
import tifffile

# USER PARAMETERS

# Give the path to the files you would like to chunk
# The path can be to raw images or atlas slices
file_path = Path(r"Z:\Labmembers\Ingvild\Testing_CellPose\test_3d\test_iba1\\")

# Define the chunk size. 
# Smaller chunks (up to 128) are preferred for more dense signals (NeuN, Sytox)
# Larger chunks (e.g. 256, 512) could be useful for medium to very sparse signals
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