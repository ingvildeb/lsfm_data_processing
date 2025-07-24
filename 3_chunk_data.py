from pathlib import Path
from utils import chunk_image


# USER PARAMETERS

# Give the path to the files you would like to chunk
# The path can be to raw images or atlas slices
file_path = Path(r"Z:\Labmembers\Ingvild\Cellpose\NeuN_model\training_sections\\")

# Define the chunk size. 
# Smaller chunks (up to 128) are preferred for more dense signals (NeuN, Sytox)
# Larger chunks (e.g. 256, 512) could be useful for medium to very sparse signals
chunk_size = 128


# MAIN CODE, do not edit

# Glob for TIFF files using pathlib
files = file_path.glob("*.tif")

# Process each file
for file in files:
    # Extract folder name using pathlib
    folder_name = file.stem

    # Define the output directory for chunked images
    image_outdir = file_path / "chunked_images" / folder_name

    # Create the output directory if it doesn't exist
    image_outdir.mkdir(parents=True, exist_ok=True)
    
    # Call chunk_image function with path objects
    chunk_image(str(file), str(image_outdir), chunk_size=chunk_size)