from pathlib import Path
from utils import chunk_image

# Define the file path using pathlib
file_path = Path(r"Z:\Labmembers\Ingvild\Testing_CellPose\test_data\Ex_488_Ch0_stitched_selected_data")

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
    chunk_image(str(file), str(image_outdir))