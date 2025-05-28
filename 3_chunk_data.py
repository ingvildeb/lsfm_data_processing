from pathlib import Path
from utils import chunk_image

# Define the file path using pathlib
file_path = Path(r"Z:\Labmembers\Ingvild\RM1\protocol_testing\202503_LargeBatch_AgingCCFBrains\pilot_analysis\561Neun\training_with_MIPs\selected_sections\\")

# Glob for TIFF files using pathlib
files = file_path.glob("*.tiff")

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