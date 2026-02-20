"""
Chunk TIFF images (2D or z-stack) for Cellpose workflows.

Supports standard images, MIPs, z-stacks, and atlas-slice images.
"""

from pathlib import Path
import tifffile
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.utils import chunk_image, chunk_z_stack
from utils.io_helpers import (
    load_script_config,
    normalize_user_path,
    require_dir,
)

# -------------------------
# CONFIG LOADING
# -------------------------
test_mode = False
cfg = load_script_config(
    Path(__file__),
    "3_chunk_data",
    test_mode=test_mode,
)

# -------------------------
# CONFIG PARAMETERS
# -------------------------

file_path = require_dir(
    normalize_user_path(cfg["file_path"]),
    "Input image folder"
)

chunk_size = cfg["chunk_size"]

# -------------------------
# OUTPUT SETUP
# -------------------------

chunk_root = file_path / f"chunked_images_{chunk_size}by{chunk_size}"
chunk_root.mkdir(exist_ok=True)

# -------------------------
# MAIN
# -------------------------

# Glob for TIFF files using pathlib
files = sorted(file_path.glob("*.tif*"))

if not files:
    raise RuntimeError(
        f"No TIFF files found in:\n{file_path}"
    )

print(f"Found {len(files)} images to chunk.\n")

# Process each file
for file in files:
    print(f"Chunking image {file}...")
    
    img = tifffile.TiffFile(file).asarray()
    shape = img.shape

    # Extract folder name using pathlib
    folder_name = file.stem

    # Define the output directory for chunked images
    image_outdir = chunk_root / folder_name

    # Create the output directory if it doesn't exist
    image_outdir.mkdir(parents=True, exist_ok=True)
    
    # Call chunk_image function with path objects
    if len(shape) == 2:
        chunk_image(file, image_outdir, chunk_size=chunk_size)

    elif len(shape) == 3:
        chunk_z_stack(file, image_outdir, chunk_size=chunk_size)
    
    else:
        raise RuntimeError(
            f"Unsupported image shape {shape} for file:\n{file}"
        )

print("\nChunking complete.")

