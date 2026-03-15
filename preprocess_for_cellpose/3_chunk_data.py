"""
Chunk TIFF images (2D or z-stack) for Cellpose workflows.

Supports standard images, MIPs, z-stacks, and atlas-slice images.
"""

from pathlib import Path
import tifffile
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from lsfm_data_processing.utils.chunking import chunk_image, chunk_z_stack
from lsfm_data_processing.utils.io_helpers import (
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
stack_mode = cfg.get("stack_mode", False)

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
print(f"stack_mode = {stack_mode}")

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
    
    # Use explicit config intent rather than inferring stack-vs-image from ndim.
    if stack_mode:
        if len(shape) != 3:
            raise RuntimeError(
                "stack_mode=true requires 3D stack images with shape (z, y, x).\n"
                f"Found shape {shape} for file:\n{file}"
            )
        chunk_z_stack(file, image_outdir, chunk_size=chunk_size)
    else:
        is_supported_2d = len(shape) == 2 or (len(shape) == 3 and shape[-1] in {3, 4})
        if not is_supported_2d:
            raise RuntimeError(
                "stack_mode=false requires a 2D image.\n"
                "Supported shapes are (y, x) for grayscale or (y, x, 3/4) for RGB/RGBA.\n"
                f"Found shape {shape} for file:\n{file}"
            )
        chunk_image(file, image_outdir, chunk_size=chunk_size)

print("\nChunking complete.")

