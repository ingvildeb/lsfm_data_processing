"""
Select representative TIFF sections from MIP folders, with optional z-stack generation.

Selection is evenly spaced with deterministic shuffling by sample ID.
The script oversamples by two planes so first/last slices can be dropped.
"""

from pathlib import Path
import shutil
import sys


parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from lsfm_data_processing.utils.stacks import tifs_to_zstack
from lsfm_data_processing.utils.io_helpers import (
    load_script_config,
    normalize_user_path,
    require_dir,
)
from lsfm_data_processing.utils.naming import get_underscore_token
from lsfm_data_processing.utils.selection import select_sections_evenly

# -------------------------
# CONFIG LOADING (shared helper)
# -------------------------
test_mode = False
cfg = load_script_config(
    Path(__file__),
    "2_select_representative_sections",
    test_mode=test_mode,
)

# -------------------------
# CONFIG PARAMETERS
# -------------------------

folder_paths = [
    require_dir(normalize_user_path(p), "Input image folder")
    for p in cfg["folder_paths"]
]

out_path = normalize_user_path(cfg["out_path"])
sample_size = cfg["sample_size"]
make_zstacks = cfg["make_zstacks"]
z_stack_number = cfg["z_stack_number"]

flag_custom_format = cfg["flag_custom_format"]
underscores_to_id_cfg = cfg["underscores_to_id"]

# validate output parent exists, then create output folder
out_path.mkdir(exist_ok=True, parents=True)

# -------------------------
# MAIN CODE
# -------------------------

for path in folder_paths:

    folder_parent = path.parent.name

    if flag_custom_format:
        underscores_to_id = underscores_to_id_cfg
    else:
        underscores_to_id = 5
    sample_id = get_underscore_token(folder_parent, underscores_to_id, "sample_id")
    
    # Load files
    files = sorted(path.glob("*.tif*"))
    n = len(files)
    
    if n == 0:
        raise RuntimeError(f"Found no images in:\n{path}\n"
                           "Did you give the MIP folder as input path?")
    
    print("-----------")
    print(f"Selecting sections from {sample_id}...")
    
    regularly_spaced_samples = select_sections_evenly(
        files=files,
        sample_id=sample_id,
        sample_size=sample_size,
        drop_edges=True,
    )
    if len(regularly_spaced_samples) == 0 and n != 0:
        continue

    # Copy each file from selected_files to the new directory
    if make_zstacks:
        
        for file in regularly_spaced_samples:

            stacked_samples = []

            # Determine the index of the current sample
            sample_idx = files.index(file)

            # Collect the surrounding images for the Z-stack
            half = z_stack_number // 2
            start_idx = max(sample_idx - half, 0)
            end_idx = min(sample_idx + half + 1, len(files))
            
            print(f"Creating z stack for {file}")

            # Collect the images to form the Z-stack
            for idx in range(start_idx, end_idx):
                stacked_samples.append(files[idx])

            # Generate the Z-stack for the collected images
            tifs_to_zstack(stacked_samples, out_path, sample_id)

        print(f"All z stacks for {sample_id} created")
        print("-----------")

    else:
        for file in regularly_spaced_samples:
            # Define the destination path for each file
            destination_path = out_path / f"{sample_id}_{file.name}"
            print(f"Copying {file} to {destination_path}")
            shutil.copy2(str(file), str(destination_path))
    
        print(f"All selected files from {sample_id} copied.")
        print("-----------")

