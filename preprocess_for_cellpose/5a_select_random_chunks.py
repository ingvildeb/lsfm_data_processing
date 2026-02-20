"""
Select a random subset of filtered image chunks.

Use this when atlas chunks are not available.
"""

from pathlib import Path
import shutil
import random
import math
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.io_helpers import load_script_config, normalize_user_path, require_dir

# -------------------------
# CONFIG LOADING
# -------------------------
test_mode = False
cfg = load_script_config(Path(__file__), "5a_select_random_chunks", test_mode=test_mode)

# -------------------------
# CONFIG PARAMETERS
# -------------------------

chunk_dir = require_dir(
    normalize_user_path(cfg["chunk_dir"]),
    "Filtered image chunks folder"
)

out_dir = normalize_user_path(cfg["out_dir"])
num_files_to_select = cfg["num_files_to_select"]

# -------------------------
# OUTPUT SETUP
# -------------------------

out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# MAIN CODE
# -------------------------

# List all chunk files
files = sorted(chunk_dir.glob("*"))

if not files:
    raise RuntimeError(f"No files found in chunk directory:\n{chunk_dir}")

# Calculate the spacing required
total_files = len(files)
if num_files_to_select > total_files:
    print("The number of files to select is greater than the number of available files. Selecting all available files.")
    num_files_to_select = total_files

spacing = total_files / num_files_to_select

# Select files based on calculated spacing
selected_files = [files[math.floor(i * spacing)] for i in range(num_files_to_select)]

# Shuffle the selected files to randomize their order
random.shuffle(selected_files)

# Iterate over the shuffled selected files and copy them with a unique prefix
for counter, file in enumerate(selected_files):
    # Use the counter as a unique prefix
    random_prefix = counter
    
    # Rename the file with a unique prefix
    destination_file_name = f"{random_prefix}_{file.name}"
    destination_path = out_dir / destination_file_name
    
    # Copy the file from source to target with the new name
    shutil.copy2(file, destination_path)
    print(f"Copied: {file} as {destination_file_name}")

print(f"Completed copying {num_files_to_select} files with randomized prefixes.")

