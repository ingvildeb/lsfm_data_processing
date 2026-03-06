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

from lsfm_data_processing.utils.io_helpers import load_script_config, normalize_user_path, require_dir

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
avoid_reselect_existing = cfg.get("avoid_reselect_existing", False)

# -------------------------
# OUTPUT SETUP
# -------------------------

out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# MAIN CODE
# -------------------------

# List all chunk files
files = sorted([p for p in chunk_dir.glob("*") if p.is_file()])

if not files:
    raise RuntimeError(f"No files found in chunk directory:\n{chunk_dir}")

# Read existing selected files from output folder and determine prefix start
existing_outputs = sorted([p for p in out_dir.glob("*") if p.is_file()])
existing_selected_names = set()
next_prefix = 0

for existing in existing_outputs:
    if "_" in existing.name:
        prefix_str, rest_name = existing.name.split("_", 1)
        if prefix_str.isdigit():
            next_prefix = max(next_prefix, int(prefix_str) + 1)
            existing_selected_names.add(rest_name)
        else:
            existing_selected_names.add(existing.name)
    else:
        existing_selected_names.add(existing.name)

# Optionally remove chunks that were already selected in previous runs
if avoid_reselect_existing:
    files = [p for p in files if p.name not in existing_selected_names]
    if not files:
        raise RuntimeError(
            "No candidate files left after excluding already selected chunks.\n"
            f"Checked output folder:\n{out_dir}"
        )

# Validate requested sample size
if num_files_to_select <= 0:
    raise RuntimeError("num_files_to_select must be > 0.")

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
current_prefix = next_prefix
for file in selected_files:
    # Rename the file with a unique prefix
    destination_file_name = f"{current_prefix}_{file.name}"
    destination_path = out_dir / destination_file_name

    # Make sure we do not overwrite existing files in out_dir
    while destination_path.exists():
        current_prefix += 1
        destination_file_name = f"{current_prefix}_{file.name}"
        destination_path = out_dir / destination_file_name

    # Copy the file from source to target with the new name
    shutil.copy2(file, destination_path)
    print(f"Copied: {file} as {destination_file_name}")
    current_prefix += 1

print(f"Completed copying {num_files_to_select} files with randomized prefixes.")

