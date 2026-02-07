from pathlib import Path
import shutil
import random
import math

"""
Written by: Ingvild Bjerke
Last modified: 1/27/2026

Purpose: Select a set of random chunks for cellpose training, validation or test sets.
Use this option if you do NOT have atlas chunks and just want a random selection.

If you made atlas chunks in the previous steps, you can instead use the script 5b_select_representative_chunks.py to get a 
representative subset of different brain regions.

"""


# USER PARAMETERS

# Give the path to your filtered image chunks
chunk_dir = Path(r"Z:\Labmembers\Ingvild\Cellpose\NeuN_model\test_256chunks\filtered_image_chunks")

# Give the path where you want the selected chunks to be saved
out_dir = Path(r"Z:\Labmembers\Ingvild\Cellpose\NeuN_model\test_256chunks\training_chunks")

# Specify the number of chunks to select
num_files_to_select = 100


# MAIN CODE, do not edit

# Create the target directory if it doesn't exist
out_dir.mkdir(exist_ok=True)

# List all files in the source directory and sort them to maintain order
files = sorted(chunk_dir.glob("*"))

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

print(f"Completed copying {num_files_to_select} files with unique prefixes.")
