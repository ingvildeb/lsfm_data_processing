from pathlib import Path
import shutil
import random

# Define source and target directories
source_directory = Path(r"Z:\Labmembers\Ingvild\Cellpose\Aldh_model\training_sections\chunked_images\filtered_image_chunks")
target_directory = Path(r"Z:\Labmembers\Ingvild\Cellpose\Aldh_model\training_chunks")
select_every = 5

# Create the target directory if it doesn't exist
target_directory.mkdir(exist_ok=True)

# List all files in the source directory and sort them to maintain order
files = sorted(source_directory.glob("*"))

# Calculate indices for every 10th file in the list
indices = [i for i in range(len(files)) if i % select_every == 0]

# Shuffle the indices to create random unique prefixes
random.shuffle(indices)

# Iterate over indices and copy corresponding files with a unique prefix
for counter, i in enumerate(indices):
    file = files[i]
    
    # Use shuffled indices as unique prefix
    random_prefix = counter
    
    # Rename the file with a unique prefix
    destination_file_name = f"{random_prefix}_{file.name}"
    destination_path = target_directory / destination_file_name
    
    # Copy the file from source to target with the new name
    shutil.copy2(file, destination_path)
    print(f"Copied: {file} as {destination_file_name}")

print(f"Completed copying every {select_every}th file with unique random prefix.")
