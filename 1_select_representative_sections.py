import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from utils import get_avg_pixel_value

base_path = r"Z:\Labmembers\Ingvild\Testing_CellPose\test_data\\"
file_path = rf"{base_path}Ex_488_Ch0_stitched\\"
files = glob(f"{file_path}*.tif")

intensity_threshold = 5
sample_size = 10

# Process and select images that are likely to be within the brain

selected_images = []

for file in files:
    pixel_value = get_avg_pixel_value(file)
    if pixel_value > intensity_threshold:
        selected_images.append(file)

# Visual check that the first and last images of the selection are within the brain. If not, increase the pixel threshold.

first_sample = Image.open(selected_images[0])
last_sample = Image.open(selected_images[-1])

plt.imshow(first_sample)
plt.axis('off')  # Turn off axis
plt.show()

plt.imshow(last_sample)
plt.axis('off')  # Turn off axis
plt.show()

# If visual check complete, move on to sample a representative subset of the images

# Ensure at least sample_size items exist in selected_images
if len(selected_images) < sample_size:
    print("Not enough images above intensity threshold to sample.")
    regularly_spaced_samples = selected_images
else:
    # Calculate the step size using integer indices correctly
    step = len(selected_images) // (sample_size - 1) if sample_size > 1 else len(selected_images)

    # Initialize selected samples empty to collect 5 samples
    regularly_spaced_samples = []

    # Collect samples using computed step
    for i in range(sample_size):
        idx = min(i * step, len(selected_images) - 1)  # Ensure indices are valid
        regularly_spaced_samples.append(selected_images[idx])



# Copy the sample subset to a new folder for further processing

# Create new path by appending '_selected_data' to the original folder name
full_folder = os.path.basename(os.path.normpath(file_path))  # Extract base directory name
new_folder_path = f"{base_path}{full_folder}_selected_data"

# Ensure the new directory exists, create if it does not
os.makedirs(new_folder_path, exist_ok=True)

# Copy each file from selected_files to the new directory
for file_path in regularly_spaced_samples:
    # Define the destination path for each file
    destination_path = os.path.join(new_folder_path, os.path.basename(file_path))
    print(f"Copying {file_path} to {destination_path}")
    shutil.copy2(file_path, destination_path)

print(f"All selected files have been copied to {new_folder_path}")