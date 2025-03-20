import numpy as np
from collections import defaultdict, Counter
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Define the path and retrieve all image files
atlas_chunk_path = Path(r"Z:\Labmembers\Ingvild\Testing_CellPose\test_data\Ex_488_Ch0_stitched_selected_data\chunked_images\selected_atlas_chunks\\")
atlas_chunks = list(atlas_chunk_path.glob("*.tif"))
atlas_ids = Path("Z:\Labmembers\Ingvild\Testing_CellPose\test_data\CCFv3_OntologyStructure_u16.xlsx")

region_counts = defaultdict(set)  # Maps region IDs to sets of image indices
image_region_ids = []  # List to keep track of regions in each image

# Load images and gather statistics on pixel values (region IDs)
for idx, image_path in enumerate(atlas_chunks):
    image_data = np.array(Image.open(image_path))
    
    regions = set(np.unique(image_data))
    image_region_ids.append(regions)

    for region in regions:
        region_counts[region].add(idx)

# Select images to maximize region coverage
selected_images = set()
covered_regions = set()

while len(selected_images) < 100 and len(covered_regions) < len(region_counts):
    best_image = None
    max_new_regions = 0

    for idx, regions in enumerate(image_region_ids):
        if idx in selected_images:
            continue

        new_regions = regions - covered_regions
        if len(new_regions) > max_new_regions:
            max_new_regions = len(new_regions)
            best_image = idx

    if best_image is not None:
        selected_images.add(best_image)
        covered_regions.update(image_region_ids[best_image])

# Ensure Proportional Representation
while len(selected_images) < 100:
    # Calculate representation score
    current_counts = Counter([region for idx in selected_images for region in image_region_ids[idx]])
    imbalance_scores = {idx: sum(current_counts.get(region, 0) - len(region_counts[region]) for region in image_region_ids[idx]) for idx in range(len(image_region_ids))}

    best_image = max(imbalance_scores, key=imbalance_scores.get)
    selected_images.add(best_image)

# Output selected image paths
selected_image_paths = [atlas_chunks[idx] for idx in selected_images]
print("Selected Images:")
for path in selected_image_paths:
    print(path)

# Calculate overall pixel frequency of regions in selected images
selected_region_counts = Counter()
for idx in selected_images:
    image_path = atlas_chunks[idx]
    image_data = np.array(Image.open(image_path))
    unique_values, counts = np.unique(image_data, return_counts=True)
    selected_region_counts.update(dict(zip(unique_values, counts)))

# Extract regions and their frequencies, ensuring only existing IDs are included
regions = list(selected_region_counts.keys())
frequencies = list(selected_region_counts.values())

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(regions, frequencies)
plt.xlabel('Region IDs')
plt.ylabel('Pixel Frequency')
plt.title('Pixel Frequency of Regions Across Selected Images')
plt.xticks(ticks=range(len(regions)), labels=regions, rotation=90)
plt.tight_layout()
plt.show()