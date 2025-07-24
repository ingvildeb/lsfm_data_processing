import numpy as np
from collections import defaultdict, Counter
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import shutil


# USER PARAMETERS

# Give the path to your filtered image chunks
chunk_dir = Path(r"Z:\Labmembers\Ingvild\Testing_CellPose\test_data\Ex_488_Ch0_stitched_selected_data\chunked_images\\")

# Give the path to your filtered atlas chunks
atlas_chunk_dir = Path(r"Z:\Labmembers\Ingvild\Testing_CellPose\test_data\Ex_488_Ch0_stitched_selected_data\chunked_images\filtered_atlas_chunks\\")
atlas_chunks = list(atlas_chunk_dir.glob("*.tif"))
#atlas_ids = Path("Z:\Labmembers\Ingvild\Testing_CellPose\test_data\CCFv3_OntologyStructure_u16.xlsx")

# Specify the number of chunks to select
number_of_chunks = 100

# MAIN CODE, do not edit

# Path setup
base_path = Path(__file__).parent.resolve()
atlas_ids = base_path / "files" / "CCFv3_OntologyStructure_u16.xlsx"

region_counts = defaultdict(set)  # Maps region IDs to sets of image indices
image_region_ids = []  # List to keep track of regions in each image

# Load atlas chunks and gather statistics on pixel values (region IDs)
for idx, image_path in enumerate(atlas_chunks):
    image_data = np.array(Image.open(image_path))
    
    regions = set(np.unique(image_data))
    image_region_ids.append(regions)

    for region in regions:
        region_counts[region].add(idx)

# Select images to maximize region coverage
selected_images = set()
covered_regions = set()

while len(selected_images) < number_of_chunks and len(covered_regions) < len(region_counts):
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

# Random selection of remaining images
available_images = set(range(len(image_region_ids))) - selected_images
while len(selected_images) < number_of_chunks:
    random_image = random.choice(list(available_images))
    selected_images.add(random_image)
    available_images.remove(random_image)

# Output selected image paths
selected_atlas_chunks = [atlas_chunks[idx] for idx in selected_images]

# Define output paths and create directories if they don't exist
image_out_path = chunk_dir / "selected_image_chunks"
atlas_out_path = chunk_dir / "selected_atlas_chunks"

image_out_path.mkdir(exist_ok=True)
atlas_out_path.mkdir(exist_ok=True)

for atlas_chunk in selected_atlas_chunks:
    # Extract chunk name and number using pathlib
    chunk_name = atlas_chunk.stem.split("_atlas")[0]
    chunk_number = atlas_chunk.stem.split("chunk_")[-1]
    corresponding_image_name = f"{chunk_name}_chunk_{chunk_number}.tif"
    image_path = atlas_chunk.parent.parent / "filtered_image_chunks" / corresponding_image_name

    # Copy files to out paths
    shutil.copy2(image_path, image_out_path / image_path.name)
    shutil.copy2(atlas_chunk, atlas_out_path / atlas_chunk.name)