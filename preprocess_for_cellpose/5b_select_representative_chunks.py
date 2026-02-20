"""
Select representative chunk pairs (image + atlas) to maximize region coverage.

Requires matching atlas chunks; otherwise use 5a_select_random_chunks.py.
"""

import numpy as np
from collections import defaultdict
from PIL import Image
from pathlib import Path
import shutil
import random
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.io_helpers import load_script_config, normalize_user_path, require_dir

# -------------------------
# CONFIG LOADING
# -------------------------
test_mode = False
cfg = load_script_config(Path(__file__), "5b_select_representative_chunks", test_mode=test_mode)

# -------------------------
# CONFIG PARAMETERS
# -------------------------

chunk_dir = require_dir(
    normalize_user_path(cfg["chunk_dir"]),
    "Filtered image chunks folder"
)

atlas_chunk_dir = require_dir(
    normalize_user_path(cfg["atlas_chunk_dir"]),
    "Filtered atlas chunks folder"
)

number_of_chunks = cfg["number_of_chunks"]

# -------------------------
# INPUT FILES
# -------------------------

atlas_chunks = sorted(atlas_chunk_dir.glob("*.tif"))

if not atlas_chunks:
    raise RuntimeError(
        f"No atlas chunk TIFF files found in:\n{atlas_chunk_dir}"
    )

print(f"Found {len(atlas_chunks)} atlas chunks.")

# -------------------------
# REGION COVERAGE ANALYSIS
# -------------------------

region_counts = defaultdict(set)
image_region_ids = []

for idx, image_path in enumerate(atlas_chunks):
    image_data = np.array(Image.open(image_path))
    regions = set(np.unique(image_data))

    image_region_ids.append(regions)

    for region in regions:
        region_counts[region].add(idx)

print(f"Found {len(region_counts)} unique atlas region IDs.")

# -------------------------
# GREEDY COVERAGE SELECTION
# -------------------------

selected_images = set()
covered_regions = set()

while (
    len(selected_images) < number_of_chunks
    and len(covered_regions) < len(region_counts)
):
    best_image = None
    max_new_regions = 0

    for idx, regions in enumerate(image_region_ids):
        if idx in selected_images:
            continue

        new_regions = regions - covered_regions

        if len(new_regions) > max_new_regions:
            max_new_regions = len(new_regions)
            best_image = idx

    if best_image is None:
        break

    selected_images.add(best_image)
    covered_regions.update(image_region_ids[best_image])

print(f"Coverage-based selected: {len(selected_images)}")

# -------------------------
# RANDOM FILL (IF NEEDED)
# -------------------------

random.seed(12345)

available_images = set(range(len(image_region_ids))) - selected_images

while len(selected_images) < number_of_chunks and available_images:
    idx = random.choice(sorted(available_images))
    selected_images.add(idx)
    available_images.remove(idx)

print(f"Total selected after fill: {len(selected_images)}")

selected_atlas_chunks = [atlas_chunks[idx] for idx in sorted(selected_images)]

# -------------------------
# OUTPUT PATHS
# -------------------------

image_out_path = chunk_dir.parent / "selected_image_chunks"
atlas_out_path = chunk_dir.parent / "selected_atlas_chunks"

image_out_path.mkdir(parents=True, exist_ok=True)
atlas_out_path.mkdir(parents=True, exist_ok=True)

# -------------------------
# COPY MATCHED PAIRS
# -------------------------

copied = 0

for atlas_chunk in selected_atlas_chunks:

    chunk_name = atlas_chunk.stem.split("_atlas")[0]
    chunk_number = atlas_chunk.stem.split("chunk_")[-1]

    corresponding_image_name = f"{chunk_name}_chunk_{chunk_number}.tif"
    image_path = chunk_dir / corresponding_image_name

    if not image_path.exists():
        raise RuntimeError(
            f"Missing corresponding image chunk:\n{image_path}"
        )

    shutil.copy2(image_path, image_out_path / image_path.name)
    shutil.copy2(atlas_chunk, atlas_out_path / atlas_chunk.name)

    copied += 1
    print(f"Copied pair: {image_path.name}")

print(f"\nFinished copying {copied} representative chunk pairs.")

