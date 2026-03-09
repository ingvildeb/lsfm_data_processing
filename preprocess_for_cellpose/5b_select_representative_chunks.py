"""
Select representative chunk pairs (image + atlas) to maximize region coverage.

Requires matching atlas chunks; otherwise use 5a_select_random_chunks.py.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import shutil
import random
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from lsfm_data_processing.utils.io_helpers import load_script_config, normalize_user_path, require_dir
from lsfm_data_processing.utils.selection import greedy_region_coverage_select, random_fill_selection

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

image_region_ids = []

for idx, image_path in enumerate(atlas_chunks):
    image_data = np.array(Image.open(image_path))
    regions = set(np.unique(image_data))

    image_region_ids.append(regions)

all_regions = set().union(*image_region_ids) if image_region_ids else set()
print(f"Found {len(all_regions)} unique atlas region IDs.")

# -------------------------
# GREEDY COVERAGE SELECTION
# -------------------------

regions_by_id = {idx: regions for idx, regions in enumerate(image_region_ids)}
selected_images, covered_regions = greedy_region_coverage_select(
    candidate_ids=list(range(len(image_region_ids))),
    regions_by_id=regions_by_id,
    limit=min(number_of_chunks, len(image_region_ids)),
)

print(f"Coverage-based selected: {len(selected_images)}")

# -------------------------
# RANDOM FILL (IF NEEDED)
# -------------------------

rng = random.Random(12345)
selected_images = random_fill_selection(
    selected=selected_images,
    all_ids=list(range(len(image_region_ids))),
    target_total=min(number_of_chunks, len(image_region_ids)),
    rng=rng,
)

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

