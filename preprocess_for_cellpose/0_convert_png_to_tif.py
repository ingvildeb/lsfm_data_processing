"""
Convert RGB PNG images to TIFF images for Cellpose preprocessing.

This is intended as a compatibility step for datasets that arrive as brightfield
PNG images. Either a single RGB channel can be extracted or the image can be
converted to grayscale, then saved as a TIFF so the rest of the Cellpose
preprocessing pipeline can keep operating on 2D TIFFs.
"""

from pathlib import Path
import sys

import numpy as np
import tifffile
from PIL import Image

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from lsfm_data_processing.utils.io_helpers import load_script_config, normalize_user_path, require_dir


def find_png_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.png" if recursive else "*.png"
    files = sorted(input_dir.glob(pattern))
    return [p for p in files if p.is_file()]


def convert_image(image_path: Path, mode: str, channel: int) -> np.ndarray:
    with Image.open(image_path) as img:
        if mode == "grayscale":
            return np.asarray(img.convert("L"))

        rgb = img.convert("RGB")
        arr = np.asarray(rgb)

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise RuntimeError(f"Expected RGB image with 3 channels:\n{image_path}")

    return arr[:, :, channel]


def build_output_path(input_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative_parent = input_path.parent.relative_to(input_dir)
    return output_dir / relative_parent / input_path.with_suffix(".tif").name


# -------------------------
# CONFIG LOADING
# -------------------------
test_mode = False
cfg = load_script_config(
    Path(__file__),
    "0_convert_png_to_tif",
    test_mode=test_mode,
)

# -------------------------
# CONFIG PARAMETERS
# -------------------------
input_dir = require_dir(
    normalize_user_path(cfg["input_dir"]),
    "Input PNG folder",
)

output_dir = normalize_user_path(cfg["output_dir"])
channel = cfg["channel"]
conversion_mode = cfg.get("conversion_mode", "single_channel")
recursive = cfg.get("recursive", False)
overwrite = cfg.get("overwrite", False)

channel_names = {0: "red", 1: "green", 2: "blue"}
if channel not in channel_names:
    raise RuntimeError("channel must be 0, 1, or 2.")
if conversion_mode not in {"single_channel", "grayscale"}:
    raise RuntimeError("conversion_mode must be 'single_channel' or 'grayscale'.")

# -------------------------
# MAIN CODE
# -------------------------
png_files = find_png_files(input_dir, recursive=recursive)

if not png_files:
    raise RuntimeError(f"No PNG files found in:\n{input_dir}")

output_dir.mkdir(parents=True, exist_ok=True)

converted = 0
skipped = 0

print(f"Found {len(png_files)} PNG files.")
if conversion_mode == "grayscale":
    print("Converting PNG images to grayscale TIFF.")
else:
    print(f"Extracting {channel_names[channel]} channel to TIFF.")

for png_path in png_files:
    out_path = build_output_path(png_path, input_dir, output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        print(f"Skipping existing file: {out_path}")
        skipped += 1
        continue

    converted_image = convert_image(
        png_path,
        mode=conversion_mode,
        channel=channel,
    )
    tifffile.imwrite(out_path, converted_image)
    converted += 1
    print(f"Converted: {png_path.name} -> {out_path.name}")

print("Conversion complete.")
print(f"TIFF files written: {converted}")
print(f"Files skipped: {skipped}")
