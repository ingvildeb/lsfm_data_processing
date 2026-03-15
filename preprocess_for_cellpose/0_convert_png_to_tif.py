"""
Convert RGB PNG images to TIFF images for Cellpose preprocessing.

This is intended as a compatibility step for datasets that arrive as brightfield
PNG images. Either the RGB image can be preserved as-is, a single RGB channel
can be extracted, or the image can be converted to grayscale, then saved as a
TIFF so the rest of the Cellpose preprocessing pipeline can keep operating on
2D TIFFs.
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
        if mode == "format_only":
            return np.asarray(img.convert("RGB"))

        if mode == "grayscale":
            return np.asarray(img.convert("L"))

        rgb = img.convert("RGB")
        arr = np.asarray(rgb)

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise RuntimeError(f"Expected RGB image with 3 channels:\n{image_path}")

    return arr[:, :, channel]


def write_tiff(out_path: Path, image_array: np.ndarray) -> None:
    if image_array.ndim == 3 and image_array.shape[2] in {3, 4}:
        tifffile.imwrite(out_path, image_array, photometric="rgb")
        return

    tifffile.imwrite(out_path, image_array)


def invert_image(image_array: np.ndarray) -> np.ndarray:
    if np.issubdtype(image_array.dtype, np.integer):
        max_value = np.iinfo(image_array.dtype).max
        return max_value - image_array

    if np.issubdtype(image_array.dtype, np.floating):
        return 1.0 - image_array

    raise TypeError(f"Unsupported dtype for inversion: {image_array.dtype}")


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
invert_output = cfg.get("invert_output", False)
recursive = cfg.get("recursive", False)
overwrite = cfg.get("overwrite", False)

if conversion_mode == "conversion_only":
    conversion_mode = "format_only"

channel_names = {0: "red", 1: "green", 2: "blue"}
if channel not in channel_names:
    raise RuntimeError("channel must be 0, 1, or 2.")
if conversion_mode not in {"format_only", "single_channel", "grayscale"}:
    raise RuntimeError(
        "conversion_mode must be 'format_only', 'conversion_only', "
        "'single_channel', or 'grayscale'."
    )

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
if conversion_mode == "format_only":
    print("Converting PNG images to RGB TIFF without changing channels.")
elif conversion_mode == "grayscale":
    print("Converting PNG images to grayscale TIFF.")
else:
    print(f"Extracting {channel_names[channel]} channel to TIFF.")
if invert_output:
    print("Inverting output intensities.")

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
    if invert_output:
        converted_image = invert_image(converted_image)
    write_tiff(out_path, converted_image)
    converted += 1
    print(f"Converted: {png_path.name} -> {out_path.name} shape={converted_image.shape}")

print("Conversion complete.")
print(f"TIFF files written: {converted}")
print(f"Files skipped: {skipped}")
