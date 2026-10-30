"""
Optionally resize, flip, and normalize all TIFF images in a folder.

This is a lightweight one-off helper for cases where you want resizing,
flipping, and/or percentile normalization without MIP generation or
sample-folder parsing.
"""

from pathlib import Path
import logging
import sys

import cv2
import tifffile

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from lsfm_data_processing.utils.image_ops import _raise_if_windows_path_too_long, normalize_array
from lsfm_data_processing.utils.io_helpers import (
    list_tiff_files,
    load_script_config,
    normalize_user_path,
    require_dir,
)

tifffile_logger = logging.getLogger("tifffile")
tifffile_logger.setLevel(logging.ERROR)

# -------------------------
# CONFIG LOADING
# -------------------------
test_mode = False
cfg = load_script_config(
    Path(__file__),
    "1b_normalize_folder",
    test_mode=test_mode,
)

# -------------------------
# CONFIG PARAMETERS
# -------------------------

input_dir = require_dir(
    normalize_user_path(cfg["input_dir"]),
    "Input image folder",
)

do_normalization = cfg.get("do_normalization", True)
min_val = cfg.get("min_val")
max_val = cfg.get("max_val")
resize_scale_factor = cfg.get("resize_scale_factor", 1.0)
flip_horizontal = cfg.get("flip_horizontal", False)
flip_vertical = cfg.get("flip_vertical", False)
convert_to_8bit = cfg.get("convert_to_8bit", True)
use_lzw_compression = cfg.get("use_lzw_compression", True)
output_format = cfg.get("output_format", "tif").lower()

if resize_scale_factor <= 0:
    raise RuntimeError("resize_scale_factor must be > 0")
if do_normalization and (min_val is None or max_val is None):
    raise RuntimeError("min_val and max_val are required when do_normalization = true")
if output_format not in {"tif", "tiff", "png"}:
    raise RuntimeError('output_format must be one of: "tif", "tiff", "png"')

# -------------------------
# OUTPUT SETUP
# -------------------------

resize_suffix = ""
if resize_scale_factor != 1.0:
    resize_suffix = f"_resize{resize_scale_factor}"

flip_suffix = ""
if flip_horizontal:
    flip_suffix += "_flip_horizontal"
if flip_vertical:
    flip_suffix += "_flip_vertical"

configured_output_dir = cfg.get("output_dir")
if configured_output_dir:
    output_dir = normalize_user_path(configured_output_dir)
else:
    normalization_suffix = f"_norm_min{min_val}_max{max_val}" if do_normalization else ""
    output_dir = input_dir.parent / (
        f"{input_dir.name}{resize_suffix}{flip_suffix}{normalization_suffix}"
    )
output_dir.mkdir(parents=True, exist_ok=True)

images = list_tiff_files(input_dir)
if not images:
    raise RuntimeError(f"No TIFF files found in:\n{input_dir}")

if sys.platform.startswith("win"):
    for image in images:
        _raise_if_windows_path_too_long(output_dir / image.with_suffix(f".{output_format}").name)

params_dict = {
    "input_dir": str(input_dir),
    "output_dir": str(output_dir),
    "do_normalization": do_normalization,
    "min_val": min_val,
    "max_val": max_val,
    "resize_scale_factor": resize_scale_factor,
    "flip_horizontal": flip_horizontal,
    "flip_vertical": flip_vertical,
    "convert_to_8bit": convert_to_8bit,
    "use_lzw_compression": use_lzw_compression,
    "output_format": output_format,
}


def resize_image(image_array, scale_factor: float):
    """Resize a 2D image or channel-last image by a uniform scale factor."""
    if scale_factor == 1.0:
        return image_array

    if image_array.ndim not in {2, 3}:
        raise RuntimeError(
            "resize_scale_factor currently supports 2D images or channel-last images.\n"
            f"Found shape {image_array.shape}"
        )

    height, width = image_array.shape[:2]
    new_width = max(1, int(round(width * scale_factor)))
    new_height = max(1, int(round(height * scale_factor)))

    return cv2.resize(
        image_array,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_LINEAR,
    )


def save_image(output_path: Path, image_array):
    """Save an image as TIFF or PNG."""
    if output_format in {"tif", "tiff"}:
        tifffile.imwrite(
            output_path,
            image_array,
            compression="lzw" if use_lzw_compression else None,
        )
        return

    if image_array.ndim == 3 and image_array.shape[-1] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    elif image_array.ndim == 3 and image_array.shape[-1] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGRA)
    elif image_array.ndim != 2:
        raise RuntimeError(
            "PNG output supports 2D grayscale images or channel-last RGB/RGBA images.\n"
            f"Found shape {image_array.shape}"
        )

    if image_array.dtype.name not in {"uint8", "uint16"}:
        raise RuntimeError(
            "PNG output supports uint8 or uint16 arrays.\n"
            f"Found dtype {image_array.dtype}. Set convert_to_8bit = true for 8-bit PNG output."
        )

    if not cv2.imwrite(str(output_path), image_array):
        raise RuntimeError(f"Failed to write PNG image:\n{output_path}")

# -------------------------
# MAIN
# -------------------------

print(f"Found {len(images)} TIFF images in:\n{input_dir}")
print(f"Writing processed images to:\n{output_dir}")
print(f"resize_scale_factor = {resize_scale_factor}")
print(f"flip_horizontal = {flip_horizontal}")
print(f"flip_vertical = {flip_vertical}")
print(f"do_normalization = {do_normalization}")
print(f"output_format = {output_format}")

for image in images:
    output_path = output_dir / image.with_suffix(f".{output_format}").name
    print(f"Processing {image.name} -> {output_path.name} ...")
    image_array = tifffile.imread(image)
    image_array = resize_image(image_array, resize_scale_factor)
    if flip_horizontal:
        image_array = cv2.flip(image_array, 1)
    if flip_vertical:
        image_array = cv2.flip(image_array, 0)
    if do_normalization:
        image_array = normalize_array(
            image_array,
            min_val=min_val,
            max_val=max_val,
            convert_to_8bit=convert_to_8bit,
        )
    save_image(output_path, image_array)

params_file = output_dir / "parameters.txt"
if sys.platform.startswith("win"):
    _raise_if_windows_path_too_long(params_file)
with open(params_file, "w", encoding="utf-8") as file:
    file.write(str(params_dict))

print("Processing complete.")
