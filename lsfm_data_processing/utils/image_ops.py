import sys
from pathlib import Path

import cv2
import numpy as np


def convert_to_uint8(image_array: np.ndarray) -> np.ndarray:
    """Scale an image array to uint8 (0..255) using min-max normalization."""
    arr = np.asarray(image_array)
    if arr.size == 0:
        return arr.astype(np.uint8)

    arr_f = arr.astype(np.float32)
    min_val = float(np.min(arr_f))
    max_val = float(np.max(arr_f))
    if max_val <= min_val:
        return np.zeros(arr.shape, dtype=np.uint8)

    out = cv2.normalize(arr_f, None, 0, 255, cv2.NORM_MINMAX)
    return np.rint(out).astype(np.uint8)


def _raise_if_windows_path_too_long(path: Path, limit: int = 260) -> None:
    """Raise a clear error when a Windows path likely exceeds legacy MAX_PATH."""
    if not sys.platform.startswith("win"):
        return
    p = str(path.resolve())
    if len(p) >= limit:
        raise RuntimeError(
            "Output path is too long for reliable Windows file operations.\n"
            f"Length: {len(p)} (limit ~{limit})\n"
            f"Path:\n{p}\n\n"
            "Use a shorter output/root path or shorter folder/file naming."
        )


def normalize_array(image_array: np.ndarray, min_val=0, max_val=99.5, convert_to_8bit=False):
    """Normalize an image array using given percentile min/max values."""
    input_dtype = image_array.dtype

    lower_threshold = np.percentile(image_array, min_val)
    upper_threshold = np.percentile(image_array, max_val)
    clipped_image = np.clip(image_array, lower_threshold, upper_threshold).astype(np.float32)

    if upper_threshold <= lower_threshold:
        return convert_to_uint8(image_array) if convert_to_8bit else image_array.astype(input_dtype)

    if convert_to_8bit:
        normalized_image = cv2.normalize(clipped_image, None, 0, 255, cv2.NORM_MINMAX)
        return np.rint(normalized_image).astype(np.uint8)

    if np.issubdtype(input_dtype, np.integer):
        dtype_info = np.iinfo(input_dtype)
        normalized_image = cv2.normalize(
            clipped_image,
            None,
            dtype_info.min,
            dtype_info.max,
            cv2.NORM_MINMAX,
        )
        return np.rint(normalized_image).astype(input_dtype)
    if np.issubdtype(input_dtype, np.floating):
        normalized_image = cv2.normalize(clipped_image, None, 0.0, 1.0, cv2.NORM_MINMAX)
        return normalized_image.astype(input_dtype)

    raise TypeError(f"Unsupported image dtype for normalization: {input_dtype}")