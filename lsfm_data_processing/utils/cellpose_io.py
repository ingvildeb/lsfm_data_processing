from pathlib import Path
import random

import cv2
import numpy as np
import tifffile

from .io_helpers import list_tiff_files


def build_prediction_index(prediction_dir: Path, prediction_required_prefix: str) -> dict[str, Path]:
    prediction_files = list_tiff_files(prediction_dir)
    if not prediction_files:
        raise RuntimeError(f"No TIFF prediction files found in:\n{prediction_dir}")

    if prediction_required_prefix:
        prefixed = [p for p in prediction_files if p.stem.startswith(prediction_required_prefix)]
        if prefixed:
            prediction_files = prefixed

    index: dict[str, Path] = {}
    for p in prediction_files:
        key = p.stem
        if prediction_required_prefix and key.startswith(prediction_required_prefix):
            key = key[len(prediction_required_prefix) :]
        if key in index:
            raise RuntimeError(f"Prediction stem collision in:\n{prediction_dir}\nKey: {key}")
        index[key] = p
    return index


def match_prediction_for_mip(
    mip_path: Path,
    prediction_index: dict[str, Path],
    prediction_required_prefix: str,
) -> Path:
    exact = prediction_index.get(mip_path.stem)
    if exact is not None:
        return exact

    candidates = [p for k, p in prediction_index.items() if k in mip_path.stem or mip_path.stem in k]
    if prediction_required_prefix:
        preferred = [p for p in candidates if p.stem.startswith(prediction_required_prefix)]
        if len(preferred) == 1:
            return preferred[0]
        if len(preferred) > 1:
            candidates = preferred

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise RuntimeError(f"No matching prediction TIFF found for MIP:\n{mip_path.name}")

    cand_text = "\n".join([str(c) for c in candidates[:10]])
    raise RuntimeError(f"Ambiguous prediction match for MIP:\n{mip_path.name}\nCandidates:\n{cand_text}")


def load_prediction_masks(prediction_path: Path) -> np.ndarray:
    masks = tifffile.TiffFile(prediction_path).asarray()
    if masks.ndim != 2:
        raise RuntimeError(f"Prediction TIFF must be 2D labels. Found shape {masks.shape} in:\n{prediction_path}")
    if not np.issubdtype(masks.dtype, np.integer):
        masks = masks.astype(np.int32)
    return masks


def create_outlines_from_masks(masks: np.ndarray) -> np.ndarray:
    outlines = np.zeros_like(masks, dtype=np.uint16)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for mask_id in np.unique(masks):
        if mask_id == 0:
            continue
        binary = (masks == mask_id).astype(np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        border = (binary > 0) & (eroded == 0)
        outlines[border] = np.uint16(mask_id)
    return outlines


def create_cellpose_npy_dict(masks: np.ndarray, out_tif_path: Path) -> dict:
    outlines = create_outlines_from_masks(masks)
    n_masks = int(len(np.unique(masks)) - (1 if np.any(masks == 0) else 0))
    rng = random.Random(12345)
    colors = [[rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)] for _ in range(max(n_masks, 0))]

    return {
        "masks": masks,
        "outlines": outlines,
        "colors": colors,
        "filename": str(out_tif_path),
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "normalize_params": {
            "lowhigh": None,
            "percentile": [1.0, 99.0],
            "normalize": True,
            "norm3D": True,
            "sharpen_radius": 0.0,
            "smooth_radius": 0.0,
            "tile_norm_blocksize": 0.0,
            "tile_norm_smooth3D": 0.0,
            "invert": False,
        },
        "restore": None,
        "ratio": 1.0,
        "diameter": None,
    }
