from pathlib import Path

import cv2
import numpy as np
import tifffile

from .image_ops import _raise_if_windows_path_too_long, convert_to_uint8, normalize_array
from .naming import get_underscore_token


def create_mips_from_folder(
    input_dir: Path,
    output_dir: Path,
    z_step_size: float,
    mip_thickness: float,
    underscores_to_plane_z: int,
    do_normalization: bool = False,
    min_val: float = 0,
    max_val: float = 99.5,
    convert_to_8bit: bool = False,
    use_lzw_compression: bool = True,
) -> None:
    """Create max-intensity projections from sequential TIFF slices in a folder."""
    slices_per_mip = int(mip_thickness / z_step_size)
    if slices_per_mip < 1:
        raise ValueError("MIP thickness must be at least equal to the z-step size.")

    output_dir.mkdir(parents=True, exist_ok=False)

    tiff_files = sorted([f for f in input_dir.glob("*.tif*")])
    num_files = len(tiff_files)

    print(f"Found {num_files} images")

    for start_slice in range(0, num_files, slices_per_mip):
        end_slice = min(start_slice + slices_per_mip, num_files)
        slices = []

        for i in range(start_slice, end_slice):
            img = cv2.imread(str(tiff_files[i]), cv2.IMREAD_UNCHANGED)
            slices.append(img)

        mip_img = np.max(np.stack(slices, axis=0), axis=0)
        if do_normalization:
            mip_img = normalize_array(
                mip_img,
                min_val=min_val,
                max_val=max_val,
                convert_to_8bit=convert_to_8bit,
            )
        elif convert_to_8bit:
            mip_img = convert_to_uint8(mip_img)

        first_plane = get_underscore_token(
            tiff_files[start_slice].stem,
            underscores_to_plane_z,
            "first plane z",
        )
        last_plane = get_underscore_token(
            tiff_files[end_slice - 1].stem,
            underscores_to_plane_z,
            "last plane z",
        )

        mip_filename = f"MIP_{first_plane}_{last_plane}.tif"
        mip_path = output_dir / mip_filename
        _raise_if_windows_path_too_long(mip_path)
        tifffile.imwrite(
            mip_path,
            mip_img,
            compression="lzw" if use_lzw_compression else None,
        )
