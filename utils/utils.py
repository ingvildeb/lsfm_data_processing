from PIL import Image
import numpy as np
import tifffile
import os
from pathlib import Path
import cv2
import nibabel as nib
import sys


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


def get_avg_pixel_value(path_to_image):
    # Load image and convert to array
    image = tifffile.TiffFile(path_to_image).asarray()
    shape = image.shape

    if len(shape) == 2:
        average_pixel_value = np.mean(image)

    elif len(shape) == 3:
        middle_z = int(shape[0] / 2)
        image = image[middle_z, :, :]
        average_pixel_value = np.mean(image)
  
    return average_pixel_value


def chunk_image(path_to_image, image_outdir, chunk_size):
    # read the image
    img = tifffile.TiffFile(path_to_image).asarray()
    image_name = path_to_image.stem

    # get the shape of the image
    shape = img.shape
    
    # crop the image and save each chunk
    for i in range(0, shape[0], chunk_size):
        for j in range(0, shape[1], chunk_size):
            chunk = img[i:i+chunk_size, j:j+chunk_size]
            tifffile.imsave("{}/{}_chunk_{}_{}.tif".format(image_outdir,image_name,i, j), chunk)

def chunk_z_stack(path_to_image, image_outdir, chunk_size):

    full_stack = tifffile.TiffFile(path_to_image).asarray()
    shape = full_stack.shape
    image_name = path_to_image.stem

    # crop the image and save each chunk
    for i in range(0, shape[1], chunk_size):
        for j in range(0, shape[2], chunk_size):
                stack_chunk = full_stack[:, i:i+chunk_size, j:j+chunk_size]
                tifffile.imsave("{}/{}_chunk_{}_{}.tif".format(image_outdir,image_name,i, j), stack_chunk)


def create_mips_from_folder(
    input_dir,
    output_dir,
    z_step_size,
    mip_thickness,
    underscores_to_plane_z,
    do_normalization=False,
    min_val=0,
    max_val=99.5,
    convert_to_8bit=False,
):
    # Calculate the number of slices to include in each MIP based on the mip_thickness
    slices_per_mip = int(mip_thickness / z_step_size)
    if slices_per_mip < 1:
        raise ValueError("MIP thickness must be at least equal to the z-step size.")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=False)

    # List all TIFF files
    tiff_files = sorted([f for f in input_dir.glob('*.tif*')])
    num_files = len(tiff_files)

    print(f"Found {num_files} images")

    # Process each MIP section
    for start_slice in range(0, num_files, slices_per_mip):
        # Determine the range of slices to include in the current MIP
        end_slice = min(start_slice + slices_per_mip, num_files)
        slices = []

        for i in range(start_slice, end_slice):
            img = cv2.imread(str(tiff_files[i]), cv2.IMREAD_UNCHANGED)
            slices.append(img)
  
        # Compute the maximum intensity projection
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

        # Extract plane identifiers for the first and last images

        first_plane = tiff_files[start_slice].stem.split('_')[underscores_to_plane_z]
        last_plane = tiff_files[end_slice - 1].stem.split('_')[underscores_to_plane_z]

        # Save MIP image with plane identifiers
        mip_filename = f"MIP_{first_plane}_{last_plane}.tif"
        mip_path = output_dir / mip_filename
        _raise_if_windows_path_too_long(mip_path)
        tifffile.imwrite(mip_path, mip_img)


def extract_atlas_plate(reg_volume, image, all_images_path, underscores_to_index, file_number_increment):

    # Calculate total number of images
    all_images = list(all_images_path.glob("*.tif"))
    no_images = len(all_images)

    # Access registered volume and find the relationship between the size of the z axis in atlas vol versus image data
    nifti_img = nib.load(reg_volume)
    data = np.asanyarray(nifti_img.dataobj)
    vol_axis_len = data.shape[1]
    axis_ratio = no_images / vol_axis_len 
  
    # calculate the absolute index of the image slice
    # # the last number in the file name uses a 0-based indexing with an increment of 20
    name = image.stem
    number = int(name.split("_")[underscores_to_index])
    print(f"Number extracted is {number}")
    image_index = (number / file_number_increment)
    print(f"Image index is {image_index}")

    # find the corresponding z axis slice in the atlas volume, scaling by the axis ratio to get the right one
    atlas_index = int(np.round(image_index / axis_ratio))
    print(f"Atlas index is {atlas_index}")
    atlas_slice = data[:, atlas_index, :]
    

    # make the image into an array and get the width and height for scaling purposes
    image_slice = np.array(Image.open(image))
    target_shape = image_slice.shape[:2]

    # rotate and scale the horizontal slice to the shape of the image slice
    # use no interpolation to ensure no changes in label values
    rotated_atlas_slice = np.rot90(atlas_slice)
    resized_atlas_slice = cv2.resize(rotated_atlas_slice, 
                                          (target_shape[1], target_shape[0]), 
                                          interpolation=cv2.INTER_NEAREST)
    
    resized_atlas_slice_16bit = resized_atlas_slice.astype(np.uint16)

    return name, resized_atlas_slice_16bit

def relabel_sequential_for_preview(label_slice):
    """
    Remap unique labels to sequential integers (1..N).
    Background (0) stays 0.
    """
    label_slice = label_slice.copy()

    unique_labels = np.unique(label_slice)
    unique_labels = unique_labels[unique_labels != 0]  # keep background as 0

    relabeled = np.zeros_like(label_slice, dtype=np.int32)

    for new_id, old_id in enumerate(unique_labels, start=1):
        relabeled[label_slice == old_id] = new_id

    return relabeled

def tifs_to_zstack(file_list, out_dir, out_prefix):

    images = []
    names = []
    for file in file_list:
        img = Image.open(file)
        images.append(np.array(img))
        name = file.stem
        names.append(name)

    zstack_array = np.stack(images)
    output_filename = out_dir / f"{out_prefix}_zstack_{names[0]}_to_{names[-1]}.tif"

    tifffile.imwrite(output_filename, zstack_array, photometric='minisblack')
