from PIL import Image
import numpy as np
import tifffile as tiff
import os
from pathlib import Path
import cv2
import nibabel as nib


def get_avg_pixel_value(file):
    # Load image and convert to array
    image = Image.open(file)
    pixels = np.array(image)
    # Calculate the average pixel value
    average_pixel_value = np.mean(image)
    return average_pixel_value

def chunk_image(path_to_image, image_outdir, chunk_size=512):
    # read the image
    img = tiff.imread(path_to_image)
    image_name = (os.path.basename(path_to_image)).split(".")[0]

    # get the shape of the image
    shape = img.shape

    # crop the image and save each chunk
    for i in range(0, shape[0], chunk_size):
        for j in range(0, shape[1], chunk_size):
            chunk = img[i:i+chunk_size, j:j+chunk_size]
            tiff.imsave("{}/{}_chunk_{}_{}.tif".format(image_outdir,image_name,i, j), chunk)

def create_mips_from_folder(input_dir, output_dir, z_step_size, mip_thickness, underscores_to_plane_z):
    # Calculate the number of slices to include in each MIP based on the mip_thickness
    slices_per_mip = int(mip_thickness / z_step_size)
    if slices_per_mip < 1:
        raise ValueError("MIP thickness must be at least equal to the z-step size.")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # List all TIFF files
    tiff_files = sorted([f for f in input_dir.glob('*.tif')])
    num_files = len(tiff_files)

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

        # Extract plane identifiers for the first and last images

        first_plane = tiff_files[start_slice].stem.split('_')[underscores_to_plane_z]
        last_plane = tiff_files[end_slice - 1].stem.split('_')[underscores_to_plane_z]

        # Save MIP image with plane identifiers
        mip_filename = f"MIP_{first_plane}_{last_plane}.tif"
        mip_path = output_dir / mip_filename
        print(f"Saving MIP {mip_path}")
        cv2.imwrite(str(mip_path), mip_img)


def normalize_image(image_path, min_val=0, max_val=99.5):
    """Normalize a single image using given min/max values."""
    image_pil = Image.open(image_path)

    # Convert image to NumPy array for manipulation
    image_array = np.array(image_pil)

    # Define clipping thresholds
    lower_threshold = np.percentile(image_array, min_val)
    upper_threshold = np.percentile(image_array, max_val)

    # Clip the image pixel values
    clipped_image = np.clip(image_array, lower_threshold, upper_threshold)

    # Normalize to range 0-255 and convert to uint8
    normalized_image = cv2.normalize(clipped_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return normalized_image


def normalize_and_save(input_image_path, output_dir, min_max_params):
    """Normalize the image with various min/max parameters and save each normalized image with the specified naming convention."""
    # Create a directory for output images
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for min_val, max_val in min_max_params:
        print(f"Normalizing with min: {min_val}, max: {max_val}")

        # Normalize the image using the given min and max
        normalized_image_array = normalize_image(input_image_path, min_val, max_val)
        
        # Create a filename based on the original file name and normalization parameters
        original_name = Path(input_image_path).stem  # Get the original file name without extension
        original_extension = Path(input_image_path).suffix  # Get the original file extension
        normalized_filename = f"{original_name}_norm_min{min_val}_max{max_val}{original_extension}"
        
        # Save the normalized image
        normalized_image_path = output_dir / normalized_filename
        cv2.imwrite(str(normalized_image_path), normalized_image_array)
        print(f"Saved normalized image to {normalized_image_path}")


def extract_atlas_plate(reg_volume, image, all_images_path, underscores_to_index):

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
    image_index = (number / 20)

    # find the corresponding z axis slice in the atlas volume, scaling by the axis ratio to get the right one
    atlas_index = int(np.round(image_index / axis_ratio))
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