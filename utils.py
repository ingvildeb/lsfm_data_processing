from PIL import Image
import numpy as np
import tifffile as tiff
import os
from pathlib import Path
import cv2


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

def create_mips_from_folder(input_dir, output_dir, z_step_size, mip_thickness):
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
        first_plane = tiff_files[start_slice].stem.split('_')[2]
        last_plane = tiff_files[end_slice - 1].stem.split('_')[2]

        # Save MIP image with plane identifiers
        mip_filename = f"MIP_{first_plane}_{last_plane}.tif"
        mip_path = output_dir / mip_filename
        print(f"Saving MIP {mip_path}")
        cv2.imwrite(str(mip_path), mip_img)


def normalize_images(in_dir, out_dir, min=0, max=99.5, suffix=".tif"):
    in_path = Path(in_dir)
    image_list = in_path.glob(f"*{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for image in image_list:
        print(f"Normalizing {image} ...")
        image_pil = Image.open(image)

        # Convert image to NumPy array for manipulation
        image_array = np.array(image_pil)

        # Define clipping thresholds
        lower_threshold = np.percentile(image_array, min)
        upper_threshold = np.percentile(image_array, max)

        # Clip the image pixel values
        clipped_image = np.clip(image_array, lower_threshold, upper_threshold)

        # Normalize to range 0-255 and convert to uint8
        normalized_image = cv2.normalize(clipped_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        img_out_path = out_dir / image.name
        print(f"Saving normalized image to {img_out_path}")

        # Save the image
        cv2.imwrite(str(img_out_path), normalized_image)
