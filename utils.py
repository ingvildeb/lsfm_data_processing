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
        cv2.imwrite(str(mip_path), mip_img)