from PIL import Image
import numpy as np
import tifffile as tiff
import os

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