import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import os
from pathlib import Path
from utils import get_avg_pixel_value


data_path = r"Z:\Labmembers\Ingvild\Testing_CellPose\test_data\Ex_488_Ch0_stitched_selected_data\chunked_images\\"
image_chunk_paths = glob(f"{data_path}*Ch0\\")
atlas_chunk_paths = glob(f"{data_path}*atlas_slice\\")

image_out_path, atlas_out_path = f"{data_path}selected_image_chunks\\", f"{data_path}selected_atlas_chunks\\"

if not os.path.exists(image_out_path):
    os.makedirs(image_out_path)

if not os.path.exists(atlas_out_path):
    os.makedirs(atlas_out_path)

for path in image_chunk_paths:
    image_chunks = glob(f"{path}*.tif")

    for chunk in image_chunks:
        chunk_name = (os.path.basename(chunk)).split("_chunk")[0]
        chunk_number = ((os.path.basename(chunk)).split(".")[0]).split("chunk_")[-1]
        average_pixel_value = get_avg_pixel_value(chunk)
        atlas_chunk = f"{Path(chunk).parent}_atlas_slice\\{chunk_name}_atlas_slice_chunk_{chunk_number}.tif"

        if average_pixel_value > 40:
            chunk_img = np.array(Image.open(chunk))
            plt.imshow(chunk_img)
            plt.axis('off')
            plt.show()

            shutil.copy2(chunk, image_out_path)
            shutil.copy2(atlas_chunk, atlas_out_path)
    break