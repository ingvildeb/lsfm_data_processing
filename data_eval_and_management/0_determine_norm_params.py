import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from utils import normalize_image, normalize_and_save




# User parameters
input_image_path = Path(r"Z:\Labmembers\Ingvild\Cellpose\Iba1_model\test_norm_params")  # Replace with the path to your image
output_directory = Path(r"Z:\Labmembers\Ingvild\Cellpose\Iba1_model\test_norm_params\\")  # Directory to save images
min_max_ranges = [(0,99.999),(0, 99.99), (0,99.9), (0,99.7)]  # Different min/max parameter pairs

images = input_image_path.glob("*.tif")

# Run code
for image in images:
    normalize_and_save(image, output_directory, min_max_ranges)
