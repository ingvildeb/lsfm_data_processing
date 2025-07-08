import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from utils import normalize_image, normalize_and_save




# Example usage
input_image_path = r"M:\SmartSPIM_Data\2025\2025_04\2025_04_24\20250424_11_45_45_IEB_IEB0031_F_P14_Aldh1_LAS_488GFP_561Bg_640Sytox_4x_5umstep_Destripe_DONE\Ex_488_Ch0_stitched\273480_467010_013840_Ch0.tif"  # Replace with the path to your image
output_directory = r"Z:\Labmembers\Ingvild\Cellpose\Aldh_model"  # Directory to save images
min_max_ranges = [(0, 99.9), (1, 99.9), (2, 99.9), (3, 99.9)]  # Different min/max parameter pairs

normalize_and_save(input_image_path, output_directory, min_max_ranges)
