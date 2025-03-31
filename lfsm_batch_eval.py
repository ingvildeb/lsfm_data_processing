from PIL import Image
from pathlib import Path
import cv2
import numpy as np

### Edit list of path (input as many as you want, separated by commas)

list_of_paths = [Path(r"M:\SmartSPIM_Data\2025\2025_03\2025_03_20\20250320_17_02_13_NB_CS0290_M_P533_C57_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_03\2025_03_20\20250320_19_47_13_NB_CS0291_M_P533_C57_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_03\2025_03_24\20250324_11_36_15_IEB_CS293_M_P533_C57_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_03\2025_03_24\20250324_14_34_33_IEB_CS0292_M_P533_C57_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_03\2025_03_25\20250325_10_58_28_NB_CS0294_F_P417_Tg_SwDI_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_03\2025_03_25\20250325_14_16_38_NB_CS0295_F_P417_Tg_SwDI_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_03\2025_03_26\20250326_10_56_52_NB_CS0296_F_P415_C57_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_03\2025_03_26\20250326_14_19_35_NB_CS0300_F_P341_Tg_SwDI_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_03\2025_03_27\20250327_15_50_47_NB_CS0301_F_P341_Tg_SwDI_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_03\2025_03_27\20250327_19_07_55_NB_CS0302_F_P428_C57_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_03\2025_03_28\20250328_10_45_32_NB_CS0303_F_P428_C57_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE")]

# Set the out path and select which channel to make collage for
out_path = Path(r"Z:\Labmembers\Ingvild\RM1\Protocol_testing\202503_LargeBatch_AgingCCFBrains")
channel = 2



##### Main code, do not edit

images_to_collage = []
text_overlays = []

max_width = 0
max_height = 0

channel_wavelengths = {0:"488", 1:"561", 2:"640"}

# Load images using PIL
for path in list_of_paths:
    # Extract sample info
    folder_name = path.name
    sample_id, sample_age, sample_geno = folder_name.split("_")[5], folder_name.split("_")[7], folder_name.split("_")[8]

    channel_image_path = path / f"Ex_{channel_wavelengths.get(channel)}_Ch{channel}_stitched"
    images = sorted(list(channel_image_path.glob("*.tif")))

    middle_index = int(len(images) / 2)
    selected_image = images[middle_index]

    image_pil = Image.open(selected_image)

    # Convert image to NumPy array for manipulation
    image_array = np.array(image_pil)

    # Define clipping thresholds
    lower_threshold = np.percentile(image_array, 1)
    upper_threshold = np.percentile(image_array, 99)

    # Clip the image pixel values
    clipped_image = np.clip(image_array, lower_threshold, upper_threshold)

    # Normalize the image to the full 0-255 range after clipping
    normalized_image = cv2.normalize(clipped_image, None, 0, 255, cv2.NORM_MINMAX)

    # Update maximum dimensions
    current_height, current_width = normalized_image.shape
    max_width = max(max_width, current_width)
    max_height = max(max_height, current_height)

    images_to_collage.append(normalized_image)

    # Prepare text overlay
    text_overlay = f"ID: {sample_id}, Age: {sample_age}, Genotype: {sample_geno}"
    text_overlays.append(text_overlay)

# Pad images to the max dimensions while keeping aspect ratios
padded_images = []
for image in images_to_collage:
    height, width = image.shape
    # Calculate padding
    pad_width = (max_width - width) // 2
    pad_height = (max_height - height) // 2

    # Pad image
    padded_image = cv2.copyMakeBorder(
        image, pad_height, max_height - height - pad_height,
        pad_width, max_width - width - pad_width,
        cv2.BORDER_CONSTANT, value=0)

    padded_images.append(padded_image)

# Determine grid size based on number of images
num_images = len(padded_images)
cols = int(np.ceil(np.sqrt(num_images)))
rows = int(np.ceil(num_images / cols))

collage_width = cols * max_width
collage_height = rows * max_height

# Create a blank collage using OpenCV
collage = np.zeros((collage_height, collage_width), dtype=np.uint8)

# Paste and overlay text on images using OpenCV
font_scale = 10  # Adjust this scale for desired font size
font_thickness = 15
font = cv2.FONT_HERSHEY_SIMPLEX

# Additional space for text placement
text_offset_y = 250  # Adjust this based on font size and desired spacing

for idx, (image, text) in enumerate(zip(padded_images, text_overlays)):
    x_offset = (idx % cols) * max_width
    y_offset = (idx // cols) * max_height
    collage[y_offset:y_offset + max_height, x_offset:x_offset + max_width] = image

    # Overlay text at a safe distance from the top of each image
    cv2.putText(collage, text, (x_offset + 10, y_offset + text_offset_y), font, font_scale, (255), font_thickness)

collage_pil = Image.fromarray(collage)

collage_pil.save(rf'{out_path}\\collage_ch{channel}.png')
