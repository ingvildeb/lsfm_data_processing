from PIL import Image
from pathlib import Path
import cv2
import numpy as np

### Edit list of path (input as many as you want, separated by commas)

list_of_paths = [Path(r"M:\SmartSPIM_Data\2025\2025_04\2025_04_15\20250415_11_06_50_NB_NB058_F_P14_C57_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_200ul_RO_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_04\2025_04_15\20250415_13_49_05_NB_NB062_M_P120_C57_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_200ul_RO_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_04\2025_04_17\20250417_15_51_44_NB_NB059_F_P14_C57_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_120ul_RO_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_04\2025_04_17\20250417_18_38_32_NB_NB060_M_P14_C57_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_100ul_RO_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_04\2025_04_23\20250423_16_45_50_NB_NB063_F_P134_C57_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_200ul_RO_Destripe_DONE"),
                 Path(r"M:\SmartSPIM_Data\2025\2025_04\2025_04_23\20250423_20_00_46_NB_NB065_M_P120_C57_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_150ul_RO_Destripe_DONE")
                 ]

# Set the out path and select which channel to make collage for
out_path = Path(r"Z:\Labmembers\Ingvild\RM1\protocol_testing\202504_MediumBatch_ROsamples")
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
    lower_threshold = np.percentile(image_array, 0)
    upper_threshold = np.percentile(image_array, 99.8)

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
