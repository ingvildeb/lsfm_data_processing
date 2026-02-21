from pathlib import Path
import tifffile
from utils.utils import normalize_array




# User parameters
input_image_path = Path(r"Z:\Labmembers\Ingvild\Cellpose\Iba1_model\test_norm_params")  # Replace with the path to your image
output_directory = Path(r"Z:\Labmembers\Ingvild\Cellpose\Iba1_model\test_norm_params\\")  # Directory to save images
min_max_ranges = [(0,99.999),(0, 99.99), (0,99.9), (0,99.7)]  # Different min/max parameter pairs

images = input_image_path.glob("*.tif")

# Run code
for image in images:
    for min_val, max_val in min_max_ranges:
        print(f"Normalizing {image.name} with min={min_val}, max={max_val}")
        image_array = tifffile.TiffFile(image).asarray()
        normalized_image = normalize_array(image_array, min_val=min_val, max_val=max_val)
        out_name = f"{image.stem}_norm_min{min_val}_max{max_val}{image.suffix}"
        tifffile.imwrite(output_directory / out_name, normalized_image)
