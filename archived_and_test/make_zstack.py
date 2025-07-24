import os
import numpy as np
from PIL import Image
import tifffile

def compile_tiff_to_zstack(directory, output_filename="zstack.tif"):
    """
    Compiles a sequence of TIFF files in a directory into a single multi-page TIFF (Z-stack).

    Args:
        directory (str): Path to the directory containing the TIFF files.
        output_filename (str, optional): Name of the output TIFF file. Defaults to "zstack.tif".
    """
    file_list = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.tif', '.tiff'))])
    file_list = file_list[266:280]
    print(len(file_list))
    if not file_list:
        raise ValueError("No TIFF files found in the specified directory.")

    images = []
    for filename in file_list:
        file_path = os.path.join(directory, filename)
        try:
            img = Image.open(file_path)
            images.append(np.array(img))
        except Exception as e:
             raise Exception(f"Error opening or processing {filename}: {e}") from e

    zstack_array = np.stack(images)

    try:
         tifffile.imwrite(output_filename, zstack_array, photometric='minisblack')
    except Exception as e:
        raise Exception(f"Error writing zstack to {output_filename}: {e}") from e

# Example usage:
directory_path = r"Z:\Labmembers\Ingvild\Testing_CellPose\test_3d\Ex_488_Ch0_stitched\\"
compile_tiff_to_zstack(directory_path)

data = tifffile.imread(r'Z:\Labmembers\Ingvild\GitHub\train\zstack_YX_5.tif')
print(data.shape)




def crop_z_stacks(file_path, crop_shape, output_dir):
    # Read the tiff file
    with tifffile.TiffFile(file_path) as tif:
        full_stack = tif.asarray()
    
    z_dim, y_dim, x_dim = full_stack.shape
    
    crop_height, crop_width = crop_shape
    
    assert crop_height <= y_dim and crop_width <= x_dim, "Crop size exceeds image dimensions"

    num_crops_y = y_dim // crop_height
    num_crops_x = x_dim // crop_width

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_crops_y):
        for j in range(num_crops_x):
            cropped_stack = full_stack[
                :,  # Keep all z-slices
                i * crop_height:(i + 1) * crop_height,
                j * crop_width:(j + 1) * crop_width
            ]
            
            cropped_filename = os.path.join(output_dir, f"crop_y{i}_x{j}.tiff")
            
            # Ensure to use the write function to account for 3D arrays in TIFF
            tifffile.imwrite(cropped_filename, cropped_stack, imagej=True)

# Example Usage
file_path = r'Z:\Labmembers\Ingvild\GitHub\zstack.tif'
crop_shape = (1000, 1000)  # Example crop size
output_dir = r'Z:\Labmembers\Ingvild\GitHub\training\\'
crop_z_stacks(file_path, crop_shape, output_dir)