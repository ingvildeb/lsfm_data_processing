import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import numpy as np
import cv2

# Load the NIfTI file
base_path = r"Z:\Labmembers\Ingvild\Testing_CellPose\test_data\\"

all_images_path = glob(f"{base_path}Ex_488_Ch0_stitched\\*.tif")
no_images = len(all_images_path)

image_path = rf"{base_path}Ex_488_Ch0_stitched_selected_data\\"
subset_images = glob(f"{image_path}*.tif")

# Access image data and find the relationship between the size of the z axis in atlas vol versus image data
file_path = rf'{base_path}ANTs_TransformedImage.nii.gz'
nifti_img = nib.load(file_path)
data = np.asanyarray(nifti_img.dataobj)
vol_axis_len = data.shape[1]
axis_ratio = no_images / vol_axis_len 

for image in subset_images:
    name = os.path.basename(image)

    # calculate the absolute index of the image slice
    # # the last number in the file name uses a 0-based indexing with an increment of 20
    number = int(name.split("_")[2])
    image_index = (number / 20)

    # find the corresponding z axis slice in the atlas volume, scaling by the axis ratio to get the right one
    atlas_index = int(np.round(image_index / axis_ratio))
    horizontal_slice = data[:, atlas_index, :]

    # make the image into an array and get the width and height for scaling purposes
    image_slice = np.array(Image.open(image))
    target_shape = image_slice.shape[:2]

    # rotate and scale the horizontal slice to the shape of the image slice
    # use no interpolation to ensure no changes in label values
    rotated_horizontal_slice = np.rot90(horizontal_slice)
    resized_horizontal_slice = cv2.resize(rotated_horizontal_slice, 
                                          (target_shape[1], target_shape[0]), 
                                          interpolation=cv2.INTER_NEAREST)

    # plot the slice using matplotlib for visual validation
    plt.imshow(resized_horizontal_slice.T, cmap="gray", origin="lower", vmin=np.min(image_slice), vmax=np.max(image_slice) * 0.1)
    plt.imshow(image_slice.T, cmap='gray', origin='lower', vmin=np.min(image_slice), vmax=np.max(image_slice) * 0.1, alpha=0.5)
    plt.title(f"Horizontal Slice at Index {atlas_index}")
    plt.axis('off')
    plt.show()

    # save the atlas slice as a 16 bit tiff image
    resized_horizontal_slice_16bit = resized_horizontal_slice.astype(np.uint16)
    cv2.imwrite(f'{name.split(".")[0]}_atlas_slice.tiff', resized_horizontal_slice_16bit)

    print("Image has been saved successfully.")