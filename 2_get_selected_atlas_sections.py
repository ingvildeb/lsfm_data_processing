import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

# USER PARAMETERS
# Give the path of your sample folder
sample_path = Path(r"Z:\LSFM\2025\2025_03\2025_03_28\20250328_10_45_32_NB_CS0303_F_P428_C57_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE\\")

# Give the path to the images for which you want matching atlas plates. 
# NB: Image names must not have been changed from their original naming format.
selected_images_path = Path(r"Z:\Labmembers\Ingvild\Testing_CellPose\selected_images\\")


# MAIN CODE, do not edit

all_images_path = sample_path / "Ex_488_Ch0_stitched"
subset_images = selected_images_path.glob("*.tif")
reg_vol_path = sample_path / "_01_registration" / "ANTs_TransformedImage.nii.gz"

def extract_atlas_plate(reg_volume, image, all_images_path):

    # Calculate total number of images
    all_images = list(all_images_path.glob("*.tif"))
    no_images = len(all_images)

    # Access registered volume and find the relationship between the size of the z axis in atlas vol versus image data
    nifti_img = nib.load(reg_volume)
    data = np.asanyarray(nifti_img.dataobj)
    vol_axis_len = data.shape[1]
    axis_ratio = no_images / vol_axis_len 
  
    # calculate the absolute index of the image slice
    # # the last number in the file name uses a 0-based indexing with an increment of 20
    name = image.name
    number = int(name.split("_")[2])
    image_index = (number / 20)

    # find the corresponding z axis slice in the atlas volume, scaling by the axis ratio to get the right one
    atlas_index = int(np.round(image_index / axis_ratio))
    atlas_slice = data[:, atlas_index, :]

    # make the image into an array and get the width and height for scaling purposes
    image_slice = np.array(Image.open(image))
    target_shape = image_slice.shape[:2]

    # rotate and scale the horizontal slice to the shape of the image slice
    # use no interpolation to ensure no changes in label values
    rotated_atlas_slice = np.rot90(atlas_slice)
    resized_atlas_slice = cv2.resize(rotated_atlas_slice, 
                                          (target_shape[1], target_shape[0]), 
                                          interpolation=cv2.INTER_NEAREST)
    
    resized_atlas_slice_16bit = resized_atlas_slice.astype(np.uint16)

    return name, resized_atlas_slice_16bit


for image in subset_images:
    name, atlas_slice = extract_atlas_plate(reg_vol_path, image, all_images_path)
    image_data = np.array(Image.open(image))
    print(name)
    # plot the slice using matplotlib for visual validation
    plt.imshow(atlas_slice.T, cmap="gray", origin="lower", vmin=np.min(image_data), vmax=np.max(image_data) * 0.1)
    plt.imshow(image_data.T, cmap='gray', origin='lower', vmin=np.min(image_data), vmax=np.max(image_data) * 0.1, alpha=0.5)
    plt.title(f"Horizontal Slice for {image.name}")
    plt.axis('off')
    plt.show()

    # save the atlas slice as a 16 bit tiff image

    cv2.imwrite(rf'{selected_images_path}\{name.split(".")[0]}_atlas_slice.tif', atlas_slice)

    print("Image has been saved successfully.")