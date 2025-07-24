import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from utils import extract_atlas_plate

# USER PARAMETERS
# Give the path of your sample folder
sample_path = Path(r"Z:\LSFM\2025\2025_03\2025_03_28\20250328_10_45_32_NB_CS0303_F_P428_C57_LAS_488Lectin_561NeuN_640Iba1_4x_4umstep_Destripe_DONE\\")

# Give the path to the images for which you want matching atlas plates. 
# NB: Image names must not have been changed from their original naming format.
selected_images_path = Path(r"Z:\Labmembers\Ingvild\Testing_CellPose\selected_images\MIPS\\")


# MAIN CODE, do not edit

all_images_path = sample_path / "Ex_488_Ch0_stitched"
subset_images = selected_images_path.glob("*.tif")
reg_vol_path = sample_path / "_01_registration" / "ANTs_TransformedImage.nii.gz"



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