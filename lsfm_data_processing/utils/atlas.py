import cv2
import nibabel as nib
import numpy as np
from PIL import Image


def extract_atlas_plate(reg_volume, image, all_images_path, underscores_to_index, file_number_increment):
    all_images = list(all_images_path.glob("*.tif"))
    no_images = len(all_images)

    nifti_img = nib.load(reg_volume)
    data = np.asanyarray(nifti_img.dataobj)
    vol_axis_len = data.shape[1]
    axis_ratio = no_images / vol_axis_len

    name = image.stem
    number = int(name.split("_")[underscores_to_index])
    print(f"Number extracted is {number}")
    image_index = number / file_number_increment
    print(f"Image index is {image_index}")

    atlas_index = int(np.round(image_index / axis_ratio))
    print(f"Atlas index is {atlas_index}")
    atlas_slice = data[:, atlas_index, :]

    image_slice = np.array(Image.open(image))
    target_shape = image_slice.shape[:2]

    rotated_atlas_slice = np.rot90(atlas_slice)
    resized_atlas_slice = cv2.resize(
        rotated_atlas_slice,
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    resized_atlas_slice_16bit = resized_atlas_slice.astype(np.uint16)

    return name, resized_atlas_slice_16bit


def relabel_sequential_for_preview(label_slice):
    """Remap labels to sequential integers (1..N); background (0) stays 0."""
    label_slice = label_slice.copy()

    unique_labels = np.unique(label_slice)
    unique_labels = unique_labels[unique_labels != 0]

    relabeled = np.zeros_like(label_slice, dtype=np.int32)

    for new_id, old_id in enumerate(unique_labels, start=1):
        relabeled[label_slice == old_id] = new_id

    return relabeled