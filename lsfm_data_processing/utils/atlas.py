import cv2
import numpy as np


def relabel_sequential_for_preview(label_slice):
    """Remap labels to sequential integers (1..N); background (0) stays 0."""
    label_slice = label_slice.copy()

    unique_labels = np.unique(label_slice)
    unique_labels = unique_labels[unique_labels != 0]

    relabeled = np.zeros_like(label_slice, dtype=np.int32)

    for new_id, old_id in enumerate(unique_labels, start=1):
        relabeled[label_slice == old_id] = new_id

    return relabeled


def atlas_slice_for_mip(
    reg_volume_data: np.ndarray,
    no_images: int,
    section_number: int,
    file_number_increment: int,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """
    Extract and resize the atlas slice corresponding to a MIP section index.
    """
    vol_axis_len = reg_volume_data.shape[1]
    axis_ratio = no_images / vol_axis_len
    image_index = section_number / file_number_increment

    atlas_index = int(np.round(image_index / axis_ratio))
    atlas_index = max(0, min(atlas_index, vol_axis_len - 1))

    atlas_slice = reg_volume_data[:, atlas_index, :]
    rotated = np.rot90(atlas_slice)
    resized = cv2.resize(rotated, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.uint16)
