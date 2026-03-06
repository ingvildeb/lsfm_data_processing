import tifffile


def get_avg_pixel_value(path_to_image):
    image = tifffile.TiffFile(path_to_image).asarray()
    shape = image.shape

    if len(shape) == 2:
        average_pixel_value = image.mean()
    elif len(shape) == 3:
        middle_z = int(shape[0] / 2)
        image = image[middle_z, :, :]
        average_pixel_value = image.mean()
    else:
        raise ValueError(f"Unsupported image shape: {shape}")

    return average_pixel_value


def chunk_image(path_to_image, image_outdir, chunk_size):
    img = tifffile.TiffFile(path_to_image).asarray()
    image_name = path_to_image.stem
    shape = img.shape

    for i in range(0, shape[0], chunk_size):
        for j in range(0, shape[1], chunk_size):
            chunk = img[i : i + chunk_size, j : j + chunk_size]
            tifffile.imwrite(f"{image_outdir}/{image_name}_chunk_{i}_{j}.tif", chunk)


def chunk_z_stack(path_to_image, image_outdir, chunk_size):
    full_stack = tifffile.TiffFile(path_to_image).asarray()
    shape = full_stack.shape
    image_name = path_to_image.stem

    for i in range(0, shape[1], chunk_size):
        for j in range(0, shape[2], chunk_size):
            stack_chunk = full_stack[:, i : i + chunk_size, j : j + chunk_size]
            tifffile.imwrite(f"{image_outdir}/{image_name}_chunk_{i}_{j}.tif", stack_chunk)