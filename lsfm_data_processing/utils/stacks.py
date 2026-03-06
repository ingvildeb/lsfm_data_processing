import numpy as np
import tifffile
from PIL import Image


def tifs_to_zstack(file_list, out_dir, out_prefix):
    images = []
    names = []
    for file in file_list:
        img = Image.open(file)
        images.append(np.array(img))
        names.append(file.stem)

    zstack_array = np.stack(images)
    output_filename = out_dir / f"{out_prefix}_zstack_{names[0]}_to_{names[-1]}.tif"

    tifffile.imwrite(output_filename, zstack_array, photometric="minisblack")