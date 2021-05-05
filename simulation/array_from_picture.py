import PIL.Image as Image
import numpy as np


def convert_img_to_array(path, filename):
    # read Image
    with Image.open("{}/{}".format(path, filename)) as img:
        img = np.asarray(img.convert("P"))
        # Convert to numpy array
        img = np.array(img, copy=True)
        # Replace white with 1
        img[img == 225] = 1
        # Replace red (255, 0, 0) with 2
        img[img == 15] = 2
        # Replace other then black (0), white (1), red (2) with 3
        img[np.where((img != 0) & (img != 1) & (img != 2))] = 3
        return img
