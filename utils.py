import os
import cv2
import numpy as np
import imageio

def load_test_data(root):
    """load bmp images from training directory into np arrays"""

    paths = os.listdir(root)

    images = []

    for path in paths:
        if "bmp" not in path.split(".")[-1]:
            continue
        image = np.array(imageio.imread(os.path.join(root, path)), dtype=np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        images.append(image)
        print(f"{os.path.join(root, path)} loaded!")

    imgs = np.asarray(images, dtype=object)
    return imgs