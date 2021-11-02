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
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        images.append(image)
        print(f"{os.path.join(root, path)} loaded!")

    imgs = np.asarray(images, dtype=object)
    return imgs



# def convolve(img, kernel):
#     columns, rows ,_ = img.shape
#     res = np.zeros(shape=(rows, columns))
#
#     for i in range(rows - 2):
#         for j in range(columns - 2):
#             res = np.sum(np.multiply(kernel, img[i:i + 3, j:j + 3]))  # x direction
#
#     return res


def convolve(img, SE, iter = 1):
    assert SE.shape[0] % 2 == 1 and SE.shape[1] % 2 == 1, "SE size should be odd"
    assert SE.shape[0] == SE.shape[1], "SE size should be square"


    img = pad_img(img, SE).astype(np.uint8)

    for i in range(iter):

        start_x = int(SE.shape[0] / 2)
        start_y = start_x

        half_kernel_size = int(SE.shape[0]/2)

        result = np.zeros(img.shape)
        for y in range(0, img.shape[0]):
            for x in range(0, img.shape[1]):
                if x < start_x or y < start_y or x >= img.shape[1] - start_x or y >= img.shape[0] - start_y:  # taking care of edges
                    continue

                region = img[y - half_kernel_size: y + half_kernel_size + 1,
                         x - half_kernel_size: x + half_kernel_size + 1]
                result[y,x] = np.sum(np.multiply(SE, region))
        img = result

    return img[start_y:-start_y, start_x:-start_x]

def pad_img(img, SE):
    return np.pad(img, int(SE.shape[0]/2), mode="edge")