import numpy as np
import os
import imageio
import cv2


def load_images(root):
    """load bmp images from root directory into np arrays"""

    paths = os.listdir(root)

    images = []

    for path in paths:
        if "bmp" not in path.split(".")[-1]:
            continue
        image = imageio.imread(os.path.join(root, path))
        images.append(np.array(image, dtype=np.uint8))

    return np.array(images, dtype=object) / 255


def dilation(img, SE):
    assert SE.shape[0] % 2 == 1 and SE.shape[1] % 2 == 1, "SE size should be odd"
    assert SE.shape[0] == SE.shape[1], "SE size should be square"

    start_x = int(SE.shape[0] / 2)
    start_y = start_x

    half_kernel_size = int(SE.shape[0]/2)

    result = np.zeros(img.shape)
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            if x < start_x or y < start_y or x == img.shape[1] - 1 or y == img.shape[0] - 1: #taking care of edges
                continue

            if img[y][x] == 1:
                result[y-half_kernel_size: y+half_kernel_size +1 , x-half_kernel_size: x+half_kernel_size +1] = SE

    return result

def erosion(img, SE):
    assert SE.shape[0] % 2 == 1 and SE.shape[1] % 2 == 1, "SE size should be odd"
    assert SE.shape[0] == SE.shape[1], "SE size should be square"

    start_x = int(SE.shape[0] / 2)
    start_y = start_x

    half_kernel_size = int(SE.shape[0] / 2)

    result = np.zeros(img.shape)
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            if x < start_x or y < start_y or x == img.shape[1] - 1 or y == img.shape[0] - 1:  # taking care of edges
                continue

            if img[y][x] == 1:
                if np.array_equal(img[y - half_kernel_size: y + half_kernel_size + 1, x - half_kernel_size: x + half_kernel_size + 1], SE):
                    result[y - half_kernel_size: y + half_kernel_size + 1, x - half_kernel_size: x + half_kernel_size + 1] = 1


    return result






def pad_img(img, SE):
    return np.pad(img, int(SE.shape[0]/2), constant_values=0)




if __name__ == '__main__':

    images = load_images("./data")
    SE = np.ones((3,3))
    for img in images:

        padded_img = pad_img(img, SE)
        dilated_img = erosion(padded_img, SE)
        cv2.imwrite("output.jpg", dilated_img*255)









