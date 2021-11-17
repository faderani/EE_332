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

def load_seq(root):
    """load jpg images of a video from root into an np array"""

    paths = os.listdir(root)
    paths = sorted(paths)

    images = []

    for path in paths:
        if "jpg" not in path.split(".")[-1]:
            continue
        image = np.array(imageio.imread(os.path.join(root, path)), dtype=np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        images.append(image)
        print(f"{os.path.join(root, path)} loaded!")

    imgs = np.asarray(images, dtype=object)
    return imgs




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

def convolution(image, kernel, average=False, verbose=False):
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))

    print("Kernel Shape : {}".format(kernel.shape))

    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    print("Output Image size : {}".format(output.shape))

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()

    return output