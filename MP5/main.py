import numpy as np

from utils import *
import numpy

from scipy.ndimage import convolve


def get_kernel(kernel_s, sigma):
    assert kernel_s % 2 == 1, "Kernel size must be odd"
    row = np.linspace(-(kernel_s - 1) / 2, (kernel_s - 1) / 2, kernel_s)
    gaussian_val = np.exp(-0.5 * np.square(row) / np.square(sigma))
    kernel = np.outer(gaussian_val, gaussian_val)
    return kernel / np.sum(kernel)

def gaussian_smoothing(img, kernel_s, sigma):
    kernel = get_kernel(kernel_s, sigma)
    num_ch = img.shape[2]
    res_map = np.zeros(img.shape)
    for ch in range(num_ch):
        res_map[:,:,ch] = convolve(img[:,:,ch], kernel)

    return res_map







if __name__ == '__main__':
    images = load_test_data("./data")


    img = gaussian_smoothing(images[0].copy(), 3, 1)
    cv2.imwrite("output.jpg", img)



