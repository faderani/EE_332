import numpy as np

from utils import *
import math



def get_gaussian_kernel(kernel_s, sigma):
    assert kernel_s % 2 == 1, "Kernel size must be odd"
    row = np.linspace(-(kernel_s - 1) / 2, (kernel_s - 1) / 2, kernel_s)
    gaussian_val = np.exp(-0.5 * np.square(row) / np.square(sigma))
    kernel = np.outer(gaussian_val, gaussian_val)
    return kernel / np.sum(kernel)

def get_sobel_kernel():
    # g_x = np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=int)
    # g_y = g_x.T
    g_x = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    g_y = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    return g_x, g_y

def gaussian_smoothing(img, kernel_s, sigma):
    kernel = get_gaussian_kernel(kernel_s, sigma)
    #num_ch = img.shape[2]
    res_map = np.zeros(img.shape)
    #for ch in range(num_ch):
    res_map = convolve(img, kernel)

    return res_map

def image_gradient(img):

    g_x, g_y = get_sobel_kernel()

    x_res = convolve(img, g_x)
    y_res = convolve(img, g_y)

    mag = np.sqrt(np.power(x_res, 2) + np.power(y_res, 2))

    cv2.imwrite("output.jpg", mag)

    #theta = np.arctan2(y_res, x_res)
    theta = np.arctan(y_res, x_res)
    return mag, theta

def look_up_table(rad):
    if rad >=math.pi/8 and rad<3*math.pi/8:
        return [(1,0),(2,1)]
    elif rad >=3*math.pi/8 and rad<5*math.pi/8:
        return [(0,0),(2,0)]
    elif rad >=5*math.pi/8 and rad<7*math.pi/8:
        return [(1,0),(0,1)]
    elif rad >=7*math.pi/8 and rad<9*math.pi/8:
        return [(0,0),(0,2)]
    elif rad >=9*math.pi/8 and rad<11*math.pi/8:
        return [(0,1),(1,2)]
    elif rad >=11*math.pi/8 and rad<13*math.pi/8:
        return [(0,2),(2,2)]
    elif rad >=13*math.pi/8 and rad<15*math.pi/8:
        return [(1,2),(2,1)]
    elif rad >=15*math.pi/8 and rad<math.pi/8:
        return [(2,0),(2,2)]


def non_maxima_sup(img, mag, theta):


    cols,rows = img.shape

    res = np.zeros(mag.shape)

    for x in range(rows):
        for y in range(cols):
            indices = look_up_table(theta[y][x])
            region = np.zeros((3,3))
            region = mag[x:x+3][y:y+3]

            #if mag[y][x] > mag[y + ]








if __name__ == '__main__':
    images = load_test_data("./data")

    img = cv2.cvtColor(images[1].astype(np.uint8), cv2.COLOR_BGR2GRAY)
    img = gaussian_smoothing(img.copy(), 5, 1)


    mag, theta = image_gradient(img)






