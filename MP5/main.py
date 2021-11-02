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
    theta = np.arctan2(y_res, x_res)
    return mag, theta

def look_up_table(rad):
    if (rad >=math.pi/8 and rad<3*math.pi/8) or (rad >=-7*math.pi/8 and rad<-5*math.pi/8):
        return [(0,0),(2,2)]
    elif (rad >=3*math.pi/8 and rad<5*math.pi/8) or (rad >=-5*math.pi/8 and rad<-3*math.pi/8):
        return [(0,1),(2,1)]
    elif (rad >=5*math.pi/8 and rad<7*math.pi/8) or (rad >=-3*math.pi/8 and rad<-1*math.pi/8):
        return [(0,2),(2,0)]
    elif (rad >=7*math.pi/8 or rad<-7*math.pi/8) or (rad >=-1*math.pi/8 and rad<math.pi/8):
        return [(1,0),(1,2)]
    else:
        print(rad/math.pi)

def non_maxima_sup(img, mag, theta):

    img = pad_img(img, np.zeros((3,3))).astype(np.uint8)

    res = np.zeros(mag.shape)

    start_x = int(3 / 2)
    start_y = start_x

    half_kernel_size = int(3 / 2)

    for y in range(0, mag.shape[0]):
        for x in range(0, mag.shape[1]):
            if x < start_x or y < start_y or x >= mag.shape[1] - start_x or y >= mag.shape[
                0] - start_y:  # taking care of edges
                continue

            region = mag[y - half_kernel_size: y + half_kernel_size + 1,
                     x - half_kernel_size: x + half_kernel_size + 1]
            rad = theta[y][x]
            indices = look_up_table(rad)
            if region[indices[0][1]][indices[0][0]] <= region[1][1] and region[indices[1][1]][indices[1][0]] <= region[1][1]:
                res[y, x] = mag[y][x]

    return res



if __name__ == '__main__':
    images = load_test_data("./data")

    img = cv2.cvtColor(images[1].astype(np.uint8), cv2.COLOR_BGR2GRAY)
    img = gaussian_smoothing(img.copy(), 5, 1)
    mag, theta = image_gradient(img)
    cv2.imwrite("output1.jpg", mag)
    res = non_maxima_sup(img, mag, theta)
    cv2.imwrite("output2.jpg", res)







