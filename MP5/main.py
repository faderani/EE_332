import numpy as np

from utils import *
import math
import cv2
from skimage import filters




def get_gaussian_kernel(kernel_s, sigma):
    assert kernel_s % 2 == 1, "Kernel size must be odd"
    row = np.linspace(-(kernel_s - 1) / 2, (kernel_s - 1) / 2, kernel_s)
    gaussian_val = np.exp(-0.5 * np.square(row) / np.square(sigma))
    kernel = np.outer(gaussian_val, gaussian_val)
    return kernel / np.sum(kernel)

def get_sobel_kernel():

    g_x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    g_y = np.flip(g_x.T, axis=0)
    return g_x, g_y

def gaussian_smoothing(img, kernel_s, sigma):
    kernel = get_gaussian_kernel(kernel_s, sigma)
    res_map = convolve(img, kernel)

    return res_map

def image_gradient(img):

    g_x, g_y = get_sobel_kernel()

    x_res = convolve(img, g_x)
    y_res = convolve(img, g_y)

    mag = np.sqrt(np.square(x_res) + np.square(y_res))
    mag *= 255.0 / mag.max()

    #mag = mag * 100 / mag.max()
    #mag = np.around(mag)

    #cv2.imwrite("output.jpg", mag)

    #theta = np.arctan2(y_res, x_res)
    theta = np.arctan2(y_res, x_res)
    return mag, theta

def look_up_table(rad):

    rad = rad + math.pi
    PI = math.pi
    if (0 <= rad < PI / 8) or (15 * PI / 8 <= rad <= 2 * PI):
        return [(1,0),(1,2)]
    elif (PI / 8 <= rad < 3 * PI / 8) or (9 * PI / 8 <= rad < 11 * PI / 8):
        return [(2,0), (0,2)]
    elif (3 * PI / 8 <= rad < 5 * PI / 8) or (11 * PI / 8 <= rad < 13 * PI / 8):
        return [(0,1),(2,1)]
    else:
        return [(0,0),(2,2)]


    # if (rad >=math.pi/8 and rad<3*math.pi/8) or (rad >=-7*math.pi/8 and rad<-5*math.pi/8):
    #     return [(0,2),(2,0)]
    # elif (rad >=3*math.pi/8 and rad<5*math.pi/8) or (rad >=-5*math.pi/8 and rad<-3*math.pi/8):
    #     return [(0,1),(2,1)]
    # elif (rad >=5*math.pi/8 and rad<7*math.pi/8) or (rad >=-3*math.pi/8 and rad<-1*math.pi/8):
    #     return [(0,0),(2,2)]
    # elif (rad >=7*math.pi/8 or rad<-7*math.pi/8) or (rad >=-1*math.pi/8 and rad<math.pi/8):
    #     return [(1,0),(1,2)]
    # else:
    #     print(rad/math.pi)


# def find_thresh(mag, non_edge_per):
#     histogram, bin_edges = np.histogram(mag, bins=256, range=(0, 255))
#
#     # configure and draw the histogram figure
#     plt.figure()
#     plt.title("Grayscale Histogram")
#     plt.xlabel("grayscale value")
#     plt.ylabel("pixels")
#     plt.xlim([0.0, 255])  # <- named arguments do not work here
#
#     plt.plot(bin_edges[0:-1], histogram)  # <- or here
#     plt.show()

def threshold(image, low, high, weak):

    output = np.zeros(image.shape)

    strong = 255

    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))

    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak

    return output


def non_max_suppression(gradient_magnitude, gradient_direction):
    image_row, image_col = gradient_magnitude.shape

    output = np.zeros(gradient_magnitude.shape)

    PI = 180

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]

            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]

            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]

            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]


    return output





def link_weak_strong(mag):

    mid = np.unique(mag)[1]
    high = np.unique(mag)[2]

    height, width = mag.shape

    final_res = np.zeros(mag.shape)

    old_mag = np.zeros(mag.shape)

    while(len(mag[mag == high]) != len(old_mag[old_mag == high])):
        old_mag = mag.copy()

        for x in range(1, width-1):
            for y in range(1, height-1):
                top = mag[y - 1][x] == 255
                top_left = mag[y-1][x-1] == 255
                top_right = mag[y-1][x+1] == 255
                bottom = mag[y + 1][x] == 255
                bottom_right = mag[y+1][x+1] == 255
                bottom_left = mag[y+1][x-1] == 255
                right = mag[y][x + 1] == 255
                left = mag[y][x - 1] == 255
                if mag[y][x] == mid:
                    if top or bottom or right or left or top_left or top_right or bottom_left or bottom_right:
                        mag[y][x] = 255

                elif mag[y][x] == high:
                    mag[y][x] = 255

        #
        # for x in range(width - 2, 0, -1):
        #     for y in range(1, height - 1):
        #         top = mag[y - 1][x] == 255
        #         top_left = mag[y-1][x-1] == 255
        #         top_right = mag[y-1][x+1] == 255
        #         bottom = mag[y + 1][x] == 255
        #         bottom_right = mag[y+1][x+1] == 255
        #         bottom_left = mag[y+1][x-1] == 255
        #         right = mag[y][x + 1] == 255
        #         left = mag[y][x - 1] == 255
        #         if mag[y][x] == mid:
        #             if top or bottom or right or left or top_left or top_right or bottom_left or bottom_right:
        #                 mag[y][x] = 255
        #
        #         elif mag[y][x] == high:
        #             mag[y][x] = 255
        #
        # for x in range(width - 2, 0, -1):
        #     for y in range(height - 2, 0, -1):
        #         top = mag[y - 1][x] == 255
        #         top_left = mag[y-1][x-1] == 255
        #         top_right = mag[y-1][x+1] == 255
        #         bottom = mag[y + 1][x] == 255
        #         bottom_right = mag[y+1][x+1] == 255
        #         bottom_left = mag[y+1][x-1] == 255
        #         right = mag[y][x + 1] == 255
        #         left = mag[y][x - 1] == 255
        #         if mag[y][x] == mid:
        #             if top or bottom or right or left or top_left or top_right or bottom_left or bottom_right:
        #                 mag[y][x] = 255
        #
        #         elif mag[y][x] == high:
        #             mag[y][x] = 255
        #
        #
        # for x in range(1, width-2):
        #     for y in range(height - 2, 0, -1):
        #         top = mag[y - 1][x] == 255
        #         top_left = mag[y-1][x-1] == 255
        #         top_right = mag[y-1][x+1] == 255
        #         bottom = mag[y + 1][x] == 255
        #         bottom_right = mag[y+1][x+1] == 255
        #         bottom_left = mag[y+1][x-1] == 255
        #         right = mag[y][x + 1] == 255
        #         left = mag[y][x - 1] == 255
        #         if mag[y][x] == mid:
        #             if top or bottom or right or left or top_left or top_right or bottom_left or bottom_right:
        #                 mag[y][x] = 255
        #
        #         elif mag[y][x] == high:
        #             mag[y][x] = 255

    mag[mag == mid] = 0
    return mag


def run_pipeline(img, save_path, low, high, weak):
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    img = gaussian_smoothing(img.copy(), 3, 1)
    mag, theta = image_gradient(img)
    res = non_max_suppression(mag, theta)
    #res = non_maxima_sup(img, mag, theta)


    res = threshold(res, low, high, weak)
    res = link_weak_strong(res)
    cv2.imwrite(save_path, res)


def run_pipeline_cv(img, save_path, low, high):
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    img = gaussian_smoothing(img.copy(), 3, 1).astype(np.uint8)
    edges = cv2.Canny(img, low, high)
    cv2.imwrite(save_path, edges)


def sobel_cv(img, save_path):
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    img = gaussian_smoothing(img.copy(), 3, 1).astype(np.uint8)
    g_x = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    g_y = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

    g_x = cv2.convertScaleAbs(g_x)
    g_y = cv2.convertScaleAbs(g_y)

    mag = cv2.addWeighted(g_x, 0.5, g_y, 0.5, 0)

    cv2.imwrite(save_path, mag)

def roberts_scikit(img, save_path):
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    img = gaussian_smoothing(img.copy(), 3, 1)
    edge_roberts = filters.roberts(img)
    cv2.imwrite(save_path, edge_roberts)


def zero_crossing(img, save_path):
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    img = gaussian_smoothing(img.copy(), 3, 1)
    lap = np.sign(filters.laplace(img))
    lap = np.pad(lap, ((0, 1), (0, 1)))
    diff_x = lap[:-1, :-1] - lap[:-1, 1:] < 0
    diff_y = lap[:-1, :-1] - lap[1:, :-1] < 0
    edges = np.logical_or(diff_x, diff_y).astype(float)

    cv2.imwrite(save_path, edges*255)



# def non_maxima_sup(img, mag, theta):
#
#     img = pad_img(img, np.zeros((3,3))).astype(np.uint8)
#
#     res = np.zeros(mag.shape)
#
#     start_x = int(3 / 2)
#     start_y = start_x
#
#     half_kernel_size = int(3 / 2)
#
#     for y in range(0, mag.shape[0]):
#         for x in range(0, mag.shape[1]):
#             if x < start_x or y < start_y or x >= mag.shape[1] - start_x or y >= mag.shape[
#                 0] - start_y:  # taking care of edges
#                 continue
#
#             region = mag[y - half_kernel_size: y + half_kernel_size + 1,
#                      x - half_kernel_size: x + half_kernel_size + 1]
#             rad = theta[y][x]
#             indices = look_up_table(rad)
#             if region[indices[0][1]][indices[0][0]] <= region[1][1] and region[indices[1][1]][indices[1][0]] <= region[1][1]:
#                 res[y, x] = mag[y][x]
#             else:
#                 res[y, x] = 0
#
#     return res



if __name__ == '__main__':
    images = load_test_data("./data")
    for idx, im in enumerate(images):
        run_pipeline(im.copy(), f"./outputs/my_implementation/{idx}.jpg", low=10, high=20, weak=80)
        run_pipeline_cv(im.copy(), f"./outputs/canny_opencv/{idx}.jpg", low=10, high=20)
        sobel_cv(im.copy(), f"./outputs/sobel/{idx}.jpg")
        roberts_scikit(im.copy(), f"./outputs/roberts/{idx}.jpg")
        zero_crossing(im.copy(), f"./outputs/zero_cross/{idx}.jpg")









