
import cv2
from MP5.main import gaussian_smoothing
import numpy as np
from utils import *
import math
from matplotlib.pylab import plt

def detect_edge(img, low, high):
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    img = gaussian_smoothing(img.copy(), 3, 1).astype(np.uint8)
    edges = cv2.Canny(img, low, high)
    return edges


def create_discrete_space(img_width, img_height):
    d = math.sqrt(img_width**2 + img_height**2)
    space = np.zeros((180, int(d)))

    return space

def get_p(x,y,theta):
    return int(x*math.cos(theta) + y*math.sin(theta))


def get_p_all(img, space):

    h, w  = img.shape

    for x in range(w):
        for y in range(h):
            if img[y][x] != 255:
                continue
            for theta in range(space.shape[0]):
                space[theta][get_p(x,y,math.radians(theta))] += 1
    return space

def thresh_space(space, thresh):
    space[space[space < thresh]] = 0
    return space

def revert_x_y(space, img):
    h_s, w_s = space.shape
    h_i, w_i = img.shape


    for theta in range(w_s):
        for p in range(h_s):
            if im





def run_pipeline(img):
    img = detect_edge(img, 20, 40)
    space = create_discrete_space(img.shape[1], img.shape[0])
    space = get_p_all(img.copy(), space)
    space = thresh_space(space, 0.2)

    return



if __name__ == '__main__':

    images = load_test_data("./data")

    run_pipeline(images[0])

