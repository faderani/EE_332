
import cv2
from MP5.main import gaussian_smoothing
import numpy as np
from utils import *
import math
from matplotlib.pylab import plt

import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage



def detect_edge(img, low, high, save_dir):
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    img = gaussian_smoothing(img.copy(), 7, 1).astype(np.uint8)
    edges = cv2.Canny(img, low, high)
    save_path = os.path.join(save_dir, "edges.jpg")
    cv2.imwrite(save_path, edges)

    return edges


def create_discrete_space(img_width, img_height, p_dim, theta_dim):
    p_max = math.sqrt(img_width**2 + img_height**2)
    theta_max = math.pi * 2
    space = np.zeros((p_dim, theta_dim))

    return space, p_max, theta_max

def create_hough_space(img,space, p_max, theta_max, save_dir):
    p_dim, theta_dim = space.shape

    img_height, img_width = img.shape

    for x in range(img_width):
        for y in range(img_height):
            if img[y][x] == 0: continue
            for itheta in range(theta_dim):
                theta = itheta * theta_max / theta_dim
                p = x * math.sin(theta) + y * math.cos(theta)
                ip = int(p_dim * p / p_max)
                itheta = int(itheta)
                space[ip][itheta] += 1

    save_path = os.path.join(save_dir, "hough_space.jpg")
    cv2.imwrite(save_path, space)

    return space


def find_maxima(space, thresh, neighbor_size, save_dir):
    data_max = filters.maximum_filter(space, neighbor_size)
    maxima = (space == data_max)

    diff = ((data_max) > thresh)
    maxima[diff == 0] = 0


    slices = np.argwhere(maxima == 1)
    slices = [tuple(x) for x in slices]

    x, y = [], []
    for dy, dx in slices:
        x_center = (dx + dx) / 2
        x.append(x_center)
        y_center = (dy + dy) / 2
        y.append(y_center)

    plt.imshow(space, origin='lower')
    plt.autoscale(False)
    plt.plot(x, y, 'rx')
    save_path = os.path.join(save_dir, "max_hough_space.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return x, y


def draw_lines(x,y, img, space, r_max, theta_max, save_dir):

    img_h, img_w = img.shape
    r_dim, theta_dim = space.shape

    idx = 1

    for i, j in zip(y, x):

        r = round((i * r_max) / r_dim, 1)
        theta = round((j * theta_max) / theta_dim, 1)

        fig, ax = plt.subplots()

        ax.imshow(img, cmap="gray")

        ax.autoscale(False)

        px = []
        py = []
        for t in range(-img_h - 40, img_h + 40, 1):
            px.append(math.cos(-theta) * t - math.sin(-theta) * r)
            py.append(math.sin(-theta) * t + math.cos(-theta) * r)

        ax.plot(px, py, linewidth=10)

        save_path = os.path.join(save_dir, "%02d" % idx + ".png")
        cv2.imwrite(save_path, space)
        plt.savefig(save_path, bbox_inches='tight')

        plt.close()

        idx += 1



def run_pipeline(img, save_dir):
    edges = detect_edge(img.copy(), 20, 40, save_dir)
    space, p_max, theta_max= create_discrete_space(edges.shape[1], edges.shape[0], 100, 200)
    space = create_hough_space(edges, space, p_max, theta_max, save_dir)
    x,y = find_maxima(space, 80, 10, save_dir)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    draw_lines(x,y,edges,space,p_max, theta_max, save_dir)
    return

if __name__ == '__main__':

    images = load_test_data("./data")

    for idx, image in enumerate(images):
        os.makedirs(f"./outputs/{idx}/", exist_ok=True)
        run_pipeline(image, f"./outputs/{idx}/")
