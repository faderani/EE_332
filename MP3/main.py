import matplotlib.pyplot as plt
import os
import imageio
from skimage import color
import numpy as np
import cv2




def save_histogram(im):
    vals = im.flatten()

    b, bins, patches = plt.hist(vals, 255)
    plt.xlim([0, 255])
    plt.show()



def load_image(path):
    """load bmp images from root directory into np arrays"""



    image = np.array(color.rgb2gray(imageio.imread(path)), dtype=np.float32)
    image = (image*255).astype(np.uint8)
    print(f"{path} loaded!")
    return image



def get_hist(im):

    hist = np.zeros((256,))
    for pix in im.flatten():
        hist[pix] += 1

    return hist

def get_cumlative_sum(hist):
    cs = []

    for idx, val in enumerate(hist):
        new_val = np.sum(hist[0:idx]) + val
        cs.append(new_val)

    return cs/max(cs) * 255


def equalize_hist(img, hist):

    new_img = np.zeros((img.shape)).flatten()


    for idx, pix in enumerate(img.flatten()):
        new_img[idx] = hist[pix]

    return new_img.reshape(img.shape)









if __name__ == "__main__":

    img = load_image("./data/moon.bmp")
    hist = get_hist(img)
    cs = get_cumlative_sum(hist)

    # plt.title("Cumulative Sum of original histogram")
    # plt.plot(cs)
    # plt.show()

    eq_img = equalize_hist(img, cs)
    cv2.imwrite("equalized_hist_img.jpg", eq_img)


