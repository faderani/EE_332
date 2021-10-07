import numpy as np
import os
import imageio
import cv2
from skimage.morphology import erosion, dilation, opening, closing
import sys




def load_images(root):
    """load bmp images from root directory into np arrays"""

    paths = os.listdir(root)

    images = []

    for path in paths:
        if "bmp" not in path.split(".")[-1]:
            continue
        image = np.array(imageio.imread(os.path.join(root, path)), dtype=np.float32)
        image /= 255
        image = np.where(image > 0.9, 1, image)
        image = np.where(image < 0.9, 0, image)
        image = image.astype(np.uint8)

        images.append(image)
        print(f"{os.path.join(root, path)} loaded!")

    imgs = np.array(images, dtype=object)



    return imgs



def dilation_op(img, SE):
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
                region = result[y-half_kernel_size: y+half_kernel_size +1 , x-half_kernel_size: x+half_kernel_size +1]
                result[y-half_kernel_size: y+half_kernel_size +1 , x-half_kernel_size: x+half_kernel_size +1] = np.logical_or(region,SE).astype(np.uint8)

    return result

def erosion_op(img, SE):
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

                region = img[y - half_kernel_size: y + half_kernel_size + 1, x - half_kernel_size: x + half_kernel_size + 1]
                and_op = np.logical_and(region, SE)


                if np.count_nonzero(and_op == 1) == np.count_nonzero(SE == 1):
                    #result[y - half_kernel_size: y + half_kernel_size + 1, x - half_kernel_size: x + half_kernel_size + 1] = 1
                    result[y][x] = 1


                # if np.array_equal(img[y - half_kernel_size: y + half_kernel_size + 1, x - half_kernel_size: x + half_kernel_size + 1], SE):
                #     result[y - half_kernel_size: y + half_kernel_size + 1, x - half_kernel_size: x + half_kernel_size + 1] = 1


    return result


def open_op(img, SE):
    er_res = erosion(img,SE)
    dil_res = dilation(er_res, SE)
    return dil_res

def close_op(img, SE):
    dil_res = dilation(img, SE)
    er_res = erosion(dil_res, SE)
    return er_res


def bound_op(img, SE):
    er_res = erosion(img, SE)
    return img - er_res









def pad_img(img, SE):
    return np.pad(img, int(SE.shape[0]/2), constant_values=0)




if __name__ == '__main__':

    images = load_images("./data")
    #SE = np.ones((3,3))
    SE = np.array(([0,1,0], [1,1,1], [0,1,0]), dtype=np.uint8)
    for idx, img in enumerate(images):

        padded_img = pad_img(img, SE).astype(np.uint8)

        open_res = open_op(padded_img.copy(), SE)

        open_true = opening(padded_img.copy(), SE)

        save_path = os.path.join("./outputs", str(idx) + "_open.jpg")
        save_path_true = os.path.join("./outputs", str(idx) + "_open_true.jpg")


        cv2.imwrite(save_path, open_res * 255)
        cv2.imwrite(save_path_true, open_true * 255)

    for idx, img in enumerate(images):
        padded_img = pad_img(img, SE).astype(np.uint8)

        close_res = close_op(padded_img.copy(), SE)

        close_true = closing(padded_img.copy(), SE)

        save_path = os.path.join("./outputs", str(idx) + "_clsoe.jpg")
        save_path_true = os.path.join("./outputs", str(idx) + "_close_true.jpg")

        cv2.imwrite(save_path, close_res * 255)
        cv2.imwrite(save_path_true, close_true * 255)





        # #dilated_img = erosion(padded_img, SE)
        # open_res = open_op(padded_img.copy(), SE)
        # close_res = close_op(padded_img.copy(), SE)
        #
        # open_true = opening(padded_img.copy(), SE)
        # close_true = closing(padded_img.copy(), SE)
        #
        #
        # # eroded_input = erosion_op(padded_img, SE)
        # # true_eroded_input = erosion(padded_img, SE)
        #
        # # dilated_input = dilation_op(padded_img.copy(), SE)
        # # true_dilated_input = dilation(padded_img.copy(), SE)
        #
        #
        # cv2.imwrite("open.jpg", open_res*255)
        # cv2.imwrite("close.jpg", close_res * 255)
        #
        # cv2.imwrite("open_true.jpg", open_true * 255)
        # cv2.imwrite("close_true.jpg", close_true * 255)









