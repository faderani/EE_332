import os
import imageio
import numpy as np
import cv2


from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt




def load_train_data(root):
    """load jpg images from training directory into np arrays"""

    paths = os.listdir(root)

    images = []

    for path in paths:
        if "jpg" not in path.split(".")[-1]:
            continue
        image = np.array(imageio.imread(os.path.join(root, path)), dtype=np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        images.append(image)
        print(f"{os.path.join(root, path)} loaded!")

    imgs = np.asarray(images, dtype=object)
    return imgs

def load_test_data(root):
    """load bmp images from training directory into np arrays"""

    paths = os.listdir(root)

    images = []

    for path in paths:
        if "bmp" not in path.split(".")[-1]:
            continue
        image = np.array(imageio.imread(os.path.join(root, path)), dtype=np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        images.append(image)
        print(f"{os.path.join(root, path)} loaded!")

    imgs = np.asarray(images, dtype=object)
    return imgs



def histogram_2d(imgs):

    flat_img = np.concatenate(imgs, axis = None)
    h = flat_img[0::3]
    s = flat_img[1::3]

    H, x_edge, y_edge = np.histogram2d(h,s, density=True)

    H = H.T
    return H, x_edge, y_edge


def segment_skin(img, H, hue_edges, sat_edges):

    h, w, _ = img.shape

    segmented_img = np.zeros(img.shape, dtype=np.float32)


    for x in range(int(w)):
        for y in range(int(h)):
            hue = img[y][x][0]
            sat = img[y][x][1]
            h_idx = np.digitize([hue], hue_edges)[0]
            s_idx = np.digitize([sat], sat_edges)[0]
            if h_idx == 0 or s_idx == 0 or h_idx == len(hue_edges)  or s_idx == len(sat_edges) :
                continue

            if H[s_idx-1][h_idx-1] > 0.000000001:
                segmented_img[y][x] = img[y][x]

    return segmented_img







if __name__ == '__main__':
    train_imgs = load_train_data("./training_data")
    H, hue_edges, sat_edges = histogram_2d(train_imgs)

    test_imgs = load_test_data("./testing_data")


    for idx, img in enumerate(test_imgs):
        seg_img = segment_skin(img.copy(), H, hue_edges, sat_edges)
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_HSV2BGR)
        cv2.imwrite(f"./results/output{idx}.jpg", seg_img)

    # fig = plt.figure(figsize=(7, 3))
    # ax = fig.add_subplot(131, title='imshow: square bins')
    # plt.imshow(H)
    # plt.show()