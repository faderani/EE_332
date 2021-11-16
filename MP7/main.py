import cv2
import numpy as np
from scipy.signal import correlate2d

from utils import *

class Template:


    def __init__(self, x1, y1, x2, y2, img, window_size):

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        self.img = img
        self.template = self.img[y1:y2, x1:x2]

        self.window_size = window_size
        self.window_x1 = max(self.x1 - window_size, 0)
        self.window_y1 = max(self.y1 - window_size, 0)
        self.window_x2 = min(self.x2 + window_size, self.img.shape[1])
        self.window_y2 = min(self.y2 + window_size, self.img.shape[0])


    def calc_ssd(self, x1, y1, x2, y2):

        patch = self.img[y1:y2, x1:x2]
        return np.sum(np.power(self.template - patch, 2))

    def calc_ssd_full(self):
        h, w = self.img.shape
        t_h, t_w = self.template.shape
        min_sum_coord = (0,0,0,0)
        min_sum = -1
        for x in range(self.window_x1, self.window_x2):
            if x < int(t_w/2) or x > w - int(t_w/2) - 1:
                continue
            for y in range(self.window_y1, self.window_y2):
                if y < int(t_h/2) or y > h - int(t_h/2) - 1:
                    continue

                x1 = x - int(t_w/2)
                x2 = x + int(t_w/2) + 1
                y1 = y - int(t_h/2)
                y2 = y + int(t_h/2) + 1

                res = self.calc_ssd(x1,y1,x2,y2)
                if min_sum == -1:
                    min_sum = res
                if res < min_sum:
                    min_sum = res
                    min_sum_coord = (x1,y1,x2,y2)

        self.window_x1 = min_sum_coord[0]
        self.window_y1 = min_sum_coord[1]
        self.window_x2 = min_sum_coord[2]
        self.window_y2 = min_sum_coord[3]
        return min_sum_coord

    def get_face(self, img):
        self.img = img
        #x1,y1,x2,y2 = self.calc_ssd_full()
        x1,y1,x2,y2 = self.calc_cross_corr()
        new_img = self.img.copy().astype(np.uint8)
        new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(new_img, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 1)
        return new_img

    def get_face_crs_corr(self, img):
        self.img = img
        res = self.calc_cross_corr()
        return res

    def correlate(self, img1, img2, norm = False):

        if norm == True:
            summation = np.sum(img1 * img2)
            dinom = np.sqrt(np.sum(np.square(img1)) * np.sum(np.square(img2)))
            return summation/dinom

        return np.sum(img1*img2)



    def calc_cross_corr(self):
        h, w = self.img.shape
        t_h, t_w = self.template.shape
        template = self.template - np.mean(self.template)
        res = np.zeros(self.img.shape)

        for x in range(self.window_x1, self.window_x2):
            if x < int(t_w / 2) or x > w - int(t_w / 2) - 1:
                continue
            for y in range(self.window_y1, self.window_y2):
                if y < int(t_h / 2) or y > h - int(t_h / 2) - 1:
                    continue

                x1 = x - int(t_w / 2)
                x2 = x + int(t_w / 2) + 1
                y1 = y - int(t_h / 2)
                y2 = y + int(t_h / 2) + 1

                corr = self.correlate(self.img[y1:y2, x1:x2], template, norm=True)
                res[y][x] = corr

        res[res == 0] = np.min(res)
        res = np.interp(res, (res.min(), res.max()), (0, 1))
        y, x = np.unravel_index(np.argmax(res), res.shape)
        x1 = int(x - self.template.shape[1]/2)
        y1 = int(y - self.template.shape[0]/2)
        x2 = int(x + self.template.shape[1]/2)
        y2 = int(y + self.template.shape[0]/2)

        self.window_x1 = x1
        self.window_y1 = y1
        self.window_x2 = x2
        self.window_y2 = y2

        return (x1,y1,x2,y2)






if __name__ == '__main__':

    X1 = 58
    Y1 = 29
    X2 = 89
    Y2 = 66
    WINDOW_SIZE = 30

    sequence = load_seq("./data")

    temp = Template(X1, Y1, X2, Y2, sequence[0], WINDOW_SIZE)

    for idx, img in enumerate(sequence):
        # new_img = img.copy().astype(np.uint8)
        # new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR)
        # cv2.rectangle(new_img, (int(X1), int(Y1)), (int(X2), int(Y2)), (255, 0, 0), 1)
        # cv2.imwrite(f"outputs/{idx}.jpg", new_img)
        # break
        res = temp.get_face(img)
        #res = temp.get_face_crs_corr(img)

        cv2.imwrite(f"outputs/{idx}.jpg", res.astype(np.uint8))
