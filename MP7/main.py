import numpy as np

from utils import *

class Template:


    def __init__(self, x1, y1, x2, y2, img):

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        self.img = img
        self.template = self.img[y1:y2, x1:x2]


    def calc_ssd(self, x1, y1, x2, y2):

        patch = self.img[y1:y2, x1:x2]
        return np.sum(np.power(self.template - patch, 2))

    def calc_ssd_full(self):
        h, w = self.img.shape
        t_h, t_w = self.template.shape
        min_sum_coord = (0,0,0,0)
        min_sum = 10000
        for x in range(w):
            if x < t_w/2 or x > t_w - t_w/2:
                continue
            for y in range(h):
                if y < t_h/2 or y > t_h - t_h/2:
                    continue

            x1 = x - t_w/2
            x2 = x + t_w/2 + 1
            y1 = y - t_h/2
            y2 = y + t_h/2 + 1

            res = self.calc_ssd(x1,y1,x2,y2)
            if res < min_sum:
                min_sum = res
                min_sum_coord = (x1,y1,x2,y2)


        return min_sum_coord







    def calc_cross_corr(self, x1, y1, x2, y2):
        patch = self.img[y1:y2, x1:x2]
        np.correlate(self.img, patch)













if __name__ == '__main__':

    sequence = load_seq()