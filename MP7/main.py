import cv2
import numpy as np
from scipy.signal import correlate2d



from tqdm import tqdm

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
        for x in range(self.window_x1+self.window_size, self.window_x2-self.window_size):
            if x < int(t_w/2) or x > w - int(t_w/2) - 1:
                continue
            for y in range(self.window_y1+self.window_size, self.window_y2-self.window_size):
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
        # self.window_x1 = max(min_sum_coord[0] - self.window_size, 0)
        # self.window_y1 = max(min_sum_coord[1] - self.window_size, 0)
        # self.window_x2 = min(min_sum_coord[2] + self.window_size, self.img.shape[1])
        # self.window_y2 = min(min_sum_coord[3] + self.window_size, self.img.shape[0])

        return min_sum_coord

    def get_face(self, img, method = "ssd"):
        self.img = img
        if method == "ssd":
            x1,y1,x2,y2 = self.calc_ssd_full()
        elif method == "cross_corr":
            x1,y1,x2,y2 = self.calc_cross_corr()
        elif method == "cross_corr_norm":
            x1, y1, x2, y2 = self.calc_cross_corr(norm=True)
        else:
            raise Exception("Method not in the list!")
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



    def calc_cross_corr(self, norm = False):
        h, w = self.img.shape
        t_h, t_w = self.template.shape
        template = self.template - np.mean(self.template)
        res = np.zeros(self.img.shape)

        for x in range(self.window_x1+self.window_size, self.window_x2-self.window_size):
            if x < int(t_w / 2) or x > w - int(t_w / 2) - 1:
                continue
            for y in range(self.window_y1+self.window_size, self.window_y2-self.window_size):
                if y < int(t_h / 2) or y > h - int(t_h / 2) - 1:
                    continue

                x1 = x - int(t_w / 2)
                x2 = x + int(t_w / 2) + 1
                y1 = y - int(t_h / 2)
                y2 = y + int(t_h / 2) + 1

                corr = self.correlate(self.img[y1:y2, x1:x2], template, norm=norm)
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

def convert_to_video(image_folder):
    video_name = f'{image_folder}/video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images = sorted(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 7, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def clean_jpg_files(image_folder):
    test = os.listdir(image_folder)

    for item in test:
        if item.endswith(".jpg"):
            os.remove(os.path.join(image_folder, item))






if __name__ == '__main__':

    X1 = 58
    Y1 = 29
    X2 = 89
    Y2 = 66
    MARGIN_SIZE = 40


    sequence = load_seq("./data")


    for idx, img in tqdm(enumerate(sequence)):
        temp = Template(X1, Y1, X2, Y2, sequence[0], MARGIN_SIZE)
        res = temp.get_face(img.copy())
        cv2.imwrite(f"output_ssd/img{idx:04d}.jpg", res.astype(np.uint8))

    for idx, img in tqdm(enumerate(sequence)):
        temp = Template(X1, Y1, X2, Y2, sequence[0], MARGIN_SIZE)
        res = temp.get_face(img.copy(), "cross_corr")
        cv2.imwrite(f"output_corr/img{idx:04d}.jpg", res.astype(np.uint8))
    for idx, img in tqdm(enumerate(sequence)):
        temp = Template(X1, Y1, X2, Y2, sequence[0], MARGIN_SIZE)
        res = temp.get_face(img.copy(), "cross_corr_norm")
        cv2.imwrite(f"output_norm_corr/img{idx:04d}.jpg", res.astype(np.uint8))

    convert_to_video("output_ssd")
    convert_to_video("output_corr")
    convert_to_video("output_norm_corr")

    clean_jpg_files("output_ssd")
    clean_jpg_files("output_corr")
    clean_jpg_files("output_norm_corr")

