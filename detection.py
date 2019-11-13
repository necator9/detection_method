from __future__ import division

import threading

import cv2
from imutils import resize
import numpy as np
import Queue

import conf
from extentions import MultipleImagesFrame, TimeCounter
import detection_logging
import pinhole_camera_model as pcm

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(2, include_bias=True)
poly.fit([[1, 2, 3, 4, 5, 6]])

DETECTION_LOG = detection_logging.create_log("detection.log", "DETECTION THREAD")

# CLASSIFIER = pickle.load(open("/home/ivan/Downloads/clf_.pcl", "rb"))
# SCALER = pickle.load(open("/home/ivan/Downloads/scaler_.pcl", "rb"))

CLASSIFIER = pickle.load(open("/home/ivan/Downloads/clf_sel.pcl", "rb"))
SCALER = pickle.load(open("/home/ivan/Downloads/scaler_sel.pcl", "rb"))

# CLASSIFIER = pickle.load(open("/home/ivan/Downloads/clf_wo_ca.pcl", "rb"))
# SCALER = pickle.load(open("/home/ivan/Downloads/scaler_wo_ca.pcl", "rb"))


def init_pcm():
    f_l = 3.6
    w_ccd = 3.4509432207429906
    h_ccd = 1.937355215491415

    # return pcm.PinholeCameraModel(rw_angle=-conf.ANGLE, f_l=f_l, w_ccd=w_ccd, h_ccd=h_ccd,
    #                               img_res=conf.IMG_RES)

    return pcm.PinholeCameraModel(rw_angle=-conf.ANGLE, f_l=40, w_ccd=36, h_ccd=26.5,
                                  img_res=conf.IMG_RES)


PINHOLE_CAM = init_pcm()


class Detection(threading.Thread):
    def __init__(self, stop_ev, data_frame_q, draw_frame_q, orig_img_q):
        super(Detection, self).__init__(name="detection")
        self.stop_event = stop_ev
        self.img_name = str()

        self.data_frame_q = data_frame_q
        self.draw_frame_q = draw_frame_q
        self.orig_img_q = orig_img_q

        self.timer = TimeCounter("detection_timer")

    def run(self):
        DETECTION_LOG.info("Detection has started")
        prepare_img = PreprocessImg()
        while self.stop_event.is_set():

            self.timer.note_time()

            # Data frame containing detected objects
            frame = DataFrame()
            # Data structure to draw images on output
            draw = MultipleImagesFrame()

            try:
                frame.orig_img = self.orig_img_q.get(timeout=2)
            except Queue.Empty:
                DETECTION_LOG.warning("Timeout reached, no items can be received from orig_img_q")

                continue

            frame.orig_img, draw.mog_mask.data, draw.filtered.data, frame.filled = prepare_img.process(frame.orig_img)
            draw.bright_mask.data, draw.extent_split_mask.data = frame.calculate()

            self.data_frame_q.put(frame, block=True)
            self.draw_frame_q.put(draw, block=True)

            self.timer.get_time()


class CountorAreaTooSmall(Exception):
    def __init__(self):
        Exception.__init__(self, "CountorAreaTooSmall")


class InfiniteDistance(Exception):
    def __init__(self):
        Exception.__init__(self, "InfiniteDistance")


class MarginCrossed(Exception):
    def __init__(self):
        Exception.__init__(self, "MarginCrossed")


class ObjParams(object):
    def __init__(self, obj_id, cnt_ao):
        self.c_a_ao = cv2.contourArea(cnt_ao)
        # if self.c_a_ao / (conf.IMG_RES[0] * conf.IMG_RES[1]) < 0.0005:  # < 0.002:
        #     raise CountorAreaTooSmall

        self.base_rect_ao = self.x_ao, self.y_ao, self.w_ao, self.h_ao = cv2.boundingRect(cnt_ao)
        # if not self.check_margin(margin=conf.MARGIN, img_res=conf.IMG_RES):
        #     raise MarginCrossed

        self.dist_ao = PINHOLE_CAM.pixels_to_distance(-conf.HEIGHT, self.y_ao + self.h_ao)
        if self.dist_ao <= 0:
            raise InfiniteDistance

        self.obj_id = obj_id

        # Calculate geom parameters of an actual object
        self.h_w_ratio_ao = self.h_ao / self.w_ao
        self.extent_ao = self.c_a_ao / (self.w_ao * self.h_ao)
        # Generate virtual cuboid and calculate its geom parameters
        self.c_a_ro, self.x_ro, self.y_ro, self.w_ro, self.h_ro = 0, 0, 0, 0, 0
        self.rect_coef_ro = -1
        self.rect_coef_diff = -1
        self.rect_coef_ao = 0

        self.w_ao_rw = PINHOLE_CAM.get_width(-conf.HEIGHT, self.dist_ao, self.base_rect_ao)
        self.h_ao_rw = PINHOLE_CAM.get_height(-conf.HEIGHT, self.dist_ao, self.base_rect_ao)

        rect_area_ao_rw = self.w_ao_rw * self.h_ao_rw
        rect_area_ao = self.w_ao * self.h_ao
        # Find from proportion
        self.c_ao_rw = self.c_a_ao * rect_area_ao_rw / rect_area_ao

        self.base_status = bool()
        self.br_status = bool()
        self.gen_status = bool()

        self.br_cr_rects = [[0, 0, 0, 0]]
        self.br_cr_area = int()
        self.br_ratio = float()

        self.o_class = int()

    def detect(self):
        o_class = self.classify()

        if o_class != 0:
            self.base_status = True
        else:
            self.base_status = False

    def classify(self):
        # if self.dist_ao < 30 and 0 < self.h_ao_rw < 5 and 0 < self.w_ao_rw < 8:
        scaled_features = SCALER.transform([[self.w_ao_rw, self.h_ao_rw,  self.c_ao_rw,
                                            self.dist_ao, -conf.HEIGHT,  -conf.ANGLE]])
        self.o_class = int(CLASSIFIER.predict(poly.transform(scaled_features)))
        # else:
        #     self.o_class = 0

        return self.o_class

    def check_margin(self, margin, img_res):
        l_m, r_m = margin, img_res[0] - margin
        u_m, d_m = margin, img_res[1] - margin
        status = l_m < self.x_ao and self.x_ao + self.w_ao < r_m and u_m < self.y_ao and self.y_ao + self.h_ao < d_m

        return status


class PreprocessImg(object):
    def __init__(self):
        # self.mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True) # , varThreshold=16
        self.mog2 = cv2.createBackgroundSubtractorKNN(detectShadows=True, history=1500)
        self.f_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.clahe_adjust = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
        self.set_ratio_done = bool()

    def process(self, orig_img):
        orig_img = resize(orig_img, height=conf.IMG_RES[1])
        # Update processing resolution according to one after resize (i.e. not correct res. is chosen by user)
        self.set_ratio(orig_img)

        orig_img = self.clahe_adjust.apply(orig_img)
        # orig_img = cv2.blur(orig_img, (5, 5))

        mog_mask = self.mog2.apply(orig_img)
        # filtered_mask = mog_mask
        filtered_mask = cv2.morphologyEx(mog_mask, cv2.MORPH_OPEN, self.f_kernel)
        # filtered_mask = cv2.blur(filtered_mask, (3, 3))

        _, filled_mask = cv2.threshold(filtered_mask, 170, 255, cv2.THRESH_BINARY)

        # filtered_mask = cv2.morphologyEx(mog_mask, cv2.MORPH_OPEN, self.f_kernel)

        filled_mask = cv2.dilate(filled_mask, None, iterations=1)
        # filled_mask = cv2.blur(filtered_mask, (3, 3))
        # filled_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, self.f1_kernel)

        return orig_img, mog_mask, filtered_mask, filled_mask

    def set_ratio(self, img):
        if not self.set_ratio_done:
            self.set_ratio_done = True
            actual_w, actual_h = img.shape[:2][1], img.shape[:2][0]
            DETECTION_LOG.info("Processing resolution: {}x{}".format(actual_w, actual_h))

            if conf.IMG_RES[0] != actual_w or conf.IMG_RES[1] != actual_h:
                conf.IMG_RES[0], conf.IMG_RES[1] = actual_w, actual_h
                global PINHOLE_CAM
                PINHOLE_CAM = init_pcm()


class DataFrame(object):
    def __init__(self):
        self.orig_img = np.dtype('uint8')
        self.filled = np.dtype('uint8')

        self.base_frame_status = None  # Can be False/True/None type
        self.ex_frame_status = None  # Can be False/True/None type
        self.base_objects = list()
        self.base_contours = list()
        self.ex_objects = list()

        self.br_rects = list()

    def calculate(self):
        self.base_objects, self.base_contours = self.basic_process(self.filled)
        # bright_mask = self.calc_bright_coeff()
        #
        split_obj_i = [[i, obj] for i, obj in enumerate(self.base_objects)
                       if obj.extent_ao < 0.5 and 2 < obj.w_ao_rw < 5 and obj.dist_ao < 30 and obj.h_ao_rw < 3]

        split_idx = [it[0] for it in split_obj_i]
        split_obj = [it[1] for it in split_obj_i]

        if len(split_obj_i) > 0:
            split_mask = self.split_object(split_obj)
            self.ex_objects, _ = self.basic_process(split_mask)

            for ele in sorted(split_idx, reverse=True):
                del self.base_objects[ele]

            self.base_objects += self.ex_objects

        self.base_frame_status = any([obj.base_status for obj in self.base_objects])

        return np.dtype('uint8'), self.filled

    @staticmethod
    def basic_process(filled_mask):
        objects = list()

        _, contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for obj_id, contour in enumerate(contours):
            try:
                obj = ObjParams(obj_id, contour)
                obj.detect()
                objects.append(obj)
            except (CountorAreaTooSmall, InfiniteDistance, MarginCrossed):
                continue

        return objects, contours

    def split_object(self, obj_to_split):
        def make_split(bin_mask, fill=0.68, tail=0.25): # fill - amount of zeros in coloumn in percent ratio
            def calc_split_point(vector):
                last_zero_ind, percent = 0, 0.0
                zero_indexes, = np.where(vector == 0)

                if zero_indexes.size > 0:
                    last_zero_ind = zero_indexes.max()
                    percent = last_zero_ind / vector.size  # Get relative size of empty area by x axis

                return last_zero_ind, percent

            rows, coloumns = bin_mask.shape

            if coloumns >= rows:
                x_mask = np.asarray([0 if (np.bincount(i)[0] / i.size) >= fill else 1 for i in bin_mask.T], dtype='int8')
                x_mask_l, x_mask_r = x_mask[:x_mask.size // 2], x_mask[x_mask.size // 2:]

                front_ind, front_percent = calc_split_point(x_mask_l)
                opposite_ind, opposite_percent = calc_split_point(x_mask_r[::-1])

                split_data = [[front_ind, front_percent], [bin_mask.shape[1] - opposite_ind, opposite_percent]]

                split_data = zip(*split_data)
                max_sp_ind = split_data[1].index(max(split_data[1]))

                if split_data[1][max_sp_ind] > tail:
                    ind = split_data[0][max_sp_ind]
                    bin_mask[:, ind] = 0

            return bin_mask

        # ex_filled_mask = np.zeros((conf.RESIZE_TO[1], conf.RESIZE_TO[0]), np.uint8) # create minimal image
        split_img = np.zeros((conf.IMG_RES[1], conf.IMG_RES[0]), np.uint8)

        for obj in obj_to_split:
            x, y, w, h = obj.base_rect_ao
            split_mask = make_split(self.filled[y:y + h, x:x + w])
            split_img[y:y+h, x:x + w] = split_mask[:, :]

        return split_img
