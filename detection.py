import threading
import cv2
import numpy as np
import queue

import conf
from extentions import TimeCounter
import logging
import pinhole_camera_model as pcm
from pre_processing import PreprocessImg
import extentions

import pickle

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(2, include_bias=True)
poly.fit([[1, 2, 3, 4, 5, 6]])

logger = logging.getLogger('detect.detect')

CLASSIFIER = pickle.load(open(conf.CLF_PATH, "rb"))
# SCALER = pickle.load(open(conf.SCALER_PATH, "rb"))

PINHOLE_CAM = pcm.PinholeCameraModel(rw_angle=-conf.ANGLE, f_l=conf.FL, w_ccd=conf.WCCD, h_ccd=conf.HCCD,
                                     img_res=conf.RES)


class Detection(threading.Thread):
    def __init__(self, stop_ev, data_frame_q, orig_img_q):
        super(Detection, self).__init__(name="detection")
        self.stop_event = stop_ev
        self.img_name = str()

        self.data_frame_q = data_frame_q
        self.orig_img_q = orig_img_q
        self.fe = pcm.FeatureExtractor(-conf.ANGLE, -conf.HEIGHT, conf.RES, (conf.WCCD, conf.HCCD), conf.FL)

        self.timer = TimeCounter("detection_timer")

    def run(self):
        logger.info("Detection has started")
        preprocessing = PreprocessImg()
        steps = dict()

        while self.stop_event.is_set():
            self.timer.note_time()

            # Data frame containing detected objects
            frame = DataFrame()

            try:
                orig_img, img_name = self.orig_img_q.get(timeout=2)
            except queue.Empty:
                logger.warning("Timeout reached, no items can be received from orig_img_q")

                continue

            steps['resized_orig'], steps['mask'], steps['filtered'], steps['filled'] = preprocessing.apply(orig_img)

            frame.calculate(steps['filled'])

            # fr = Frame(steps['filled'], self.fe)
            # fr.process()

            self.data_frame_q.put(frame, block=True)

            if conf.WRITE_IMG:
                extentions.write_steps(steps, frame, img_name)

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
        # if conf.CNT_AREA_FILTERING > 0:
        #     if self.c_a_ao / (conf.RES[0] * conf.RES[1]) < conf.CNT_AREA_FILTERING:
        #         raise CountorAreaTooSmall

        self.base_rect_ao = self.x_ao, self.y_ao, self.w_ao, self.h_ao = cv2.boundingRect(cnt_ao)
        # if conf.MARGIN > 0:
        #     if not self.check_margin(margin=conf.MARGIN, img_res=conf.RES):
        #         raise MarginCrossed

        self.dist_ao = PINHOLE_CAM.pixels_to_distance(-conf.HEIGHT, self.y_ao + self.h_ao)
        # if self.dist_ao <= 0:
        #     raise InfiniteDistance

        # if conf.MAX_DISTANCE > 0:
        #     if self.dist_ao > conf.MAX_DISTANCE:
        #         raise InfiniteDistance

        self.obj_id = obj_id

        # Calculate geom parameters of an actual object
        self.h_w_ratio_ao = self.h_ao / self.w_ao
        self.extent_ao = self.c_a_ao / (self.w_ao * self.h_ao)

        self.w_ao_rw = PINHOLE_CAM.get_width(-conf.HEIGHT, self.dist_ao, self.base_rect_ao)
        self.h_ao_rw = PINHOLE_CAM.get_height(-conf.HEIGHT, self.dist_ao, self.base_rect_ao)

        rect_area_ao_rw = self.w_ao_rw * self.h_ao_rw
        rect_area_ao = self.w_ao * self.h_ao

        self.c_ao_rw = self.c_a_ao * rect_area_ao_rw / rect_area_ao

        self.o_class = self.classify()
        self.o_class_nm = conf.o_class_mapping[self.o_class]
        self.binary_status = self.o_class > 0

    def classify(self):
        # if  0 < self.h_ao_rw < 5 and 0 < self.w_ao_rw < 8:

        # scaled_features = [[self.w_ao_rw, self.h_ao_rw,  #self.c_ao_rw,
        #                     self.dist_ao, -conf.HEIGHT,  -conf.ANGLE]]
        feature_vector = [[self.w_ao_rw, self.h_ao_rw,  self.c_ao_rw, self.dist_ao, -conf.HEIGHT,  -conf.ANGLE]]  #
        logger.debug('Feature vector native: {}'.format(feature_vector))
        # scaled_features = SCALER.transform(feature_vector)
        # logger.debug('Scaled features: {}'.format(scaled_features))
        scaled_features = feature_vector
        poly_features = poly.transform(scaled_features)
        # logger.debug('Poly features: {}'.format(poly_features))
        self.o_class = int(CLASSIFIER.predict(poly_features))


        # else:
        #     self.o_class = 0

        return self.o_class

    def check_margin(self, margin, img_res):
        l_m, r_m = margin, img_res[0] - margin
        u_m, d_m = margin, img_res[1] - margin
        status = l_m < self.x_ao and self.x_ao + self.w_ao < r_m and u_m < self.y_ao and self.y_ao + self.h_ao < d_m

        return status


class Frame(object):
    def __init__(self, mask, fe_ext):
        self.mask = mask
        self.fe_ext = fe_ext

    def find_basic_params(self):
        cnts, _ = cv2.findContours(self.mask, mode=0, method=1)
        c_areas = [cv2.contourArea(cnt) for cnt in cnts]
        b_rects = [cv2.boundingRect(b_r) for b_r in cnts]

        return np.asarray(c_areas), np.asarray(b_rects)

    def process(self):
        c_areas, b_rects = self.find_basic_params()
        if len(c_areas) > 0:  # Calculate features when something is present on the mask
            z_est, x_est, width_est, height_est, rw_ca_est = self.fe_ext.extract_features(c_areas, b_rects)
            feature_vector = np.stack((width_est, height_est, rw_ca_est, z_est,
                                                     np.ones(z_est.shape) * -conf.HEIGHT,
                                                     np.ones(z_est.shape) * -conf.ANGLE), axis=1)
            logger.debug('Feature vector new: {}'.format(feature_vector))
            poly_features = poly.transform(feature_vector)
            # logger.debug('Poly features: {}'.format(poly_features))
            o_class = CLASSIFIER.predict(poly_features)
            logger.debug('New method: {}'.format(o_class))


class DataFrame(object):
    def __init__(self):
        self.base_frame_status = None  # Can be False/True/None type
        self.base_objects = list()
        self.base_contours = list()
        self.ex_objects = list()

        self.br_rects = list()

    def calculate(self, filled):
        self.base_objects, self.base_contours = self.basic_process(filled)
        logger.debug('Native method: {}'.format([obj.o_class for obj in self.base_objects]))

        # split_obj_i = [[i, obj] for i, obj in enumerate(self.base_objects)
        #                if obj.extent_ao < 0.5 and 2 < obj.w_ao_rw < 5 and obj.dist_ao < 30 and obj.h_ao_rw < 3]
        #
        # split_idx = [it[0] for it in split_obj_i]
        # split_obj = [it[1] for it in split_obj_i]
        #
        # if len(split_obj_i) > 0:
        #     split_mask = self.split_object(split_obj, filled)
        #     self.ex_objects, _ = self.basic_process(split_mask)
        #
        #     for ele in sorted(split_idx, reverse=True):
        #         del self.base_objects[ele]
        #
        #     self.base_objects += self.ex_objects

        self.base_frame_status = any([obj.binary_status for obj in self.base_objects])

        return np.dtype('uint8'), filled

    def basic_process(self, filled_mask):
        objects = list()

        # _, contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours,  _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        for obj_id, contour in enumerate(contours):
            try:
                objects.append(ObjParams(obj_id, contour))
            except (CountorAreaTooSmall, InfiniteDistance, MarginCrossed):
                continue

        return objects, contours

    def split_object(self, obj_to_split, filled):
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
                x_mask = np.asarray([0 if (np.count_nonzero(i == 0) / i.size) >= fill else 1 for i in bin_mask.T],
                                    dtype='int8')
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

        split_img = np.zeros((conf.RES[1], conf.RES[0]), np.uint8)

        for obj in obj_to_split:
            x, y, w, h = obj.base_rect_ao
            split_mask = make_split(filled[y:y + h, x:x + w])
            split_img[y:y+h, x:x + w] = split_mask[:, :]

        return split_img
