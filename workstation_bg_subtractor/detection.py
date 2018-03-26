import threading
import time
import glob
import os
import copy

import cv2
from imutils import resize
import numpy as np
import conf
import Queue

from extentions import DrawImgStructure
import detection_logging

DETECTION_LOG = detection_logging.create_log("detection.log", "DETECTION THREAD")


class Detection(threading.Thread):
    def __init__(self, stop_ev, data_frame_q, draw_frame_q):
        super(Detection, self).__init__()
        self.stop_event = stop_ev
        self.img_name = str()

        self.data_frame_q = data_frame_q
        self.draw_frame_q = draw_frame_q

    # Main thread routine
    def run(self):
        DETECTION_LOG.info("Detection has started")
        img_fr = PreProcess()

        while self.stop_event.is_set():
            if conf.IMG_BUFF.processed or not conf.IMG_BUFF.inserted:
                DETECTION_LOG.info("Waiting for a new frame. Buff has read - {}; Image was passed into buffer - {}"
                                   .format(conf.IMG_BUFF.processed, conf.IMG_BUFF.inserted))
                time.sleep(0.1)

                continue

            data_frame = DataFrame()

            data_frame.orig_img = copy.copy(conf.IMG_BUFF.image)
            conf.IMG_BUFF.processed = True
            DETECTION_LOG.debug("image id {}".format(conf.IMG_BUFF.id))
            # data_frame.__init__ = DataFrame.__init__

            draw_frame = DrawImgStructure()
            draw_frame.mog_mask.data, draw_frame.filtered_mask.data = img_fr.process(data_frame)
            draw_frame.bright_mask.data, draw_frame.extent_split_mask.data = data_frame.calculate()

            try:
                self.data_frame_q.put_nowait(data_frame)
            except Queue.Full:
                DETECTION_LOG.error("Data queue is full. Queue size: {}".format(self.data_frame_q.qsize()))

            try:
                self.draw_frame_q.put_nowait(draw_frame)
            except Queue.Full:
                DETECTION_LOG.error("Draw queue is full. Queue size: {}".format(self.data_frame_q.qsize()))

            DETECTION_LOG.debug("Detection iteration number {}".format(conf.COUNTER))
            conf.COUNTER += 1

        self.quit()

    # Stop and quit the thread operation.
    def quit(self):
        DETECTION_LOG.info("Exiting the Detection thread...")
        self.stop_event.clear()


class ObjParams(object):
    def __init__(self, obj_id=int()):
        self.obj_id = obj_id
        self.base_status = bool()
        self.br_status = bool()
        self.gen_status = bool()

        self.h_w_ratio = float()
        self.base_rect = tuple()

        self.br_cr_rects = [[0, 0, 0, 0]]
        self.br_cr_area = int()
        self.br_ratio = float()

        self.contour_area = float()
        self.rect_coef = float()
        self.extent = float()
        self.rect_area = float()
        self.rect_perimeter = float()

    def process_obj(self, contour):
        self.calc_params(contour)
        self.detect()

    def calc_params(self, contour):
        self.contour_area = cv2.contourArea(contour)
        self.base_rect = x, y, w, h = cv2.boundingRect(contour)
        self.h_w_ratio = round(float(h) / w, 2)
        self.rect_area = w * h
        self.rect_perimeter = 2 * (h + w)
        self.extent = round(float(self.contour_area) / self.rect_area, 2)

        if float(h) / w > 0.7:
            k = 1.0
        else:
            k = -1.0

        # coeff(k * ((2.0 * w * h + 2 * w ** 2 + h) / w), 1) # Kirill suggestion
        self.rect_coef = round(self.contour_area * k * ((h ** 2 + 2 * h * w + w ** 2) /
                                                        (h * w * 4.0)), 3)

    # TODO Transfer into dataframe class

    def detect(self):
        is_rect_coeff_belongs = self.check_rect_coeff(self.rect_coef)
        is_extent_belongs = self.check_extent(self.extent)
        is_margin_crossed = self.check_margin(self.base_rect[0], self.base_rect[2])

        if is_rect_coeff_belongs and not is_margin_crossed and is_extent_belongs:
            self.base_status = True
        else:
            self.base_status = False

    @staticmethod
    def check_rect_coeff(coeff):

        return conf.COEFF_RANGE[0] < coeff < conf.COEFF_RANGE[1]

    @staticmethod
    def check_extent(extent):

        return extent > conf.EXTENT_THRESHOLD

    @staticmethod
    def check_margin(x, w):
        x_margin = conf.X_MARGIN
        x_left = not (x_margin < x < conf.PROC_IMG_RES[0] - x_margin)
        x_right = not (x_margin < x + w < conf.PROC_IMG_RES[0] - x_margin)

        return x_left or x_right


class PreProcess(object):
    def __init__(self):
        try:  # Handle CV2 version, for 3.x version
            self.__mog = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        except AttributeError:  # Handle CV2 version, for 2.4.x version
            self.__mog = cv2.BackgroundSubtractorMOG()

        self.__filtering_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, conf.F_KERNEL_SIZE)
        self.set_ratio_done = bool()

    def process(self, d_frame):
        orig_img = resize(d_frame.orig_img, width=conf.PROC_IMG_RES[0], height=conf.PROC_IMG_RES[1])
        self.set_ratio(orig_img)

        mog_mask = self.__mog.apply(orig_img)
        _, mog_mask = cv2.threshold(mog_mask, 127, 255, cv2.THRESH_BINARY)

        if conf.F_KERNEL_SIZE[0] > 0 and conf.F_KERNEL_SIZE[1] > 0:
            filtered_mask = cv2.morphologyEx(mog_mask, cv2.MORPH_OPEN, self.__filtering_kernel)
        else:
            filtered_mask = copy.copy(mog_mask)

        filled_mask = cv2.dilate(filtered_mask, None, iterations=conf.DILATE_ITERATIONS)

        d_frame.orig_img = orig_img
        d_frame.filled_mask = filled_mask

        return mog_mask, filtered_mask

    def set_ratio(self, img):
        if not self.set_ratio_done:
            self.set_ratio_done = True
            actual_w, actual_h = img.shape[:2][1], img.shape[:2][0]

            if conf.PROC_IMG_RES[0] != actual_w or conf.PROC_IMG_RES[1] != actual_h:
                conf.PROC_IMG_RES[0] = actual_w
                conf.PROC_IMG_RES[1] = actual_h


class DataFrame(object):
    def __init__(self):
        self.orig_img = np.dtype('uint8')
        self.filled_mask = np.dtype('uint8')

        self.base_frame_status = None  # Can be False/True/None type
        self.ex_frame_status = None  # Can be False/True/None type
        self.base_objects = list()
        self.ex_objects = list()
        self.base_contours = list()

        self.br_rects = list()

    def calculate(self):
        self.base_objects, self.base_contours = self.__basic_process(self.filled_mask)
        bright_mask = self.__brightness_process(self.base_objects)
        ex_filled_mask = self.__extent_split_process()
        self.ex_objects, _ = self.__basic_process(ex_filled_mask)
        _ = self.__brightness_process(self.ex_objects)

        self.detect()

        return bright_mask, ex_filled_mask

    @staticmethod
    def __basic_process(filled_mask):
        objects = list()
        _, contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for obj_id, contour in enumerate(contours):
            obj = ObjParams(obj_id)
            obj.process_obj(contour)
            objects.append(obj)

        return objects, contours

    # TODO remake to crop and analyze only one object, only problem object should be considered further
    def __extent_split_process(self):
        ex_filled_mask = np.zeros((conf.PROC_IMG_RES[1], conf.PROC_IMG_RES[0]), np.uint8) # create minimal image
        for obj in self.base_objects:
            is_extent = obj.extent < 0.6
            is_rect_coeff = -20000 < obj.rect_coef < -10000  # Try to reduce to -5000 or so

            if is_extent and is_rect_coeff and not obj.base_status and not obj.br_status:
                x, y, w, h = obj.base_rect

                ex_filled_mask[y:y+h, x:x + w] = self.filled_mask[y:y+h, x:x + w]
                cv2.line(ex_filled_mask, (x + int(w / 2), 0), (x + int(w / 2), conf.PROC_IMG_RES[1]), (0, 0, 0), 3)

        return ex_filled_mask

    def __brightness_process(self, objects):
        brightness_mask = np.zeros((conf.PROC_IMG_RES[1], conf.PROC_IMG_RES[0], 3), np.uint8)
        # if len(self.base_objects) > 0:  # keep it for optimization for BBB
        brightness_mask[np.where((self.orig_img > [220, 220, 220]).all(axis=2))] = [255]
        brightness_mask = cv2.cvtColor(brightness_mask, cv2.COLOR_BGR2GRAY)
        _, contours, _ = cv2.findContours(brightness_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        brightness_mask = cv2.cvtColor(brightness_mask, cv2.COLOR_GRAY2BGR)

        self.br_rects = [cv2.boundingRect(contour) for contour in contours]

        for obj in objects:
            # if obj.base_status: # keep it for optimization for BBB
            obj.br_cr_rects = [self.intersection(obj.base_rect, br_rect) for br_rect in self.br_rects]
            obj.br_cr_area = sum([rect[2] * rect[3] for rect in obj.br_cr_rects])
            obj.br_ratio = round(float(obj.br_cr_area) / obj.rect_area, 3)
            obj.br_status = obj.br_ratio > conf.BRIGHTNESS_THRESHOLD

        return brightness_mask

    def detect(self):
        self.base_frame_status = self.__take_frame_status(self.base_objects)
        self.ex_frame_status = self.__take_frame_status(self.ex_objects)

    @staticmethod
    def __take_frame_status(objects):
        status_arr = list()
        for obj in objects:
            obj.gen_status = obj.base_status and not obj.br_status
            status_arr.append(obj.gen_status)
        return any(status_arr)

    @staticmethod
    def intersection(area_1, area_2):
        x = max(area_1[0], area_2[0])
        y = max(area_1[1], area_2[1])
        w = min(area_1[0] + area_1[2], area_2[0] + area_2[2]) - x
        h = min(area_1[1] + area_1[3], area_2[1] + area_2[3]) - y

        if w < 0 or h < 0:
            return [0, 0, 0, 0]

        return [x, y, w, h]














