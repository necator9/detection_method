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
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(6, include_bias=True)

DETECTION_LOG = detection_logging.create_log("detection.log", "DETECTION THREAD")

CLASSIFIER = pickle.load(open("/home/ivan/Downloads/classifier.pcl", "rb"))

PINHOLE_CAM = pcm.PinholeCameraModel(rw_angle=-conf.ANGLE, f_l=40, w_ccd=36, h_ccd=26.5,
                                     img_res=conf.RESIZE_TO)


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
            draw.bright_mask.data, draw.extent_split_mask.data = frame.calculate

            try:
                self.data_frame_q.put_nowait(frame)
            except Queue.Full:
                DETECTION_LOG.error("Data queue is full. Queue size: {}".format(self.data_frame_q.qsize()))

            try:
                self.draw_frame_q.put_nowait(draw)
            except Queue.Full:
                DETECTION_LOG.error("Draw queue is full. Queue size: {}".format(self.data_frame_q.qsize()))

            DETECTION_LOG.debug("Detection iteration performed")

            self.timer.get_time()


class ObjParams(object):
    def __init__(self, obj_id, cnt_ao):
        self.obj_id = obj_id

        # Calculate geom parameters of an actual object
        self.c_a_ao = cv2.contourArea(cnt_ao)
        self.base_rect_ao = self.x_ao, self.y_ao, self.w_ao, self.h_ao = cv2.boundingRect(cnt_ao)
        # Define y-coord as a middle of b.r.
        # self.y_ao = self.y_ao + self.h_ao / 2
        self.h_w_ratio_ao = float(self.h_ao) / self.w_ao
        self.rect_coef_ao = self.calc_rect_coef(self.c_a_ao, self.h_ao, self.w_ao, self.h_w_ratio_ao)

        self.extent_ao = float(self.c_a_ao) / (self.w_ao * self.h_ao)

        # Estimate distance of the actual object
        # self.dist_ao = PRED_DIST_F(self.y_ao)
        self.dist_ao = PINHOLE_CAM.pixels_to_distance(-conf.HEIGHT, self.y_ao + self.h_ao)
        # def pixels_to_distance(n=10, h=10., r=0, Sh_px=480., FL=35., Sh=26.5):

        # Generate virtual cuboid and calculate its geom parameters
        if self.dist_ao > 0:
            self.c_a_ro, self.x_ro, self.y_ro, self.w_ro, self.h_ro = 0, 0, 0, 0, 0
            self.rect_coef_ro = -1
            self.rect_coef_diff = -1
            # self.c_a_ro, self.x_ro, self.y_ro, self.w_ro, self.h_ro = PINHOLE_CAM.get_ref_val(self.dist_ao)
            # self.rect_coef_ro = self.calc_rect_coef(self.c_a_ro, self.h_ro, self.w_ro, float(self.h_ro) / self.w_ro)
            # self.rect_coef_diff = self.rect_coef_ro / self.rect_coef_ao

            self.w_ao_rw = PINHOLE_CAM.get_width(-conf.HEIGHT, self.dist_ao, self.base_rect_ao)
            self.h_ao_rw = PINHOLE_CAM.get_height(-conf.HEIGHT, self.dist_ao, self.base_rect_ao)
            rect_area_ao_rw = self.w_ao_rw * self.h_ao_rw
            rect_area_ao = self.w_ao * self.h_ao
            # Find from proportion
            self.c_ao_rw = self.c_a_ao * rect_area_ao_rw / rect_area_ao
        else:
            self.c_a_ro, self.x_ro, self.y_ro, self.w_ro, self.h_ro = 0, 0, 0, 0, 0
            self.rect_coef_ro = -1
            self.rect_coef_diff = -1

            self.w_ao_rw = float()
            self.h_ao_rw = float()
            self.c_ao_rw = float()

        self.base_status = bool()
        self.br_status = bool()
        self.gen_status = bool()

        self.br_cr_rects = [[0, 0, 0, 0]]
        self.br_cr_area = int()
        self.br_ratio = float()

        self.o_class = int()

    @staticmethod
    def scale_param(reg_f, val, pred_dist, ref_dist=6):
        ref_val = reg_f(ref_dist)
        pred_val = reg_f(pred_dist)
        scaled_val = val * (ref_val / pred_val)

        return scaled_val

    def process_obj(self):
        self.detect()

    def calc_rect_coef_old(self, h, w, c, k):
        '''
        y --- number of pixel till left low corner
        '''
        return round(c * k * ((h ** 2 + 2 * h * w + w ** 2) / (h * w * 4.0)), 3)

    @staticmethod
    def calc_rect_coef(c_a, h, w, h_w_ratio):
        # k = 1 if (h_w_ratio > 0.7) else -1
        k = 1
        rect_coef = c_a * k * ((h ** 2 + 2 * h * w + w ** 2) / (h * w * 4.0))

        return round(rect_coef, 3)
    # TODO Transfer into dataframe class

    def detect(self):
        self.classify()
        is_rect_coeff_belongs = self.check_rect_coeff(self.rect_coef_diff)
        is_extent_belongs = self.check_extent(self.extent_ao)
        is_margin_crossed = self.check_margin(self.base_rect_ao[0], self.base_rect_ao[2])

        #if is_rect_coeff_belongs and not is_margin_crossed and is_extent_belongs:
        if is_rect_coeff_belongs:
            self.base_status = True
        else:
            self.base_status = False

    def classify(self):
        if self.dist_ao < 30 and self.rect_coef_diff < 3:
            self.o_class = int(CLASSIFIER.predict(poly.fit_transform([[self.rect_coef_diff, self.h_w_ratio_ao]])))
        else:
            self.o_class = 3


    @staticmethod
    def check_rect_coeff(coeff):

        return conf.COEFF_RANGE[0] < coeff < conf.COEFF_RANGE[1]

    @staticmethod
    def check_extent(extent):

        return extent > conf.EXTENT_THRESHOLD

    @staticmethod
    def check_margin(x, w):
        x_margin = conf.X_MARGIN
        x_left = not (x_margin < x < conf.RESIZE_TO[0] - x_margin)
        x_right = not (x_margin < x + w < conf.RESIZE_TO[0] - x_margin)

        return x_left or x_right


class PreprocessImg(object):
    def __init__(self):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.f_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, conf.F_KERNEL_SIZE)
        # self.f1_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 1))
        self.clahe_adjust = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
        self.set_ratio_done = bool()

    def process(self, orig_img):
        orig_img = resize(orig_img, height=conf.RESIZE_TO[1])
        # Update processing resolution according to one after resize (i.e. not correct res. is chosen by user)
        self.set_ratio(orig_img)

        orig_img = self.clahe_adjust.apply(orig_img)

        mog_mask = self.mog2.apply(orig_img)
        _, mog_mask = cv2.threshold(mog_mask, 127, 255, cv2.THRESH_BINARY)

        filtered_mask = cv2.morphologyEx(mog_mask, cv2.MORPH_OPEN, self.f_kernel)
        # filled_mask = cv2.dilate(filtered_mask, None, iterations=conf.DILATE_ITERATIONS)
        filled_mask = filtered_mask

        return orig_img, mog_mask, filtered_mask, filled_mask

    def set_ratio(self, img):
        if not self.set_ratio_done:
            self.set_ratio_done = True
            actual_w, actual_h = img.shape[:2][1], img.shape[:2][0]
            DETECTION_LOG.info("Processing resolution: {}x{}".format(actual_w, actual_h))

            if conf.RESIZE_TO[0] != actual_w or conf.RESIZE_TO[1] != actual_h:
                conf.RESIZE_TO[0], conf.RESIZE_TO[1] = actual_w, actual_h


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

    @property
    def calculate(self):
        self.base_objects, self.base_contours = self.basic_process(self.filled)
        bright_mask = self.calc_bright_coeff()
        #
        for obj in self.base_objects:
            is_extent = obj.extent_ao < 0.5
            is_width = 3 < obj.w_ao_rw < 5
            is_distance = obj.dist_ao < 30

            if is_extent and is_width and is_distance:
                self.filled = self.split_object()
                self.ex_objects, _ = self.basic_process(self.filled)
                self.base_objects = self.ex_objects

        self.calc_bright_coeff()

        self.detect()

        return bright_mask, self.filled

    @staticmethod
    def basic_process(filled_mask):
        objects = list()

        _, contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for obj_id, contour in enumerate(contours):
            obj = ObjParams(obj_id, contour)
            obj.process_obj()
            objects.append(obj)

        return objects, contours

    # TODO remake to crop and analyze only one object, only problem object should be considered further
    def split_object(self):
        def make_split(bin_mask, fill=0.68, tail=0.25): # fill - amount of zeros in coloumn in percent ratio
            def calc_split_point(vector):
                last_zero_ind, percent = 0, 0.0
                zero_indexes, = np.where(vector == 0)

                if zero_indexes.size > 0:
                    last_zero_ind = zero_indexes.max()
                    percent = last_zero_ind / float(vector.size)  # Get relative size of empty area by x axis

                return last_zero_ind, percent

            rows, coloumns = bin_mask.shape

            if coloumns >= rows:
                x_mask = np.asarray([0 if (np.bincount(i)[0] / float(i.size)) >= fill else 1 for i in bin_mask.T], dtype='int8')
                #print x_mask
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

        for obj in self.base_objects:
            is_extent = obj.extent_ao < 0.5
            is_width = obj.w_ao_rw > 3
            is_dist = obj.dist_ao < 30

            if is_extent and is_width and is_dist:
                x, y, w, h = obj.base_rect_ao
                split_mask = make_split(self.filled[y:y + h, x:x + w])
                self.filled[y:y+h, x:x + w] = split_mask[:, :]
                #ex_filled_mask[y:y+h, x:x + w] = self.filled_mask[y:y+h, x:x + w]
#                cv2.line(ex_filled_mask, (x + int(w / 2), 0), (x + int(w / 2), conf.RESIZE_TO[1]), (0, 0, 0), 3)

        return self.filled

    def calc_bright_coeff(self):
        # if len(self.base_objects) > 0:
        #     brightness_mask = np.zeros((conf.RESIZE_TO[1], conf.RESIZE_TO[0]), np.uint8)
        #
        #     brightness_mask[np.where(self.orig_img > 265)] = [255]   # Chose later appropriate values
        #     _, contours, _ = cv2.findContours(brightness_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     self.br_rects = [cv2.boundingRect(contour) for contour in contours]
        #
        for obj in self.base_objects:
        #         obj.br_cr_rects = [self.intersection(obj.base_rect_ao, br_rect) for br_rect in self.br_rects]
        #         obj.br_cr_area = sum([rect[2] * rect[3] for rect in obj.br_cr_rects])
            segment = self.orig_img[obj.y_ao: obj.y_ao + obj.h_ao, obj.x_ao: obj.x_ao + obj.w_ao]
            obj.br_ratio = segment.mean()
        #         obj.br_status = obj.br_ratio > conf.BRIGHTNESS_THRESHOLD
        #
        # else:
        #     brightness_mask = np.dtype('uint8')

        brightness_mask = np.dtype('uint8')

        return brightness_mask

    def detect(self):
        self.base_frame_status = self.take_frame_status(self.base_objects)
        self.ex_frame_status = self.take_frame_status(self.ex_objects)

    @staticmethod
    def take_frame_status(objects):
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














