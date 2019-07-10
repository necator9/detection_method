import threading
import copy

import cv2
from imutils import resize
import numpy as np
import Queue

import conf
from extentions import MultipleImagesFrame, TimeCounter
import detection_logging
import pinhole_camera_model as pcm

DETECTION_LOG = detection_logging.create_log("detection.log", "DETECTION THREAD")


pinhole_cam = pcm.PinholeCameraModel()
pred_dist_f = pinhole_cam.init_y_regress()


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
        img_fr = PreProcess()
        while self.stop_event.is_set():

            self.timer.note_time()

            data_frame = DataFrame()

            try:
                data_frame.orig_img = self.orig_img_q.get(timeout=2)
            except Queue.Empty:
                DETECTION_LOG.warning("Timeout reached, no items can be received from orig_img_q")

                continue

            draw_frame = MultipleImagesFrame()
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

            DETECTION_LOG.debug("Detection iteration performed")

            self.timer.get_time()

        # self.quit()

    # Stop and quit the thread operation.
    def quit(self):
        DETECTION_LOG.info("Exiting the Detection thread...")
        self.stop_event.clear()


class ObjParams(object):
    def __init__(self, obj_id, cnt_ao):
        self.obj_id = obj_id

        # Calculate geom parameters of an actual object
        self.c_a_ao = cv2.contourArea(cnt_ao)
        self.base_rect_ao = self.x_ao, self.y_ao, self.w_ao, self.h_ao = cv2.boundingRect(cnt_ao)
        # Define y-coord as a middle of b.r.
        self.y_ao = self.y_ao + self.h_ao / 2
        self.h_w_ratio_ao = float(self.h_ao) / self.w_ao
        self.rect_coef_ao = self.calc_rect_coef(self.c_a_ao, self.h_ao, self.w_ao, self.h_w_ratio_ao)

        self.extent_ao = float(self.c_a_ao) / (self.w_ao * self.h_ao)

        # Estimate distance of the actual object
        self.dist_ao = pred_dist_f(self.y_ao)

        # Generate virtual cuboid and calculate its geom parameters
        self.c_a_ro, self.x_ro, self.y_ro, self.w_ro, self.h_ro = pinhole_cam.get_ref_val(self.dist_ao)
        self.rect_coef_ro = self.calc_rect_coef(self.c_a_ro, self.h_ro, self.w_ro, float(self.h_ro) / self.w_ro)
        self.rect_coef_diff = (self.rect_coef_ro - self.rect_coef_ao) / self.rect_coef_ro

        self.base_status = bool()
        self.br_status = bool()
        self.gen_status = bool()

        self.br_cr_rects = [[0, 0, 0, 0]]
        self.br_cr_area = int()
        self.br_ratio = float()

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
        k = 1 if (h_w_ratio > 0.7) and (h_w_ratio < 3.2) else -1
        rect_coef = c_a * k * ((h ** 2 + 2 * h * w + w ** 2) / (h * w * 4.0))

        return round(rect_coef, 3)
    # TODO Transfer into dataframe class

    def detect(self):
        is_rect_coeff_belongs = self.check_rect_coeff(self.rect_coef_diff)
        is_extent_belongs = self.check_extent(self.extent_ao)
        is_margin_crossed = self.check_margin(self.base_rect_ao[0], self.base_rect_ao[2])

        #if is_rect_coeff_belongs and not is_margin_crossed and is_extent_belongs:
        if is_rect_coeff_belongs:
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
        x_left = not (x_margin < x < conf.RESIZE_TO[0] - x_margin)
        x_right = not (x_margin < x + w < conf.RESIZE_TO[0] - x_margin)

        return x_left or x_right


class PreProcess(object):
    def __init__(self):
        self.__mog = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        if not(0 in conf.F_KERNEL_SIZE):
            self.__filtering_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, conf.F_KERNEL_SIZE)
        else:
            self.__filter = copy.copy

        self.set_ratio_done = bool()

    def process(self, d_frame):
        # orig_img = resize(d_frame.orig_img, width=conf.RESIZE_TO[0], height=conf.RESIZE_TO[1])
        orig_img = resize(d_frame.orig_img, height=conf.RESIZE_TO[1])


        # orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

        self.set_ratio(orig_img)

        # orig_img = self.adjust_gamma(orig_img, 2)
        # orig_img = self.increase_brightness(orig_img, 20)

        orig_img = self.clahe_contrast(orig_img)

        mog_mask = self.__mog.apply(orig_img)
        _, mog_mask = cv2.threshold(mog_mask, 127, 255, cv2.THRESH_BINARY)

        filtered_mask = self.__filter(mog_mask)

        filled_mask = cv2.dilate(filtered_mask, None, iterations=conf.DILATE_ITERATIONS)
        #filled_mask = filtered_mask 

        d_frame.orig_img = orig_img
        d_frame.filled_mask = filled_mask

        return mog_mask, filtered_mask

    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    @staticmethod
    def increase_brightness(image, value=30):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return image

    @staticmethod
    def clahe_contrast(image):
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
        # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
        return clahe.apply(image)

    def set_ratio(self, img):
        if not self.set_ratio_done:
            self.set_ratio_done = True
            actual_w, actual_h = img.shape[:2][1], img.shape[:2][0]
            DETECTION_LOG.warning("Actual resolution used for processing is {}x{}".format(actual_w, actual_h))

            if conf.RESIZE_TO[0] != actual_w or conf.RESIZE_TO[1] != actual_h:
                conf.RESIZE_TO[0] = actual_w
                conf.RESIZE_TO[1] = actual_h

    def __filter(self, mog_mask):

        return cv2.morphologyEx(mog_mask, cv2.MORPH_OPEN, self.__filtering_kernel)


class DataFrame(object):
    def __init__(self):
        self.orig_img = np.dtype('uint8')
        self.filled_mask = np.dtype('uint8')

        self.base_frame_status = None  # Can be False/True/None type
        self.ex_frame_status = None  # Can be False/True/None type
        self.base_objects = list()
        self.base_contours = list()
        self.ex_objects = list()

        self.br_rects = list()

    def calculate(self):
        self.base_objects, self.base_contours = self._basic_process(self.filled_mask)


        bright_mask = self.calc_bright_coeff(self.base_objects)


        ex_filled_mask = self._extent_split_process()
        self.ex_objects, _ = self._basic_process(ex_filled_mask)
        self.calc_bright_coeff(self.ex_objects)

        self.detect()

        return bright_mask, ex_filled_mask

    @staticmethod
    def _basic_process(filled_mask):
        objects = list()

        _, contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for obj_id, contour in enumerate(contours):
            obj = ObjParams(obj_id, contour)
            obj.process_obj()
            objects.append(obj)

        return objects, contours

    # TODO remake to crop and analyze only one object, only problem object should be considered further
    def _extent_split_process(self):
        def make_split(bin_mask, fill=0.68, tail=0.25): # fill - amount of zeros in coloumn in percent ratio  
            def calc_split_point(vector):
                last_zero_ind, percent = 0, 0.0
                zero_indexes, = np.where(vector == 0)

                if zero_indexes.size > 0:
                    last_zero_ind = zero_indexes.max()
                    percent = last_zero_ind / float(vector.size) # Get relative size of empty area by x axis

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
        ex_filled_mask = np.zeros((conf.RESIZE_TO[1], conf.RESIZE_TO[0]), np.uint8) # create minimal image
        for obj in self.base_objects:
            is_extent = obj.extent_ao < 0.6
            is_rect_coeff = -10000 < obj.rect_coef_ao < -2000  # Try to reduce to -5000 or so

            if is_extent and is_rect_coeff and not obj.base_status and not obj.br_status:
                x, y, w, h = obj.base_rect_ao
                split_mask = make_split(self.filled_mask[y:y+h, x:x + w])
                ex_filled_mask[y:y+h, x:x + w] = split_mask[:, :]
                #ex_filled_mask[y:y+h, x:x + w] = self.filled_mask[y:y+h, x:x + w]
#                cv2.line(ex_filled_mask, (x + int(w / 2), 0), (x + int(w / 2), conf.RESIZE_TO[1]), (0, 0, 0), 3)

        return ex_filled_mask

    def calc_bright_coeff(self, objects):
        if len(self.base_objects) > 0:
            brightness_mask = np.zeros((conf.RESIZE_TO[1], conf.RESIZE_TO[0]), np.uint8)

            brightness_mask[np.where(self.orig_img > 265)] = [255]   # Chose later appropriate values
            _, contours, _ = cv2.findContours(brightness_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.br_rects = [cv2.boundingRect(contour) for contour in contours]

            for obj in objects:
                obj.br_cr_rects = [self.intersection(obj.base_rect_ao, br_rect) for br_rect in self.br_rects]
                obj.br_cr_area = sum([rect[2] * rect[3] for rect in obj.br_cr_rects])
                obj.br_ratio = round(float(obj.br_cr_area) / obj.w_ao * obj.h_ao, 3)
                obj.br_status = obj.br_ratio > conf.BRIGHTNESS_THRESHOLD

        else:
            brightness_mask = np.dtype('uint8')

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














