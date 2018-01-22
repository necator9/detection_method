import logging
import threading
import time

import cv2
from imutils import resize
import config
import glob
import numpy as np
import copy
import sqlite3
import os
import csv

logger = logging.getLogger(__name__)


class Detector(threading.Thread):
    def __init__(self, stop_ev):
        super(Detector, self).__init__(name="Detector")
        self.running = False
        self.stop_event = stop_ev
        self.img_name = str()

        if config.WRITE_TO_DB:
            db_name = self.gen_name("Database")
            self.db = DbStore(db_name)
            logger.info("Database name: %s" % db_name)

        if config.WRITE_TO_CSV:
            csv_name = self.gen_name("csv_file")
            self.csv = CSVstore(csv_name)

        self.x_range = (config.MARGIN[0], config.PROC_IMG_RES[0] - config.MARGIN[0])
        self.y_range = (config.MARGIN[1], config.PROC_IMG_RES[1] - config.MARGIN[1])

        self.res_orig_img = list()
        self.res_orig_img_2 = list()
        self.mog_mask = list()
        self.filled_img = list()
        self.filtered_img = list()
        self.ex_filled_img = list()
        self.orig_status_img = list()
        self.ex_status_img = list()
        self.ex_orig_img = list()
        self.brightness_mask = list()

        self.red_x_border = list()
        self.red_y_border = list()

        self.coeffs = list()

    # Main thread routine
    def run(self):
        logger.info("Detection has started")
        self.running = True

        pre_process = PreProcess()

        while config.COUNTER < config.IMG_IN_DIR and self.running:
            path_to_img = glob.glob(os.path.join(config.IN_DIR, "img_%s_*.jpeg" % config.COUNTER))[0]
            self.img_name = path_to_img.split("/")[-1]

            orig_img = cv2.imread(path_to_img)

            img_fr = ImgFrame()  # TODO test to pass without instance directly
            img_fr = pre_process.perform(img_fr, orig_img)

            data_frame = DataFrame(img_fr)
            data_frame.gen_process()

            if config.SHOW_IMG or config.SAVE_IMG:
                out_img = self.form_out_img(data_frame)

                if config.SHOW_IMG:
                    cv2.imshow('Detection', out_img)
                    cv2.waitKey(1)
                    time.sleep(0.1)

                if config.SAVE_IMG:
                    self.save_image(out_img)

            # if config.WRITE_TO_DB:
            #     self.db.db_write(self.coeffs[0], config.COUNTER)
            #
            # if config.WRITE_TO_CSV:
            #     self.csv.write(self.coeffs[1], self.img_name)
            #
            # if not self.check_length(self.coeffs[0]):
            #     logger.error("Exiting main loop. Check on length failed")
            #     break

            config.COUNTER += 1

        if config.WRITE_TO_DB:
            self.db.quit()

        if config.WRITE_TO_CSV:
            self.csv.quit()

        self.quit()

    @staticmethod
    def gen_name(db_name):
        i = 0
        while True:
            name_plus_counter = db_name + "_%s" % str(i).zfill(3)
            path_plus_name = os.path.join(config.OUT_DIR, name_plus_counter)
            if not os.path.exists(path_plus_name):
                return path_plus_name
            else:
                i += 1

    @staticmethod
    def __check_length(coeffs):
        reference_len = len(coeffs.o_status_arr)
        for attr, value in coeffs.__dict__.iteritems():
            if len(value) != reference_len:
                logger.error("Data structure error. Length of %s != reference_len" % attr)

                return False

        return True

    def form_out_img(self, frame):
        # !!!!!!!!!!!!!!
        self.ex_filled_img = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0]), np.uint8)
        self.ex_orig_img = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0], 3), np.uint8)
        # !!!!!!!!!!!!!
        self.res_orig_img_2 = copy.copy(self.res_orig_img)
        self.ex_status_img = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0], 3), np.uint8)
        self.orig_status_img = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0], 3), np.uint8)

        self.red_x_border = np.zeros((config.PROC_IMG_RES[1], 1, 3), np.uint8)
        self.red_x_border[:] = (0, 0, 255)
        self.red_y_border = np.zeros((1, config.PROC_IMG_RES[0] * 3 + 2, 3), np.uint8)
        self.red_y_border[:] = (0, 0, 255)

        self.put_name(self.mog_mask, "MOG2 mask")
        self.put_name(self.filtered_img, "Filtered image")
        self.put_name(self.filled_img, "Filled image")
        self.put_name(self.res_orig_img, "Rectangle area")
        self.put_name(self.res_orig_img_2, "Contour area")
        self.put_name(self.ex_filled_img, "Division by extent (Filled image)")
        self.put_name(self.ex_orig_img, "Division by extent (Original image)")
        self.put_name(self.orig_status_img, "Original status")
        self.put_name(self.ex_status_img, "Status after division by extent")

        self.put_margin(self.res_orig_img)

        self.put_status(self.orig_status_img, frame)
        # self.put_status(self.ex_status_img, self.coeffs[1])

        self.draw_rect_areas(self.res_orig_img, frame)
        # self.draw_rect_area(self.ex_orig_img, self.coeffs[1])
        self.draw_contour_areas(self.res_orig_img_2, frame)

        self.ex_filled_img = cv2.cvtColor(self.ex_filled_img, cv2.COLOR_GRAY2BGR)
        self.mog_mask = cv2.cvtColor(self.mog_mask, cv2.COLOR_GRAY2BGR)
        self.filtered_img = cv2.cvtColor(self.filtered_img, cv2.COLOR_GRAY2BGR)
        self.filled_img = cv2.cvtColor(self.filled_img, cv2.COLOR_GRAY2BGR)

        # print 1, self.mog_mask.shape[:2]
        # print 2, self.filled_img.shape[:2]
        # print 3, self.filled_img.shape[:2]

        h_stack1 = np.hstack((self.mog_mask, self.red_x_border, self.filtered_img, self.red_x_border, self.filled_img))
        h_stack2 = np.hstack(
            (self.res_orig_img_2, self.red_x_border, self.res_orig_img, self.red_x_border, self.orig_status_img))
        h_stack3 = np.hstack(
            (self.ex_filled_img, self.red_x_border, self.ex_orig_img, self.red_x_border, self.ex_status_img))

        out_img = np.vstack((h_stack1, self.red_y_border, h_stack2, self.red_y_border, h_stack3))

        return out_img

    @staticmethod
    def put_name(img, text):
        cv2.putText(img, text, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def draw_rect_areas(img, frame):
        for obj in frame.base_objects:
            cv2.rectangle(img, (obj.x, obj.y), (obj.x + obj.w, obj.y + obj.h), (0, 255, 0), 1)
            cv2.putText(img, str(obj.obj_id + 1), (obj.x + 5, obj.y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)

    @staticmethod
    def draw_contour_areas(img, frame):
        cv2.drawContours(img, frame.base_contours, -1, (0, 255, 0), 1)

    @staticmethod
    def put_status(img, frame):
        cv2.putText(img, str(frame.base_frame_status), (80, 95), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def save_image(self, out_img):
        # Save JPEG with proper name
        img_name = self.img_name
        path = os.path.join(config.OUT_DIR, img_name)
        cv2.imwrite(path, out_img)

    @staticmethod
    def put_margin(img):
        x_left_up = config.X_MARGIN
        y_left_up = 0
        x_left_down = x_left_up
        y_left_down = config.PROC_IMG_RES[1]

        x_right_up = config.PROC_IMG_RES[0] - config.X_MARGIN - 1
        y_right_up = 0
        x_right_down = x_right_up
        y_right_down = y_left_down

        cv2.line(img, (x_left_up, y_left_up), (x_left_down, y_left_down), (255, 0, 0), 1)
        cv2.line(img, (x_right_up, y_right_up), (x_right_down, y_right_down), (255, 0, 0), 1)

    # Stop and quit the thread operation.
    def quit(self):
        self.running = False
        self.stop_event.clear()
        logger.info("Grabber has quit")


class ObjParams(object):
    def __init__(self, obj_id=int()):
        self.obj_id = obj_id
        self.obj_status = bool()

        self.x = int()
        self.y = int()
        self.w = int()
        self.h = int()

        self.x_br_cross = int()
        self.y_br_cross = int()
        self.w_br_cross = int()
        self.h_br_cross = int()

        self.contour_area = float()
        self.rect_coef = float()
        self.extent = float()
        self.rect_area = float()
        self.rect_perimeter = float()

    def add_br_crossing(self, x_br_cross, y_br_cross, w_br_cross, h_br_cross):
        self.x_br_cross = x_br_cross
        self.y_br_cross = y_br_cross
        self.w_br_cross = w_br_cross
        self.h_br_cross = h_br_cross

    def process_obj(self, contour):
        self.calc_params(contour)
        self.detect()

    def calc_params(self, contour):
        self.contour_area = cv2.contourArea(contour)
        self.x, self.y, self.w, self.h = cv2.boundingRect(contour)
        self.rect_area = self.w * self.h
        self.rect_perimeter = 2 * (self.h + self.w)
        self.extent = round(float(self.contour_area) / self.rect_area, 3)

        if float(self.h) / self.w > 0.7:
            k = 1.0
        else:
            k = -1.0

        # coeff(k * ((2.0 * w * h + 2 * w ** 2 + h) / w), 1) # Kirill suggestion
        self.rect_coef = round(self.contour_area * k * ((self.h ** 2 + 2 * self.h * self.w + self.w ** 2) /
                                                        (self.h * self.w * 4.0)), 3)

    def detect(self):
        is_rect_coeff_belongs = self.check_rect_coeff(self.rect_coef)
        is_extent_belongs = self.check_extent(self.extent)
        is_margin_crossed = self.check_margin(self.x, self.w)

        if is_rect_coeff_belongs and is_extent_belongs and not is_margin_crossed:
            self.obj_status = True
        else:
            self.obj_status = False

    @staticmethod
    def check_rect_coeff(coeff):

        return config.COEFF_RANGE[0] < coeff < config.COEFF_RANGE[1]

    @staticmethod
    def check_extent(extent):

        return extent > config.EXTENT_THRESHOLD

    @staticmethod
    def check_margin(x, w):
        x_margin = config.X_MARGIN
        x_left = not (x_margin < x < config.PROC_IMG_RES[0] - x_margin)
        x_right = not (x_margin < x + w < config.PROC_IMG_RES[0] - x_margin)

        return x_left or x_right


# class ImgFrame(object):
#     def __init__(self):
#         self.r_img = list()
#         self.br_mask = list()
#         self.mog_mask = list()
#         self.filtered_mask = list()
#         self.filled_mask = list()


class PreProcess(object):
    def __init__(self):
        self.__mog = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.__filtering_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.F_KERNEL_SIZE)

        self.r_img = list()
        self.mog_mask = list()
        self.filtered_mask = list()
        self.filled_mask = list()
        self.br_mask = list()

    def perform(self, frame, orig_img):
        frame.r_img = resize(orig_img, width=config.PROC_IMG_RES[0], height=config.PROC_IMG_RES[1])
        mog_mask = self.__mog.apply(frame.r_img)
        _, frame.mog_mask = cv2.threshold(mog_mask, 127, 255, cv2.THRESH_BINARY)

        if config.F_KERNEL_SIZE[0] > 0 and config.F_KERNEL_SIZE[1] > 0:
            frame.filtered_mask = cv2.morphologyEx(frame.mog_mask, cv2.MORPH_OPEN, self.__filtering_kernel)
        else:
            frame.filtered_mask = copy.copy(frame.mog_mask)

        frame.filled_mask = cv2.dilate(frame.filtered_mask, None, iterations=config.DILATE_ITERATIONS)

        return frame

 # def check_ratio(self):
    #     if config.PROC_IMG_RES[0] != self.res_in_img.shape[:2][1] or config.PROC_IMG_RES[1] != self.res_in_img.shape[:2][0]:  # Remake to run once in loop
    #         config.PROC_IMG_RES[0] = self.res_in_img.shape[:2][1]
    #         config.PROC_IMG_RES[1] = self.res_in_img.shape[:2][0]


class DataFrame(object):
    def __init__(self, img_fr):
        self.img_fr = img_fr
        self.base_frame_status = None  # Can be False/True type
        self.ex_frame_status = None  # Can be False/True type

        self.base_objects = list()
        self.ex_objects = list()

        self.base_contours = list()

    def gen_process(self):
        self.base_objects, self.base_contours = self.__basic_process(filled_mask)
        ex_filled_mask = self.__extent_split_process(filled_mask)
        self.ex_objects, _ = self.__basic_process(ex_filled_mask)
        self.base_frame_status = self.take_frame_status(self.base_objects)
        self.ex_frame_status = self.take_frame_status(self.ex_objects)

    @staticmethod
    def __basic_process(filled_mask):
        objects = list()
        _, contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for obj_id, contour in enumerate(contours):
            obj = ObjParams(obj_id)
            obj.process_obj(contour)
            objects.append(obj)

        return objects, contours

    def __extent_split_process(self, filled_mask):
        ex_filled = np.zeros((1, 1, 1), np.uint8)
        for obj_id, obj in enumerate(self.base_objects):
            is_extent = obj.extent < 0.6
            is_rect_coeff = -20000 < obj.rect_coef < -10000  # Try to reduce to -5000 or so

            if is_extent and is_rect_coeff:
                ex_filled = copy.copy(filled_mask)
                cv2.line(ex_filled, (obj.x + int(obj.w / 2), 0), (obj.x + int(obj.w / 2), config.PROC_IMG_RES[1]),
                         (0, 0, 0), 3)

        return ex_filled

    def __brightness_process(self):
        brightness_mask[np.where((self.res_in_img > [220, 220, 220]).all(axis=2))] = [255]
        brightness_mask = cv2.cvtColor(brightness_mask, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def __take_frame_status(objects):
        status_arr = list()
        for obj in objects:
            status_arr.append(obj.obj_status)
        return any(status_arr)


class DbStore(object):
    def __init__(self, db_name):
        self.db_name = db_name
        self.db = sqlite3.connect(self.db_name)
        self.table_name = str()
        self.d_parameters = list()

    def db_write(self, coeffs, counter):
        self.table_name = "img_%s" % str(counter).zfill(4)
        self.d_parameters = coeffs
        self.db = sqlite3.connect(self.db_name)
        cur = self.db.cursor()

        cur.execute('''CREATE TABLE %s (Status TEXT, Rect_coeff REAL, hw_ratio REAL, Contour_area REAL, Rect_area REAL, 
                                        Rect_perimeter REAL, Extent_coeff REAL, x REAL, y REAL, w REAL, h REAL )'''
                    % self.table_name)

        cur.executemany('''INSERT INTO %s(Status, Rect_coeff, hw_ratio, Contour_area, Rect_area, Rect_perimeter,
                                        Extent_coeff, x, y, w, h) VALUES(?,?,?,?,?,?,?,?,?,?,?)'''
                        % self.table_name, (self.d_parameters.get_arr()))

        self.db.commit()

    def quit(self):
        self.db.commit()
        self.db.close()


class CSVstore(object):
    def __init__(self, name):
        self.name = name + ".csv"
        print self.name
        fieldnames = ["Img_name", "Object_no", "Status", "Rect_coeff", "hw_ratio", "Contour_area", "Rect_area",
                      "Rect_perimeter", "Extent_coeff", "x", "y", "w", "h"]
        self.f = open(name, 'w')
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.writer.writeheader()

    def write(self, coeffs, img_name):
        for i in range(len(coeffs.o_status_arr)):
            if coeffs.rect_coef_arr[i] < -15000:
                continue
            self.writer.writerow({"Img_name": img_name, "Object_no": i + 1, "Status": coeffs.o_status_arr[i],
                                  "Rect_coeff": coeffs.rect_coef_arr[i],
                                  "hw_ratio": round(float(coeffs.h_arr[i]) / coeffs.w_arr[i], 3),
                                  "Contour_area": coeffs.contour_a_arr[i], "Rect_area": coeffs.rect_a_arr[i],
                                  "Rect_perimeter": coeffs.rect_p_arr[i], "Extent_coeff": coeffs.extent_arr[i],
                                  "x": coeffs.x_arr[i], "y": coeffs.y_arr[i], "w": coeffs.w_arr[i],
                                  "h": coeffs.h_arr[i]})

    def quit(self):
        self.f.close()

# def get_arr(self):
#     arr = list()
#     if len(self.o_status) > 0:
#         for i in range(len(self.o_status)):
#             arr.append([self.o_status[i], self.rect_coef[i], round(float(self.h[i]) / self.w[i], 3),
#                         self.contour_area[i], self.rect_area[i], self.rect_perimeter[i], self.extent[i],
#                         self.x[i], self.y[i], self.w[i], self.h[i]])
#     else:
#         arr.append([None] * 11)
#
#     return arr


# if brightness_mask is not None:
#                _, br_cnts, _ = cv2.findContours(brightness_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#            if brightness_mask is not None:
#               X = set(range(x, x + w))
#               Y = set(range(y, y + h))
#
#               for br_contour in br_cnts:
#                   x_br, y_br, w_br, h_br = cv2.boundingRect(br_contour)
#                   X_br = set(range(x_br, x_br + w_br))
#                   Y_br = set(range(y_br, y_br + h_br))
#
#                   if len(X & X_br and Y & Y_br) > 0:
#                       x_cross_range = list(X & X_br and Y & Y_br)
#                       y_cross_range = list(Y & Y_br and X & X_br)
#                       x_cross_b_rect = x_cross_range[0]
#                       w_cross_b_rect = x_cross_range[-1] - x_cross_range[0]
#                       y_cross_b_rect = y_cross_range[0]
#                       h_cross_b_rect = y_cross_range[-1] - y_cross_range[0]
#                       coef_ob.add(x_br_cross=x_cross_b_rect, y_br_cross=y_cross_b_rect, w_br_cross=w_cross_b_rect,
#                                   h_br_cross=h_cross_b_rect)
#                       pass
