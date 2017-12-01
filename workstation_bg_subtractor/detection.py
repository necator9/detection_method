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


logger = logging.getLogger(__name__)


class Detector(threading.Thread):
    def __init__(self, stop_ev):
        super(Detector, self).__init__(name="Detector")
        self.running = False
        self.stop_event = stop_ev

        self.mog = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        if config.F_KERNEL_SIZE[0] > 0 and config.F_KERNEL_SIZE[1] > 0:
            self.filtering_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.F_KERNEL_SIZE)

        self.img_name = str()
        self.x_range = (config.MARGIN[0], config.PROC_IMG_RES[0] - config.MARGIN[0])
        self.y_range = (config.MARGIN[1], config.PROC_IMG_RES[1] - config.MARGIN[1])

        self.res_orig_img = list()
        self.res_orig_img_2 = list()
        self.mog_mask = list()
        self.filled_img = list()
        self.filtered_img = list()
        self.out_img = list()
        self.ex_filled_img = list()
        self.orig_status_img = list()
        self.ex_status_img = list()
        self.ex_orig_img = list()

        self.counter = 0

    # Main thread routine
    def run(self):

        self.db_name = os.path.join(config.IMG_OUT_DIR, self.db_name)
        self.db = DbStore(self.db_name)

        logger.info("Grabber started")
        self.running = True
        files_in_dir = (len(glob.glob(os.path.join(config.IMG_IN_DIR, "*.jpeg")))) - 1

        while self.counter < files_in_dir and self.running:
            path_to_img = glob.glob(os.path.join(config.IMG_IN_DIR, "img_%s_*.jpeg" % self.counter))[0]
            self.img_name = path_to_img.split("/")[-1]
            record_name = "img_%s" % str(self.counter).zfill(4)

            # print self.img_name
            orig_img = cv2.imread(path_to_img)
            self.res_orig_img, self.mog_mask, self.filtered_img, self.filled_img = self.process_img(orig_img)
            coeffs = DStructure()
            self.coeff_calc(self.filled_img, coeffs)
            self.detect(coeffs)

            self.ex_filled_img, self.ex_orig_img = self.check_on_extent(coeffs)
            e_coeffs = DStructure()
            self.coeff_calc(self.ex_filled_img, e_coeffs)
            self.detect(e_coeffs)

            if config.SHOW_IMG:
                self.form_out_img(coeffs, e_coeffs)
                cv2.imshow('Detection', self.out_img)
                cv2.waitKey(1)

            if config.WRITE_TO_DB:
                self.db.db_write(coeffs, record_name)

            if config.SAVE_IMG:
                self.save_image()

            self.counter += 1
            time.sleep(0)

        self.db.quit()
        self.quit()

    def check_on_exist(self):
        i = 0
        db_name = os.path.join(config.IMG_OUT_DIR, self.db_name)
        while os.path.exists(self.db_name):
            self.db_name += ""

    def process_img(self, orig_img):
        r_orig_img = resize(orig_img, width=config.PROC_IMG_RES[0], height=config.PROC_IMG_RES[1])
        mog_mask = self.mog.apply(r_orig_img)
        _, mog_mask = cv2.threshold(mog_mask, 127, 255, cv2.THRESH_BINARY)

        if config.F_KERNEL_SIZE[0] > 0 and config.F_KERNEL_SIZE[1] > 0:
            filtered_img = cv2.morphologyEx(mog_mask, cv2.MORPH_OPEN, self.filtering_kernel)
        else:
            filtered_img = copy.copy(mog_mask)

        filled_img = cv2.dilate(filtered_img, None, iterations=3)

        return r_orig_img, mog_mask, filtered_img, filled_img

    @staticmethod
    def detect(coeffs):
        for coeff in coeffs.rect_coef_arr:
            is_coeff_belongs = config.COEFF_RANGE[0] < coeff < config.COEFF_RANGE[1]

            if is_coeff_belongs:
                coeffs.add_o_status(o_status=True)
                logging.info("Motion detected")
            else:
                coeffs.add_o_status(o_status=False)

    def check_on_extent(self, coeffs):
        for extent, rect_coeff, x, y, w, h in zip(coeffs.extent_arr, coeffs.rect_coef_arr,
                                                  coeffs.x_arr, coeffs.y_arr, coeffs.w_arr, coeffs.h_arr):

            is_extent = extent < 0.6
            is_rect_coeff = -20000 < rect_coeff < -10000

            if is_extent and is_rect_coeff:
                ex_filled = copy.copy(self.filled_img)
                cv2.line(ex_filled, (x + int(w/2), 0), (x + int(w/2), config.PROC_IMG_RES[1]), (0, 0, 0), 3)

                return ex_filled, copy.copy(self.res_orig_img)

            else:

                return np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0]), np.uint8), \
                       np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0], 3), np.uint8)
        else:

            return np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0]), np.uint8), \
                   np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0], 3), np.uint8)

    @staticmethod
    def coeff_calc(filled_img, coeffs):
        _, cnts, _ = cv2.findContours(filled_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            contour_a = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            rect_a = w * h
            rect_p = 2 * (h + w)
            extent = round(float(contour_a) / rect_a, 3)

            if float(h) / w > 0.7:
                k = 1.0
            else:
                k = -1.0

            # coeff(k * ((2.0 * w * h + 2 * w ** 2 + h) / w), 1) # Kirill suggestion
            rect_coef = round(contour_a * k * ((h ** 2 + 2 * h * w + w ** 2) / (h * w * 4.0)), 3)
            coeffs.add(contour=contour, x=x, y=y, w=w, h=h, rect_coef=rect_coef, extent=extent,
                       contour_a=contour_a, rect_a=rect_a, rect_p=rect_p)

    def form_out_img(self, coeffs, e_coeffs):
        self.res_orig_img_2 = copy.copy(self.res_orig_img)
        self.ex_status_img = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0], 3), np.uint8)
        self.orig_status_img = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0], 3), np.uint8)

        self.put_name(self.mog_mask, "MOG2 mask")
        self.put_name(self.filtered_img, "Filtered image")
        self.put_name(self.filled_img, "Filled image")
        self.put_name(self.res_orig_img, "Rectangle area")
        self.put_name(self.res_orig_img_2, "Contour area")
        self.put_name(self.ex_filled_img, "Division by extent (Filled image)")
        self.put_name(self.ex_orig_img, "Division by extent (Original image)")
        self.put_name(self.orig_status_img, "Original status")
        self.put_name(self.ex_status_img, "Status after division by extent")

        self.put_status(self.orig_status_img, coeffs)
        self.put_status(self.ex_status_img, e_coeffs)

        self.draw_rect_area(self.res_orig_img, coeffs)
        self.draw_rect_area(self.ex_orig_img, e_coeffs)
        self.draw_contour_area(self.res_orig_img_2, coeffs)

        self.ex_filled_img = cv2.cvtColor(self.ex_filled_img, cv2.COLOR_GRAY2BGR)
        self.mog_mask = cv2.cvtColor(self.mog_mask, cv2.COLOR_GRAY2BGR)
        self.filtered_img = cv2.cvtColor(self.filtered_img, cv2.COLOR_GRAY2BGR)
        self.filled_img = cv2.cvtColor(self.filled_img, cv2.COLOR_GRAY2BGR)

        # print 1, self.filled_img.shape[:2]
        # print 2, self.ex_status_img.shape[:2]
        # print 3, self.ex_filled_img.shape[:2]

        h_stack1 = np.hstack((self.mog_mask, self.filtered_img, self.filled_img))
        h_stack2 = np.hstack((self.res_orig_img_2, self.res_orig_img, self.orig_status_img))
        h_stack3 = np.hstack((self.ex_filled_img, self.ex_orig_img, self.ex_status_img))

        self.out_img = np.vstack((h_stack1, h_stack2, h_stack3))

    @staticmethod
    def put_name(img, text):
        cv2.putText(img, text, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def draw_rect_area(img, coeffs):
        for i in range(len(coeffs.rect_coef_arr)):
            x, y, w, h = coeffs.x_arr[i], coeffs.y_arr[i], coeffs.w_arr[i], coeffs.h_arr[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(img, str(i + 1), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)

    @staticmethod
    def draw_contour_area(img, coeffs):
        cv2.drawContours(img, coeffs.contour_arr, -1, (0, 255, 0), 1)

    @staticmethod
    def put_status(img, coeffs):
        if len(coeffs.o_status_arr) > 0:
            status = any(coeffs.o_status_arr)
        else:
            status = None
        cv2.putText(img, str(status), (80, 95), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def save_image(self):
        # Save JPEG with proper name
        img_name = self.img_name
        path = os.path.join(config.IMG_OUT_DIR, img_name)
        cv2.imwrite(path, self.out_img)

    @staticmethod
    def form_data(det_res):
        string = ""
        for i in range(len(det_res)):
            for k in range(len(det_res[i])):
                for l in range(len(det_res[i][k])):
                    string += str(det_res[i][k][l]) + ", "
            string = string[:-2]
            string += "\n"
        return string

    # Stop and quit the thread operation.
    def quit(self):
        self.running = False
        self.stop_event.clear()
        logger.info("Grabber has quit")


class DStructure:
    def __init__(self):
        self.o_status_arr = []
        self.contour_arr = []
        self.x_arr = []
        self.y_arr = []
        self.w_arr = []
        self.h_arr = []
        self.rect_coef_arr = []
        self.extent_arr = []
        self.contour_a_arr = []
        self.rect_a_arr = []
        self.rect_p_arr = []

    def add(self, contour, x, y, w, h, rect_coef, extent, contour_a, rect_a, rect_p):
        self.contour_arr.append(contour)
        self.x_arr.append(x)
        self.y_arr.append(y)
        self.w_arr.append(w)
        self.h_arr.append(h)
        self.rect_coef_arr.append(rect_coef)
        self.extent_arr.append(extent)
        self.contour_a_arr.append(contour_a)
        self.rect_a_arr.append(rect_a)
        self.rect_p_arr.append(rect_p)

    def add_o_status(self, o_status):
        self.o_status_arr.append(o_status)

    def get_arr(self):
        arr = []
        if len(self.o_status_arr) > 0:
            for i in range(len(self.o_status_arr)):
                arr.append((self.o_status_arr[i], self.rect_coef_arr[i], round(float(self.h_arr[i])/self.w_arr[i], 3),
                            self.contour_a_arr[i], self.rect_a_arr[i], self.rect_p_arr[i], self.extent_arr[i],
                            self.x_arr[i], self.y_arr[i], self.w_arr[i], self.h_arr[i]))
        else:
            arr.append((None, None, None, None, None, None, None, None, None, None, None))

        return arr


class DbStore:
    def __init__(self, db_name):
        self.db_name = db_name
        self.db = sqlite3.connect(self.db_name)
        self.img_name = str()
        self.d_parameters = list()

    def db_write(self, coeffs, img_name):
        self.d_parameters = coeffs
        self.img_name = img_name
        self.db = sqlite3.connect(self.db_name)
        cur = self.db.cursor()

        cur.execute('''CREATE TABLE %s (Status TEXT, Rect_coeff REAL, hw_ratio REAL, Contour_area REAL, Rect_area REAL, 
                                        Rect_perimeter REAL, Extent_coeff REAL, x REAL, y REAL, w REAL, h REAL )'''
                                        % self.img_name)

        cur.executemany('''INSERT INTO %s(Status, Rect_coeff, hw_ratio, Contour_area, Rect_area, Rect_perimeter,
                                        Extent_coeff, x, y, w, h) VALUES(?,?,?,?,?,?,?,?,?,?,?)'''
                                        % self.img_name, (self.d_parameters.get_arr()))

    def quit(self):
        self.db.commit()
        self.db.close()


