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

        self.mog = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        if config.F_KERNEL_SIZE[0] > 0 and config.F_KERNEL_SIZE[1] > 0:
            self.filtering_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.F_KERNEL_SIZE)

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
        self.out_img = list()
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

        while config.COUNTER < config.IMG_IN_DIR and self.running:
            path_to_img = glob.glob(os.path.join(config.IN_DIR, "img_%s_*.jpeg" % config.COUNTER))[0]
            self.img_name = path_to_img.split("/")[-1]

            orig_img = cv2.imread(path_to_img)
            self.res_orig_img, self.mog_mask, self.filtered_img, self.filled_img, self.brightness_mask = \
                self.process_img(orig_img)

            self.coeffs = list()
            self.coeffs.append(self.coeff_calc(self.filled_img, self.brightness_mask))

            self.ex_filled_img, self.ex_orig_img = self.check_on_extent(self.coeffs)
            self.coeffs.append(self.coeff_calc(self.ex_filled_img))

            self.detect()

            if config.SHOW_IMG or config.SAVE_IMG:
                self.form_out_img()

                if config.SHOW_IMG:
                    cv2.imshow('Detection', self.out_img)
                    cv2.waitKey(1)
                    time.sleep(0.1)

                if config.SAVE_IMG:
                    self.save_image()

            if config.WRITE_TO_DB:
                self.db.db_write(self.coeffs[0], config.COUNTER)

            if config.WRITE_TO_CSV:
                self.csv.write(self.coeffs[1], self.img_name)

            if not self.check_length(self.coeffs[0]):
                logger.error("Exiting main loop. Check on length failed")
                break

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

    def process_img(self, orig_img):
        r_orig_img = resize(orig_img, width=config.PROC_IMG_RES[0], height=config.PROC_IMG_RES[1])

        if config.PROC_IMG_RES[0] != r_orig_img.shape[:2][1] or config.PROC_IMG_RES[1] != r_orig_img.shape[:2][0]: # Remake to run once in loop
            config.PROC_IMG_RES[0] = r_orig_img.shape[:2][1]
            config.PROC_IMG_RES[1] = r_orig_img.shape[:2][0]

        mog_mask = self.mog.apply(r_orig_img)
        _, mog_mask = cv2.threshold(mog_mask, 127, 255, cv2.THRESH_BINARY)

        if config.F_KERNEL_SIZE[0] > 0 and config.F_KERNEL_SIZE[1] > 0:
            filtered_img = cv2.morphologyEx(mog_mask, cv2.MORPH_OPEN, self.filtering_kernel)
        else:
            filtered_img = copy.copy(mog_mask)

        brightness_mask = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0], 3), np.uint8)

        brightness_mask[np.where((r_orig_img > [220, 220, 220]).all(axis=2))] = [255]
        brightness_mask = cv2.cvtColor(brightness_mask, cv2.COLOR_BGR2GRAY)

        filled_img = cv2.dilate(filtered_img, None, iterations=config.DILATE_ITERATIONS)

        return r_orig_img, mog_mask, filtered_img, filled_img, brightness_mask

    def detect(self):
        for coeffs in self.coeffs:
            for coeff, extent, x, w in zip(coeffs.rect_coef_arr, coeffs.extent_arr, coeffs.x_arr, coeffs.w_arr):
                is_rect_coeff_belongs = self.check_rect_coeff(coeff)
                is_extent_belongs = self.check_extent(extent)
                is_margin_crossed = self.check_margin(x, w)

                if is_rect_coeff_belongs and is_extent_belongs and not is_margin_crossed:
                    coeffs.add_o_status(o_status=True)
                else:
                    coeffs.add_o_status(o_status=False)

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

    def check_on_extent(self, d_structs):
        coeffs = d_structs[0]
        for extent, rect_coeff, x, y, w, h in zip(coeffs.extent_arr, coeffs.rect_coef_arr,
                                                  coeffs.x_arr, coeffs.y_arr, coeffs.w_arr, coeffs.h_arr):

            is_extent = extent < 0.6
            is_rect_coeff = -20000 < rect_coeff < -10000  # Try to reduce to -5000 or so

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
    def coeff_calc(filled_img, brightness_mask=None):
        coef_ob = Obj_properties()
        _, cnts, _ = cv2.findContours(filled_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if brightness_mask is not None:
            _, br_cnts, _ = cv2.findContours(brightness_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            contour_a = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if brightness_mask is not None:
               X = set(range(x, x + w))
               Y = set(range(y, y + h))

               for br_contour in br_cnts:
                   x_br, y_br, w_br, h_br = cv2.boundingRect(br_contour)
                   X_br = set(range(x_br, x_br + w_br))
                   Y_br = set(range(y_br, y_br + h_br))

                   if len(X & X_br and Y & Y_br) > 0:
                       x_cross_range = list(X & X_br and Y & Y_br)
                       y_cross_range = list(Y & Y_br and X & X_br)
                       x_cross_b_rect = x_cross_range[0]
                       w_cross_b_rect = x_cross_range[-1] - x_cross_range[0]
                       y_cross_b_rect = y_cross_range[0]
                       h_cross_b_rect = y_cross_range[-1] - y_cross_range[0]
                       coef_ob.add(x_br_cross=x_cross_b_rect, y_br_cross=y_cross_b_rect, w_br_cross=w_cross_b_rect,
                                   h_br_cross=h_cross_b_rect)
                       pass

            rect_a = w * h
            rect_p = 2 * (h + w)
            extent = round(float(contour_a) / rect_a, 3)

            if float(h) / w > 0.7:
                k = 1.0
            else:
                k = -1.0


            # coeff(k * ((2.0 * w * h + 2 * w ** 2 + h) / w), 1) # Kirill suggestion
            rect_coef = round(contour_a * k * ((h ** 2 + 2 * h * w + w ** 2) / (h * w * 4.0)), 3)
            coef_ob.add(contour=contour, x=x, y=y, w=w, h=h, rect_coef=rect_coef, extent=extent,
                        contour_a=contour_a, rect_a=rect_a, rect_p=rect_p)


            # mutual_area = list()
            #
            # for br_contour in br_cnts:
            #     for br_x, br_y in br_contour[0]:
            #         for x, y, w, h in zip(coef_ob.x_arr, coef_ob.y_arr, coef_ob.w_arr, coef_ob.h_arr):
            #             if x <= br_x <= x + w:
            #                 if y <= br_y <= y + h:
            #                     pass

        return coef_ob

    @staticmethod
    def check_length(coeffs):
        reference_len = len(coeffs.o_status_arr)
        for attr, value in coeffs.__dict__.iteritems():
            if len(value) != reference_len:
                logger.error("Data structure error. Length of %s != reference_len" % attr)

                return False

        return True

    def form_out_img(self):
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

        self.put_line(self.res_orig_img)

        self.put_status(self.orig_status_img, self.coeffs[0])
        self.put_status(self.ex_status_img, self.coeffs[1])

        self.draw_rect_area(self.res_orig_img, self.coeffs[0])
        self.draw_rect_area(self.ex_orig_img, self.coeffs[1])
        self.draw_contour_area(self.res_orig_img_2, self.coeffs[0])

        self.ex_filled_img = cv2.cvtColor(self.ex_filled_img, cv2.COLOR_GRAY2BGR)
        self.mog_mask = cv2.cvtColor(self.mog_mask, cv2.COLOR_GRAY2BGR)
        self.filtered_img = cv2.cvtColor(self.filtered_img, cv2.COLOR_GRAY2BGR)
        self.filled_img = cv2.cvtColor(self.filled_img, cv2.COLOR_GRAY2BGR)

        # print 1, self.mog_mask.shape[:2]
        # print 2, self.filled_img.shape[:2]
        # print 3, self.filled_img.shape[:2]

        h_stack1 = np.hstack((self.mog_mask, self.red_x_border, self.filtered_img, self.red_x_border, self.filled_img))
        h_stack2 = np.hstack((self.res_orig_img_2, self.red_x_border, self.res_orig_img, self.red_x_border, self.orig_status_img))
        h_stack3 = np.hstack((self.ex_filled_img, self.red_x_border, self.ex_orig_img, self.red_x_border, self.ex_status_img))

        self.out_img = np.vstack((h_stack1, self.red_y_border, h_stack2, self.red_y_border, h_stack3))

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
        path = os.path.join(config.OUT_DIR, img_name)
        cv2.imwrite(path, self.out_img)

    @staticmethod
    def put_line(img):
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


class Obj_properties:
    # TODO change structure from lis
    def __init__(self):
        self.o_status_arr = list()
        self.contour_arr = list()

        self.x_arr = list()
        self.y_arr = list()
        self.w_arr = list()
        self.h_arr = list()

        self.x_br_cross = list()
        self.y_br_cross = list()
        self.w_br_cross = list()
        self.h_br_cross = list()

        self.rect_coef_arr = list()
        self.extent_arr = list()
        self.contour_a_arr = list()
        self.rect_a_arr = list()
        self.rect_p_arr = list()

    def add(self, contour, x, y, w, h, rect_coef, extent, contour_a, rect_a, rect_p, x_br_cross, y_br_cross,
            w_br_cross, h_br_cross):
        self.contour_arr.append(contour)

        self.x_arr.append(x)
        self.y_arr.append(y)
        self.w_arr.append(w)
        self.h_arr.append(h)

        self.x_br_cross.append(x_br_cross)
        self.y_br_cross.append(y_br_cross)
        self.w_br_cross.append(w_br_cross)
        self.h_br_cross.append(h_br_cross)

        self.rect_coef_arr.append(rect_coef)
        self.extent_arr.append(extent)
        self.contour_a_arr.append(contour_a)
        self.rect_a_arr.append(rect_a)
        self.rect_p_arr.append(rect_p)

    def add_o_status(self, o_status):
        self.o_status_arr.append(o_status)

    def get_arr(self):
        arr = list()
        if len(self.o_status_arr) > 0:
            for i in range(len(self.o_status_arr)):
                arr.append([self.o_status_arr[i], self.rect_coef_arr[i], round(float(self.h_arr[i])/self.w_arr[i], 3),
                            self.contour_a_arr[i], self.rect_a_arr[i], self.rect_p_arr[i], self.extent_arr[i],
                            self.x_arr[i], self.y_arr[i], self.w_arr[i], self.h_arr[i]])
        else:
            arr.append([None] * 11)

        return arr


class DbStore:
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


class CSVstore:
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
                                  "hw_ratio": round(float(coeffs.h_arr[i])/coeffs.w_arr[i], 3),
                                  "Contour_area": coeffs.contour_a_arr[i], "Rect_area": coeffs.rect_a_arr[i],
                                  "Rect_perimeter": coeffs.rect_p_arr[i], "Extent_coeff": coeffs.extent_arr[i],
                                  "x": coeffs.x_arr[i], "y": coeffs.y_arr[i], "w": coeffs.w_arr[i], "h": coeffs.h_arr[i]})


    def quit(self):
        self.f.close()