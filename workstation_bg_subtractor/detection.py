import logging
import threading
import time

import cv2
from imutils import resize
import config
import glob
import pyexiv2
import os
import numpy as np
import copy
import sqlite3

logger = logging.getLogger(__name__)


class Detector(threading.Thread):
    def __init__(self, stop_ev):
        super(Detector, self).__init__(name="Detector")
        self.running = False
        self.stop_event = stop_ev

        self.mog = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        if config.F_KERNEL_SIZE[0] > 0 and config.F_KERNEL_SIZE[1] > 0:
            self.filtering_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.F_KERNEL_SIZE)

        self.out_data = ""
        self.x_range = (config.MARGIN[0], config.PROC_IMG_RES[0] - config.MARGIN[0])
        self.y_range = (config.MARGIN[1], config.PROC_IMG_RES[1] - config.MARGIN[1])

        self.r_orig_img = []
        self.orig_img_2 = []
        self.mog_mask = []
        self.filled_img = []
        self.filtered_img = []
        self.out_img = []
        self.extent_img = []
        self.data_img = []
        self.blank_img = []

        self.counter = 0

    # Main thread routine
    def run(self):
        logger.info("Grabber started")
        self.running = True

        while (self.counter < ((len(glob.glob(os.path.join(config.IMG_IN_DIR, "*.jpeg")))) - 1)) and self.running:
            orig_img = cv2.imread(glob.glob(os.path.join(config.IMG_IN_DIR, "img_%s_*.jpeg" % self.counter))[0])
            self.r_orig_img, self.mog_mask, self.filtered_img, self.filled_img = self.process_img(orig_img)

            coeffs = DStructure()
            self.coeff_calc(self.filled_img, coeffs)
            config.MOTION_STATUS = self.detect(coeffs)

            # self.extent_img = self.check_on_extent(coeffs)
            # e_coeffs = DStructure()
            # self.coeff_calc(self.extent_img, coeffs)
            # config.MOTION_STATUS = self.detect(e_coeffs)

            self.form_out_img(coeffs)
            cv2.imshow('image', self.out_img)
            cv2.waitKey(1)




            self.counter += 1
            time.sleep(0)

            # db = DbStore("db1")
            # db.set_params("abc", 1)
            # db.db_write()
            # exit(0)

            # cv2.imshow('image', self.filled_img)
            # cv2.waitKey(0)


            # print self.extent_img.shape[:2]

            # self.e_result = self.coeff_calc(self.extent_img)


            # self.save_image()


            # self.out_data += self.formData(det_res)
        # file = open("/home/ivan/test_ir/data_1000.txt", "w")
        # file.write(self.out_data)
        # file.close()

        self.quit()

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
        config.MOTION_STATUS = False

        for coeff in coeffs.rect_coef_arr:
            is_coeff_belongs = config.COEFF_RANGE[0] < coeff < config.COEFF_RANGE[1]

            if is_coeff_belongs:
                coeffs.add_o_status(o_status=True)
                logging.info("Motion detected")
            else:
                config.MOTION_STATUS = False
                coeffs.add_o_status(o_status=False)

        return any(coeffs.o_status_arr)

    def check_on_extent(self, coeffs):
        for extent, rect_coeff, x, y, w, h in zip(coeffs.extent_arr, coeffs.rect_coef_arr,
                                                  coeffs.x_arr, coeffs.y_arr, coeffs.w_arr, coeffs.h_arr):

            is_extent = extent < 0.6
            is_rect_coeff = -20000 < rect_coeff < -10000

            if is_extent and is_rect_coeff:
                ex_filled = copy.copy(self.filled_img[y:y + h, x:x + w])
                cv2.line(ex_filled, (int(w/2), 0), (int(w/2), h), (0, 0, 0), 2)
                # cv2.imshow('img', ex_filled)
                # cv2.waitKey(0)
            else:
                ex_filled = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0]), np.uint8)
        else:
            ex_filled = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0]), np.uint8)

        return ex_filled


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
            coeffs.add(contour=contour, x=x, y=y, w=w, h=h, rect_coef=rect_coef, extent=extent, contour_a=contour_a,
                       rect_a=rect_a, rect_p=rect_p)

    def form_out_img(self, coeffs):
        self.orig_img_2 = copy.copy(self.r_orig_img)
        self.blank_img = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0], 3), np.uint8)
        self.data_img = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0], 3), np.uint8)
        # self.extent_img = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0]), np.uint8)

        self.draw_on_mog_mask()
        self.draw_on_filtered_img()
        self.draw_on_filled_img()
        self.draw_on_orig_img_1(coeffs)
        self.draw_on_orig_img_2(coeffs)
        self.draw_on_extent_img()
        self.draw_on_data_img(coeffs)
        # self.draw_on_blank_img()

        self.extent_img = cv2.cvtColor(self.extent_img, cv2.COLOR_GRAY2BGR)
        self.mog_mask = cv2.cvtColor(self.mog_mask, cv2.COLOR_GRAY2BGR)
        self.filtered_img = cv2.cvtColor(self.filtered_img, cv2.COLOR_GRAY2BGR)
        self.filled_img = cv2.cvtColor(self.filled_img, cv2.COLOR_GRAY2BGR)

        print 1, self.filled_img.shape[:2]
        print 2, self.blank_img.shape[:2]
        print 3, self.extent_img.shape[:2]

        h_stack1 = np.hstack((self.mog_mask, self.filtered_img, self.filled_img))
        h_stack2 = np.hstack((self.extent_img, self.data_img, self.blank_img))
        h_stack3 = np.hstack((self.orig_img_2, self.r_orig_img, self.blank_img))

        self.out_img = np.vstack((h_stack1, h_stack2, h_stack3))

    def draw_on_mog_mask(self):
        cv2.putText(self.mog_mask, str("MOG2 mask"), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def draw_on_filtered_img(self):
        cv2.putText(self.filtered_img, str("Filtered image"), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def draw_on_filled_img(self):
        cv2.putText(self.filled_img, str("Filled image"), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def draw_on_orig_img_1(self, coeffs):
        cv2.putText(self.r_orig_img, "Rectangle area", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 255), 1, cv2.LINE_AA)

        for i in range(len(coeffs.rect_coef_arr)):
            x, y, w, h = coeffs.x_arr[i], coeffs.y_arr[i], coeffs.w_arr[i], coeffs.h_arr[i]
            cv2.rectangle(self.r_orig_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(self.r_orig_img, str(i), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(self.r_orig_img, str(coeffs.rect_coef_arr[i]), (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)

    def draw_on_orig_img_2(self, coeffs):
        cv2.putText(self.orig_img_2, "Contour area", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 255), 1, cv2.LINE_AA)
        cv2.drawContours(self.orig_img_2, coeffs.contour_arr, -1, (0, 255, 0), 1)

    def draw_on_extent_img(self):
        cv2.putText(self.extent_img, "Split", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 0, cv2.LINE_AA)

    def draw_on_data_img(self, coeffs):
        cv2.putText(self.data_img, str(any(coeffs.o_status_arr)), (80, 95), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def save_image(self):
        # Save JPEG with proper name
        img_name = "img_%s.jpeg" % self.counter
        path = os.path.join(config.IMG_OUT_DIR, img_name)
        cv2.imwrite(path, self.out_img)
        # string = ""
        # for i in range(len(det_res)):
        #     if i > 0:
        #         string += "\n"
        #     for k in range(len(det_res[i])):
        #         for l in range(len(det_res[i][k])):
        #             string += str(det_res[i][k][l]) + ", "
        #     string = string[:-2]
        #
        # # Parser for csv in exif
        # # a = []
        # # reader = csv.reader(string.split('\n'), delimiter=',')
        # # for row in reader:
        # #     a.append(row)
        #
        # # Write exif to saved JPEG
        # metadata = pyexiv2.ImageMetadata(path)
        # metadata.read()
        # metadata['Exif.Image.Software'] = pyexiv2.ExifTag('Exif.Image.Software', 'OpenCV-3.2.0-dev, pyexiv2')
        # metadata['Exif.Image.Artist'] = pyexiv2.ExifTag('Exif.Image.Artist', 'Ivan Matveev')
        # metadata['Exif.Photo.UserComment'] = pyexiv2.ExifTag('Exif.Photo.UserComment', string)
        # metadata.write()




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


class DbStore:
    def __init__(self, db_name):
        self.db_name = db_name
        self.db = sqlite3.connect(self.db_name)
        self.img_name = []
        self.d_parameters = []

    def set_params(self, img_name, d_parameters):
        self.img_name = img_name
        self.d_parameters = d_parameters

    def db_write(self, ):
        self.db = sqlite3.connect(self.db_name)
        cur = self.db.cursor()


        cur.execute('''CREATE TABLE %s (Status TEXT, 
                                        Rect_coeff REAL, 
                                        hw_ratio REAL, 
                                        Contour_area REAL, 
                                        Rect_area REAL, 
                                        Rect_perimeter REAL, 
                                        Extent_coeff REAL, 
                                        x REAL, 
                                        y REAL, 
                                        w REAL, 
                                        h REAL )''' % self.img_name)
        # self.d_result.append([arr, (x, y, w, h), rect_coeff, extent, solidity, a_contour, a_rect, p_rect])

        cur.execute('''INSERT INTO %s(Status, 
                                        Rect_coeff, 
                                        hw_ratio, 
                                        Contour_area,
                                        Rect_area,
                                        Rect_perimeter,
                                        Extent_coeff,
                                        x,
                                        y,
                                        w,
                                        h) VALUES(?,?,?,?,?,?,?,?,?,?,?)'''
                    % self.img_name, (str(config.MOTION_STATUS), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))

        self.db.commit()

    def quit(self):
        self.db.close()


