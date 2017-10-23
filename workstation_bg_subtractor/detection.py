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

logger = logging.getLogger(__name__)


class Detector(threading.Thread):
    def __init__(self, stop_ev):
        super(Detector, self).__init__(name="Detector")
        self.running = False
        self.stop_event = stop_ev

        self.mog = cv2.createBackgroundSubtractorMOG2()
        if config.F_KERNEL_SIZE[0] > 0 and config.F_KERNEL_SIZE[1] > 0:
            self.filtering_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.F_KERNEL_SIZE)

        self.out_data = ""
        self.x_range = (config.MARGIN[0], config.PROC_IMG_RES[0] - config.MARGIN[0])
        self.y_range = (config.MARGIN[1], config.PROC_IMG_RES[1] - config.MARGIN[1])

        self.orig_img_1 = []
        self.orig_img_2 = []
        self.mog_mask = []
        self.filled_img = []
        self.filtered_img = []
        self.out_img = []
        self.extent_img = []
        self.data_img = []
        self.blank_img = []
        self.bg_img = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0], 3), np.uint8)
        self.rect_mask = []

        self.d_result = []
        self.e_result = []
        self.frame_m_status = False
        self.counter = 0

    # Main thread routine
    def run(self):
        logger.info("Grabber started")
        self.running = True

        while (self.counter < ((len(glob.glob(os.path.join(config.IMG_IN_DIR, "*.jpeg")))) - 1)) and self.running:

            logger.debug("Taking image...")
            self.orig_img_1 = cv2.imread(glob.glob(os.path.join(config.IMG_IN_DIR, "img_%s_*.jpeg" % self.counter))[0])
            self.preprocessing()
            self.d_result = self.coeff_calc(self.filled_img)

            self.extent_img = self.check_on_extent(self.filled_img, self.d_result)
            print self.extent_img.shape[:2]
            #
            self.e_result = self.coeff_calc(self.extent_img)

            self.detect(self.d_result)

            self.form_out_img()

            cv2.imshow('image', self.out_img)
            cv2.waitKey(1)



            # self.save_image()

            self.counter += 1
            time.sleep(0.2)
            # self.out_data += self.formData(det_res)
        # file = open("/home/ivan/test_ir/data_1000.txt", "w")
        # file.write(self.out_data)
        # file.close()

        self.quit()

    def preprocessing(self):
        self.orig_img_1 = resize(self.orig_img_1, width=config.PROC_IMG_RES[0], height=config.PROC_IMG_RES[1])
        self.mog_mask = self.mog.apply(self.orig_img_1)

        if config.F_KERNEL_SIZE[0] > 0 and config.F_KERNEL_SIZE[1] > 0:
            self.filtered_img = cv2.morphologyEx(self.mog_mask, cv2.MORPH_OPEN, self.filtering_kernel)
        else:
            self.filtered_img = copy.copy(self.mog_mask)

        self.filled_img = cv2.dilate(self.filtered_img, None, iterations=3)

    def detect(self, d_result):
        config.MOTION_STATUS = False
        self.frame_m_status = False

        for i in range(len(d_result)):
            is_coeff_belongs = config.COEFF_RANGE[0] < d_result[i][2] < config.COEFF_RANGE[1]
            if is_coeff_belongs:
                config.MOTION_STATUS = True
                self.frame_m_status = True
                logging.info("Motion detected")
            else:
                config.MOTION_STATUS = False

    def check_on_extent(self, filled_img, d_result):
        print type(filled_img)
        if type(filled_img) is np.ndarray:
            print "kek"
        else:
            print "shmek"
        if len(d_result) > 0 and type(filled_img) is np.ndarray > 0:
            n = 1
            for i in range(len(d_result)):

                is_coeff_belongs = -20000 < d_result[i][2] < -10000
                if d_result[i][3] < 0.6 and is_coeff_belongs:
                    x, y, w, h = d_result[i][1]
                    self.rect_mask = np.zeros((h, w), np.uint8)
                    self.rect_mask[:, :] = filled_img[y:y + h, x:x + w]

                    self.bg_img = np.zeros((h, w), np.uint8)
                    self.bg_img[:, :w/2 - n] = self.rect_mask[:, :w / 2 - n]
                    self.bg_img[:, w / 2 + n:] = self.rect_mask[:, w / 2 + n:]

                    filled_img[y:y + h, x:x + w] = self.bg_img[:, :]

                    return filled_img
        else:
            filled_img = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0]), np.uint8)
            return filled_img

                # cv2.imshow('im', self.filled_img)
                # cv2.waitKey(0)


                # self.bg_img[:, : w/2 - n] = self.filled_img[y:y + h, x: (x + w) / 2 - 1]

                # self.bg_img[:, w / 2 - 1] = self.filled_img[y:y + h, x:((x + w / 2) - 1)]
                #self.test_img[:, :] = self.filled_img[y:y + h, ((x + w / 2) + 1):x+w]


                # black_image[:, :160 - n] = white_image[:, :160 - n]
                # black_image[:, 160 + n:] = white_image[:, 160 + n:]
        # if extent[0] > 0:
        #
        #     image1 = self.mask[coord[1]:coord[1] + coord[3], coord[0]:coord[0] + coord[2]]
        #     img_name = "img_%s.jpeg" % self.counter
        #     path = os.path.join(config.IMG_OUT_DIR, img_name)
        #     cv2.imwrite(path, image1)
        #     cv2.imshow('image', image1)
        #     cv2.waitKey(1)
            # is_x_belongs = self.x_range[0] < coord[0] < self.x_range[1]
            # is_x_max_belongs = self.x_range[0] < coord[0] + coord[2] < self.x_range[1]
            # is_y_belongs = self.y_range[0] < coord[1] < self.y_range[1]
            # is_y_max_belongs = self.y_range[0] < coord[1] + coord[3] < self.y_range[1]

    def coeff_calc(self, filled_img):
        d_result = list()
        _, cnts, _ = cv2.findContours(filled_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for arr in cnts:
            a_contour = cv2.contourArea(arr)
            x, y, w, h = cv2.boundingRect(arr)
            a_rect = w * h
            p_rect = 2 * (h + w)

            if float(h) / w > 0.7:
                k = 1.0
            else:
                k = -1.0

            hull = cv2.convexHull(arr)
            hull_area = cv2.contourArea(hull)
            solidity = round(float(a_contour) / hull_area, 3)
            extent = round(float(a_contour) / a_rect, 3)
            # rect = cv2.minAreaRect(arr)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)

            # coeff(k * ((2.0 * w * h + 2 * w ** 2 + h) / w), 1) # Kirill suggestion
            rect_coeff = round(a_contour * k * ((h ** 2 + 2 * h * w + w ** 2) / (h * w * 4.0)), 3)
            d_result.append([arr, (x, y, w, h), rect_coeff, extent, solidity, a_contour, a_rect, p_rect])

        return d_result

    def form_out_img(self):
        self.orig_img_2 = copy.copy(self.orig_img_1)
        self.data_img = np.zeros((config.PROC_IMG_RES[1] * 4, config.PROC_IMG_RES[0], 3), np.uint8)
        self.blank_img = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0] * 2, 3), np.uint8)
        #self.extent_img = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0]), np.uint8)

        self.draw_on_mog_mask()
        self.draw_on_filtered_img()
        self.draw_on_filled_img()
        self.draw_on_orig_img_1()
        self.draw_on_orig_img_2()
        self.draw_on_extent_img()
        self.draw_on_data_img()
        self.draw_on_blank_img()

        self.extent_img = cv2.cvtColor(self.extent_img, cv2.COLOR_GRAY2BGR)
        self.mog_mask = cv2.cvtColor(self.mog_mask, cv2.COLOR_GRAY2BGR)
        self.filtered_img = cv2.cvtColor(self.filtered_img, cv2.COLOR_GRAY2BGR)
        self.filled_img = cv2.cvtColor(self.filled_img, cv2.COLOR_GRAY2BGR)

        print 1, self.filled_img.shape[:2]
        print 2, self.blank_img.shape[:2]
        h_stack1 = np.hstack((self.mog_mask, self.filtered_img))
        h_stack2 = np.hstack((self.filled_img, self.extent_img))
        h_stack3 = np.hstack((self.orig_img_2, self.orig_img_1))
        h_stack4 = self.blank_img

        self.out_img = np.vstack((h_stack1, h_stack2, h_stack3, h_stack4))
        self.out_img = np.hstack((self.out_img, self.data_img))

    def draw_on_mog_mask(self):
        cv2.putText(self.mog_mask, str("MOG2 mask"), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def draw_on_filtered_img(self):
        cv2.putText(self.filtered_img, str("Filtered image"), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def draw_on_filled_img(self):
        cv2.putText(self.filled_img, str("Filled image"), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def draw_on_orig_img_1(self):
        cv2.putText(self.orig_img_1, "Rectangle area", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 255), 1, cv2.LINE_AA)

        for i in range(len(self.d_result)):
            x, y, w, h = self.d_result[i][1]
            cv2.rectangle(self.orig_img_1, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(self.orig_img_1, str(i), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)

    def draw_on_orig_img_2(self):
        cv2.putText(self.orig_img_2, "Contour area", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 255), 1, cv2.LINE_AA)
        cntrs = []
        for i in range(len(self.d_result)):
            cntrs.append(self.d_result[i][0])
        cv2.drawContours(self.orig_img_2, cntrs, -1, (0, 255, 0), 1)

    def draw_on_extent_img(self):
        cv2.putText(self.extent_img, "Split", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 0, cv2.LINE_AA)

    def draw_on_data_img(self):
        obj_y0 = 15
        obj_dy = 0
        interval = 20

        for i in range(len(self.d_result)):
            cv2.putText(self.data_img, "Object %s:" % i, (15, obj_y0 + obj_dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)
            obj_dy += interval
            cv2.putText(self.data_img, "x: %s, y: %s, w: %s, h: %s, " % (self.d_result[i][1][0], self.d_result[i][1][1],
                                                                         self.d_result[i][1][2],
                                                                         self.d_result[i][1][3]), (15, obj_y0 + obj_dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            obj_dy += interval
            cv2.putText(self.data_img, "h/w ratio: %s" % round((float(self.d_result[i][1][3])/self.d_result[i][1][2]), 3), (15, obj_y0 + obj_dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            obj_dy += interval
            is_coeff_belongs = config.COEFF_RANGE[0] < self.d_result[i][2] < config.COEFF_RANGE[1]
            if is_coeff_belongs:
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)
            cv2.putText(self.data_img, "Rectangle coeff: %s" % self.d_result[i][2], (15, obj_y0 + obj_dy), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        color, 1, cv2.LINE_AA)
            obj_dy += interval
            cv2.putText(self.data_img, "Rectangle area: %s" % self.d_result[i][6], (15, obj_y0 + obj_dy), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            obj_dy += interval
            cv2.putText(self.data_img, "Rectangle perimeter: %s" % self.d_result[i][7], (15, obj_y0 + obj_dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            obj_dy += interval
            cv2.putText(self.data_img, "Contour area: %s" % self.d_result[i][5], (15, obj_y0 + obj_dy), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            obj_dy += interval
            cv2.putText(self.data_img, "Extent coeff: %s" % self.d_result[i][3], (15, obj_y0 + obj_dy), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            obj_dy += interval

            # self.d_result.append([arr, (x, y, w, h), rect_coeff, extent, solidity, a_contour, a_rect, p_rect])
    def draw_on_blank_img(self):
        cv2.putText(self.blank_img, str(self.frame_m_status), (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
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

