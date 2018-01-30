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
            self.db = DbSave(db_name)
            logger.info("Database name: %s" % db_name)

        if config.WRITE_TO_CSV:
            csv_name = self.gen_name("csv_file")
            self.csv = CsvSave(csv_name)

        self.x_range = (config.MARGIN[0], config.PROC_IMG_RES[0] - config.MARGIN[0])
        self.y_range = (config.MARGIN[1], config.PROC_IMG_RES[1] - config.MARGIN[1])

        self.red_x_border = list()
        self.red_y_border = list()

    # Main thread routine
    def run(self):
        logger.info("Detection has started")
        self.running = True

        img_fr = PreProcess()

        while config.COUNTER < config.IMG_IN_DIR and self.running:
            path_to_img = glob.glob(os.path.join(config.IN_DIR, "img_%s_*.jpeg" % config.COUNTER))[0]
            self.img_name = path_to_img.split("/")[-1]

            data_frame = DataFrame()
            draw_img = Draw()

            data_frame.orig_img.data = cv2.imread(path_to_img)

            draw_img.mog_mask.data, draw_img.filtered_mask.data = img_fr.process(data_frame)

            draw_img.ex_filled_mask.data = data_frame.calculate()

            if config.SHOW_IMG or config.SAVE_IMG:
                draw_img.form_out_img(data_frame)

                if config.SHOW_IMG:
                    draw_img.show()
                    time.sleep(0.1)

                if config.SAVE_IMG:
                    draw_img.save(self.img_name)

            if config.WRITE_TO_DB:
                self.db.db_write(data_frame, config.COUNTER)

            if config.WRITE_TO_CSV:
                self.csv.write(data_frame.base_objects, self.img_name)

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

    # Stop and quit the thread operation.
    def quit(self):
        self.running = False
        self.stop_event.clear()
        logger.info("Detector has quit")


class ObjParams(object):
    def __init__(self, obj_id=int()):
        self.obj_id = obj_id
        self.obj_status = bool()

        self.x = int()
        self.y = int()
        self.w = int()
        self.h = int()

        self.h_w_ratio = float()
        self.base_rects = [[0, 0, 0, 0]]

        self.br_cr_rects = [[0, 0, 0, 0]]
        self.br_cr_area = int()

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
        self.x, self.y, self.w, self.h = cv2.boundingRect(contour)
        self.h_w_ratio = round(float(self.h) / self.w, 3)
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

    # TODO Transfer into dataframe class

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


class PreProcess(object):
    def __init__(self):
        self.__mog = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.__filtering_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.F_KERNEL_SIZE)

    def process(self, d_frame):

        orig_img = resize(d_frame.orig_img.data, width=config.PROC_IMG_RES[0], height=config.PROC_IMG_RES[1])

        mog_mask = self.__mog.apply(orig_img)
        _, mog_mask = cv2.threshold(mog_mask, 127, 255, cv2.THRESH_BINARY)

        if config.F_KERNEL_SIZE[0] > 0 and config.F_KERNEL_SIZE[1] > 0:
            filtered_mask = cv2.morphologyEx(mog_mask, cv2.MORPH_OPEN, self.__filtering_kernel)
        else:
            filtered_mask = copy.copy(mog_mask)

        filled_mask = cv2.dilate(filtered_mask, None, iterations=config.DILATE_ITERATIONS)

        d_frame.orig_img.data = orig_img
        d_frame.filled_mask.data = filled_mask

        return mog_mask, filtered_mask


# TODO merge DataFrame and Preprocess classes


class DataFrame(object):
    def __init__(self):
        self.orig_img = ImgStructure("Original image")
        self.filled_mask = ImgStructure("Filled MOG mask")

        self.base_frame_status = None  # Can be False/True type
        self.ex_frame_status = None  # Can be False/True type
        self.base_objects = list()
        self.ex_objects = list()
        self.base_contours = list()

        self.br_rects = list()

    def calculate(self):
        self.base_objects, self.base_contours = self.__basic_process(self.filled_mask.data)

        ex_filled_mask = self.__extent_split_process()
        self.ex_objects, _ = self.__basic_process(ex_filled_mask)

        self.ex_frame_status = self.__take_frame_status(self.ex_objects)
        self.base_frame_status = self.__take_frame_status(self.base_objects)

        ex_filled_mask = self.__brightness_process()

        return ex_filled_mask

    @staticmethod
    def __basic_process(filled_mask):
        objects = list()
        _, contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for obj_id, contour in enumerate(contours):
            obj = ObjParams(obj_id)
            obj.process_obj(contour)
            objects.append(obj)

        return objects, contours

    def __extent_split_process(self):
        ex_filled_mask = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0]), np.uint8)
        for obj_id, obj in enumerate(self.base_objects):
            is_extent = obj.extent < 0.6
            is_rect_coeff = -20000 < obj.rect_coef < -10000  # Try to reduce to -5000 or so

            if is_extent and is_rect_coeff:
                ex_filled_mask = copy.copy(self.filled_mask.data)
                cv2.line(ex_filled_mask,
                         (obj.x + int(obj.w / 2), 0), (obj.x + int(obj.w / 2), config.PROC_IMG_RES[1]), (0, 0, 0), 3)

        return ex_filled_mask

    def __brightness_process(self):
        brightness_mask = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0], 3), np.uint8)
        brightness_mask[np.where((self.orig_img.data > [250, 250, 250]).all(axis=2))] = [255]
        brightness_mask = cv2.cvtColor(brightness_mask, cv2.COLOR_BGR2GRAY)
        # filtering_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.F_KERNEL_SIZE)
        #
        # brightness_mask = cv2.morphologyEx(brightness_mask, cv2.MORPH_OPEN, filtering_kernel)
        # brightness_mask = cv2.dilate(brightness_mask, None, iterations=config.DILATE_ITERATIONS)

        _, contours, _ = cv2.findContours(brightness_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        brightness_mask = cv2.cvtColor(brightness_mask, cv2.COLOR_GRAY2BGR)

        self.br_rects = [cv2.boundingRect(contour) for contour in contours]

        for obj in self.base_objects:
            base_rect = [obj.x, obj.y, obj.w, obj.h]
            obj.br_cr_rects = [self.intersection(base_rect, br_rect) for br_rect in self.br_rects]

        #     obj.br_cr_rect = self.intersection(base_rect, obj.br_rect)
        #
        #     obj.br_cr_area = sum([w * h for w, h in zip(obj.br_cr_w, obj.br_cr_h)])
            a = obj.br_cr_rects
        return brightness_mask

    @staticmethod
    def __take_frame_status(objects):
        status_arr = list()
        for obj in objects:
            status_arr.append(obj.obj_status)
        return any(status_arr)

    def get_base_params(self):
        db_arr = list()
        if len(self.base_objects) > 0:
            for obj in self.base_objects:
                db_arr.append([obj.obj_status, obj.rect_coef, obj.h_w_ratio, obj.contour_area, obj.rect_area,
                               obj.rect_perimeter, obj.extent, obj.x, obj.y, obj.w, obj.h])

        else:
            db_arr.append([None] * 11)

        return db_arr

    @staticmethod
    def intersection(area_1, area_2):
        x = max(area_1[0], area_2[0])
        y = max(area_1[1], area_2[1])
        w = min(area_1[0] + area_1[2], area_2[0] + area_2[2]) - x
        h = min(area_1[1] + area_1[3], area_2[1] + area_2[3]) - y

        if w < 0 or h < 0:
            return [0, 0, 0, 0]

        return [x, y, w, h]

    def calc(self, areas_1, areas_2):
        for area_1 in areas_1:
            intersection = [self.intersection(area_1, area_2) for area_2 in areas_2]
            print intersection


class ImgStructure(object):
    def __init__(self, name=str()):
        self.data = np.dtype('uint8')
        self.name = name


class Draw(object):
    def __init__(self):
        self.mog_mask = ImgStructure("MOG mask")
        self.filtered_mask = ImgStructure("Filtered MOG mask")
        self.filled_mask = ImgStructure("Filled MOG mask")

        self.cont = ImgStructure("Contour area")
        self.rect_cont = ImgStructure("Rectangle area")
        self.status = ImgStructure("Original status")

        self.ex_filled_mask = ImgStructure("Extent-split mask")
        self.ex_rect_cont = ImgStructure("Extent-split image")
        self.ex_status = ImgStructure("Extent-split status")

        self.out_img = ImgStructure("Detection result")

    def form_out_img(self, d_frame):

        # def check_ratio(self):
        #     if config.PROC_IMG_RES[0] != self.res_in_img.shape[:2][1] or config.PROC_IMG_RES[1]
        # != self.res_in_img.shape[:2][0]:  # Remake to run once in loop
        #         config.PROC_IMG_RES[0] = self.res_in_img.shape[:2][1]
        #         config.PROC_IMG_RES[1] = self.res_in_img.shape[:2][0]

        self.filled_mask.data = copy.copy(d_frame.filled_mask.data)
        self.cont.data = copy.copy(d_frame.orig_img.data)
        self.rect_cont.data = copy.copy(d_frame.orig_img.data)
        self.ex_rect_cont.data = copy.copy(d_frame.orig_img.data)

        x_border = np.zeros((config.PROC_IMG_RES[1], 1, 3), np.uint8)
        x_border[:] = (0, 0, 255)
        y_border = np.zeros((1, config.PROC_IMG_RES[0] * 3 + 2, 3), np.uint8)
        y_border[:] = (0, 0, 255)

        for attr, value in self.__dict__.iteritems():
            if attr != "out_img":
                if len(value.data.shape) == 2:  # TODO make it correctly
                    value.data = cv2.cvtColor(value.data, cv2.COLOR_GRAY2BGR)
                if len(value.data.shape) == 0:
                    value.data = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0], 3), np.uint8)

                self.__put_name(value.data, value.name)

        self.__put_margin(self.rect_cont.data)

        self.__put_status(self.status.data, d_frame.base_frame_status)
        self.__put_status(self.ex_status.data, d_frame.ex_frame_status)

        self.__draw_rect_areas(self.rect_cont.data, d_frame.base_objects)
        self.__draw_rect_areas(self.ex_rect_cont.data, d_frame.ex_objects)
        self.__draw_rect_areas(self.cont.data, d_frame.br_rects)

        for obj in d_frame.base_objects:
            for rect in obj.br_cr_rects:
                x, y, w, h = rect[0], rect[1], rect[2], rect[3]
                cv2.rectangle(self.rect_cont.data, (x, y), (x + w, y + h), (0, 0, 255), -1)

        # self.__draw_contour_areas(self.cont.data, d_frame.base_contours)
        # self.__draw_contour_areas(self.rect_cont.data, d_frame.base_contours)

        h_stack1 = np.hstack((self.mog_mask.data, x_border, self.filtered_mask.data, x_border, self.filled_mask.data))
        h_stack2 = np.hstack((self.cont.data, x_border, self.rect_cont.data, x_border, self.status.data))
        h_stack3 = np.hstack(
            (self.ex_filled_mask.data, x_border, self.ex_rect_cont.data, x_border, self.ex_status.data))

        self.out_img.data = np.vstack((h_stack1, y_border, h_stack2, y_border, h_stack3))

        return self.out_img

    @staticmethod
    def __put_name(img, text):
        cv2.putText(img, text, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def __draw_rect_areas(img, objects):
        for obj in objects:
            cv2.rectangle(img, (obj.x, obj.y), (obj.x + obj.w, obj.y + obj.h), (0, 255, 0), 2)
            cv2.putText(img, str(obj.obj_id + 1), (obj.x + 5, obj.y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)

    @staticmethod
    def __draw_contour_areas(img, contours):
        cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

    @staticmethod
    def __put_status(img, status):
        cv2.putText(img, str(status), (80, 95), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def __put_margin(img):
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

    def show(self):
        cv2.imshow(self.out_img.name, self.out_img.data)
        cv2.waitKey(1)

    def save(self, img_name):
        # Save JPEG with proper name
        path = os.path.join(config.OUT_DIR, img_name)
        cv2.imwrite(path, self.out_img.data)


class DbSave(object):
    def __init__(self, db_name):
        self.db_name = db_name
        self.db = sqlite3.connect(self.db_name)
        self.table_name = str()
        self.d_frame = list()

    def db_write(self, d_frame, counter):
        self.table_name = "img_%s" % str(counter).zfill(4)
        self.d_frame = d_frame
        self.db = sqlite3.connect(self.db_name)
        cur = self.db.cursor()
        db_arr = self.d_frame.get_base_params()

        cur.execute('''CREATE TABLE %s (Status TEXT, Rect_coeff REAL, hw_ratio REAL, Contour_area REAL, Rect_area REAL, 
                                        Rect_perimeter REAL, Extent_coeff REAL, x REAL, y REAL, w REAL, h REAL )'''
                    % self.table_name)

        cur.executemany('''INSERT INTO %s(Status, Rect_coeff, hw_ratio, Contour_area, Rect_area, Rect_perimeter,
                                        Extent_coeff, x, y, w, h) VALUES(?,?,?,?,?,?,?,?,?,?,?)'''
                        % self.table_name, db_arr)

        self.db.commit()

    def quit(self):
        self.db.commit()
        self.db.close()


class CsvSave(object):
    def __init__(self, name):
        self.name = name + ".csv"
        print self.name
        fieldnames = ["Img_name", "Object_no", "Status", "Rect_coeff", "hw_ratio", "Contour_area", "Rect_area",
                      "Rect_perimeter", "Extent", "x", "y", "w", "h"]
        self.f = open(name, 'w')
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.writer.writeheader()

    def write(self, base_objects, img_name):
        for i, obj in enumerate(base_objects):
            self.writer.writerow({"Img_name": img_name, "Object_no": i + 1, "Status": obj.obj_status,
                                  "Rect_coeff": obj.rect_coef, "hw_ratio": obj.h_w_ratio,
                                  "Contour_area": obj.contour_area, "Rect_area": obj.rect_area,
                                  "Rect_perimeter": obj.rect_perimeter, "Extent": obj.extent,
                                  "x": obj.x, "y": obj.y, "w": obj.w, "h": obj.h})

    def quit(self):
        self.f.close()
