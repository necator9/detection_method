import logging
import threading
import time
import pickle

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

        if config.WRITE_TO_PICKLE:
            self.pickle_data = list()

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

            draw_img.bright_mask.data, draw_img.extent_split_mask.data = data_frame.calculate()

            if config.SHOW_IMG or config.SAVE_IMG:
                draw_img.form_out_img(data_frame)

                if config.SHOW_IMG:
                    draw_img.show()
                    time.sleep(0.1)

                if config.SAVE_IMG:
                    draw_img.save(self.img_name)

            if config.WRITE_TO_DB:
                self.db.db_write(data_frame)

            if config.WRITE_TO_CSV:
                self.csv.write(data_frame.base_objects, self.img_name)

            if config.WRITE_TO_PICKLE:
                self.pickle_data.append(data_frame.base_objects)

            config.COUNTER += 1

        if config.WRITE_TO_DB:
            self.db.quit()

        if config.WRITE_TO_CSV:
            self.csv.quit()

        if config.WRITE_TO_PICKLE:
            name = "%s_%s_data.pkl" % (config.OUT_DIR.split("/")[-2], config.OUT_DIR.split("/")[-1])
            path = os.path.join(config.OUT_DIR, name)
            with open(path, 'wb') as output:
                pickle.dump(self.pickle_data, output, pickle.HIGHEST_PROTOCOL)

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
        self.set_ratio(orig_img)  # TODO Remake to run once in loop
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

    @ staticmethod
    def set_ratio(img):
        actual_w, actual_h = img.shape[:2][1], img.shape[:2][0]
        if config.PROC_IMG_RES[0] != actual_w or config.PROC_IMG_RES[1] != actual_h:
            config.PROC_IMG_RES[0] = actual_w
            config.PROC_IMG_RES[1] = actual_h


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
        ex_filled_mask = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0]), np.uint8) # create minimal image
        for obj in self.base_objects:
            is_extent = obj.extent < 0.6
            is_rect_coeff = -20000 < obj.rect_coef < -10000  # Try to reduce to -5000 or so

            if is_extent and is_rect_coeff and not obj.base_status and not obj.br_status:
                x, y, w, h = obj.base_rect

                ex_filled_mask[y:y+h, x:x + w] = self.filled_mask.data[y:y+h, x:x + w]
                cv2.line(ex_filled_mask, (x + int(w / 2), 0), (x + int(w / 2), config.PROC_IMG_RES[1]), (0, 0, 0), 3)

        return ex_filled_mask

    def __brightness_process(self, objects):
        brightness_mask = np.zeros((config.PROC_IMG_RES[1], config.PROC_IMG_RES[0], 3), np.uint8)
        # if len(self.base_objects) > 0:  # keep it for optimization for BBB
        brightness_mask[np.where((self.orig_img.data > [220, 220, 220]).all(axis=2))] = [255]
        brightness_mask = cv2.cvtColor(brightness_mask, cv2.COLOR_BGR2GRAY)
        _, contours, _ = cv2.findContours(brightness_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        brightness_mask = cv2.cvtColor(brightness_mask, cv2.COLOR_GRAY2BGR)

        self.br_rects = [cv2.boundingRect(contour) for contour in contours]

        for obj in objects:
            # if obj.base_status: # keep it for optimization for BBB
            obj.br_cr_rects = [self.intersection(obj.base_rect, br_rect) for br_rect in self.br_rects]
            obj.br_cr_area = sum([rect[2] * rect[3] for rect in obj.br_cr_rects])
            obj.br_ratio = round(float(obj.br_cr_area) / obj.rect_area, 3)
            obj.br_status = obj.br_ratio > config.BRIGHTNESS_THRESHOLD

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
        self.mog_mask = ImgStructure("Original MOG mask")
        self.filtered_mask = ImgStructure("Filtered mask")
        self.filled_mask = ImgStructure("Dilated mask")

        self.extent_split_mask = ImgStructure("Extent-split mask")
        self.rect_cont = ImgStructure(" ")  # Basic detection + Bright areas
        self.status = ImgStructure("Original status")

        self.bright_mask = ImgStructure("Brightness mask")
        self.ex_rect_cont = ImgStructure("Extent-split")
        self.ex_status = ImgStructure("Extent-split status")

        self.out_img = ImgStructure("Detection result")

    def form_out_img(self, d_frame):

        self.filled_mask.data = copy.copy(d_frame.filled_mask.data)
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

        self.__draw_rects(self.rect_cont.data, d_frame.base_objects)
        self.__draw_rects(self.ex_rect_cont.data, d_frame.ex_objects)

        self.__draw_rects_br_cr(self.rect_cont.data, d_frame.base_objects)

        # self.__draw_contour_areas(self.cont.data, d_frame.base_contours)
        # self.__draw_contour_areas(self.rect_cont.data, d_frame.base_contours)

        h_stack1 = np.hstack((self.mog_mask.data, x_border, self.filtered_mask.data, x_border, self.filled_mask.data))
        h_stack2 = np.hstack((self.bright_mask.data, x_border, self.rect_cont.data, x_border, self.status.data))
        h_stack3 = np.hstack(
            (self.extent_split_mask.data, x_border, self.ex_rect_cont.data, x_border, self.ex_status.data))

        self.out_img.data = np.vstack((h_stack1, y_border, h_stack2, y_border, h_stack3))

        return self.out_img

    @staticmethod
    def __put_name(img, text):
        cv2.putText(img, text, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def __draw_rects(img, objects):

        for obj in objects:
            if obj.gen_status:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            x, y, w, h = obj.base_rect
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, str(obj.obj_id), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)

    @staticmethod
    def __draw_rects_br_cr(img, objects):
        for obj in objects:
            for rect in obj.br_cr_rects:
                x, y, w, h = rect
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), -1)

    @staticmethod
    def __draw_rects_br(img, rects):
        for rect in rects:
            x, y, w, h = rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), -1)

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
        self.table_name = "Data_for_" + config.OUT_DIR.split("/")[-1]
        self.d_frame = DataFrame()
        self.cur = self.db.cursor()

        self.cur.execute('''CREATE TABLE %s (Img_name TEXT, Obj_id INT, Status TEXT, Base_status TEXT, Br_status TEXT,  Rect_coeff REAL, Extent_coeff REAL, 
                                            Br_ratio REAL, hw_ratio REAL, Contour_area REAL, Rect_area INT, 
                                            Rect_perimeter INT, Br_cross_area INT, x INT, y INT, w INT, h INT )'''
                                            % self.table_name)

    def db_write(self, d_frame):

        self.d_frame = d_frame
        self.db = sqlite3.connect(self.db_name)

        img_name = str(config.COUNTER)
        db_arr = self.get_base_params(self.d_frame.base_objects, img_name)

        self.cur = self.db.cursor()

        self.cur.executemany('''INSERT INTO %s(Img_name, Obj_id, Status, Base_status, Br_status,  Rect_coeff, Extent_coeff, Br_ratio, hw_ratio, 
                                              Contour_area, Rect_area, Rect_perimeter, Br_cross_area, x, y, w, h) 
                                              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''' % self.table_name, db_arr)

        if len(d_frame.ex_objects) > 0:
            img_name += "_split"
            db_split_arr = self.get_base_params(self.d_frame.ex_objects, img_name)
            self.cur.executemany('''INSERT INTO %s(Img_name, Obj_id, Status, Base_status, Br_status, Rect_coeff, Extent_coeff, Br_ratio, hw_ratio, 
                                                         Contour_area, Rect_area, Rect_perimeter, Br_cross_area, x, y, w, h) 
                                                         VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''' % self.table_name,
                                 db_split_arr)

        self.db.commit()


    @staticmethod
    def get_base_params(objects, img_name):

        # img_name = str(config.COUNTER).zfill(4)
        db_arr = list()
        if len(objects) > 0:
            for obj in objects:
                db_arr.append([img_name, obj.obj_id, str(obj.gen_status), str(obj.base_status), str(obj.br_status), obj.rect_coef, obj.extent, obj.br_ratio,
                               obj.h_w_ratio, obj.contour_area, obj.rect_area, obj.rect_perimeter, obj.br_cr_area,
                               obj.base_rect[0], obj.base_rect[1], obj.base_rect[2], obj.base_rect[3]])

        return db_arr

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
