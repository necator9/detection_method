import sqlite3
import os
import conf
import pickle
import threading
import numpy as np
import cv2
import copy
import glob
import time
import Queue

import detection_logging

SAVER_LOG = detection_logging.create_log("saver.log", "Saver")
SAVE_COUNTER = int()


def blank_fn(*args, **kwargs):
    pass


class Saving(threading.Thread):
    def __init__(self, data_frame_q, draw_frame_q):
        super(Saving, self).__init__(name="saving")

        self._is_running = bool()
        self.data_frame_q = data_frame_q
        self.draw_frame_q = draw_frame_q

        self.db_obj = Database(self.__gen_name("sql_database"))
        self.pickle_obj = PickleWrap(self.__gen_name("pickle_data.pkl"))
        self.draw_obj = Draw()

    def run(self):
        SAVER_LOG.info("Starting the Saving thread...")
        self._is_running = True
        self.check_if_dir_exists()

        while self._is_running:
            self.write()
            # SAVER_LOG.debug("Entry has been written")

        self.finish_writing()
        self.db_obj.quit()
        self.pickle_obj.quit()

        SAVER_LOG.info("Saving thread has been finished")

    def write(self):
        try:
            data_frame = self.data_frame_q.get(timeout=2)
        except Queue.Empty:
            SAVER_LOG.warning("Exception has raised, data_frame_q is empty")

            return 1

        try:
            draw_frame = self.draw_frame_q.get(timeout=2)
        except Queue.Empty:
            SAVER_LOG.warning("Exception has raised, draw_frame_q is empty")

            return 1

        self.db_obj.write(data_frame)

        self.pickle_obj.add(data_frame)

        self.draw_obj.form_out_img(data_frame, draw_frame)
        self.draw_obj.save()

        global SAVE_COUNTER
        SAVE_COUNTER += 1

    def finish_writing(self):
        SAVER_LOG.info("Writing finishing...")
        if not self.data_frame_q.empty():
            q_size = self.data_frame_q.qsize()
            for i in range(self.data_frame_q.qsize()):
                SAVER_LOG.info("{} elements in queue are remaining to write".format(q_size - i))
                self.write()

        SAVER_LOG.warning("{} elements HAVE BEEN NOT WRITTEN".format(self.data_frame_q.qsize()))

    def check_if_dir_exists(self):
        if not os.path.isdir(conf.OUT_DIR):
            os.makedirs(conf.OUT_DIR)
            SAVER_LOG.info("OUTPUT directory does not exists. New folder has been created")

    def quit(self):
        self._is_running = False

    @staticmethod
    def __gen_name(name):
        i = 0
        while True:
            name_plus_counter = ("{0}{1}" + name).format(str(i).zfill(3), "_")
            path_plus_name = os.path.join(conf.OUT_DIR, name_plus_counter)
            if not os.path.exists(path_plus_name):
                return path_plus_name
            else:
                i += 1


class Database(object):
    def __init__(self, db_name):
        if conf.WRITE_TO_DB:
            self.db_name = db_name
            SAVER_LOG.info("Database name: {}".format(self.db_name))
            self.db = sqlite3.connect(self.db_name, check_same_thread=False)
            self.cur = self.db.cursor()
            self.table_name = "Data_for_{}".format(conf.OUT_DIR.split("/")[-1])
            self.write_table()
        else:
            self.write = blank_fn
            self.quit = blank_fn

    def write_table(self):
        SAVER_LOG.debug("Database table has been written")

        self.cur.execute('''CREATE TABLE {} (Img_name TEXT, Obj_id INT, Status TEXT, Base_status TEXT, Br_status TEXT,  
                                        Rect_coeff REAL, Extent_coeff REAL, Br_ratio REAL, hw_ratio REAL, 
                                        Contour_area REAL, Rect_area INT, Rect_perimeter INT, Br_cross_area INT, 
                                        x INT, y INT, w INT, h INT )'''.format(self.table_name))

        self.db.commit()

    def write(self, d_frame):

        img_name = str(SAVE_COUNTER)
        db_arr = self.get_base_params(d_frame.base_objects, img_name)

        self.cur.executemany('''INSERT INTO {}(Img_name, Obj_id, Status, Base_status, Br_status,  Rect_coeff, 
                                          Extent_coeff, Br_ratio, hw_ratio, Contour_area, Rect_area, Rect_perimeter, 
                                          Br_cross_area, x, y, w, h) 
                                          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''.format(self.table_name), db_arr)

        if len(d_frame.ex_objects) > 0:
            img_name += "_split"
            db_split_arr = self.get_base_params(d_frame.ex_objects, img_name)
            self.cur.executemany('''INSERT INTO {}(Img_name, Obj_id, Status, Base_status, Br_status, Rect_coeff, 
                                              Extent_coeff, Br_ratio, hw_ratio, Contour_area, Rect_area, Rect_perimeter, 
                                              Br_cross_area, x, y, w, h) 
                                              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''.format(
                self.table_name),
                            db_split_arr)

        self.db.commit()

    @staticmethod
    def get_base_params(objects, img_name):
        # img_name = str(config.COUNTER).zfill(4)
        db_arr = list()
        if len(objects) > 0:
            for obj in objects:
                db_arr.append(
                    [img_name, obj.obj_id, str(obj.gen_status), str(obj.base_status), str(obj.br_status), obj.rect_coef,
                     obj.extent, obj.br_ratio,
                     obj.h_w_ratio, obj.contour_area, obj.rect_area, obj.rect_perimeter, obj.br_cr_area,
                     obj.base_rect[0], obj.base_rect[1], obj.base_rect[2], obj.base_rect[3]])

        return db_arr

    def quit(self):
        SAVER_LOG.info("Closing the database...")
        self.db.commit()
        self.db.close()


class PickleWrap(object):
    def __init__(self, pickle_name):
        if conf.WRITE_TO_PICKLE:
            self.pickle_name = pickle_name
            self.pickle_data = list()
        else:
            self.add = blank_fn
            self.quit = blank_fn

    def add(self, data_frame):
        self.pickle_data.append(data_frame.base_objects)

    def quit(self):
        with open(self.pickle_name, 'wb') as output:
            pickle.dump(self.pickle_data, output, pickle.HIGHEST_PROTOCOL)


class ImgStructure(object):
    def __init__(self, name=str()):
        self.data = np.dtype('uint8')
        self.name = name


class DrawImgStructure(object):
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


class Draw(object):
    def __init__(self):
        if conf.SAVE_IMG:
            self.out_img = ImgStructure("Detection result")
            self.draw_img_structure = DrawImgStructure()
            self.img_name = str()
            self.x_border = np.dtype('uint8')
            self.y_border = np.dtype('uint8')
            self.borders_updated_flag = bool()

        else:
            self.form_out_img = blank_fn
            self.save = blank_fn

    def update_borders(self):
        if not self.borders_updated_flag:
            self.borders_updated_flag = True

            self.x_border = np.zeros((conf.PROC_IMG_RES[1], 1, 3), np.uint8)
            self.x_border[:] = (0, 0, 255)
            self.y_border = np.zeros((1, conf.PROC_IMG_RES[0] * 3 + 2, 3), np.uint8)
            self.y_border[:] = (0, 0, 255)

    def form_out_img(self, data_frame, draw_frame):

        self.update_borders()

        self.draw_img_structure = draw_frame

        self.img_name = os.path.join(conf.OUT_DIR, "img_{}.jpeg".format(SAVE_COUNTER))

        self.draw_img_structure.filled_mask.data = copy.copy(data_frame.filled_mask)
        self.draw_img_structure.rect_cont.data = copy.copy(data_frame.orig_img)
        self.draw_img_structure.ex_rect_cont.data = copy.copy(data_frame.orig_img)

        for attr, value in self.draw_img_structure.__dict__.iteritems():
            if len(value.data.shape) == 2:
                value.data = cv2.cvtColor(value.data, cv2.COLOR_GRAY2BGR)

            if len(value.data.shape) == 0:
                value.data = np.zeros((conf.PROC_IMG_RES[1], conf.PROC_IMG_RES[0], 3), np.uint8)

            self.__put_name(value.data, value.name)

        self.__put_margin(self.draw_img_structure.rect_cont.data)

        self.__put_status(self.draw_img_structure.status.data, data_frame.base_frame_status)
        self.__put_status(self.draw_img_structure.ex_status.data, data_frame.ex_frame_status)

        self.__draw_rects(self.draw_img_structure.rect_cont.data, data_frame.base_objects)
        self.__draw_rects(self.draw_img_structure.ex_rect_cont.data, data_frame.ex_objects)

        self.__draw_rects_br_cr(self.draw_img_structure.rect_cont.data, data_frame.base_objects)

        # self.__draw_contour_areas(self.cont.data, d_frame.base_contours)
        # self.__draw_contour_areas(self.rect_cont.data, d_frame.base_contours)

        h_stack1 = np.hstack((self.draw_img_structure.mog_mask.data, self.x_border,
                              self.draw_img_structure.filtered_mask.data,
                              self.x_border, self.draw_img_structure.filled_mask.data))
        h_stack2 = np.hstack((self.draw_img_structure.bright_mask.data, self.x_border,
                              self.draw_img_structure.rect_cont.data,
                              self.x_border, self.draw_img_structure.status.data))
        h_stack3 = np.hstack(
            (self.draw_img_structure.extent_split_mask.data, self.x_border, self.draw_img_structure.ex_rect_cont.data,
             self.x_border, self.draw_img_structure.ex_status.data))

        self.out_img.data = np.vstack((h_stack1, self.y_border, h_stack2, self.y_border, h_stack3))

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
        cv2.putText(img, str(status), (80, 95), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA)


    @staticmethod
    def __put_margin(img):
        x_left_up = conf.X_MARGIN
        y_left_up = 0
        x_left_down = x_left_up
        y_left_down = conf.PROC_IMG_RES[1]

        x_right_up = conf.PROC_IMG_RES[0] - conf.X_MARGIN - 1
        y_right_up = 0
        x_right_down = x_right_up
        y_right_down = y_left_down

        cv2.line(img, (x_left_up, y_left_up), (x_left_down, y_left_down), (255, 0, 0), 1)
        cv2.line(img, (x_right_up, y_right_up), (x_right_down, y_right_down), (255, 0, 0), 1)

    # def show(self):
    #     cv2.imshow(self.out_img.name, self.out_img.data)
    #     cv2.waitKey(1)
    #     time.sleep(1)

    def save(self):
        # Save JPEG with proper name
        path = os.path.join(conf.OUT_DIR, self.img_name)
        cv2.imwrite(path, self.out_img.data)


# class Csv(object):
#     def __init__(self, name):
#         self.name = name + ".csv"
#         fieldnames = ["Img_name", "Object_no", "Status", "Rect_coeff", "hw_ratio", "Contour_area", "Rect_area",
#                       "Rect_perimeter", "Extent", "x", "y", "w", "h"]
#         self.f = open(name, 'w')
#         self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
#         self.writer.writeheader()
#
#     def write(self, base_objects, img_name):
#         for i, obj in enumerate(base_objects):
#             self.writer.writerow({"Img_name": img_name, "Object_no": i + 1, "Status": obj.obj_status,
#                                   "Rect_coeff": obj.rect_coef, "hw_ratio": obj.h_w_ratio,
#                                   "Contour_area": obj.contour_area, "Rect_area": obj.rect_area,
#                                   "Rect_perimeter": obj.rect_perimeter, "Extent": obj.extent,
#                                   "x": obj.x, "y": obj.y, "w": obj.w, "h": obj.h})
#
#     def quit(self):
#         self.f.close()

class TimeCounter(object):
    def __init__(self, watch_name):
        if conf.WATCH_LOGS:
            self.watch_name = watch_name
            self.watch_log = detection_logging.create_log("{}.log".format(self.watch_name), self.watch_name)
            self.start_time = float()
            self.res_time = float()
        else:
            self.note_time = blank_fn
            self.get_time = blank_fn

    def note_time(self):
        self.start_time = time.time()

    def get_time(self):
        self.res_time = time.time() - self.start_time
        self.watch_log.info("{} takes {}s".format(self.watch_name, self.res_time))


















