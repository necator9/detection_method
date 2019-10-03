import sqlite3
import os
import pickle
import threading
import numpy as np
import cv2
import copy
import time
import Queue

import conf
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

        self.check_if_dir_exists()
        self.db_obj = Database(self.gen_name("sql_database"))
        self.pickle_obj = PickleWrap(self.gen_name("pickle_data.pkl"))
        self.draw_obj = Draw()

    def run(self):
        SAVER_LOG.info("Starting the Saving thread...")
        self._is_running = True

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

        self.draw_obj.save_multiple(data_frame, draw_frame)

        self.draw_obj.save_single(data_frame)

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
    def gen_name(name):
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
            self.table_name = 'object_parameters'
            self.write_table()
        else:
            self.write = blank_fn
            self.quit = blank_fn

    def write_table(self):
        SAVER_LOG.debug("Database table has been written")

        self.cur.execute('''CREATE TABLE {} (Img_name TEXT, Obj_id INT, Rect_coeff_diff REAL, Rect_coeff_ro REAL, 
        Rect_coeff_ao REAL, dist_ao REAL, c_a_ro REAL, c_a_ao REAL, Extent REAL, Status TEXT, Base_status TEXT, 
        Br_status TEXT, Br_ratio REAL, h_w_ratio_ao REAL, Br_cross_area INT, x_ao INT, y_ao INT, w_ao INT, h_ao INT, 
        x_ro INT, y_ro INT, w_ro INT, h_ro INT, o_class INT, c_a_rw REAL, w_rw REAL, h_rw REAL)'''.format(self.table_name))

        self.db.commit()

    def write(self, d_frame):

        img_name = str(SAVE_COUNTER)
        db_arr = self.get_base_params(d_frame.base_objects, img_name)

        self.cur.executemany('''INSERT INTO {}(Img_name, Obj_id, Rect_coeff_diff, Rect_coeff_ro, Rect_coeff_ao, dist_ao, 
        c_a_ro, c_a_ao, Extent, Status, Base_status, Br_status, Br_ratio, h_w_ratio_ao, Br_cross_area, x_ao, y_ao, w_ao, 
        h_ao, x_ro, y_ro, w_ro, h_ro, o_class, c_a_rw, w_rw, h_rw) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''.
                             format(self.table_name), db_arr)

        if len(d_frame.ex_objects) > 0:
            img_name += "_split"
            db_split_arr = self.get_base_params(d_frame.ex_objects, img_name)
            self.cur.executemany('''INSERT INTO {}(Img_name, Obj_id, Rect_coeff_diff, Rect_coeff_ro, Rect_coeff_ao, 
            dist_ao, c_a_ro, c_a_ao, Extent, Status, Base_status, Br_status, Br_ratio, h_w_ratio_ao,Br_cross_area, x_ao,
            y_ao, w_ao, h_ao, x_ro, y_ro, w_ro, h_ro, o_class, c_a_rw, w_rw, h_rw) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''.
                                 format(self.table_name), db_split_arr)

        self.db.commit()

    @staticmethod
    def get_base_params(objects, img_name):
        # img_name = str(config.COUNTER).zfill(4)
        db_arr = list()
        if len(objects) > 0:
            for obj in objects:
                db_arr.append(
                    [img_name, obj.obj_id, obj.rect_coef_diff, obj.rect_coef_ro, obj.rect_coef_ao, obj.dist_ao,
                     obj.c_a_ro, obj.c_a_ao, obj.extent_ao, str(obj.gen_status), str(obj.base_status),
                     str(obj.br_status), obj.br_ratio, obj.h_w_ratio_ao, obj.br_cr_area, obj.x_ao, obj.y_ao, obj.w_ao,
                     obj.h_ao, obj.x_ro, obj.y_ro, obj.w_ro, obj.h_ro, obj.o_class, obj.c_ao_rw, obj.w_ao_rw, obj.h_ao_rw])

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


class MultipleImagesFrame(object):
    def __init__(self):
        self.mog_mask = ImgStructure("Original MOG mask")
        self.filtered = ImgStructure("Filtered mask")
        self.filled_mask = ImgStructure("Dilated mask")

        self.extent_split_mask = ImgStructure("Extent-split mask")
        self.rect_cont = ImgStructure(" ")  # Basic detection + Bright areas
        self.status = ImgStructure("Original status")

        self.bright_mask = ImgStructure("Brightness mask")
        self.ex_rect_cont = ImgStructure("Extent-split")
        self.ex_status = ImgStructure("Extent-split status")


class Draw(object):
    def __init__(self):
        if conf.SAVE_VERBOSE:
            self.out_img = ImgStructure("Detection result")
            self.draw_img_structure = MultipleImagesFrame()
            self.x_border = np.dtype('uint8')
            self.y_border = np.dtype('uint8')
            self.borders_updated_flag = bool()

        if not conf.SAVE_VERBOSE:
            self.save_multiple = blank_fn

        if not conf.SAVE_SINGLE:
            self.save_single = blank_fn

        self.color_map = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (0, 0, 0), 4: (0, 255, 255)}

    def update_borders(self):
        if not self.borders_updated_flag:
            self.borders_updated_flag = True
            self.x_border = np.zeros((conf.RESIZE_TO[1], 1, 3), np.uint8)
            self.x_border[:] = (0, 0, 255)
            self.y_border = np.zeros((1, conf.RESIZE_TO[0] * 3 + 2, 3), np.uint8)
            self.y_border[:] = (0, 0, 255)

    def draw_multiple_images(self, data_frame, draw_frame):
        self.update_borders()

        self.draw_img_structure = draw_frame

        self.draw_img_structure.filled_mask.data = copy.copy(data_frame.filled)
        self.draw_img_structure.rect_cont.data = copy.copy(data_frame.orig_img)
        self.draw_img_structure.ex_rect_cont.data = copy.copy(data_frame.orig_img)

        for attr, value in self.draw_img_structure.__dict__.iteritems():
            if len(value.data.shape) == 2:
                value.data = cv2.cvtColor(value.data, cv2.COLOR_GRAY2BGR)
                # pass

            if len(value.data.shape) == 0:
                value.data = np.zeros((conf.RESIZE_TO[1], conf.RESIZE_TO[0], 3), np.uint8)

            self.put_name(value.data, value.name)

        self.put_margin(self.draw_img_structure.rect_cont.data)

        self.put_status(self.draw_img_structure.status.data, data_frame.base_frame_status)
        self.put_status(self.draw_img_structure.ex_status.data, data_frame.ex_frame_status)

        self.draw_rects(self.draw_img_structure.rect_cont.data, data_frame.base_objects)
        self.draw_virual_object(self.draw_img_structure.rect_cont.data, data_frame.base_objects)
        self.draw_rects(self.draw_img_structure.ex_rect_cont.data, data_frame.ex_objects)

        self.draw_rects_br_cr(self.draw_img_structure.rect_cont.data, data_frame.base_objects)

        # self.__draw_contour_areas(self.cont.data, d_frame.base_contours)
        # self.__draw_contour_areas(self.rect_cont.data, d_frame.base_contours)

        h_stack1 = np.hstack((self.draw_img_structure.mog_mask.data, self.x_border,
                              self.draw_img_structure.filtered.data,
                              self.x_border, self.draw_img_structure.filled_mask.data))
        h_stack2 = np.hstack((self.draw_img_structure.bright_mask.data, self.x_border,
                              self.draw_img_structure.rect_cont.data,
                              self.x_border, self.draw_img_structure.status.data))
        h_stack3 = np.hstack(
            (self.draw_img_structure.extent_split_mask.data, self.x_border, self.draw_img_structure.ex_rect_cont.data,
             self.x_border, self.draw_img_structure.ex_status.data))

        self.out_img.data = np.vstack((h_stack1, self.y_border, h_stack2, self.y_border, h_stack3))

        return self.out_img.data

    @staticmethod
    def draw_single_image(data_frame):
        data_frame = copy.copy(data_frame)
        Draw.put_name(data_frame.orig_img, " ")
        Draw.draw_rects(data_frame.orig_img, data_frame.base_objects)
        Draw.draw_rects_br_cr(data_frame.orig_img, data_frame.base_objects)

        return data_frame.orig_img

    @staticmethod
    def put_name(img, text):
        cv2.putText(img, text, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_rects(self, img, objects):
        for obj in objects:
            color = self.color_map[obj.o_class]

            x, y, w, h = obj.base_rect_ao
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, str(obj.obj_id), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)
            # Put distance value above the rectangle
            cv2.putText(img, str(round(obj.dist_ao, 1)), (x + 5, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 1, cv2.LINE_AA)

    def draw_virual_object(self, img, objects):
        for obj in objects:
            color = (255, 0, 0)
            x, y, w, h = (obj.x_ao + obj.w_ao / 2) - obj.w_ro / 2, obj.y_ro, obj.w_ro, obj.h_ro
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

    @staticmethod
    def draw_rects_br_cr(img, objects):
        for obj in objects:
            for rect in obj.br_cr_rects:
                x, y, w, h = rect
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)

    @staticmethod
    def draw_rects_br(img, rects):
        for rect in rects:
            x, y, w, h = rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), -1)

    @staticmethod
    def draw_contour_areas(img, contours):
        cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

    @staticmethod
    def put_status(img, status):
        cv2.putText(img, str(status), (80, 95), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA)


    @staticmethod
    def put_margin(img):
        x_left_up = conf.X_MARGIN
        y_left_up = 0
        x_left_down = x_left_up
        y_left_down = conf.RESIZE_TO[1]

        x_right_up = conf.RESIZE_TO[0] - conf.X_MARGIN - 1
        y_right_up = 0
        x_right_down = x_right_up
        y_right_down = y_left_down

        cv2.line(img, (x_left_up, y_left_up), (x_left_down, y_left_down), (255, 0, 0), 1)
        cv2.line(img, (x_right_up, y_right_up), (x_right_down, y_right_down), (255, 0, 0), 1)

    # def show(self):
    #     cv2.imshow(self.out_img.name, self.out_img.data)
    #     cv2.waitKey(1)
    #     time.sleep(1)

    def save(self, out_img, img_name):
        # Save JPEG with proper name
        SAVER_LOG.debug("Entry has been written")
        path = os.path.join(conf.OUT_DIR, "{}_{}.jpeg".format(img_name, SAVE_COUNTER))
        cv2.imwrite(path, out_img)

    def save_multiple(self, data_frame, draw_frame):
        out_img = self.draw_multiple_images(data_frame, draw_frame)
        self.save(out_img, "m_img")

    def save_single(self, data_frame):
        out_img = self.draw_single_image(data_frame)
        self.save(out_img, "s_img")


class TimeCounter(object):
    def __init__(self, watch_name):
        if conf.TIMERS:
            self.watch_name = watch_name
            self.watch_log = detection_logging.create_log("{}.log".format(self.watch_name), self.watch_name)
            self.start_time = float()
            self.res_time = float()
            self.average_time_list = list()
        else:
            self.note_time = blank_fn
            self.get_time = blank_fn
            self.get_average_time = blank_fn

    def note_time(self):
        self.start_time = time.time()

    def get_time(self):
        self.res_time = time.time() - self.start_time
        self.average_time_list.append(self.res_time)

        if len(self.average_time_list) == conf.TIME_WINDOW:
            self.__get_average_time()
            self.average_time_list = list()

    def __get_average_time(self):
        average_time = round(np.mean(self.average_time_list), 3)
        self.watch_log.info("{} iteration t: {}s, FPS: {}. Window size: {} ".format(self.watch_name, average_time,
                                                                                    round(1/average_time, 2),
                                                                                    conf.TIME_WINDOW))






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











