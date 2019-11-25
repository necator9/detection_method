import sqlite3
import os
import pickle
import threading
import numpy as np
import cv2
import time
import Queue

import conf
import detection_logging

SAVER_LOG = detection_logging.create_log("saver.log", "Saver")
SAVE_COUNTER = int()


def blank_fn(*args, **kwargs):
    pass


class Saving(threading.Thread):
    def __init__(self, data_frame_q):
        super(Saving, self).__init__(name="saving")

        self._is_running = bool()
        self.data_frame_q = data_frame_q

        self.check_if_dir_exists()
        self.db_obj = Database(self.gen_name("sql_database"))
        self.pickle_obj = PickleWrap(self.gen_name("pickle_data.pkl"))

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

        self.db_obj.write(data_frame)

        self.pickle_obj.add(data_frame)

        global SAVE_COUNTER
        SAVE_COUNTER += 1

    def finish_writing(self):
        SAVER_LOG.info("Finishing writing ...")
        if not self.data_frame_q.empty():
            q_size = self.data_frame_q.qsize()
            for i in range(self.data_frame_q.qsize()):
                SAVER_LOG.debug("{} elements in queue are remaining to write".format(q_size - i))
                self.write()

        SAVER_LOG.warning("{} elements HAVE BEEN NOT WRITTEN".format(self.data_frame_q.qsize()))
        self._is_running = False

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

        self.cur.execute('''CREATE TABLE {} (Img_name TEXT, Obj_id INT,
        Rect_coeff_ao REAL, dist_ao REAL, c_a_ao REAL, Extent REAL, Binary_status TEXT, 
        h_w_ratio_ao REAL, x_ao INT, y_ao INT, w_ao INT, h_ao INT, 
        o_class INT, c_a_rw REAL, w_rw REAL, h_rw REAL)'''.format(self.table_name))

        self.db.commit()

    def write(self, d_frame):

        img_name = str(SAVE_COUNTER)
        db_arr = self.get_base_params(d_frame.base_objects, img_name)

        self.cur.executemany('''INSERT INTO {}(Img_name, Obj_id,  dist_ao, 
        c_a_ao, Extent, Binary_status, h_w_ratio_ao, x_ao, y_ao, w_ao, 
        h_ao, o_class, c_a_rw, w_rw, h_rw) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''.
                             format(self.table_name), db_arr)

        if len(d_frame.ex_objects) > 0:
            img_name += "_split"
            db_split_arr = self.get_base_params(d_frame.ex_objects, img_name)
            self.cur.executemany('''INSERT INTO {}(Img_name, Obj_id, 
            dist_ao, c_a_ao, Extent, Binary_status, h_w_ratio_ao, x_ao,
            y_ao, w_ao, h_ao, o_class, c_a_rw, w_rw, h_rw) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''.
                                 format(self.table_name), db_split_arr)

        self.db.commit()

    @staticmethod
    def get_base_params(objects, img_name):
        # img_name = str(config.COUNTER).zfill(4)
        db_arr = list()
        if len(objects) > 0:
            for obj in objects:
                db_arr.append(
                    [img_name, obj.obj_id, obj.dist_ao,
                     obj.c_a_ao, obj.extent_ao, str(obj.binary_status),
                     obj.h_w_ratio_ao, obj.x_ao, obj.y_ao, obj.w_ao,
                     obj.h_ao, obj.o_class, obj.c_ao_rw, obj.w_ao_rw, obj.h_ao_rw])

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


color_map = {0: (0, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0), 4: (0, 255, 255)}


def draw_rects(img, objects):
    for obj in objects:
        color = color_map[obj.o_class]

        x, y, w, h = obj.base_rect_ao
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, str(obj.obj_id), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1, cv2.LINE_AA)
        # Put distance value above the rectangle
        cv2.putText(img, str(round(obj.dist_ao, 1)), (x + 5, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 1, cv2.LINE_AA)


def put_obj_status(img, objects):
    cntr = 0.1
    for obj in objects:
        if obj.o_class > 0:
            cv2.putText(img, str(obj.o_class_nm), (int(conf.RES[0] * 0.2), int(conf.RES[1] * cntr)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, str(obj.obj_id), (int(conf.RES[0] * 0.08), int(conf.RES[1] * cntr)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[obj.o_class], 1, cv2.LINE_AA)
            cntr += 0.1


def write_steps(steps, frame, img_name):
    blank_img_left = np.zeros((conf.RES[1], conf.RES[0], 3), np.uint8)
    blank_img_right = np.zeros((conf.RES[1], conf.RES[0], 3), np.uint8)

    for key, img in steps.items():
        steps[key] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Name frames
        cv2.putText(steps[key], key, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Put binary detection status
    cv2.putText(blank_img_right, str(frame.base_frame_status), (int(conf.RES[0] * 0.3), int(conf.RES[1] * 0.5)),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA)

    # Put particular obj status
    put_obj_status(blank_img_left, frame.base_objects)

    draw_rects(steps['resized_orig'], frame.base_objects)

    h_stack1 = np.hstack((steps['mask'], steps['filtered'], steps['filled']))
    h_stack2 = np.hstack((blank_img_left, steps['resized_orig'], blank_img_right))

    out_img = np.vstack((h_stack1, h_stack2))

    cv2.imwrite(os.path.join(conf.OUT_DIR, img_name), out_img)
