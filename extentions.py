import os
import threading
import numpy as np
import cv2
import time

try:
   import queue
except ImportError:
   import Queue as queue

import conf
import logging

logger = logging.getLogger('detect.ext')
SAVE_COUNTER = int()


def blank_fn(*args, **kwargs):
    pass


class Saving(threading.Thread):
    def __init__(self, data_frame_q):
        super(Saving, self).__init__(name="saving")

        self._is_running = bool()
        self.data_frame_q = data_frame_q

        self.check_if_dir_exists()
        self.writer = WriteCsv()

    def run(self):
        logger.info("Starting the Saving thread...")
        self._is_running = True

        while self._is_running:
            self.write()
            # SAVER_LOG.debug("Entry has been written")

        self.finish_writing()

        logger.info("Saving thread has been finished")

    def write(self):
        try:
            data_frame = self.data_frame_q.get(timeout=2)
        except queue.Empty:
            logger.warning("Exception has raised, data_frame_q is empty")

            return 1

        self.writer.write(data_frame)

        global SAVE_COUNTER
        SAVE_COUNTER += 1

    def finish_writing(self):
        logger.info("Finishing writing ...")
        if not self.data_frame_q.empty():
            q_size = self.data_frame_q.qsize()
            for i in range(self.data_frame_q.qsize()):
                logger.debug("{} elements in queue are remaining to write".format(q_size - i))
                self.write()

        logger.warning("{} elements HAVE BEEN NOT WRITTEN".format(self.data_frame_q.qsize()))
        self._is_running = False

    def check_if_dir_exists(self):
        if not os.path.isdir(conf.OUT_DIR):
            os.makedirs(conf.OUT_DIR)
            logger.info("OUTPUT directory does not exists. New folder has been created")

    def quit(self):
        self._is_running = False
        self.writer.quit()

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


class WriteCsv(object):
    def __init__(self):
        self.fd = open('h{}_a{}.csv'.format(conf.HEIGHT, conf.ANGLE), 'a')
        self.fmt = '%d,%d,%.2f,%.2f,%.2f,%.1f,%.2f,%d,%d,%d,%d,%d,%.2f,%d'
        self.fd.write("img,o_num,rw_w,rw_h,rw_ca,rw_z,rw_x,x,y,w,h,ca,o_prob,o_class\n")

    def write(self, data):
        np.savetxt(self.fd, data, fmt=self.fmt)

    def quit(self):
        self.fd.close()


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
            self.watch_log = logger
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


color_map = {0: (0, 0, 0), 1: (0, 255, 0), 2: (255, 204, 33), 3: (0, 255, 255)}


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


def draw_rects_new(img, data_frame):
    for row in data_frame.tolist():
        o_class = row[-1]
        if o_class == 0:
            continue

        o_prob = row[-2]

        o_class_nm = conf.o_class_mapping[o_class]
        x, y, w, h = [int(param) for param in row[7:11]]
        p1 = (x, y)
        p2 = (x + w, y + h)
        color = color_map[o_class]
        x_rw, y_rw, z_rw = row[6], -conf.HEIGHT, row[5]

        cv2.rectangle(img, p1, p2, color, 1)
        cv2.putText(img, '({0:.2f},{1:.2f},{2:.2f})'.format(x_rw, y_rw, z_rw), (x, p2[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
        cv2.putText(img, '{0:.2f} {1}'.format(o_prob, o_class_nm), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1,
                    cv2.LINE_AA)

def draw_tracking(img, objects, prob_q):
    if len(objects) == 0:
        return

    for obj_key in prob_q.keys():
        average_prob = np.mean(prob_q[obj_key])
        row = objects[obj_key]
        x, y, w, h = [int(param) for param in row[7:11]]

        cv2.putText(img, '{0} '.format(obj_key), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 0, 255), 1, cv2.LINE_AA)
        if len(prob_q[obj_key]) == 5:
            cv2.putText(img, '{0:.2f}'.format(average_prob), (x + 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 255), 1, cv2.LINE_AA)

def put_obj_status(img, objects):
    cntr = 0.1
    for obj in objects:
        if obj.o_class > 0:
            cv2.putText(img, str(obj.o_class_nm), (int(conf.RES[0] * 0.2), int(conf.RES[1] * cntr)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, str(obj.obj_id), (int(conf.RES[0] * 0.08), int(conf.RES[1] * cntr)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[obj.o_class], 1, cv2.LINE_AA)
            cntr += 0.1


def write_steps(steps, frame, img_name, objects, prob_q):
    blank_img_left = np.zeros((conf.RES[1], conf.RES[0], 3), np.uint8)
    blank_img_right = np.zeros((conf.RES[1], conf.RES[0], 3), np.uint8)

    for key, img in steps.items():
        steps[key] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Name frames
        cv2.putText(steps[key], key, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Put binary detection status
    # cv2.putText(blank_img_right, str(frame.base_frame_status), (int(conf.RES[0] * 0.3), int(conf.RES[1] * 0.5)),
    #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA)

    # Put particular obj status
    # put_obj_status(blank_img_left, frame)

    draw_rects_new(steps['resized_orig'], frame)
    draw_tracking(steps['resized_orig'], objects, prob_q)

    h_stack1 = np.hstack((steps['mask'], steps['filtered']))
    h_stack2 = np.hstack((steps['filled'], steps['resized_orig']))

    out_img = np.vstack((h_stack1, h_stack2))

    cv2.imwrite(os.path.join(conf.OUT_DIR, img_name), out_img)
