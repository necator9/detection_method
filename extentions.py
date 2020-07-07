import os
import threading
import numpy as np
import cv2
import queue

import conf
import logging

logger = logging.getLogger('detect.ext')
SAVE_COUNTER = int()


class Saving(threading.Thread):
    def __init__(self, data_frame_q):
        super(Saving, self).__init__(name="saving")

        self._is_running = bool()
        self.data_frame_q = data_frame_q
        self.writer = WriteCsv()

    def run(self):
        logger.info("Starting the Saving thread...")
        self._is_running = True

        while self._is_running:
            self.write()

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
        self.fd = open(os.path.join(conf.OUT_DIR, 'h{}_a{}.csv'.format(conf.HEIGHT, conf.ANGLE)), 'a')
        self.fmt = '%d,%d,%.2f,%.2f,%.2f,%.1f,%.2f,%d,%d,%d,%d,%d,%.2f,%d'
        self.fd.write("img,o_num,rw_w,rw_h,rw_ca,rw_z,rw_x,x,y,w,h,ca,o_prob,o_class\n")

    def write(self, data):
        np.savetxt(self.fd, data, fmt=self.fmt)

    def quit(self):
        self.fd.close()


color_map = {0: (0, 0, 0), 1: (0, 255, 0), 2: (255, 204, 33), 3: (0, 255, 255)}


def draw_rects_new(img, data_frame, padding):
    for row in data_frame.tolist():
        o_class = row[-1]
        # if o_class == 0:
        #     continue

        o_prob = row[-2]

        o_class_nm = conf.o_class_mapping[o_class]
        x, y, w, h = [int(param) for param in row[7:11]]
        x += padding
        y += padding
        p1 = (x, y)
        p2 = (x + w, y + h)
        color = color_map[o_class]
        x_rw, y_rw, z_rw = row[6], -conf.HEIGHT, row[5]

        cv2.rectangle(img, p1, p2, color, 1)
        cv2.putText(img, '({0:.2f},{1:.2f},{2:.2f})'.format(x_rw, y_rw, z_rw), (x, p2[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
        cv2.putText(img, '{0:.2f} {1}'.format(o_prob, o_class_nm), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1,
                    cv2.LINE_AA)


def draw_tracking(img, objects, prob_q, padding):
    if len(objects) == 0:
        return

    for obj_key in prob_q.keys():
        average_prob = np.mean(prob_q[obj_key])
        row = objects[obj_key]
        x, y, w, h = [int(param) for param in row[7:11]]
        x += padding
        y += padding

        cv2.putText(img, '{0} '.format(obj_key), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 0, 255), 1, cv2.LINE_AA)
        if len(prob_q[obj_key]) == 3:
            cv2.putText(img, '{0:.2f}'.format(average_prob), (x + 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 255), 1, cv2.LINE_AA)


def add_padding(image, paddind_size):
    h_pad = np.zeros((paddind_size, conf.RES[0]), np.uint8)
    stack1 = np.vstack((h_pad, image, h_pad))

    v_pad = np.zeros((stack1.shape[0], paddind_size), np.uint8)
    stack2 = np.hstack((v_pad, stack1, v_pad))

    return stack2


def write_steps(steps, frame, img_name, objects, prob_q):

    for key, img in steps.items():
        steps[key] = cv2.undistort(img, conf.intrinsic_orig, conf.dist, None, conf.intrinsic_target)

    padding = 30
    for key, img in steps.items():
        steps[key] = add_padding(img, padding)

    for key, img in steps.items():
        steps[key] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Name frames
    for key, img in steps.items():
        cv2.putText(steps[key], key, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    draw_rects_new(steps['resized_orig'], frame, padding)
    draw_tracking(steps['resized_orig'], objects, prob_q, padding)

    h_stack1 = np.hstack((steps['mask'], steps['filtered']))
    h_stack2 = np.hstack((steps['filled'], steps['resized_orig']))

    out_img = np.vstack((h_stack1, h_stack2))

    cv2.imwrite(os.path.join(conf.OUT_DIR, img_name), out_img)
