import cv2
import threading
import Queue
import time
import glob
import os
import numpy as np

import detection_logging
import conf
import global_vars
from extentions import TimeCounter

CAPTURING_LOG = detection_logging.create_log("capturing.log", "CAPTURING THREAD")


class VirtualCamera(threading.Thread):
    def __init__(self, orig_img_q, stop_ev, h_w_ratio=(480, 640)):
        super(VirtualCamera, self).__init__(name="virtual_camera")

        # self.dir_path = '../../rendering/render_v3/{}/{}'.format(round(height, 1), angle)
        self.dir_path = conf.IN_DIR
        self._check_dir(self.dir_path)
        img_paths = glob.glob(os.path.join(self.dir_path, '*.jpg'))
        print img_paths
        # self.img_names_float = sorted([float(img_name.split('/')[-1][:-4]) for img_name in img_paths])

        self.img_names_float = sorted([float(os.path.split(img_name)[1][:-4]) for img_name in img_paths])
        print self.img_names_float

        CAPTURING_LOG.info("Files in directory: {}".format(len(self.img_names_float)))

        self.iterator = 0
        self.blank_iterator = 0
        self.blank_img = np.zeros((h_w_ratio[0], h_w_ratio[1]))

        self.stop_event = stop_ev
        self.orig_img_q = orig_img_q

    def run(self):
        CAPTURING_LOG.info("Virtual camera thread has started")

        while self.stop_event.is_set():
            image, _ = self.get_image()

            if image is None:
                break

            CAPTURING_LOG.debug("Image {} has been taken".format(global_vars.COUNTER))

            try:
                self.orig_img_q.put(image, timeout=10)
            except Queue.Full:
                CAPTURING_LOG.warning("orig_img_q is full, next iteration")
                continue

            global_vars.COUNTER += 1

        self.quit()

    def get_image(self, add_blank=False):
        if add_blank and self.blank_iterator < 100:
            self.blank_iterator += 1

            return self.blank_img, np.nan

        else:
            name = os.path.join(self.dir_path, '{}.jpg'.format(self.img_names_float[self.iterator]))
            image = cv2.imread(name, 0)
            self.iterator += 1

            if self.iterator >= len(self.img_names_float):
                return None, None

            return image, round(self.img_names_float[self.iterator], 1)

    def _check_dir(self, dir_path):
        if not os.path.isdir(dir_path):
            CAPTURING_LOG.error("INPUT directory does not exists. Path: {}".format(dir_path))
            time.sleep(2)
            self.stop_event.clear()

    def quit(self):
        self.stop_event.clear()


class Camera(threading.Thread):
    def __init__(self, orig_img_q, stop_ev):
        super(Camera, self).__init__(name="camera")
        self.stop_event = stop_ev
        self.orig_img_q = orig_img_q
        self.camera = cv2.VideoCapture(conf.DEVICE)  # Initialize the camera capture object
        self.timer = TimeCounter("camera_timer")

    # Main thread routine
    def run(self):
        CAPTURING_LOG.info("Camera thread has started...")
        self.cam_setup()

        while self.stop_event.is_set():
            self.timer.note_time()
            read_ok, image = self.camera.read()

            if not read_ok:
                CAPTURING_LOG.error("Capturing failed")

                break

            try:
                self.orig_img_q.put(image, timeout=2)
            except Queue.Full:
                CAPTURING_LOG.warning("orig_img_q is full, next iteration")

                continue

            global_vars.COUNTER += 1

            self.timer.get_time()

    def cam_setup(self):
        # Check on successful camera initialization
        if not self.camera.isOpened():
            CAPTURING_LOG.error("Cannot initialize camera: {}".format(conf.DEVICE))

            self.stop_event.clear()

        # Initial camera configuration
        self.camera.set(3, conf.RESOLUTION[0])
        self.camera.set(4, conf.RESOLUTION[1])
        self.camera.set(5, conf.FPS)

    def quit(self):
        self.camera.release()
        CAPTURING_LOG.info("Exiting the Camera thread...")
