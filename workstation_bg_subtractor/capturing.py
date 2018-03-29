import cv2
import threading
import Queue
import time
import numpy as np
import glob
import os

import detection_logging
import conf
import global_vars
from extentions import TimeCounter

CAPTURING_LOG = detection_logging.create_log("capturing.log", "CAPTURING THREAD")
IMAGE_BUFFER = list()


class ImageBuffer(threading.Thread):
    def __init__(self, img_q, stop_ev):
        super(ImageBuffer, self).__init__(name="image_queue")
        self.orig_img_q = img_q
        self.stop_event = stop_ev

    def run(self):
        while self.stop_event.is_set():
            if len(IMAGE_BUFFER) > 0:
                try:
                    self.orig_img_q.put(IMAGE_BUFFER, timeout=2)
                except Queue.Full:
                    CAPTURING_LOG.warning("orig_img_q is full, next iteration")

                    continue


class VirtualCamera(threading.Thread):
    def __init__(self, orig_img_q, stop_ev):
        super(VirtualCamera, self).__init__(name="virtual_camera")
        self.stop_event = stop_ev
        self.orig_img_q = orig_img_q
        self.__check_dir()

    def run(self):
        i = 0
        CAPTURING_LOG.info("Virtual camera thread has started")
        images_in_dir = (len(glob.glob(os.path.join(conf.IN_DIR, "*.jpeg")))) - 1
        CAPTURING_LOG.info("Files in directory: {}".format(images_in_dir))

        while i < images_in_dir and self.stop_event.is_set():
            path_to_img = glob.glob(os.path.join(conf.IN_DIR, "img_{}_*.jpeg".format(i)))[0]

            image = cv2.imread(path_to_img)
            CAPTURING_LOG.debug("Image {} has been taken".format(i))

            try:
                self.orig_img_q.put(image, timeout=2)
            except Queue.Full:
                CAPTURING_LOG.warning("orig_img_q is full, next iteration")

                continue

            i += 1

        self.quit()

    def __check_dir(self):
        if not os.path.isdir(conf.IN_DIR):
            CAPTURING_LOG.error("INPUT directory does not exists. Path: {}".format(conf.IN_DIR))
            time.sleep(2)
            self.stop_event.clear()

    def quit(self):
        self.stop_event.clear()


class Camera(threading.Thread):
    def __init__(self, stop_ev):
        super(Camera, self).__init__(name="camera")
        self.stop_event = stop_ev
        self.camera = cv2.VideoCapture(conf.IN_DEVICE)  # Initialize the camera capture object
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

            global IMAGE_BUFFER
            IMAGE_BUFFER = image

            self.timer.get_time()

    def cam_setup(self):
        # Check on successful camera initialization
        if not self.camera.isOpened():
            CAPTURING_LOG.error("Cannot initialize camera: {}".format(conf.IN_DEVICE))

            self.stop_event.clear()

        # Initial camera configuration
        self.camera.set(3, conf.ORIG_IMG_RES[0])
        self.camera.set(4, conf.ORIG_IMG_RES[1])
        self.camera.set(5, conf.FPS)

    def quit(self):
        self.camera.release()
        CAPTURING_LOG.info("Exiting the Camera thread...")
