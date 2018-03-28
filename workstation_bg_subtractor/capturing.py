import cv2
import threading
import time
import numpy as np
import glob
import os

import detection_logging
import conf
import global_vars
from extentions import TimeCounter

CAPTURING_LOG = detection_logging.create_log("capturing.log", "CAPTURING THREAD")


class VirtualCamera(threading.Thread):
    def __init__(self, stop_ev):
        super(VirtualCamera, self).__init__(name="virtual_camera")
        self.stop_event = stop_ev
        self.check_dir()

    def run(self):
        i = 0
        CAPTURING_LOG.info("Virtual camera thread has started")
        images_in_dir = (len(glob.glob(os.path.join(conf.IN_DIR, "*.jpeg")))) - 1
        CAPTURING_LOG.info("Files in directory: {}".format(images_in_dir))

        while i < images_in_dir and self.stop_event.is_set():
            if not global_vars.IMG_BUFF.processed:
                time.sleep(0.01)

                continue
            path_to_img = glob.glob(os.path.join(conf.IN_DIR, "img_{}_*.jpeg".format(global_vars.COUNTER)))[0]

            image = cv2.imread(path_to_img)
            img_buff = ImgBuff()
            img_buff.put(image)
            CAPTURING_LOG.info("Image {} has been taken".format(i))
            conf.IMG_BUFF = img_buff
            time.sleep(0.2)
            i += 1

        self.quit()

    def check_dir(self):
        if not os.path.isdir(conf.IN_DIR):
            CAPTURING_LOG.error("INPUT directory does not exists. Path: {}".format(conf.IN_DIR))
            time.sleep(2)
            exit(1)

    def quit(self):
        self.stop_event.clear()


class ImgBuff(object):
    def __init__(self):
        self.image = np.dtype('uint8')
        self.processed = bool()
        self.inserted = bool()

    def get(self):
        self.processed = True

        return self.image

    def put(self, image):
        self.image = image
        self.inserted = True


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
            read_ok, img = self.camera.read()

            if not read_ok:
                CAPTURING_LOG.error("Capturing failed")

                break

            img_buff = ImgBuff()
            img_buff.put(img)

            global_vars.IMG_BUFF = img_buff

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
