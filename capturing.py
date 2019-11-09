import cv2
import threading
import Queue
import glob
import os

import detection_logging
import conf
import global_vars
from extentions import TimeCounter

CAPTURING_LOG = detection_logging.create_log("capturing.log", "CAPTURING THREAD")


def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        CAPTURING_LOG.error("No such path: {}".format(dir_path))
        raise StartAppError

    return dir_path


class StartAppError(Exception):
    def __init__(self):
        Exception.__init__(self, "StartAppError")


class VirtualCamera(threading.Thread):
    def __init__(self, orig_img_q, stop_ev):
        super(VirtualCamera, self).__init__(name="virtual_camera")
        self.dir_path = check_dir(conf.IN_DIR)

        self.img_paths = glob.glob(os.path.join(self.dir_path, '*.jpeg'))
        img_names_digits = [int(os.path.split(img_name)[1].split('.')[0]) for img_name in self.img_paths]
        img_names_digits, self.img_paths = zip(*sorted(zip(img_names_digits, self.img_paths)))

        self.stop_event = stop_ev
        self.orig_img_q = orig_img_q

    def run(self):
        CAPTURING_LOG.info("Files to process: {}".format(len(self.img_paths)))
        CAPTURING_LOG.debug("Virtual camera thread has started")

        for img_path in self.img_paths:
            image = cv2.imread(img_path) # , 0
            self.orig_img_q.put(image, block=True)

            global_vars.COUNTER += 1

            if not self.stop_event.is_set():
                break

        self.quit()

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
