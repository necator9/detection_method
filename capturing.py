import cv2
import threading
import queue
import os

import logging
import conf
import global_vars

logger = logging.getLogger('detect.capture')


def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        logger.error("No such path: {}".format(dir_path))
        raise StartAppError

    return dir_path


class StartAppError(Exception):
    def __init__(self):
        Exception.__init__(self, "StartAppError")


class Camera(threading.Thread):
    def __init__(self, orig_img_q, stop_ev):
        super(Camera, self).__init__(name="camera")
        self.stop_event = stop_ev
        self.orig_img_q = orig_img_q
        self.camera = cv2.VideoCapture(conf.DEVICE)  # Initialize the camera capture object

    # Main thread routine
    def run(self):
        logger.info("Camera thread has started...")
        self.cam_setup()

        while not self.stop_event.is_set():
            read_ok, image = self.camera.read()

            if not read_ok:
                logger.error("Capturing failed")
                self.quit()

                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            try:
                self.orig_img_q.put((image, '{}.jpeg'.format(global_vars.COUNTER)))  # , block=True
                # self.orig_img_q.put(image, block=True)
            except queue.Full:
                logger.warning("orig_img_q is full, next iteration")

                continue

            global_vars.COUNTER += 1

    def cam_setup(self):
        # Check on successful camera initialization
        if not self.camera.isOpened():
            logger.error("Cannot initialize camera: {}".format(conf.DEVICE))

            self.stop_event.clear()

        # Initial camera configuration
        self.camera.set(3, conf.RES[0])
        self.camera.set(4, conf.RES[1])
        self.camera.set(5, conf.FPS)

    def quit(self):
        self.camera.release()
        self.stop_event.set()
        logger.info("Exiting the Camera thread...")
