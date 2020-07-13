import cv2
import threading
import queue

import logging
import conf

logger = logging.getLogger('detect.capture')


class Camera(threading.Thread):
    def __init__(self, orig_img_q, stop_ev):
        super(Camera, self).__init__(name="Capturing")
        self.stop_event = stop_ev
        self.orig_img_q = orig_img_q

        self.camera = cv2.VideoCapture(conf.DEVICE)  # Initialize the camera capture object
        self.cam_setup()

    # Main thread routine
    def run(self):
        logger.info("Camera thread has started...")

        while not self.stop_event.is_set():
            read_ok, image = self.camera.read()

            if not read_ok:
                logger.error("Capturing failed")
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            try:
                self.orig_img_q.put(image, timeout=0.5)
            except queue.Full:
                logger.warning("Capturing queue is full, next iteration")
                continue

        self.quit()

    def cam_setup(self):
        # Check on successful camera initialization
        if not self.camera.isOpened():
            logger.error("Cannot initialize the camera: {}".format(conf.DEVICE))
            self.quit()

        else:
            # Initial camera configuration
            self.camera.set(3, conf.RES[0])
            self.camera.set(4, conf.RES[1])
            self.camera.set(5, conf.FPS)

    def quit(self):
        self.camera.release()
        self.stop_event.set()
        logger.info("Exiting the capturing thread")
