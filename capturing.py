# Created by Ivan Matveev at 01.05.20
# E-mail: ivan.matveev@hs-anhalt.de

# Capturing images from a camera in separate thread

import logging
import cv2
import threading
import queue
import time

logger = logging.getLogger('detect.capture')


class Camera(threading.Thread):
    def __init__(self, orig_img_q, stop_ev, config):
        super(Camera, self).__init__(name="Capturing")
        self.stop_event = stop_ev
        self.orig_img_q = orig_img_q
        self.config = config

        self.camera = cv2.VideoCapture(self.config['device'])  # Initialize the camera capture object
        self.cam_setup()

    # Main thread routine
    def run(self):
        logger.info("Camera thread has started...")

        while not self.stop_event.is_set():
            
            # Control FPS manually, because sometimes the FPS assignment performed by CV2 does not work 
            time.sleep(1.0 / self.config['fps'])

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
            msg = "Cannot initialize the camera: {}".format(self.config['device'])
            logger.error(msg)
            self.quit()
            raise Exception(msg)

        else:
            # Initial camera configuration
            self.camera.set(3, self.config['resolution'][0])
            self.camera.set(4, self.config['resolution'][1])
            # The microcontroller controlls cameras does not support FPS assignment
            #self.camera.set(5, self.config['fps'])

    def quit(self):
        self.camera.release()
        self.stop_event.set()
        logger.info("Exiting the capturing thread")
