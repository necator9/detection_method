import copy
import logging
import threading
import time

import cv2
from imutils import resize
import config

logger = logging.getLogger(__name__)


class Grabber(threading.Thread):
    def __init__(self, stop_ev):
        super(Grabber, self).__init__(name="Grabber")
        # Thread state status flag
        self.running = False
        self.stop_event = stop_ev
        # Initialize the camera capture object
        self.camera = cv2.VideoCapture(config.DEVICE)

    # Main thread routine
    def run(self):
        logger.info("Grabber started")
        self.running = True
        self.cam_setup()
        while self.running:
            start_time = time.time()
            logger.debug("Taking image...")

            # Getting of an image into img
            read_ok, img = self.camera.read()
            if not read_ok:
                logger.error("Capturing failed")
                break

            config.IMG_BUFF = resize(img, width=config.IMG_WIDTH_SAVE, height=config.IMG_HEIGHT_SAVE)

            processing_t = time.time() - start_time
            config.T_GRABBER.append(processing_t)
            logger.debug("Image shooting takes %s s", processing_t)

    # Camera configuration in accordance to OpenCV version
    def cam_setup(self):
        # Check on successful camera initialization
        if not self.camera.isOpened():
            logger.error("Cannot initialize camera object")
            self.quit()
        # Initial camera configuration
        cv_version = int(cv2.__version__.split(".")[0])
        if cv_version == 3:
            self.camera.set(3, config.IMG_WIDTH)
            self.camera.set(4, config.IMG_HEIGHT)
            self.camera.set(5, config.FPS)
        if cv_version == 2:
            self.camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, config.IMG_WIDTH)
            self.camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, config.IMG_HEIGHT)
            self.camera.set(cv2.cv.CV_CAP_PROP_FPS, config.FPS)

    # Stop and quit the thread operation.
    def quit(self):
        self.running = False
        self.stop_event.clear()
        self.camera.release()
        logger.info("Grabber has quit")


class Detector(threading.Thread):
    def __init__(self, stop_ev):
        super(Detector, self).__init__(name="Detector")
        # Thread state status flag
        self.running = False
        self.stop_event = stop_ev
        self.start_t = time.time()
        self.mog = cv2.createBackgroundSubtractorMOG2()
        self.filtering_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.FILTERED_OBJ_SIZE)

    def run(self):
        self.running = True
        logger.info("Detector started")
        while self.running:
            detection_t = time.time()
            if not self.check_on_buffer():
                time.sleep(1)
                continue
            img = copy.copy(config.IMG_BUFF)
            mask = self.mog.apply(img)
            filtered = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.filtering_kernel)
            filled = cv2.dilate(filtered, None, iterations=8)
            _, cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if self.detect(cnts):
                config.MOTION_STATUS = True
                logging.info("Motion detected")
            else:
                config.MOTION_STATUS = False

            if config.IMG_SAVE:
                self.save_image(cnts, img, self.start_t)
            time.sleep(0.3)
            processing_t = time.time() - detection_t
            config.T_DETECTOR.append(processing_t)
            logger.debug("Image processing takes: %s s", processing_t)

    @staticmethod
    def check_on_buffer():
        if len(config.IMG_BUFF) == 0:
            logging.warning("No available images in buffer")
            return False
        else:
            return True

    @staticmethod
    def detect(cnts):
        for arr in cnts:
            if cv2.contourArea(arr) > config.DETECTED_OBJ_SIZE:
                return True
            else:
                return False

    @staticmethod
    def save_image(cnts, img, start_t):
        for arr in cnts:
            (x, y, w, h) = cv2.boundingRect(arr)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, str(cv2.contourArea(arr)), (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                        cv2.LINE_AA)
            current_time = round(time.time() - start_t, 2)
            cv2.imwrite(config.BBB_SYNC_DIRECTORY + "img/img_%s_%s.jpeg" % (current_time, cv2.contourArea(arr)), img)

    def quit(self):
        self.running = False
        self.stop_event.clear()
        logger.info("Detector has quit")















