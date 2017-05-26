import logging
import cv2
import time
from imutils import resize
import config
import threading

logger = logging.getLogger(__name__)


class Grabber(threading.Thread):
    def __init__(self, stop_ev):
        super(Grabber, self).__init__(name="Grabber")
        # Thread state status flag
        self.running = False
        self.stop_event = stop_ev
        # Initialize the camera capture object
        self.camera = cv2.VideoCapture(config.DEV)

    # Main thread routine
    def run(self):
        logger.info("Grabber started")
        self.running = True
        self.cam_setup()
        while self.running:
            start_time = time.time()
            logger.debug("Taking image...")
            # Getting of an image into img
            ret, img = self.camera.read()
            config.IMG_BUFF = resize(img, width=100)
            if len(config.T_GRABBER) < config.ST_WINDOW:
                config.T_GRABBER.append(time.time() - start_time)
            logger.debug("Image shooting takes %s s", time.time() - start_time)

    # Stop and quit the thread operation.
    def quit(self):
        self.running = False
        self.stop_event.clear()
        self.camera.release()
        logger.info("Grabber finished")

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
        while self.running and self.stop_event.is_set():
            detection_t = time.time()
            if self.check_on_buffer():
                img = config.IMG_BUFF
            else:
                time.sleep(1)
                continue
            mask = self.mog.apply(img)
            filtered = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.filtering_kernel)
            filled = cv2.dilate(filtered, None, iterations=8)
            _, cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if self.detect(cnts):
                logging.info("Motion detected")

            if config.IMG_SAVE:
                self.save_image(cnts, img, self.start_t)
            processing_t = round(time.time() - detection_t, 4)

            if len(config.T_DETECTOR) < 300:
                config.T_DETECTOR.append(processing_t)
            logger.debug("Image processing takes: %s s", processing_t)

    def quit(self):
        self.running = False
        logger.info("Detector finished")

    @staticmethod
    def check_on_buffer():
        if len(config.IMG_BUFF) == 0:
            logging.warning("No available images in buffer")
            time.sleep(1)
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
            current_time = round(time.time() - start_t, 3)
            cv2.imwrite(config.PATH_TO_SHARE + "img/img_%s.jpeg" % current_time, img)


















