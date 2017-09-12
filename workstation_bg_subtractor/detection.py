import copy
import logging
import threading
import time

import cv2
from imutils import resize
import config
import glob

logger = logging.getLogger(__name__)


class Detector(threading.Thread):
    def __init__(self, stop_ev):
        super(Detector, self).__init__(name="Detector")
        # Thread state status flag
        self.running = False
        self.stop_event = stop_ev

        self.mog = cv2.createBackgroundSubtractorMOG2()
        self.filtering_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.F_KERNEL_SIZE)
        self.counter = 0

    # Main thread routine
    def run(self):
        logger.info("Grabber started")
        self.running = True
        print self.counter
        print len(glob.glob("/home/ivan/test_ir/img/*.jpeg"))
        while (self.counter < ((len(glob.glob("/home/ivan/test_ir/share/img/*.jpeg"))) - 1)) and self.running:

            start_time = time.time()
            logger.debug("Taking image...")

            image = cv2.imread(glob.glob("/home/ivan/test_ir/share/img/img_%s_*.jpeg" % self.counter)[0])
            img = copy.copy(image)
            img = resize(img, width=config.PROC_IMG_RES[0], height=config.PROC_IMG_RES[1])
            mask = self.mog.apply(img)
            filtered = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.filtering_kernel)
            filled = cv2.dilate(filtered, None, iterations=8)
            _, cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if self.detect(cnts):
                config.MOTION_STATUS = True
                logging.info("Motion detected")
                for arr in cnts:
                    (x, y, w, h) = cv2.boundingRect(arr)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, str(cv2.contourArea(arr)), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (0, 0, 255),
                                1,
                                cv2.LINE_AA)
            else:
                config.MOTION_STATUS = False

            processing_t = time.time() - start_time
            logger.debug("Image shooting takes %s s", processing_t)

            cv2.imshow('image', img)
            cv2.waitKey(1)
            # cv2.imwrite("/home/ivan/test_ir/img_%s.jpeg" % self.counter, img)

            self.counter += 1
            time.sleep(0.2)
            print self.counter

        self.quit()

    # Stop and quit the thread operation.
    def quit(self):
        self.running = False
        self.stop_event.clear()
        logger.info("Grabber has quit")

    @staticmethod
    def detect(cnts):
        for arr in cnts:
            if cv2.contourArea(arr) > config.D_OBJ_SIZE:
                return True
            else:
                return False






