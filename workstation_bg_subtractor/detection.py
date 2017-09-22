import copy
import logging
import threading
import time

import cv2
from imutils import resize
import config
import glob
import pyexiv2
import os

logger = logging.getLogger(__name__)


class Detector(threading.Thread):
    def __init__(self, stop_ev):
        super(Detector, self).__init__(name="Detector")
        self.running = False
        self.stop_event = stop_ev

        self.mog = cv2.createBackgroundSubtractorMOG2()
        self.filtering_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.F_KERNEL_SIZE)
        self.counter = 0
        self.img = []

    # Main thread routine
    def run(self):
        logger.info("Grabber started")
        self.running = True

        while (self.counter < ((len(glob.glob(os.path.join(config.IMG_IN_DIR, "*.jpeg")))) - 1)) and self.running:

            logger.debug("Taking image...")

            self.img = cv2.imread(glob.glob(os.path.join(config.IMG_IN_DIR, "img_%s_*.jpeg" % self.counter))[0])
            self.img = resize(self.img, width=config.PROC_IMG_RES[0], height=config.PROC_IMG_RES[1])
            mask = self.mog.apply(self.img)
            filtered = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.filtering_kernel)
            filled = cv2.dilate(filtered, None, iterations=8)
            _, cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            det_res = self.detect(cnts)

            if config.MOTION_STATUS:
                self.draw_on_img(det_res)

            cv2.imshow('image', self.img)
            cv2.waitKey(1)

            self.save_image(det_res)

            self.counter += 1
            time.sleep(0.2)

        self.quit()

    @staticmethod
    def detect(cnts):
        detect_res = []
        for arr in cnts:
            contour_area = cv2.contourArea(arr)
            if contour_area > config.D_OBJ_SIZE:
                coord = cv2.boundingRect(arr)
                detect_res.append([contour_area, coord])
                config.MOTION_STATUS = True
                logging.info("Motion detected")
                logger.info(str(detect_res))
            else:
                config.MOTION_STATUS = False
        return detect_res

    def draw_on_img(self, det_res):
        for i in range(len(det_res)):
            x, y, w, h = det_res[i][1]
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(self.img, str(det_res[i][0]), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 255), 1, cv2.LINE_AA)
            # cv2.putText(img, "w = %s" % str(w), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            #             (0,50, 200), 1, cv2.LINE_AA)
            # cv2.putText(img, "h = %s" % str(h), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            #             (0, 50, 200), 1, cv2.LINE_AA)
            # cv2.putText(img, "h/w = %s" % str(float(h)/w), (100, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            #             (0, 50, 200), 1, cv2.LINE_AA)
            # cv2.drawContours(img, cnts, -1, (0,255,0), 3)

    def save_image(self, det_res):
        # Save JPEG with proper name
        img_name = "img_%s.jpeg" % self.counter
        path = os.path.join(config.IMG_OUT_DIR, img_name)
        cv2.imwrite(path, self.img)

        # Write exif to saved JPEG
        metadata = pyexiv2.ImageMetadata(path)
        metadata.read()
        metadata['Exif.Image.Software'] = pyexiv2.ExifTag('Exif.Image.Software', 'OpenCV-3.2.0-dev, pyexiv2')
        metadata['Exif.Image.Artist'] = pyexiv2.ExifTag('Exif.Image.Artist', 'Ivan Matveev')
        metadata['Exif.Photo.UserComment'] = pyexiv2.ExifTag('Exif.Photo.UserComment', "status:%s data:%s" %
                                                             (config.MOTION_STATUS, det_res))

        metadata['Exif.Photo.UserComment'] = pyexiv2.ExifTag('Exif.Photo.UserComment', 'status:%s' %
                                                             config.MOTION_STATUS)
        metadata.write()

    # Stop and quit the thread operation.
    def quit(self):
        self.running = False
        self.stop_event.clear()
        logger.info("Grabber has quit")








