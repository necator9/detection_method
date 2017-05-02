#!/usr/bin/env python

import threading
import time
import logging.config
import cv2
import os
import sys
import numpy

img_buff = []  # shared var which grabber uses for writing and detector for reading

command = "rsync -avzhe 'ssh -p 2122' --delete ../share/ ivan@192.168.100.119:~/share_rpi/"

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)


def cam_setup(camera, width, height, fps):
    camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.cv.CV_CAP_PROP_FPS, fps)


def grabber(stop_ev):
    logger.info("Grabber start")
    global img_buff
    mean_time = []

    camera = cv2.VideoCapture(0)  # Initialize the camera capture object
    # camera = cv2.VideoCapture("/home/pi/out.m4v")
    if not camera.isOpened():  # Check on successful camera initialization
        logger.error("Cannot initialize camera object")
        os.system(command)
        stop_ev.clear()
        sys.exit(-1)

    cam_setup(camera, 320, 240, 7)  # Initial camera configuration: function(object, width, height, fps)

    while stop_ev.is_set():
        start_time = time.time()
        logger.debug("Taking image...")
        ret, img_buff = camera.read()
        tm = round(time.time() - start_time, 3)
        mean_time.append(tm)
        logger.debug("Image shooting takes %s s", tm)
    camera.release()
    logger.info("Mean capturing time %s s", numpy.mean(mean_time))


def detector(stop_ev):
    logger.info("Detector start")
    global img_buff
    mean_time = []

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while stop_ev.is_set():
        logger.debug("Detection process...")
        start_time = time.time()
        while stop_ev.is_set():
            if len(img_buff) == 0:
                logging.warning("No available images in buffer")
                time.sleep(1)
            else:
                img_proc = img_buff
                break

        (rects, weights) = hog.detectMultiScale(img_proc, winStride=(8, 8), padding=(8, 8), scale=1.06)
        tm = round((time.time() - start_time), 3)
        logger.debug("Image processing takes: %s s", tm)
        mean_time.append(tm)

        # draw the bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(img_proc, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("my_window", img_proc)
        cv2.waitKey(1)

    logger.info("Mean detection time: %s s", numpy.mean(mean_time))

stop_ev = threading.Event()
stop_ev.set()
grabberThr = threading.Thread(target=grabber, name="Grabber", args=(stop_ev,))
detectorThr = threading.Thread(target=detector, name="Detector", args=(stop_ev,))

logger.info("Program started")

grabberThr.start()
detectorThr.start()

try:
    while stop_ev.is_set():
        time.sleep(1)
except KeyboardInterrupt:
    logger.warning("Keyboard Interrupt, threads are going to stop")

stop_ev.clear()

grabberThr.join()
detectorThr.join()

# Sync saved images with workstation via rsync command
logger.info("Sync with workstation...")
os.system(command)
logger.info("Program finished")
