#!/usr/bin/env python

import threading
import time
import logging.config
import cv2
import os
import sys
import numpy

image = []  # global var used by both threads

command = "rsync -avzhe 'ssh -p 2122' --delete ../share/ ivan@192.168.100.119:~/share_rpi/"

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)


def capture(stop_ev, pool_sema):
    logger.info("Start")
    global image
    mean_time = []

    camera = cv2.VideoCapture(0)  # Initialize the camera capture object with the cv2.VideoCapture class.
    # camera = cv2.VideoCapture("/home/pi/out.m4v")  # Initialize the camera capture object with the cv2.VideoCapture class.

    if not camera.isOpened():  # Check on successful camera object initialization
        logger.error("Cannot initialize camera object")
        os.system(command)
        sys.exit(-1)

    # Set resolution of the capture #352x288
    camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320)
    camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240)

    hog = cv2.HOGDescriptor()               # Hot descriptor initialization
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    i = 0
    while stop_ev.is_set():
        pool_sema.acquire()
        start_time = time.time()
        logger.info("Taking image...")
        ret, image = camera.read()                     # Take the actual image we want to keep
        tm = round(time.time() - start_time, 3)
        mean_time.append(tm)
        logger.info("Image shooting takes %s s", tm)
        pool_sema.release()
        i += 1
    del camera
    logger.info("Mean time %s s", numpy.mean(mean_time))



def detect(stop_ev, pool_sema):
    logger.info("Start")
    global image
    mean_time = []

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    i = 0
    while stop_ev.is_set():
        pool_sema.acquire()

        logger.info("Detection process... %s" % i)
        start_time = time.time()

        (rects, weights) = hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.06)
        tm = round((time.time() - start_time), 3)
        logger.info("Image processing takes: %s s", tm)
        mean_time.append(tm)

        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        #cv2.imwrite("../share/img/single_%s.jpg" % i, image)
        cv2.imshow("my_window", image)
        cv2.waitKey(1)
        i += 1
        pool_sema.release()

    logger.info("Mean time: %s s", numpy.mean(mean_time))

pool_sema = threading.BoundedSemaphore(1)
stop_ev = threading.Event()
stop_ev.set()
captureThread = threading.Thread(target=capture, name="CaptureThread", args=(stop_ev, pool_sema))
detectThread = threading.Thread(target=detect, name="DetectThread", args=(stop_ev, pool_sema))

logger.info("Program started")

captureThread.start()
time.sleep(0.5)
detectThread.start()

i = 0
try:
    while True:
    # while i < 20:
        time.sleep(1)
        i += 1
except KeyboardInterrupt:
    logger.warning("Keyboard Interrupt, threads are going to stop")
stop_ev.clear()

captureThread.join()
detectThread.join()

# Sync saved images with workstation via rsync command
logger.info("Sync with workstation...")
os.system(command)
logger.info("Program finished")
