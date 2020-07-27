#!/usr/bin/env python3.7

import logging
import queue
import cv2
import threading
import os

import capturing
import detection
import conf


def check_if_dir_exists():
    if not os.path.isdir(conf.OUT_DIR):
        os.makedirs(conf.OUT_DIR)
        print("Output directory does not exists. New folder has been created.")


check_if_dir_exists()

# Set up logging,
logger = logging.getLogger('detect')
logger.setLevel(conf.LOG_LEVEL)
file_handler = logging.FileHandler(os.path.join(conf.OUT_DIR, 'detection.log'))
ch = logging.StreamHandler()

formatter = logging.Formatter('%(levelname)s %(asctime)s %(threadName)s - %(message)s')
file_handler.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(file_handler)

logger.debug("Program started")
logger.info('OpenCV version: {} '.format(cv2.__version__))

stop_event = threading.Event()
orig_img_q = queue.Queue(maxsize=1)

detection_routine = detection.Detection(stop_event, orig_img_q)  # Not a thread!
capturing_thread = capturing.Camera(orig_img_q, stop_event)

capturing_thread.start()

try:
    detection_routine.run()

except KeyboardInterrupt:
    logger.warning('Interrupt received, stopping the threads')
    stop_event.set()

finally:
    capturing_thread.join()
    logger.debug("Program finished")

