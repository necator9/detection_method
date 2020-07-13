#!/usr/bin/env python3.7

import threading
import time
import conf
import global_vars
import os

import capturing
import detection
import logging
import queue

import cv2


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

formatter = logging.Formatter('%(asctime)s %(threadName)s - %(message)s')
file_handler.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(file_handler)

logger.info('OpenCV version: {} '.format(cv2.__version__))


def main():
    logger.debug("Program has started")

    stop_event = threading.Event()
    orig_img_q = queue.Queue(maxsize=1)

    detection_thread = detection.Detection(stop_event, orig_img_q)
    capturing_thread = capturing.Camera(orig_img_q, stop_event)

    capturing_thread.start()
    detection_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(1)

    except KeyboardInterrupt:
        stop_event.set()
        logger.warning('Keyboard interrupt, stopping the threads')

    capturing_thread.join()
    detection_thread.join()

    logger.debug("Program finished")


if __name__ == '__main__':
    main()

