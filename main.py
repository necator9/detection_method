#!/usr/bin/env python3.7

import threading
import time
import conf
import global_vars
import os


import capturing
import detection
import extentions
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


def blank_fn(*args, **kwargs):
    pass


def main():
    logger.debug("Program has started")

    stop_event = threading.Event()
    # stop_event.set()

    data_frame_q = queue.Queue(maxsize=50)
    orig_img_q = queue.Queue(maxsize=1)

    detection_thread = detection.Detection(stop_event, data_frame_q, orig_img_q)
    saver_thread = extentions.Saving(data_frame_q)

    try:
        capturing_thread = capturing.Camera(orig_img_q, stop_event)

        if not conf.WRITE_TO_CSV:
            saver_thread.start = blank_fn
            saver_thread.quit = blank_fn
            saver_thread.join = blank_fn
            data_frame_q.put = blank_fn
            data_frame_q.get = blank_fn

    except capturing.StartAppError:
        stop_event.set()
        exit(1)


    saver_thread.start()
    capturing_thread.start()
    detection_thread.start()

    try:
        while not stop_event.is_set():
            logger.info("{} images captured".format(global_vars.COUNTER))
            time.sleep(10)
    except KeyboardInterrupt:
        logger.warning("Keyboard Interrupt, threads are going to stop")
        stop_event.set()

    capturing_thread.quit()
    saver_thread.quit()

    capturing_thread.join()
    detection_thread.join()
    saver_thread.join()

    logger.debug("Program finished")

    time.sleep(1)


if __name__ == '__main__':
    main()

