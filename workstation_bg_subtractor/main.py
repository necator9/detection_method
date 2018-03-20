#!/usr/bin/env python

import glob

import threading
import time
import conf
import os

import detection
import extentions
import detection_logging
import Queue


logger = detection_logging.create_log("smartlight.log", "root")


def check_if_dir_exists():
    if not os.path.isdir(conf.IN_DIR):
        logger.error("INPUT directory does not exists. Path: %s" % conf.IN_DIR)
        exit(1)

    if not os.path.isdir(conf.OUT_DIR):
        os.makedirs(conf.OUT_DIR)
        logger.info("OUTPUT directory does not exists. New folder has been created")


def blank_fn(*args, **kwargs):
    pass


def main():

    logger.info("Program has started")

    check_if_dir_exists()

    logger.info("INPUT directory: %s" % conf.IN_DIR)
    logger.info("OUTPUT directory: %s" % conf.OUT_DIR)

    conf.IMG_IN_DIR = (len(glob.glob(os.path.join(conf.IN_DIR, "*.jpeg")))) - 1
    logger.info("Files in directory: %s" % conf.IMG_IN_DIR)

    stop_event = threading.Event()
    stop_event.set()

    detection_logging.init_log_thread()

    data_frame_q = Queue.Queue(maxsize=5000)
    draw_frame_q = Queue.Queue(maxsize=1000)

    detection_thread = detection.Detection(stop_event, data_frame_q, draw_frame_q)
    saver_thread = extentions.Saving(data_frame_q, draw_frame_q)

    if not (conf.WRITE_TO_DB or conf.WRITE_TO_PICKLE or conf.SAVE_IMG):
        saver_thread.start = blank_fn
        saver_thread.quit = blank_fn
        saver_thread.join = blank_fn
        data_frame_q.put_nowait = blank_fn
        data_frame_q.get = blank_fn

    if not conf.SAVE_IMG:
        draw_frame_q.put_nowait = blank_fn
        draw_frame_q.get = blank_fn

    detection_thread.start()
    saver_thread.start()

    try:
        while stop_event.is_set():
            logger.info("Progress - %s/%s images" % (conf.COUNTER, conf.IMG_IN_DIR))
            time.sleep(1)
    except KeyboardInterrupt:
        logger.warning("Keyboard Interrupt, threads are going to stop")

    detection_thread.quit()
    saver_thread.quit()

    detection_thread.join()
    saver_thread.join()

    detection_logging.stop_log_thread()

    logger.info("Program finished")


if __name__ == '__main__':
    main()

