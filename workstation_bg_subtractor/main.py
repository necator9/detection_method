#!/usr/bin/env python

import glob

import threading
import time
import conf

import capturing
import detection
import extentions
import detection_logging
import Queue
import cv2


MAIN_LOGGER = detection_logging.create_log("main.log", "root")


def blank_fn(*args, **kwargs):
    pass


def check_cv_version():
    cv_version = cv2.__version__[0]
    if cv_version is not "3":
        MAIN_LOGGER.error("The program works only with OpenCV v3.x.x or higher. Current v:{}".format(cv2.__version__))

        return True


def main():
    detection_logging.init_log_thread()

    if check_cv_version():
        time.sleep(1)
        detection_logging.stop_log_thread()

        exit(1)

    MAIN_LOGGER.info("Program has started")

    stop_event = threading.Event()
    stop_event.set()

    data_frame_q = Queue.Queue(maxsize=5000)
    draw_frame_q = Queue.Queue(maxsize=1000)

    detection_thread = detection.Detection(stop_event, data_frame_q, draw_frame_q)
    saver_thread = extentions.Saving(data_frame_q, draw_frame_q)

    if conf.VIRTUAL_CAMERA:
        capturing_thread = capturing.VirtualCamera(stop_event)
    else:
        capturing_thread = capturing.Camera(stop_event)

    if not (conf.WRITE_TO_DB or conf.WRITE_TO_PICKLE or conf.SAVE_IMG):
        saver_thread.start = blank_fn
        saver_thread.quit = blank_fn
        saver_thread.join = blank_fn
        data_frame_q.put_nowait = blank_fn
        data_frame_q.get = blank_fn

    if not conf.SAVE_IMG:
        draw_frame_q.put_nowait = blank_fn
        draw_frame_q.get = blank_fn

    saver_thread.start()
    capturing_thread.start()
    detection_thread.start()

    try:
        while stop_event.is_set():
            MAIN_LOGGER.info("Processed {} images".format(conf.COUNTER))
            time.sleep(1)
    except KeyboardInterrupt:
        MAIN_LOGGER.warning("Keyboard Interrupt, threads are going to stop")

    detection_thread.quit()
    saver_thread.quit()

    detection_thread.join()
    saver_thread.join()

    detection_logging.stop_log_thread()

    MAIN_LOGGER.info("Program finished")


if __name__ == '__main__':
    main()

