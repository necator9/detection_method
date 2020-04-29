#!/usr/bin/env python3.7

import threading
import time
import conf
import global_vars


# from train_regression import TrainRegression
import capturing
import detection
import extentions
import logging
import queue

import cv2


# Set up logging,
logger = logging.getLogger('detect')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('detection.log')
ch = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s %(threadName)s - %(message)s')
file_handler.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(file_handler)

logger.info('OpenCV version: {} '.format(cv2.__version__))


def blank_fn(*args, **kwargs):
    pass


def check_cv_version():
    cv_version = cv2.__version__[0]
    if int(cv_version) < 3:
        logger.error("The program works only with OpenCV v3.x.x or higher. Current v:{}".format(cv2.__version__))

        return True


def main():

#    if check_cv_version():
#        time.sleep(1)
#        detection_logging.stop_log_thread()

        # exit(1)

    logger.debug("Program has started")

    stop_event = threading.Event()
    stop_event.set()

    data_frame_q = queue.Queue(maxsize=50)
    orig_img_q = queue.Queue(maxsize=1)

    detection_thread = detection.Detection(stop_event, data_frame_q, orig_img_q)
    saver_thread = extentions.Saving(data_frame_q)

    try:
        if conf.VIRTUAL_CAMERA:
            capturing_thread = capturing.VirtualCamera(orig_img_q, stop_event)
        else:
            capturing_thread = capturing.Camera(orig_img_q, stop_event)

        if not (conf.WRITE_TO_DB or conf.WRITE_TO_PICKLE):
            saver_thread.start = blank_fn
            saver_thread.quit = blank_fn
            saver_thread.join = blank_fn
            data_frame_q.put_nowait = blank_fn
            data_frame_q.get = blank_fn


    except capturing.StartAppError:
        stop_event.clear()
        exit(1)


    saver_thread.start()
    capturing_thread.start()
    detection_thread.start()

    try:
        while stop_event.is_set():
            logger.info("{} images captured".format(global_vars.COUNTER))
            time.sleep(1)
    except KeyboardInterrupt:
        logger.warning("Keyboard Interrupt, threads are going to stop")

    capturing_thread.quit()
    saver_thread.quit()

    capturing_thread.join()
    detection_thread.join()
    saver_thread.join()


    logger.debug("Program finished")

    time.sleep(1)


if __name__ == '__main__':
    main()

