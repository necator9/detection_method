#!/usr/bin/env python

import threading
import time
import conf
import global_vars


# from train_regression import TrainRegression
import capturing
import detection
import extentions
import detection_logging
try:
   import queue
except ImportError:
   import Queue as queue
import cv2

CV_VERSION = int(cv2.__version__[0])
MAIN_LOGGER = detection_logging.create_log("main.log", "root")

MAIN_LOGGER.info('OpenCV version: {} '.format(cv2.__version__))

def blank_fn(*args, **kwargs):
    pass


def check_cv_version():
    cv_version = cv2.__version__[0]
    if int(cv_version) < 3:
        MAIN_LOGGER.error("The program works only with OpenCV v3.x.x or higher. Current v:{}".format(cv2.__version__))

        return True


def main():
    detection_logging.init_log_thread()

#    if check_cv_version():
#        time.sleep(1)
#        detection_logging.stop_log_thread()

        # exit(1)

    MAIN_LOGGER.info("Program has started")

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
            MAIN_LOGGER.info("{} images captured".format(global_vars.COUNTER))
            time.sleep(1)
    except KeyboardInterrupt:
        MAIN_LOGGER.warning("Keyboard Interrupt, threads are going to stop")

    capturing_thread.quit()
    saver_thread.quit()

    capturing_thread.join()
    detection_thread.join()
    saver_thread.join()


    MAIN_LOGGER.info("Program finished")

    time.sleep(1)
    detection_logging.stop_log_thread()



if __name__ == '__main__':
    main()

