#!/usr/bin/env python

import threading
import logging.config
from config import T_GRABBER, T_DETECTOR, COMMAND, IMG_SAVE, PATH_TO_SHARE
import detection
import time
import os
from numpy import mean
import display


def check_dir():
    img_path = PATH_TO_SHARE + "img/"
    if not os.path.isdir(PATH_TO_SHARE):
        logger.error("No such directory: %s" % PATH_TO_SHARE)
        return False
    if not os.path.isdir(PATH_TO_SHARE + "img/"):
        logger.error("No such directory: %s" % img_path)
        return False
    else:
        return True


def clear_img_dir():
    img_path = PATH_TO_SHARE + "img/"
    files_n = len([name for name in os.listdir(img_path) if os.path.isfile(name)])
    if files_n > 0:
        os.system("rm " + img_path + "*")
        logger.debug("Previous files are removed in dir: %s" % img_path)
    else:
        logger.debug("No images detected in dir: %s" % img_path)

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

logger.info("Program started")

if not check_dir():
    exit(1)

if IMG_SAVE:
    clear_img_dir()

stop_event = threading.Event()
stop_event.set()

grabberThr = detection.Grabber(stop_event)
detectorThr = detection.Detector(stop_event)
displayThr = display.Display(stop_event)

grabberThr.start()
detectorThr.start()
displayThr.start()

try:
    while stop_event.is_set():
        time.sleep(1)
except KeyboardInterrupt:
    logger.warning("Keyboard Interrupt, threads are going to stop")

grabberThr.quit()
detectorThr.quit()
displayThr.quit()



logger.info("Program finished")

if IMG_SAVE:
    os.system(COMMAND)
