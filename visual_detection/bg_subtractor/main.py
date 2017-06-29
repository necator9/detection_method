#!/usr/bin/env python

import threading
import logging.config
import config
import detection
import time
import os
import glob
import extensions


def check_dir():
    img_path = config.PATH_TO_SHARE + "img/"
    if not os.path.isdir(config.PATH_TO_SHARE):
        logger.error("No such directory: %s" % config.PATH_TO_SHARE)
        return False
    if not os.path.isdir(config.PATH_TO_SHARE + "img/"):
        logger.error("No such directory: %s" % img_path)
        return False
    else:
        return True


def clear_img_dir():
    img_path = '../share/img/*'
    files_n = len(glob.glob(img_path))
    if files_n > 0:
        os.system("rm " + img_path)
        logger.info("Previous files are removed in dir: %s" % img_path)
    else:
        logger.info("No images detected in dir: %s" % img_path)

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

logger.info("Program started")

if not check_dir():
    exit(1)

if config.IMG_SAVE:
    clear_img_dir()

stop_event = threading.Event()
stop_event.set()

grabberThr = detection.Grabber(stop_event)
detectorThr = detection.Detector(stop_event)
if config.UI:
    displayThr = extensions.Display(stop_event)

grabberThr.start()
detectorThr.start()
if config.UI:
    displayThr.start()

try:
    while stop_event.is_set():
        time.sleep(1)
except KeyboardInterrupt:
    logger.warning("Keyboard Interrupt, threads are going to stop")

grabberThr.quit()
detectorThr.quit()
if config.UI:
    displayThr.quit()

logger.info("Program finished")

if config.IMG_SAVE:
    os.system(config.COMMAND)
