#!/usr/bin/env python

import glob
import os
import logging.config
import threading
import time
import config
import os

import detection


def check_if_dir_exists():
    if not os.path.isdir(config.IN_DIR):
        logger.error("INPUT directory does not exists. Path: %s" % config.IN_DIR)
        exit(1)

    if not os.path.isdir(config.OUT_DIR):
        os.makedirs(config.OUT_DIR)
        logger.info("OUTPUT directory does not exists. New folder has been created")


logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

logger.info("Program has started")

check_if_dir_exists()

logger.info("INPUT directory: %s" % config.IN_DIR)
logger.info("OUTPUT directory: %s" % config.OUT_DIR)

config.IMG_IN_DIR = (len(glob.glob(os.path.join(config.IN_DIR, "*.jpeg")))) - 1
logger.info("Files in directory: %s" % config.IMG_IN_DIR)

stop_event = threading.Event()
stop_event.set()

detectionThr = detection.Detector(stop_event)

detectionThr.start()

try:
    while stop_event.is_set():
        logger.info("Progress - %s/%s images" % (config.COUNTER, config.IMG_IN_DIR))
        time.sleep(1)
except KeyboardInterrupt:
    logger.warning("Keyboard Interrupt, threads are going to stop")

detectionThr.quit()


logger.info("Program finished")


