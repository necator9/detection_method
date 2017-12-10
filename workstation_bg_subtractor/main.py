#!/usr/bin/env python

import glob
import os
import logging.config
import threading
import time
import config

import detection


logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

logger.info("Program has started")
logger.info("INPUT directory: %s" % config.IN_DIR)
logger.info("OUT directory: %s" % config.OUT_DIR)

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


