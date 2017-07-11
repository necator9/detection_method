#!/usr/bin/env python

import logging.config
import os
import threading
import time

import config
import detection
import extensions


logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

logger.info("Program started")

extensions.parse()

if not extensions.check_dir():
    exit(1)

if config.IMG_SAVE:
    extensions.clear_dir()

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

time.sleep(0.05)
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

if config.SYNC:
    os.system(config.COMMAND)
