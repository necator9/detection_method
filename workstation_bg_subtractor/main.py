#!/usr/bin/env python

import logging.config
import threading
import time

import detection


logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

logger.info("Program started")

stop_event = threading.Event()
stop_event.set()

grabberThr = detection.Detector(stop_event)

grabberThr.start()


time.sleep(0.05)
try:
    while stop_event.is_set():
        time.sleep(1)
except KeyboardInterrupt:
    logger.warning("Keyboard Interrupt, threads are going to stop")

grabberThr.quit()


logger.info("Program finished")


