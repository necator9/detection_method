#!/usr/bin/env python

import threading
import logging.config
from config import T_GRABBER, T_DETECTOR, COMMAND, IMG_SAVE, PATH_TO_SHARE
import detection
import time
import os
from numpy import mean


logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

logger.info("Program started")

if IMG_SAVE:
    os.system("rm ../share/img/*")

stop_event = threading.Event()
stop_event.set()

grabberThr = detection.Grabber(stop_event)
detectorThr = detection.Detector(stop_event)

grabberThr.start()
detectorThr.start()

try:
    while stop_event.is_set():
        time.sleep(1)
except KeyboardInterrupt:
    logger.warning("Keyboard Interrupt, threads are going to stop")

grabberThr.quit()
detectorThr.quit()

# Timing calculations
T_GRABBER = round(mean(T_GRABBER), 3)
T_DETECTOR = round(mean(T_DETECTOR), 3)
mean_it_time = T_GRABBER + T_DETECTOR

logger.info("Mean capturing time %s s", T_GRABBER)
logger.info("Mean detection time: %s s", T_DETECTOR)
logger.info("Mean iteration time %s s" % mean_it_time)
logger.info("Mean FPS %s" % round(1/mean_it_time, 3))

logger.info("Program finished")

if IMG_SAVE:
    os.system(COMMAND)
