#!/usr/bin/env python

import threading
import logging.config

from config import *
import config
import grabber
import detector

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

logger.info("Program started")

q = Queue.Queue()

stop_ev = threading.Event()
stop_ev.set()

grabberThr = threading.Thread(target=grabber.capture, name="Grabber", args=(stop_ev,))
detectorThr = threading.Thread(target=detector.detect, name="Detector", args=(stop_ev, ))

grabberThr.start()
detectorThr.start()

try:
    while stop_ev.is_set():
        time.sleep(1)
except KeyboardInterrupt:
    logger.warning("Keyboard Interrupt, threads are going to stop")

stop_ev.clear()

grabberThr.join()
detectorThr.join()

# Timing calculations
config.t_grabber = round(numpy.mean(config.t_grabber), 3)
config.t_detector = round(numpy.mean(config.t_detector), 3)
mean_it_time = config.t_grabber + config.t_detector

logger.info("Mean capturing time %s s", config.t_grabber)
logger.info("Mean detection time: %s s", config.t_detector)
logger.info("Mean iteration time %s s" % mean_it_time)
logger.info("Mean FPS %s" % round(1/mean_it_time, 3))

logger.info("Program finished")
