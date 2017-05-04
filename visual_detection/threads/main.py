#!/usr/bin/env python

import threading
import logging.config

from config import *
import config
import grabber
import detector
import tracker

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

q = Queue.Queue()

stop_ev = threading.Event()
stop_ev.set()
grabberThr = threading.Thread(target=grabber.capture, name="Grabber", args=(stop_ev,))
detectorThr = threading.Thread(target=detector.detect, name="Detector", args=(stop_ev, q))
trackerThr = threading.Thread(target=tracker.track, name="Tracker", args=(stop_ev, q,))

logger.info("Program started")

grabberThr.start()
detectorThr.start()
trackerThr.start()

try:
    while stop_ev.is_set():
        time.sleep(1)
except KeyboardInterrupt:
    logger.warning("Keyboard Interrupt, threads are going to stop")

stop_ev.clear()

grabberThr.join()
detectorThr.join()
trackerThr.join()

# Timing calculations
config.mean_t_grabber = round(numpy.mean(config.mean_t_grabber), 3)
config.mean_t_detector = round(numpy.mean(config.mean_t_detector), 3)
mean_it_time = config.mean_t_grabber + config.mean_t_detector

logger.info("Mean capturing time %s s", config.mean_t_grabber)
logger.info("Mean detection time: %s s", config.mean_t_detector)
logger.info("Mean iteration time %s s" % mean_it_time)
logger.info("Mean FPS %s" % round(1/mean_it_time, 3))


# Sync saved images with workstation via rsync command
# logger.info("Sync with workstation...")
# os.system(command)
# logger.info("Program finished")
