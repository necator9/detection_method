#!/usr/bin/env python

from module_lib import Module
import threading
import time
import logging.config


logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

stop_ev = threading.Event()
stop_ev.set()

module = Module(stop_ev, pir=True, control=True)
module.set_statistic_lvl(pir_pol=True)

st_time = time.time()
logger.info("Detection module started")
module.start()

try:
    while time.time() - st_time < 20:
        time.sleep(1)
        print module.get_status()
except KeyboardInterrupt:
    logger.warning("Keyboard Interrupt, threads are going to stop")
stop_ev.clear()

module.join()
logger.info("Detection module finished")
