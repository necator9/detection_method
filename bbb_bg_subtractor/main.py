#!/usr/bin/env python
import argparse
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

parser = argparse.ArgumentParser(description='Motion detection setup for Beaglebone Black',
                                 epilog='Hochschule Anhalt, 2017')

parser.add_argument('--ui', action='store_false',
                    help='Disable command line interface')
parser.add_argument('--save', action='store_false',
                    help='Disable image saving')
parser.add_argument('--sync', action='store_false',
                    help='Disable sync with workstation')
parser.add_argument('--dev', metavar='/dev/videoX', default=config.DEVICE,
                    help='Input camera device')
args = parser.parse_args()

if not args.ui:
    config.UI = False
if not args.save:
    config.IMG_SAVE = False
if not args.sync:
    config.SYNC_DIR = False
if args.dev:
    config.DEVICE = args.dev



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

if config.SYNC_DIR:
    os.system(config.COMMAND)
