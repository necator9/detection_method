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
parser.add_argument('--dev', metavar='/path/', default=config.DEVICE,
                    help='Input camera device or file')
parser.add_argument('--ores', metavar='int', nargs=2, type=int, default=config.ORIG_IMG_RES,
                    help='Resolution of an image to capture from camera (width and height)')
parser.add_argument('--pres', metavar='int', nargs=2, type=int, default=config.PROC_IMG_RES,
                    help='Resolution of an image for processing (resize, process and save)')
parser.add_argument('--ofps', metavar='int', type=int, default=config.FPS,
                    help='FPS of a camera')
parser.add_argument('--fkernel', metavar='int', nargs=2, type=int, default=config.F_KERNEL_SIZE,
                    help='Size of elliptical filtering kernel in pixels')
parser.add_argument('--osize', metavar='int', type=int, default=config.D_OBJ_SIZE,
                    help='Object size to be detected')
parser.add_argument('--bsync', metavar='/path/', default=config.BBB_SYNC_DIR,
                    help='Path to synchronizing directory on Beaglebone Black')
parser.add_argument('--bsave', metavar='/path/', default=config.BBB_IMG_DIR,
                    help='Path to image saving directory on Beaglebone Black')
parser.add_argument('--wsync', metavar='/path/', default=config.W_SYNC_DIR,
                    help='Path to synchronizing directory on workstation')
parser.add_argument('--wuip', metavar='username@ip', default=config.W_USER_IP,
                    help='Workstation username@ip')
parser.add_argument('--wport', metavar='int', default=config.W_PORT,
                    help='Workstation ssh port')

args = parser.parse_args()

if not args.ui:
    config.UI = False
if not args.save:
    config.IMG_SAVE = False
if not args.sync:
    config.SYNC = False
config.DEVICE = args.dev
config.ORIG_IMG_RES = args.ores
config.PROC_IMG_RES = args.pres
config.FPS = args.ofps
config.F_KERNEL_SIZE = args.fkernel
config.D_OBJ_SIZE = args.osize
config.BBB_SYNC_DIR = args.bsync
config.BBB_IMG_DIR = args.bsave
config.W_SYNC_DIR = args.wsync
config.W_USER_IP = args.wuip
config.W_PORT = args.wport

W_PORT = "2122"                                     #






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
