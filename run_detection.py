#!/usr/bin/env python3

# Created by Ivan Matveev at 01.05.20
# E-mail: ivan.matveev@hs-anhalt.de

# The main thread which starts capturing and detection routines.

import cv2
import yaml

import logging
from logging.handlers import RotatingFileHandler

import queue
import threading
import os
import argparse

import capturing
import detection


def check_if_dir_exists(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        print("Output directory does not exist, new folder created.")


parser = argparse.ArgumentParser(description='Run the lightweight detection algorithm')
parser.add_argument('-p', '--path', action='store',
                    help="path to the configuration file (default: ./configs/config.yml)",
                    default='./configs/config.yml')
args = parser.parse_args()

config = yaml.safe_load(open(args.path))

out_dir = config['out_dir']
check_if_dir_exists(out_dir)

# Set up logging,
logger = logging.getLogger('detect')
logger.setLevel(config['log_level'])
file_handler = RotatingFileHandler(os.path.join(out_dir, 'detection.log'), mode='a', maxBytes=5 * 1024 * 1024,
                                   backupCount=3)
ch = logging.StreamHandler()

formatter = logging.Formatter('%(levelname)s %(asctime)s %(threadName)s - %(message)s')
file_handler.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(file_handler)

logger.debug("Program started")
logger.debug('OpenCV version: {} '.format(cv2.__version__))

stop_event = threading.Event()
orig_img_q = queue.Queue(maxsize=1)

try:
    detection_routine = detection.Detection(stop_event, orig_img_q, config)  # Not a thread!
    capturing_thread = capturing.Camera(orig_img_q, stop_event, config)

    capturing_thread.start()
    detection_routine.run()
    capturing_thread.join()

except Exception as crash_err:
    crash_msg = '\n{0}\nAPP CRASH. Error msg:\n{1}\n{0}'.format(100 * '-', crash_err)
    logger.exception(crash_msg)
    stop_event.set()
    exit(1)

except KeyboardInterrupt:
    logger.warning('Interrupt received, stopping the threads')
    stop_event.set()

finally:
    logger.debug("Program finished")
