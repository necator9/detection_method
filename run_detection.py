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
import sys
import signal

import capturing
import detection


class ServiceExit(Exception):
    """
    Custom exception which is used to trigger the clean exit
    of all running threads and the main program.
    """
    pass


def service_shutdown(signum, frame):
    print('Caught signal %d' % signum)
    raise ServiceExit


def check_if_dir_exists(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        print("Output directory does not exist, new folder created: {}".format(dir_path))


parser = argparse.ArgumentParser(description='Run the lightweight detection algorithm')
parser.add_argument('config', action='store', help="path to the configuration file")
parser.add_argument('-c', '--clf', action='store',
                    help="override path to the pickled classifier file given in config")

args = parser.parse_args()

config = yaml.safe_load(open(args.config))

if args.clf:
    config['clf'] = args.clf  # Override config value by passed argument

out_dir = config['out_dir']
check_if_dir_exists(out_dir)

# Set up logging
logger = logging.getLogger('detect')
logger.setLevel(config['log_level'])
file_handler = RotatingFileHandler(os.path.join(out_dir, 'detection.log'), mode='a', maxBytes=5 * 1024 * 1024,
                                   backupCount=3)
ch = logging.StreamHandler()

formatter = logging.Formatter('%(levelname)s %(asctime)s %(threadName)s - %(message)s')
file_handler.setFormatter(formatter)
ch.setFormatter(formatter)
ch.setLevel(logging.WARNING)

logger.addHandler(ch)
logger.addHandler(file_handler)

# Register the signal handlers
signal.signal(signal.SIGTERM, service_shutdown)
signal.signal(signal.SIGINT, service_shutdown)

logger.info("Program started")
logger.debug('OpenCV version: {} '.format(cv2.__version__))

stop_event = threading.Event()
orig_img_q = queue.Queue(maxsize=1)

capturing_thread = None

try:
    detection_routine = detection.Detection(stop_event, orig_img_q, config)  # Not a thread!
    capturing_thread = capturing.Camera(orig_img_q, stop_event, config)

    capturing_thread.start()
    detection_routine.run()

except ServiceExit:
    # Terminate the running threads.
    # Set the shutdown flag on each thread to trigger a clean shutdown of each thread.
    stop_event.set()

except Exception as crash_err:
    crash_msg = '\n{0}\nAPP CRASH. Error msg:\n{1}\n{0}'.format(100 * '-', crash_err)
    logger.exception(crash_msg)
    stop_event.set()
    sys.stderr.write(crash_msg)

    if capturing_thread:
        capturing_thread.join()

    sys.exit(1)

finally:
    if capturing_thread:
        capturing_thread.join()
    logger.info("Program finished")
