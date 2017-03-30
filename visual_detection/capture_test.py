# 0.05 s takes image capture = 20 fps

import cv2                              # OpenCV library
import time
import os
import sys
import logging.config
import numpy

command = "rsync -avzhe 'ssh -p 2122' --delete share/ ivan@192.168.8.108:~/share_BBB/"


def get_image():
    retval, im = camera.read()          # Get a full image out of a VideoCapture object
    return im                           # Returns image in PIL format

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

logger.info("Start")
camera = cv2.VideoCapture(0)  # Initialize the camera capture object with the cv2.VideoCapture class.
if not camera.isOpened():  # Check on successful camera object initialization
    logger.error("Cannot initialize camera object")
    os.system(command)
    sys.exit(-1)

# Set resolution of the capture (Logitech C910 supports 640x480 and 1920x1080)
camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 400)
camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 300)

hog = cv2.HOGDescriptor()               # Hot descriptor initialization
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

i = 0
mean_time = []

while i < 40:
    start_time = time.time()
    logger.info("Taking image...")
    image = get_image()                     # Take the actual image we want to keep
    tm = round(time.time() - start_time, 3)
    mean_time.append(tm)
    logger.info("Image shooting takes %s s", tm)

    logger.info("Images writing...")
    cv2.imwrite("./share/img/processed_%s.jpg" % i, image)  # Writing images

    time.sleep(1)
    i += 1

del camera
logger.info("Mean time %s s", numpy.mean(mean_time))

# Sync saved images with workstation via rsync command
logger.info("Sync with workstation...")
os.system(command)







