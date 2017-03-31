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


#camera.set(cv2.cv.CV_CAP_PROP_FOURCC, cv2.cv.CV_FOURCC('M', 'J', 'P', 'G') );
    #camera.set(cv2.CV_CAP_PROP_FPS, 15);
camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240)

hog = cv2.HOGDescriptor()               # Hot descriptor initialization
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

i = 0
mean_time = []

try:
    while i < 1:
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
except KeyboardInterrupt:
    logger.warning("Keyboard Interrupt, threads are going to stop")

cv2.imshow('my webcam', image)
cv2.waitKey(0)

#del camera
logger.info("Mean time %s s", numpy.mean(mean_time))

# Sync saved images with workstation via rsync command
logger.info("Sync with workstation...")
os.system(command)






