import cv2
import time
import os
import sys
import imutils
import logging.config

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

ramp_frames = 10

def get_image():
    retval, im = camera.read()          # Get a full image out of a VideoCapture object
    return im                           # Returns image in PIL format

logger.info("Start")

start_time = time.time()
camera = cv2.VideoCapture(0)            # Initialize the camera capture object with the cv2.VideoCapture class.

if not camera.isOpened():               # Check on successful camera object initialization
    logger.error("Cannot initialize camera object")
    sys.exit(-1)

# Set resolution of the capture (Logitech C910 supports 640x480 and 1920x1080)
camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

for i in xrange(ramp_frames):           # Get idle ramp frames to achieve better quality of the image
    temp = get_image()

logger.info("Taking image...")
image = get_image()                     # Take the actual image we want to keep

del camera

original = image.copy()                 # Copy original image for further estimation

logger.info("Image shooting takes %s s", round(time.time() - start_time, 3))

start_time = time.time()
image = imutils.resize(image, width=min(400, image.shape[1]))   # Resize image to achieve better performance

hog = cv2.HOGDescriptor()               # Hot descriptor initialization
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

logger.info("Detection process...")
(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
logger.info("Image processing takes: %s s", round((time.time() - start_time), 3))

for (x, y, w, h) in rects:              # Draw rectangles for visual estimation
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

logger.info("Images writing...")
cv2.imwrite("img/processed.jpg", image)  # Writing images
cv2.imwrite("img/original.jpg", original)

# Sync saved images with workstation via rsync command
logger.info("Sync with workstation...")
command_1 = "rsync -avzhe ssh --delete img/ fila@192.168.8.107:~/img_BBB/"
command_2 = "rsync -avzhe ssh --delete logs/ fila@192.168.8.107:~/img_BBB/logs/"
os.system(command_1)
os.system(command_2)



