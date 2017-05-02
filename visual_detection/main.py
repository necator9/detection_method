import cv2  # OpenCV library
import time
import sys
import logging.config

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)


def get_image():
    retval, im = camera.read()  # Get a full image out of a VideoCapture object
    return im  # Returns image in PIL format

ramp_frames = 30  # Amount of idle shots before capturing of image

logger.info("Start")

camera = cv2.VideoCapture(0)  # Initialize the camera capture object with the cv2.VideoCapture class.

if not camera.isOpened():  # Check on successful camera object initialization
    logger.error("Cannot initialize camera object")
    sys.exit(-1)

# Set resolution of the capture (Logitech C910 supports 640x480 and 1920x1080)
camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.cv.CV_CAP_PROP_FPS, 5)


for i in xrange(ramp_frames):  # Get idle ramp frames to achieve better quality of the image
    temp = get_image()

hog = cv2.HOGDescriptor()  # Hot descriptor initialization
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

try:
    i = 0
    while True:
        # capturing
        start_time = time.time()
        img_buff = get_image()  # Take the actual image we want to keep
        logger.info("Image shooting takes %s s", round(time.time() - start_time, 3))

        # # detection
        # start_time = time.time()
        # # image = imutils.resize(image, width=min(400, image.shape[1]))   # Resize image to achieve better performance
        # logger.info("Detection process...")
        # (rects, weights) = hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.06)
        # logger.info("Image processing takes: %s s", round((time.time() - start_time), 3))
        #
        # for (x, y, w, h) in rects:  # Draw rectangles for visual estimation
        #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # logger.info("Images writing...")
        # cv2.imwrite("./share/img/processed_%s.jpg" % i, image)   # Writing images

        #time.sleep(2)
        # i += 1

        # cv2.imshow("my_window", image)
        # cv2.waitKey(1)

        num_frames = 120;

        print "Capturing {0} frames".format(num_frames)

        # Start time
        start = time.time()

        # Grab a few frames
        for i in xrange(0, num_frames):
            ret, frame = camera.read()

        # End time
        end = time.time()

        # Time elapsed
        seconds = end - start
        print "Time taken : {0} seconds".format(seconds)

        # Calculate frames per second
        fps = num_frames / seconds
        print "Estimated frames per second : {0}".format(fps)

except KeyboardInterrupt:
    logger.warning("Keyboard interrupt")
    camera.release()
    sys.exit(0)
