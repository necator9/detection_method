# detection while winStride=(8, 8), padding=(8, 8), scale=1.05 takes 2.56 s

import numpy
import cv2
import os
import time
import logging.config

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

logger.info("Start")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

mean_time = []
i = 0
while i < 40:

    imagePath = "share/img/processed_%s.jpg" % i
    img_buff = cv2.imread(imagePath)

    logger.info("Detection process... %s" % i)
    start_time = time.time()
    (rects, weights) = hog.detectMultiScale(img_buff, winStride=(8, 8), padding=(8, 8), scale=1.1)
    tm = round((time.time() - start_time), 3)
    logger.info("Image processing takes: %s s", tm)
    mean_time.append(tm)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(img_buff, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imwrite("share/img/single_%s.jpg" % i, img_buff)
    i += 1

logger.info("Mean time: %s s", numpy.mean(tm))
command = "rsync -avzhe 'ssh -p 2122' --delete share/ ivan@192.168.8.108:~/share_BBB/"
os.system(command)
