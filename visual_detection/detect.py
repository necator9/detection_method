# import the necessary packages
#from __future__ import print_function
#from imutils.object_detection import non_max_suppression
#from imutils import paths
import numpy as np
import imutils
import cv2
import os
import time

print ("start...")
st_time = time.time()
imagePath = "/root/2.jpg"

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

image = cv2.imread(imagePath)
image = imutils.resize(image, width=min(400, image.shape[1]))


# detect people in the image
(rects, weights) = hog.detectMultiScale(image, winStride=(8, 8),
                                        padding=(8, 8), scale=1.05)

print (time.time() - st_time)
# draw the original bounding boxes
for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imwrite("img/kek.jpg", image)
cv2.waitKey(0)
command = "rsync -avzhe ssh --delete img/ fila@192.168.8.107:~/img_BBB/"
os.system(command)
