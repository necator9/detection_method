import logging
import cv2
import os
import sys
import time
import numpy
import Queue

# Shared vars
img_buff = []                   # Grabber writes in the var/detector reads from the var

# Camera parameters
dev = 0                         # Number of a device (camera) to use (/dev/video<number>)
# dev = "/home/pi/out.m4v"
# dev = "/home/ivan/out.m4v"
width = 320                     # Width of an image to capture
height = 240                    # Height of an image to capture
fps = 7                         # Capturing frequency (frames per second)

# Detector parameters
winStr = (8, 8)                 # Step size in both the x and y location of the sliding window
pad = (8, 8)                    # Number of pixels in x and y directions in which window is padded
scale = 1.06                    # Influence on the number of levels in the image pyramid.

# Tracker parameters
shift_min = 0                   # Low threshold in distance between pixels
shift_max = 25                  # High threshold in distance between pixels
deq_len = 5                     # Size of buffer containing coordinated for analysis of movement

# Statistic parameters
st_window = 300                 # Window size for timing calculation
t_detector = []                 # Times of detection duration are stored here
t_grabber = []                  # Times of capturing duration are stored here
detections_amount = 0           # Amount of detection during execution (the value is incremented)




