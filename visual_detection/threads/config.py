import logging
import cv2
import os
import sys
import time
import numpy
import Queue


img_buff = [] # shared var which grabber uses for writing and detector for reading
command = "rsync -avzhe 'ssh -p 2122' --delete ../share/ ivan@192.168.100.119:~/share_rpi/"
mean_t_detector = []
mean_t_grabber = []
