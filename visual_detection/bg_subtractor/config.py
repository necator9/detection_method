from collections import deque

# Shared vars
IMG_BUFF = []                   # Grabber writes in the var/detector reads from the var

# Camera parameters
DEV = 0                         # Number of a device (camera) to use (/dev/video<number>)
# dev = "/home/pi/out.m4v"
# dev = "/home/ivan/out.m4v"
IMG_WIDTH = 320                 # Width of an image to capture
IMG_HEIGHT = 240                # Height of an image to capture
FPS = 7                         # Capturing frequency (frames per second)

# Detector parameters
FILTERED_OBJ_SIZE = (3, 3)    # Size of elliptical filtering kernel in pixels
DETECTED_OBJ_SIZE = 30         # Detection threshold. Minimal obj size to be detected

# Interactive parameters
IMG_SAVE = True                # Save frames where motion is detected
UI = True                       # Display user interface

# Internal parameters
COMMAND = "rsync -avzhe 'ssh -p 2122' --delete ../share/ ivan@192.168.100.119:~/share_BBB/"
PATH_TO_SHARE = "../share/"

# Global variables for internal usage
MOTION_STATUS = False            # Initial status of motion
T_DETECTOR = deque(maxlen=30)    # Times of detection duration are stored here
T_GRABBER = deque(maxlen=30)     # Times of capturing duration are stored here
# Avoid operations with empty deque
T_DETECTOR.append(0)
T_GRABBER.append(0)




